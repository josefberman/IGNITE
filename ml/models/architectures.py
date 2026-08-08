"""Graph Transformer-style parent encoder and hierarchical modification planner (PyTorch)."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem


ATOM_LIST = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]
# H, B, C, N, O, F, Si, P, S, Cl, Br, I

HYBRIDIZATION_LIST = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]

RING_SIZES = (3, 4, 5, 6, 7, 8)

# element one-hot (+ unknown) + hybridization (+ other) + scalars + ring-size flags
ATOM_FEAT_DIM = (
    len(ATOM_LIST)
    + 1
    + len(HYBRIDIZATION_LIST)
    + 1
    + 12
    + len(RING_SIZES)
)


def _atom_features(atom: Chem.Atom) -> List[float]:
    z = atom.GetAtomicNum()
    feats: List[float] = [1.0 if z == a else 0.0 for a in ATOM_LIST]
    feats.append(0.0 if z in ATOM_LIST else 1.0)

    hyb = atom.GetHybridization()
    feats.extend(1.0 if hyb == h else 0.0 for h in HYBRIDIZATION_LIST)
    feats.append(0.0 if hyb in HYBRIDIZATION_LIST else 1.0)

    feats.extend(
        [
            float(atom.GetTotalDegree()),
            float(atom.GetFormalCharge()),
            float(atom.GetTotalValence()),
            float(atom.GetValence(Chem.ValenceType.EXPLICIT)),
            float(atom.GetValence(Chem.ValenceType.IMPLICIT)),
            float(atom.GetTotalNumHs()),
            float(atom.GetNumRadicalElectrons()),
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
            float(atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED),
            float(atom.GetMass()) / 100.0,
            float(z) / 100.0,
        ]
    )
    feats.extend(float(atom.IsInRingSize(s)) for s in RING_SIZES)
    assert len(feats) == ATOM_FEAT_DIM
    return feats


def mol_to_graph(smiles: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    atom_feats = [_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_feats, dtype=torch.float32)
    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])
    if not edges:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return x, edge_index


DEFAULT_HIDDEN = 64
DEFAULT_LAYERS = 3
DEFAULT_HEADS = 4
DEFAULT_DROPOUT = 0.15


class BondMessagePass(nn.Module):
    """One round of bond-neighbor mean aggregation before self-attention."""

    def __init__(self, hidden: int):
        super().__init__()
        self.self_lin = nn.Linear(hidden, hidden)
        self.nei_lin = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h: torch.Tensor, edge_index: Optional[torch.Tensor]) -> torch.Tensor:
        # h: [B, N, H]
        if edge_index is None or edge_index.numel() == 0:
            return self.norm(F.gelu(self.self_lin(h)))
        B, N, H = h.shape
        if B != 1:
            raise ValueError("BondMessagePass currently supports batch size 1")
        h0 = h.squeeze(0)
        src, dst = edge_index[0], edge_index[1]
        agg = torch.zeros(N, H, device=h.device, dtype=h.dtype)
        deg = torch.zeros(N, 1, device=h.device, dtype=h.dtype)
        agg.index_add_(0, dst, h0[src])
        deg.index_add_(0, dst, torch.ones(len(dst), 1, device=h.device, dtype=h.dtype))
        agg = agg / deg.clamp_min(1.0)
        mixed = self.self_lin(h0) + self.nei_lin(agg)
        return self.norm(F.gelu(mixed)).unsqueeze(0)


class AttentionPool(nn.Module):
    """Learnable query attention over atom embeddings."""

    def __init__(self, hidden: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden) * 0.02)

    def forward(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # h: [B, N, H] -> [B, H]
        q = self.query.unsqueeze(0).unsqueeze(0).expand(h.size(0), 1, -1)
        scores = torch.bmm(q, h.transpose(1, 2)) / math.sqrt(h.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), -1e9)
        weights = F.softmax(scores, dim=-1)
        return torch.bmm(weights, h).squeeze(1)


class GraphTransformerEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int = ATOM_FEAT_DIM,
        hidden: int = DEFAULT_HIDDEN,
        layers: int = DEFAULT_LAYERS,
        heads: int = DEFAULT_HEADS,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError(f"hidden ({hidden}) must be divisible by heads ({heads})")
        self.hidden = hidden
        self.input = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden))
        self.bond_mp = BondMessagePass(hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            batch_first=True,
            dropout=dropout,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.pool = AttentionPool(hidden)
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        *,
        return_atom_emb: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, N, F]
        h = self.input(x)
        h = self.bond_mp(h, edge_index)
        h = self.encoder(h, src_key_padding_mask=mask)
        atom_emb = self.out(h)
        graph_emb = self.pool(atom_emb, mask)
        if return_atom_emb:
            return graph_emb, atom_emb
        return graph_emb


class TargetDeltaEncoder(nn.Module):
    def __init__(self, in_dim: int = 4, hidden: int = DEFAULT_HIDDEN, dropout: float = DEFAULT_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        return self.net(delta)


class ModificationPlanner(nn.Module):
    """Hierarchical: family → site → reagent (family-conditioned)."""

    def __init__(
        self,
        n_families: int,
        n_reagents: int,
        hidden: int = DEFAULT_HIDDEN,
        layers: int = DEFAULT_LAYERS,
        heads: int = DEFAULT_HEADS,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()
        self.hidden = hidden
        self.n_families = n_families
        self.n_reagents = n_reagents
        enc_kw = dict(hidden=hidden, layers=layers, heads=heads, dropout=dropout)
        self.parent_enc = GraphTransformerEncoder(**enc_kw)
        self.delta_enc = TargetDeltaEncoder(hidden=hidden, dropout=dropout)
        self.fuse = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.family_head = nn.Linear(hidden, n_families)
        self.reagent_head = nn.Sequential(
            nn.Linear(hidden + n_families, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_reagents),
        )
        self.site_head = nn.Linear(hidden, 1)
        self.site_cross = nn.MultiheadAttention(
            hidden, num_heads=min(heads, hidden // 8), batch_first=True, dropout=dropout
        )

    def encode(
        self,
        atom_x: torch.Tensor,
        delta: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # atom_x: [1, N, F] -> z [1, H], parent_z [1, H], atom_emb [1, N, H]
        parent_z, atom_emb = self.parent_enc(atom_x, edge_index, return_atom_emb=True)
        delta_z = self.delta_enc(delta)
        z = self.fuse(torch.cat([parent_z, delta_z], dim=-1))
        return z, parent_z, atom_emb

    def forward_family(self, z: torch.Tensor) -> torch.Tensor:
        return self.family_head(z)

    def _family_features(
        self,
        z: torch.Tensor,
        family_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if family_idx is not None:
            return F.one_hot(family_idx, self.n_families).to(dtype=z.dtype, device=z.device)
        fam_logits = self.family_head(z)
        return F.softmax(fam_logits, dim=-1)

    def forward_reagent(
        self,
        z: torch.Tensor,
        *,
        family_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        fam_feat = self._family_features(z, family_idx)
        return self.reagent_head(torch.cat([z, fam_feat], dim=-1))

    def forward_sites(self, atom_emb: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # atom_emb: [B, N, H], z: [B, H] -> [N] logits (batch size 1 at train/infer)
        ctx, _ = self.site_cross(z.unsqueeze(1), atom_emb, atom_emb)
        fused = atom_emb + ctx.expand_as(atom_emb)
        logits = self.site_head(fused).squeeze(-1)
        if logits.dim() == 2 and logits.size(0) == 1:
            return logits.squeeze(0)
        return logits


class DeltaPropertyPredictor(nn.Module):
    def __init__(
        self,
        hidden: int = DEFAULT_HIDDEN,
        out_dim: int = 4,
        layers: int = DEFAULT_LAYERS,
        heads: int = DEFAULT_HEADS,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()
        self.hidden = hidden
        enc_kw = dict(hidden=hidden, layers=layers, heads=heads, dropout=dropout)
        self.parent_enc = GraphTransformerEncoder(**enc_kw)
        self.child_enc = GraphTransformerEncoder(**enc_kw)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(
        self,
        parent_x: torch.Tensor,
        child_x: torch.Tensor,
        parent_edge: Optional[torch.Tensor] = None,
        child_edge: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        p = self.parent_enc(parent_x, parent_edge)
        c = self.child_enc(child_x, child_edge)
        return self.head(torch.cat([p, c], dim=-1))


class UncertaintyEnsemble(nn.Module):
    def __init__(self, models: List[DeltaPropertyPredictor]):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, parent_x: torch.Tensor, child_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        preds = torch.stack([m(parent_x, child_x) for m in self.models], dim=0)
        mean = preds.mean(dim=0)
        std = preds.std(dim=0)
        return mean, std
