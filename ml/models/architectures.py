"""Simple graph / gap encoders and hierarchical modification planner (PyTorch)."""

from __future__ import annotations

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
DEFAULT_HEADS = 4  # retained for CLI / checkpoint meta compatibility; unused by GraphEncoder
DEFAULT_DROPOUT = 0.15


class BondMessagePass(nn.Module):
    """One residual round of bond-neighbor mean aggregation."""

    def __init__(self, hidden: int):
        super().__init__()
        self.self_lin = nn.Linear(hidden, hidden)
        self.nei_lin = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h: torch.Tensor, edge_index: Optional[torch.Tensor]) -> torch.Tensor:
        # h: [B, N, H]. Residual keeps the atom's own identity for the site head.
        if edge_index is None or edge_index.numel() == 0:
            return h + self.norm(F.gelu(self.self_lin(h)))
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
        return (h0 + self.norm(F.gelu(mixed))).unsqueeze(0)


class GraphEncoder(nn.Module):
    """Atom projection, residual bond message passing, mean+max readout."""

    def __init__(
        self,
        in_dim: int = ATOM_FEAT_DIM,
        hidden: int = DEFAULT_HIDDEN,
        layers: int = DEFAULT_LAYERS,
        heads: int = DEFAULT_HEADS,  # unused; kept for call-site compatibility
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()
        _ = heads  # encoder no longer uses multi-head attention
        self.hidden = hidden
        self.input = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden))
        self.mp_layers = nn.ModuleList([BondMessagePass(hidden) for _ in range(max(1, layers))])
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(hidden * 2, hidden)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        *,
        return_atom_emb: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, N, F]
        local = self.input(x)
        h = local
        for mp in self.mp_layers:
            h = self.dropout(mp(h, edge_index))
        atom_emb = h
        if mask is not None:
            keep = (~mask).unsqueeze(-1).to(dtype=h.dtype)
            denom = keep.sum(dim=1).clamp_min(1.0)
            mean = (h * keep).sum(dim=1) / denom
            masked = h.masked_fill(~keep.bool(), -1e9)
            mx = masked.max(dim=1).values
        else:
            mean = h.mean(dim=1)
            mx = h.max(dim=1).values
        graph_emb = self.readout(torch.cat([mean, mx], dim=-1))
        if return_atom_emb:
            return graph_emb, atom_emb, local
        return graph_emb


# Backward-compatible alias
GraphTransformerEncoder = GraphEncoder


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
        gap_dim: int = 4,
    ):
        super().__init__()
        self.hidden = hidden
        self.n_families = n_families
        self.n_reagents = n_reagents
        enc_kw = dict(hidden=hidden, layers=layers, heads=heads, dropout=dropout)
        self.parent_enc = GraphEncoder(**enc_kw)
        self.gap_dim = gap_dim
        # FiLM: the 4-d gap scales and shifts the molecule vector so a small
        # property direction cannot be ignored next to a 64-d embedding.
        self.film = nn.Linear(gap_dim, hidden * 2)
        # Score each atom from its pre-message-passing features and the gap,
        # then pool. The reacting functional group is local; mean readout is not.
        self.atom_score = nn.Linear(hidden + gap_dim, 1)
        self.fuse = nn.Sequential(
            nn.Linear(hidden * 2 + gap_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Heads also see the raw gap so the property direction cannot be
        # squeezed out of z by the shared encoder.
        self.family_head = nn.Linear(hidden + gap_dim, n_families)
        # atom features passed to the heads are local || contextual (2H).
        self.reagent_head = nn.Sequential(
            nn.Linear(hidden + n_families + gap_dim + hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_reagents),
        )
        self.site_mix = nn.Sequential(
            nn.Linear(hidden * 2 + hidden + gap_dim, hidden),
            nn.GELU(),
        )
        self.site_head = nn.Linear(hidden, 1)

    def encode(
        self,
        atom_x: torch.Tensor,
        delta: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # atom_x: [1, N, F], delta: [1, 4] -> z [1, H], parent_z [1, H], atom_emb [1, N, H]
        parent_z, atom_ctx, atom_local = self.parent_enc(atom_x, edge_index, return_atom_emb=True)
        gamma, beta = self.film(delta).chunk(2, dim=-1)
        parent_cond = parent_z * (1.0 + gamma) + beta
        n_atoms = atom_local.size(1)
        gap_exp = delta.unsqueeze(1).expand(-1, n_atoms, -1)
        attn = self.atom_score(torch.cat([atom_local, gap_exp], dim=-1)).squeeze(-1)
        alpha = torch.softmax(attn, dim=-1)
        attended = torch.bmm(alpha.unsqueeze(1), atom_local).squeeze(1)
        z = self.fuse(torch.cat([parent_cond, attended, delta], dim=-1))
        atom_emb = torch.cat([atom_local, atom_ctx], dim=-1)
        return z, parent_z, atom_emb

    def _gap(self, z: torch.Tensor, gap: Optional[torch.Tensor]) -> torch.Tensor:
        if gap is None:
            return torch.zeros(*z.shape[:-1], self.gap_dim, device=z.device, dtype=z.dtype)
        return gap

    def forward_family(self, z: torch.Tensor, gap: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.family_head(torch.cat([z, self._gap(z, gap)], dim=-1))

    def _family_features(
        self,
        z: torch.Tensor,
        family_idx: Optional[torch.Tensor] = None,
        gap: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if family_idx is not None:
            return F.one_hot(family_idx, self.n_families).to(dtype=z.dtype, device=z.device)
        fam_logits = self.forward_family(z, gap)
        return F.softmax(fam_logits, dim=-1)

    def forward_reagent(
        self,
        z: torch.Tensor,
        *,
        family_idx: Optional[torch.Tensor] = None,
        gap: Optional[torch.Tensor] = None,
        atom_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        fam_feat = self._family_features(z, family_idx, gap)
        gap_t = self._gap(z, gap)
        if atom_emb is None:
            local = torch.zeros_like(z)
        else:
            local = atom_emb.max(dim=1).values
        return self.reagent_head(torch.cat([z, fam_feat, gap_t, local], dim=-1))

    def forward_sites(
        self,
        atom_emb: torch.Tensor,
        z: torch.Tensor,
        gap: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # atom_emb: [B, N, H], z: [B, H] -> [N] logits (batch size 1 at train/infer)
        n_atoms = atom_emb.size(1)
        z_exp = z.unsqueeze(1).expand(-1, n_atoms, -1)
        gap_exp = self._gap(z, gap).unsqueeze(1).expand(-1, n_atoms, -1)
        mixed = self.site_mix(torch.cat([atom_emb, z_exp, gap_exp], dim=-1))
        logits = self.site_head(mixed).squeeze(-1)
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
        self.parent_enc = GraphEncoder(**enc_kw)
        self.child_enc = GraphEncoder(**enc_kw)
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
