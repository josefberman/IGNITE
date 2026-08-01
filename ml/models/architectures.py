"""Graph Transformer-style parent encoder and hierarchical modification planner (PyTorch)."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
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


DEFAULT_HIDDEN = 32


class GraphTransformerEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int = ATOM_FEAT_DIM,
        hidden: int = DEFAULT_HIDDEN,
        layers: int = 3,
        heads: int = 4,
    ):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError(f"hidden ({hidden}) must be divisible by heads ({heads})")
        self.input = nn.Linear(in_dim, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden * 2, batch_first=True, dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.out = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, N, F] — for simplicity callers can pass single-graph unsqueezed
        h = self.input(x)
        h = self.encoder(h, src_key_padding_mask=mask)
        # mean pool over atoms
        if mask is not None:
            valid = (~mask).float().unsqueeze(-1)
            h = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        else:
            h = h.mean(dim=1)
        return self.out(h)


class TargetDeltaEncoder(nn.Module):
    def __init__(self, in_dim: int = 4, hidden: int = DEFAULT_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        return self.net(delta)


class ModificationPlanner(nn.Module):
    """Hierarchical: family → site → reagent."""

    def __init__(self, n_families: int, n_reagents: int, hidden: int = DEFAULT_HIDDEN):
        super().__init__()
        self.hidden = hidden
        self.parent_enc = GraphTransformerEncoder(hidden=hidden)
        self.delta_enc = TargetDeltaEncoder(hidden=hidden)
        self.fuse = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU())
        self.family_head = nn.Linear(hidden, n_families)
        self.site_head = nn.Linear(hidden, 1)  # applied per-atom via broadcast in training loop
        self.reagent_head = nn.Linear(hidden, n_reagents)
        self.atom_proj = nn.Linear(ATOM_FEAT_DIM, hidden)

    def encode(self, atom_x: torch.Tensor, delta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # atom_x: [1, N, F]
        parent_z = self.parent_enc(atom_x)
        delta_z = self.delta_enc(delta)
        z = self.fuse(torch.cat([parent_z, delta_z], dim=-1))
        return z, parent_z

    def forward_family(self, z: torch.Tensor) -> torch.Tensor:
        return self.family_head(z)

    def forward_reagent(self, z: torch.Tensor) -> torch.Tensor:
        return self.reagent_head(z)

    def forward_sites(self, atom_x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # returns [N] logits for reacting-site probability
        atom_h = torch.relu(self.atom_proj(atom_x.squeeze(0)))
        context = z.squeeze(0)
        gated = atom_h * context
        return self.site_head(gated).squeeze(-1)


class DeltaPropertyPredictor(nn.Module):
    def __init__(self, hidden: int = DEFAULT_HIDDEN, out_dim: int = 4):
        super().__init__()
        self.hidden = hidden
        self.parent_enc = GraphTransformerEncoder(hidden=hidden)
        self.child_enc = GraphTransformerEncoder(hidden=hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, parent_x: torch.Tensor, child_x: torch.Tensor) -> torch.Tensor:
        p = self.parent_enc(parent_x)
        c = self.child_enc(child_x)
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
