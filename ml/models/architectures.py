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

    # rdkit-pypi 2022.x uses GetExplicitValence/GetImplicitValence;
    # RDKit 2024+ renamed those to GetValence(ValenceType.*).
    if hasattr(atom, "GetValence") and hasattr(Chem, "ValenceType"):
        explicit_v = atom.GetValence(Chem.ValenceType.EXPLICIT)
        implicit_v = atom.GetValence(Chem.ValenceType.IMPLICIT)
    else:
        explicit_v = atom.GetExplicitValence()
        implicit_v = atom.GetImplicitValence()
    feats.extend(
        [
            float(atom.GetTotalDegree()),
            float(atom.GetFormalCharge()),
            float(atom.GetTotalValence()),
            float(explicit_v),
            float(implicit_v),
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


BOND_TYPE_LIST = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
# Listed bond types plus "other".
N_BOND_TYPES = len(BOND_TYPE_LIST) + 1


def _bond_type_index(bond: Chem.Bond) -> int:
    bt = bond.GetBondType()
    return BOND_TYPE_LIST.index(bt) if bt in BOND_TYPE_LIST else len(BOND_TYPE_LIST)


def mol_to_graph(smiles: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Atom features and a ``[3, E]`` edge tensor: source, target, bond-type index.

    Encoders without bond embeddings read only the first two rows.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    atom_feats = [_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_feats, dtype=torch.float32)
    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = _bond_type_index(bond)
        edges.append([i, j, bt])
        edges.append([j, i, bt])
    if not edges:
        edge_index = torch.zeros((3, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return x, edge_index


DEFAULT_HIDDEN = 128
DEFAULT_LAYERS = 3
DEFAULT_BOND_TYPES = N_BOND_TYPES
DEFAULT_SITE_REAGENT = True
DEFAULT_HEADS = 4  # retained for CLI / checkpoint meta compatibility; unused by GraphEncoder
DEFAULT_DROPOUT = 0.15
# Hidden layers before the logits. 0 is a single linear map.
DEFAULT_FAMILY_LAYERS = 1
DEFAULT_REAGENT_LAYERS = 1
DEFAULT_SITE_LAYERS = 1


def _mlp_head(in_dim: int, hidden: int, out_dim: int, n_hidden: int, dropout: float) -> nn.Module:
    """``n_hidden`` hidden layers, then a linear map to ``out_dim``. ``0`` is one linear layer."""
    if n_hidden <= 0:
        return nn.Linear(in_dim, out_dim)
    mods: List[nn.Module] = []
    width = in_dim
    for _ in range(n_hidden):
        mods.extend((nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout)))
        width = hidden
    mods.append(nn.Linear(width, out_dim))
    return nn.Sequential(*mods)


def _site_mix(in_dim: int, hidden: int, n_hidden: int) -> nn.Module:
    """Hidden stack for the site head. At least one layer; no dropout, matching the shipped head."""
    if n_hidden < 1:
        raise ValueError("site head needs at least one hidden layer")
    mods: List[nn.Module] = []
    width = in_dim
    for _ in range(n_hidden):
        mods.extend((nn.Linear(width, hidden), nn.GELU()))
        width = hidden
    return nn.Sequential(*mods)


class BondMessagePass(nn.Module):
    """One residual round of bond-neighbor mean aggregation.

    With ``bond_types > 0`` each message adds an embedding of its bond type;
    ``0`` keeps the untyped messages of older checkpoints.
    """

    def __init__(self, hidden: int, bond_types: int = 0):
        super().__init__()
        self.self_lin = nn.Linear(hidden, hidden)
        self.nei_lin = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.bond_emb = nn.Embedding(bond_types, hidden) if bond_types > 0 else None

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
        msg = h0[src]
        if self.bond_emb is not None and edge_index.size(0) > 2:
            msg = msg + self.bond_emb(edge_index[2])
        agg.index_add_(0, dst, msg)
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
        bond_types: int = DEFAULT_BOND_TYPES,
    ):
        super().__init__()
        _ = heads  # encoder no longer uses multi-head attention
        self.hidden = hidden
        self.input = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden))
        self.mp_layers = nn.ModuleList(
            [BondMessagePass(hidden, bond_types) for _ in range(max(1, layers))]
        )
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
    """Hierarchical: family → site → reagent (family-conditioned).

    FiLM on the graph vector, atom attention + fuse, and heads that also see the
    raw 4-d gap. ``bond_types=0``, ``site_reagent=False``, and
    ``family_layers=0`` rebuild checkpoints written before typed bonds and the
    reagent-conditioned site head.
    """

    def __init__(
        self,
        n_families: int,
        n_reagents: int,
        hidden: int = DEFAULT_HIDDEN,
        layers: int = DEFAULT_LAYERS,
        heads: int = DEFAULT_HEADS,
        dropout: float = DEFAULT_DROPOUT,
        gap_dim: int = 4,
        family_layers: int = DEFAULT_FAMILY_LAYERS,
        reagent_layers: int = DEFAULT_REAGENT_LAYERS,
        site_layers: int = DEFAULT_SITE_LAYERS,
        bond_types: int = DEFAULT_BOND_TYPES,
        site_reagent: bool = DEFAULT_SITE_REAGENT,
    ):
        super().__init__()
        self.hidden = hidden
        self.layers = layers
        self.heads = heads
        self.dropout = dropout
        self.n_families = n_families
        self.n_reagents = n_reagents
        self.gap_dim = gap_dim
        self.family_layers = family_layers
        self.reagent_layers = reagent_layers
        self.site_layers = site_layers
        self.bond_types = bond_types
        self.site_reagent = site_reagent
        enc_kw = dict(hidden=hidden, layers=layers, heads=heads, dropout=dropout, bond_types=bond_types)
        self.parent_enc = GraphEncoder(**enc_kw)
        self.film = nn.Linear(gap_dim, hidden * 2)
        self.atom_score = nn.Linear(hidden + gap_dim, 1)
        self.fuse = nn.Sequential(
            nn.Linear(hidden * 2 + gap_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.family_head = _mlp_head(hidden + gap_dim, hidden, n_families, family_layers, dropout)
        self.reagent_head = _mlp_head(
            hidden + n_families + gap_dim + hidden * 2,
            hidden,
            n_reagents,
            reagent_layers,
            dropout,
        )
        site_in = hidden * 2 + hidden + gap_dim
        if site_reagent:
            self.site_reagent_emb = nn.Embedding(n_reagents, hidden)
            site_in += hidden
        else:
            self.site_reagent_emb = None
        self.site_mix = _site_mix(site_in, hidden, site_layers)
        self.site_head = nn.Linear(hidden, 1)

    def arch_meta(self) -> dict:
        """Constructor settings a checkpoint needs to rebuild this model."""
        return {
            "architecture": "legacy",
            "hidden": self.hidden,
            "layers": self.layers,
            "heads": self.heads,
            "dropout": self.dropout,
            "family_layers": self.family_layers,
            "reagent_layers": self.reagent_layers,
            "site_layers": self.site_layers,
            "bond_types": self.bond_types,
            "site_reagent": self.site_reagent,
        }

    def encode(
        self,
        atom_x: torch.Tensor,
        delta: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # atom_x: [1, N, F], delta: [1, 4] -> z [1, H], parent_z [1, H], atom_emb [1, N, 2H]
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
        return F.softmax(self.forward_family(z, gap), dim=-1)

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
        reagent_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-atom site logits. ``reagent_idx`` is the reagent being placed; ignored by reagent-blind heads."""
        n_atoms = atom_emb.size(1)
        z_exp = z.unsqueeze(1).expand(-1, n_atoms, -1)
        gap_exp = self._gap(z, gap).unsqueeze(1).expand(-1, n_atoms, -1)
        parts = [atom_emb, z_exp, gap_exp]
        if self.site_reagent_emb is not None:
            if reagent_idx is None:
                reag = torch.zeros(z.size(0), self.hidden, device=z.device, dtype=z.dtype)
            else:
                reag = self.site_reagent_emb(reagent_idx.to(device=z.device).view(-1))
            parts.append(reag.unsqueeze(1).expand(-1, n_atoms, -1))
        mixed = self.site_mix(torch.cat(parts, dim=-1))
        logits = self.site_head(mixed).squeeze(-1)
        if logits.dim() == 2 and logits.size(0) == 1:
            return logits.squeeze(0)
        return logits


# ---------------------------------------------------------------------------
# Multitask planner: hard parameter sharing with a configurable fork point
# ---------------------------------------------------------------------------

MT_TASKS = ("family", "reagent", "site")

# Defaults for ``--arch multitask``; chosen by the validation search in
# docs/multitask_search.md.
DEFAULT_MT_SHARED_MP = 3
DEFAULT_MT_BRANCH_MP = 0
DEFAULT_MT_SHARED_FC = 0
DEFAULT_MT_TOWER_LAYERS = 1
DEFAULT_MT_TOWER_HIDDEN = DEFAULT_HIDDEN
# "hierarchical": reagent tower sees the family, site tower sees the reagent
# (the ModificationPlanner chain). "none": three parallel, unconditioned heads.
MT_CONDITIONING = ("hierarchical", "none")
DEFAULT_MT_CONDITIONING = "hierarchical"


def _seg_sum(v: torch.Tensor, seg: torch.Tensor, n: int) -> torch.Tensor:
    out = v.new_zeros((n,) + tuple(v.shape[1:]))
    return out.index_add(0, seg, v)


def _seg_mean(v: torch.Tensor, seg: torch.Tensor, n: int) -> torch.Tensor:
    ones = torch.ones(seg.numel(), device=v.device, dtype=v.dtype)
    cnt = _seg_sum(ones, seg, n).clamp_min(1.0)
    return _seg_sum(v, seg, n) / cnt.view(-1, *([1] * (v.dim() - 1)))


def _seg_max(v: torch.Tensor, seg: torch.Tensor, n: int) -> torch.Tensor:
    idx = seg.view(-1, *([1] * (v.dim() - 1))).expand_as(v)
    out = v.new_full((n,) + tuple(v.shape[1:]), float("-inf"))
    return out.scatter_reduce(0, idx, v, reduce="amax", include_self=True)


def segment_logsumexp(v: torch.Tensor, seg: torch.Tensor, n: int) -> torch.Tensor:
    """Per-segment logsumexp of a 1-d tensor. Empty segments give a large negative, not -inf."""
    m = _seg_max(v.detach(), seg, n)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    s = _seg_sum(torch.exp(v - m[seg]), seg, n)
    return torch.log(s.clamp_min(1e-30)) + m


class _GraphContext(nn.Module):
    """Graph readout plus gap conditioning: mean+max readout, FiLM, gap-aware atom attention, fuse.

    The same computation as ``GraphEncoder.readout`` followed by the FiLM /
    attention / fuse block of ``ModificationPlanner.encode``, on a disjoint
    union of graphs.
    """

    def __init__(self, hidden: int, gap_dim: int, dropout: float):
        super().__init__()
        self.readout = nn.Linear(hidden * 2, hidden)
        self.film = nn.Linear(gap_dim, hidden * 2)
        self.atom_score = nn.Linear(hidden + gap_dim, 1)
        self.fuse = nn.Sequential(
            nn.Linear(hidden * 2 + gap_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, h, local, batch, n_graphs, gap):
        mean = _seg_mean(h, batch, n_graphs)
        mx = _seg_max(h, batch, n_graphs)
        parent_z = self.readout(torch.cat([mean, mx], dim=-1))
        gamma, beta = self.film(gap).chunk(2, dim=-1)
        parent_cond = parent_z * (1.0 + gamma) + beta
        attn = self.atom_score(torch.cat([local, gap[batch]], dim=-1)).squeeze(-1)
        attn = attn - _seg_max(attn.detach(), batch, n_graphs)[batch]
        w = torch.exp(attn)
        alpha = w / _seg_sum(w, batch, n_graphs)[batch]
        attended = _seg_sum(alpha.unsqueeze(-1) * local, batch, n_graphs)
        z = self.fuse(torch.cat([parent_cond, attended, gap], dim=-1))
        return z, parent_z


def _shared_fc(hidden: int, n_layers: int, dropout: float) -> nn.Module:
    if n_layers <= 0:
        return nn.Identity()
    mods: List[nn.Module] = []
    for _ in range(n_layers):
        mods.extend((nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)))
    return nn.Sequential(*mods)


class MultitaskPlanner(nn.Module):
    """Shared trunk that forks into family, reagent, and site branches.

    Trunk: atom projection and ``shared_mp`` residual bond message-passing
    layers, shared by all three tasks.

    Fork point:

    * ``branch_mp == 0`` forks after the graph vector. One readout / FiLM /
      attention / fuse block and ``shared_fc`` dense layers are shared, then
      each task gets its own tower.
    * ``branch_mp > 0`` forks inside the encoder. Each task continues with
      ``branch_mp`` private message-passing layers and its own readout / FiLM /
      attention / fuse block (``shared_fc`` must be 0).

    Towers: family MLP, reagent MLP, per-atom site MLP, with
    ``*_tower_layers`` hidden layers of width ``tower_hidden``. With
    ``conditioning="hierarchical"`` the reagent tower also gets the family
    (one-hot of the true family in training, the predicted family at
    inference) and the site tower gets an embedding of the reagent, as in
    ``ModificationPlanner``. With ``conditioning="none"`` the three heads are
    parallel and see only their branch of the shared representation and the
    gap; ``family_idx`` / ``reagent_idx`` are ignored and the site head is
    reagent-blind (``site_reagent`` is False).

    ``shared_mp=L, branch_mp=0, shared_fc=0``, one-layer towers and
    ``tower_hidden=hidden`` compute exactly what ``ModificationPlanner`` with
    ``layers=L`` and one-layer heads computes (see
    ``legacy_to_multitask_state_dict``).

    The single-molecule API (``encode`` / ``forward_family`` /
    ``forward_reagent`` / ``forward_sites``) matches ``ModificationPlanner``,
    so the backend uses either model unchanged. ``z`` and ``atom_emb`` hold
    the three branch representations side by side.
    """

    architecture = "multitask"

    def __init__(
        self,
        n_families: int,
        n_reagents: int,
        hidden: int = DEFAULT_HIDDEN,
        shared_mp: int = DEFAULT_MT_SHARED_MP,
        branch_mp: int = DEFAULT_MT_BRANCH_MP,
        shared_fc: int = DEFAULT_MT_SHARED_FC,
        tower_hidden: Optional[int] = None,
        family_tower_layers: int = DEFAULT_MT_TOWER_LAYERS,
        reagent_tower_layers: int = DEFAULT_MT_TOWER_LAYERS,
        site_tower_layers: int = DEFAULT_MT_TOWER_LAYERS,
        dropout: float = DEFAULT_DROPOUT,
        gap_dim: int = 4,
        bond_types: int = DEFAULT_BOND_TYPES,
        site_reagent: bool = DEFAULT_SITE_REAGENT,
        heads: int = DEFAULT_HEADS,
        conditioning: str = DEFAULT_MT_CONDITIONING,
    ):
        super().__init__()
        if conditioning not in MT_CONDITIONING:
            raise ValueError(f"conditioning must be one of {MT_CONDITIONING}, got {conditioning!r}")
        if conditioning == "none":
            site_reagent = False
        self.conditioning = conditioning
        if shared_mp < 0 or branch_mp < 0 or shared_fc < 0:
            raise ValueError("shared_mp, branch_mp and shared_fc must be >= 0")
        if shared_mp + branch_mp < 1:
            raise ValueError("need at least one message-passing layer (shared_mp + branch_mp >= 1)")
        if branch_mp > 0 and shared_fc > 0:
            raise ValueError("shared_fc applies only when the fork is after the graph vector (branch_mp=0)")
        self.hidden = hidden
        self.shared_mp = shared_mp
        self.branch_mp = branch_mp
        self.shared_fc_layers = shared_fc
        self.tower_hidden = int(tower_hidden or hidden)
        self.family_tower_layers = family_tower_layers
        self.reagent_tower_layers = reagent_tower_layers
        self.site_tower_layers = site_tower_layers
        self.dropout = dropout
        self.gap_dim = gap_dim
        self.bond_types = bond_types
        self.site_reagent = site_reagent
        self.heads = heads
        self.n_families = n_families
        self.n_reagents = n_reagents
        self.layers = shared_mp + branch_mp

        self.input = nn.Sequential(nn.Linear(ATOM_FEAT_DIM, hidden), nn.LayerNorm(hidden))
        self.trunk_mp = nn.ModuleList([BondMessagePass(hidden, bond_types) for _ in range(shared_mp)])
        self.drop = nn.Dropout(dropout)
        if branch_mp > 0:
            self.branch_layers = nn.ModuleDict(
                {
                    t: nn.ModuleList([BondMessagePass(hidden, bond_types) for _ in range(branch_mp)])
                    for t in MT_TASKS
                }
            )
            self.contexts = nn.ModuleDict({t: _GraphContext(hidden, gap_dim, dropout) for t in MT_TASKS})
            self.context = None
            self.shared_fc = None
        else:
            self.branch_layers = None
            self.contexts = None
            self.context = _GraphContext(hidden, gap_dim, dropout)
            self.shared_fc = _shared_fc(hidden, shared_fc, dropout)

        th = self.tower_hidden
        self.family_tower = _mlp_head(hidden + gap_dim, th, n_families, family_tower_layers, dropout)
        fam_in = n_families if conditioning == "hierarchical" else 0
        self.reagent_tower = _mlp_head(
            hidden + fam_in + gap_dim + hidden * 2, th, n_reagents, reagent_tower_layers, dropout
        )
        site_in = hidden * 2 + hidden + gap_dim
        if site_reagent:
            self.site_reagent_emb = nn.Embedding(n_reagents, hidden)
            site_in += hidden
        else:
            self.site_reagent_emb = None
        self.site_mix = _site_mix(site_in, th, site_tower_layers)
        self.site_head = nn.Linear(th, 1)

    def arch_meta(self) -> dict:
        return {
            "architecture": "multitask",
            "hidden": self.hidden,
            "layers": self.layers,
            "heads": self.heads,
            "dropout": self.dropout,
            "bond_types": self.bond_types,
            "site_reagent": self.site_reagent,
            "mt_shared_mp": self.shared_mp,
            "mt_branch_mp": self.branch_mp,
            "mt_shared_fc": self.shared_fc_layers,
            "mt_tower_hidden": self.tower_hidden,
            "mt_family_tower_layers": self.family_tower_layers,
            "mt_reagent_tower_layers": self.reagent_tower_layers,
            "mt_site_tower_layers": self.site_tower_layers,
            "mt_conditioning": self.conditioning,
        }

    # ---- batched API (disjoint union of graphs) -------------------------
    def encode_batch(self, x, edge_index, batch, n_graphs, gap):
        """Per-task ``(z [B, H], atom_emb [N, 2H])`` for a disjoint union of graphs.

        ``x`` is ``[N, F]``, ``edge_index`` ``[3, E]`` with offset atom indices,
        ``batch`` maps each atom to its graph, ``gap`` is ``[B, gap_dim]``.
        """
        local = self.input(x)
        h = local.unsqueeze(0)
        for mp in self.trunk_mp:
            h = self.drop(mp(h, edge_index))
        out = {}
        if self.branch_layers is None:
            h0 = h.squeeze(0)
            z, _ = self.context(h0, local, batch, n_graphs, gap)
            z = self.shared_fc(z)
            atom_emb = torch.cat([local, h0], dim=-1)
            for t in MT_TASKS:
                out[t] = (z, atom_emb)
            return out
        for t in MT_TASKS:
            hb = h
            for mp in self.branch_layers[t]:
                hb = self.drop(mp(hb, edge_index))
            hb = hb.squeeze(0)
            z, _ = self.contexts[t](hb, local, batch, n_graphs, gap)
            out[t] = (z, torch.cat([local, hb], dim=-1))
        return out

    def family_logits(self, z_f, gap):
        return self.family_tower(torch.cat([z_f, gap], dim=-1))

    def reagent_logits(self, z_r, fam_feat, gap, atom_emb_r, batch, n_graphs):
        local = _seg_max(atom_emb_r, batch, n_graphs)
        if self.conditioning == "none":
            return self.reagent_tower(torch.cat([z_r, gap, local], dim=-1))
        return self.reagent_tower(torch.cat([z_r, fam_feat, gap, local], dim=-1))

    def site_logits(self, atom_emb_s, z_s, gap, batch, reagent_idx=None):
        parts = [atom_emb_s, z_s[batch], gap[batch]]
        if self.site_reagent_emb is not None:
            if reagent_idx is None:
                reag = z_s.new_zeros(z_s.size(0), self.hidden)
            else:
                reag = self.site_reagent_emb(reagent_idx.to(device=z_s.device).view(-1))
            parts.append(reag[batch])
        return self.site_head(self.site_mix(torch.cat(parts, dim=-1))).squeeze(-1)

    # ---- single-molecule API (same signatures as ModificationPlanner) ----
    def _slice(self, t: torch.Tensor, task: str, width: int) -> torch.Tensor:
        i = MT_TASKS.index(task)
        return t[..., i * width : (i + 1) * width]

    def _gap(self, z: torch.Tensor, gap: Optional[torch.Tensor]) -> torch.Tensor:
        if gap is None:
            return torch.zeros(z.size(0), self.gap_dim, device=z.device, dtype=z.dtype)
        return gap

    def encode(self, atom_x, delta, edge_index=None):
        x = atom_x.squeeze(0)
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        if edge_index is None:
            edge_index = torch.zeros((3, 0), dtype=torch.long, device=x.device)
        out = self.encode_batch(x, edge_index, batch, 1, delta)
        z = torch.cat([out[t][0] for t in MT_TASKS], dim=-1)
        atom_emb = torch.cat([out[t][1] for t in MT_TASKS], dim=-1).unsqueeze(0)
        return z, out["family"][0], atom_emb

    def forward_family(self, z, gap=None):
        return self.family_logits(self._slice(z, "family", self.hidden), self._gap(z, gap))

    def forward_reagent(self, z, *, family_idx=None, gap=None, atom_emb=None):
        gap_t = self._gap(z, gap)
        z_r = self._slice(z, "reagent", self.hidden)
        if atom_emb is None:
            local = torch.zeros(z.size(0), self.hidden * 2, device=z.device, dtype=z.dtype)
        else:
            local = self._slice(atom_emb, "reagent", self.hidden * 2).max(dim=1).values
        if self.conditioning == "none":
            return self.reagent_tower(torch.cat([z_r, gap_t, local], dim=-1))
        if family_idx is not None:
            fam_feat = F.one_hot(family_idx, self.n_families).to(dtype=z.dtype, device=z.device)
        else:
            fam_feat = F.softmax(self.forward_family(z, gap_t), dim=-1)
        return self.reagent_tower(torch.cat([z_r, fam_feat, gap_t, local], dim=-1))

    def forward_sites(self, atom_emb, z, gap=None, reagent_idx=None):
        emb = self._slice(atom_emb, "site", self.hidden * 2)
        B, N, _ = emb.shape
        batch = torch.arange(B, device=emb.device).repeat_interleave(N)
        logits = self.site_logits(
            emb.reshape(B * N, -1),
            self._slice(z, "site", self.hidden),
            self._gap(z, gap),
            batch,
            reagent_idx,
        ).view(B, N)
        if B == 1:
            return logits.squeeze(0)
        return logits


def legacy_to_multitask_state_dict(state: dict) -> dict:
    """Rename a ``ModificationPlanner`` state dict for the equivalent ``MultitaskPlanner``.

    Valid for ``shared_mp=layers, branch_mp=0, shared_fc=0``, one-layer towers
    and ``tower_hidden=hidden``.
    """
    renames = (
        ("parent_enc.input.", "input."),
        ("parent_enc.mp_layers.", "trunk_mp."),
        ("parent_enc.readout.", "context.readout."),
        ("film.", "context.film."),
        ("atom_score.", "context.atom_score."),
        ("fuse.", "context.fuse."),
        ("family_head.", "family_tower."),
        ("reagent_head.", "reagent_tower."),
    )
    out = {}
    for key, val in state.items():
        new = key
        for old, rep in renames:
            if key.startswith(old):
                new = rep + key[len(old):]
                break
        out[new] = val
    return out


def build_planner(meta: dict, n_families: int, n_reagents: int) -> nn.Module:
    """Rebuild the planner described by checkpoint metadata.

    ``architecture == "multitask"`` builds ``MultitaskPlanner``; anything else
    builds ``ModificationPlanner``. Keys missing from older checkpoints fall
    back to the layouts those checkpoints were written with: linear family
    head, untyped bonds, reagent-blind site head.
    """
    hidden = int(meta.get("hidden", DEFAULT_HIDDEN))
    heads = int(meta.get("heads", DEFAULT_HEADS))
    dropout = float(meta.get("dropout", DEFAULT_DROPOUT))
    bond_types = int(meta.get("bond_types", 0))
    site_reagent = bool(meta.get("site_reagent", False))
    if meta.get("architecture") == "multitask":
        return MultitaskPlanner(
            n_families=n_families,
            n_reagents=n_reagents,
            hidden=hidden,
            shared_mp=int(meta["mt_shared_mp"]),
            branch_mp=int(meta.get("mt_branch_mp", 0)),
            shared_fc=int(meta.get("mt_shared_fc", 0)),
            tower_hidden=int(meta.get("mt_tower_hidden", hidden)),
            family_tower_layers=int(meta.get("mt_family_tower_layers", DEFAULT_MT_TOWER_LAYERS)),
            reagent_tower_layers=int(meta.get("mt_reagent_tower_layers", DEFAULT_MT_TOWER_LAYERS)),
            site_tower_layers=int(meta.get("mt_site_tower_layers", DEFAULT_MT_TOWER_LAYERS)),
            dropout=dropout,
            bond_types=bond_types,
            site_reagent=site_reagent,
            heads=heads,
            conditioning=str(meta.get("mt_conditioning", DEFAULT_MT_CONDITIONING)),
        )
    return ModificationPlanner(
        n_families=n_families,
        n_reagents=n_reagents,
        hidden=hidden,
        layers=int(meta.get("layers", DEFAULT_LAYERS)),
        heads=heads,
        dropout=dropout,
        family_layers=int(meta["family_layers"]) if "family_layers" in meta else 0,
        reagent_layers=int(meta.get("reagent_layers", DEFAULT_REAGENT_LAYERS)),
        site_layers=int(meta.get("site_layers", DEFAULT_SITE_LAYERS)),
        bond_types=bond_types,
        site_reagent=site_reagent,
    )
