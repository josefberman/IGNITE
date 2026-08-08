"""Typed domain contracts for IGNITE search, scoring, and evidence."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


PROPERTY_KEYS = ("density", "det_velocity", "det_pressure", "hf_solid")
PROPERTY_UNITS = {
    "density": "g/cm^3",
    "det_velocity": "m/s",
    "det_pressure": "GPa",
    "hf_solid": "kJ/mol",
}
PROPERTY_RANGES = {
    "density": (0.8, 2.5),
    "det_velocity": (1000.0, 12000.0),
    "det_pressure": (0.1, 60.0),
    "hf_solid": (-1000.0, 1200.0),
}


class PredictionSource(str, Enum):
    DATASET = "dataset"
    LITERATURE = "literature"
    XGBOOST = "xgboost"
    DELTA_MODEL = "delta_model"
    UNKNOWN = "unknown"
    # Deprecated — retained so old job JSON still deserializes
    ANALOGUE = "analogue"


class NoveltyStatus(str, Enum):
    KNOWN = "known"
    NO_MATCH_FOUND = "no_match_found"
    INCOMPLETE_SEARCH = "incomplete_search"
    # Deprecated — retained for old job JSON
    CLOSE_ANALOGUE = "close_analogue"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TargetProperties(BaseModel):
    density: float = Field(..., description="Target density g/cm^3")
    det_velocity: float = Field(..., description="Target detonation velocity m/s")
    det_pressure: float = Field(..., description="Target detonation pressure GPa")
    hf_solid: float = Field(..., description="Target solid heat of formation kJ/mol")

    @field_validator(*PROPERTY_KEYS)
    @classmethod
    def check_range(cls, v: float, info) -> float:
        lo, hi = PROPERTY_RANGES[info.field_name]
        if not (lo <= v <= hi):
            raise ValueError(f"{info.field_name} must be in [{lo}, {hi}], got {v}")
        return float(v)

    def as_dict(self) -> Dict[str, float]:
        return {
            "density": self.density,
            "det_velocity": self.det_velocity,
            "det_pressure": self.det_pressure,
            "hf_solid": self.hf_solid,
        }


class BeamParameters(BaseModel):
    depth: int = Field(10, ge=1, le=50, description="Beam depth")
    width: int = Field(5, ge=1, le=100, description="Expansions per node")
    top_k: int = Field(5, ge=1, le=50, description="Survivors per iteration K")
    mape_threshold: float = Field(1.0, ge=0.0, le=100.0)
    random_seed: int = Field(42, ge=0)
    enable_literature: bool = True
    enable_patent_search: bool = True
    novelty_as_gate: bool = False
    early_stop_on_mape: bool = Field(
        False,
        description="Stop search when best MAPE meets threshold (after ≥1 expansion). "
        "Disable to always run the full depth.",
    )


class ScoreBreakdown(BaseModel):
    mape: float
    mape_normalized: float
    sascore_penalty: float
    minimal_edit_penalty: float
    total: float
    weights: Dict[str, float] = Field(default_factory=dict)


class PropertyEstimate(BaseModel):
    value: float
    source: PredictionSource = PredictionSource.UNKNOWN
    # Retained for internal/legacy job JSON; not reported in the GUI.
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    lower: Optional[float] = None
    upper: Optional[float] = None
    citation: Optional[str] = None


class SafetyFinding(BaseModel):
    """Internal hard-gate / SAScore signal — not returned to clients."""

    score: float = Field(..., ge=0.0, le=1.0, description="0=safer, 1=higher risk")
    is_acceptable: bool = True
    sascore: float = 5.0
    unstable_motifs: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class NoveltyEvidence(BaseModel):
    """Internal novelty-gate signal — not returned to clients."""

    status: NoveltyStatus = NoveltyStatus.INCOMPLETE_SEARCH
    score: float = Field(0.0, ge=0.0, le=1.0, description="1=more novel")
    max_tanimoto_to_seeds: Optional[float] = None
    nearest_seed_smiles: Optional[str] = None
    literature_hits: List[Dict[str, Any]] = Field(default_factory=list)
    patent_hits: List[Dict[str, Any]] = Field(default_factory=list)
    providers_checked: List[str] = Field(default_factory=list)
    providers_unavailable: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class ModificationRecord(BaseModel):
    reaction_family: str
    reactive_site: Optional[int] = None
    reagent: Optional[str] = None
    smarts: Optional[str] = None
    edit_distance: int = 1
    planner: str = "rule_based"


class CandidateMolecule(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    smiles: str
    canonical_smiles: str
    inchikey: Optional[str] = None
    generation: int = 0
    parent_id: Optional[str] = None
    parent_smiles: Optional[str] = None
    properties: Dict[str, PropertyEstimate] = Field(default_factory=dict)
    mape: float = 100.0
    score: ScoreBreakdown | None = None
    safety: SafetyFinding | None = Field(default=None, exclude=True)
    novelty: NoveltyEvidence | None = Field(default=None, exclude=True)
    modification: ModificationRecord | None = None
    lineage: List[str] = Field(default_factory=list)
    rejected: bool = False
    rejection_reason: Optional[str] = None

    def property_values(self) -> Dict[str, float]:
        return {k: v.value for k, v in self.properties.items()}

    def strip_internal_signals(self) -> "CandidateMolecule":
        """Drop safety/novelty and uncertainty fields before client-facing export."""
        self.safety = None
        self.novelty = None
        cleaned: Dict[str, PropertyEstimate] = {}
        for key, est in self.properties.items():
            cleaned[key] = PropertyEstimate(
                value=est.value,
                source=est.source,
                confidence=0.5,
                lower=None,
                upper=None,
                citation=None,
            )
        self.properties = cleaned
        return self


class SearchRequest(BaseModel):
    targets: TargetProperties
    beam: BeamParameters = Field(default_factory=BeamParameters)


class SearchProgress(BaseModel):
    job_id: str
    status: JobStatus
    iteration: int = 0
    max_iterations: int = 0
    beam_size: int = 0
    candidates_generated: int = 0
    rejected_count: int = 0
    best_mape: Optional[float] = None
    best_score: Optional[float] = None
    message: str = ""


class BeamTreeNode(BaseModel):
    """Lightweight node for beam-search tree visualization."""

    id: str
    parent_id: Optional[str] = None
    generation: int = 0
    canonical_smiles: str
    mape: float = 100.0
    score_total: Optional[float] = None
    reaction_family: Optional[str] = None
    planner: Optional[str] = None
    in_beam: bool = False


class BeamSearchTree(BaseModel):
    """Directed forest of accepted expansions; layers list beam survivor ids per iteration."""

    nodes: List[BeamTreeNode] = Field(default_factory=list)
    beam_layers: List[List[str]] = Field(default_factory=list)
    best_id: Optional[str] = None
    max_generation: int = 0
    iterations_completed: int = 0


class SearchResult(BaseModel):
    job_id: str
    status: JobStatus
    best: Optional[CandidateMolecule] = None
    top_k: List[CandidateMolecule] = Field(default_factory=list)
    progress: SearchProgress
    beam_tree: Optional[BeamSearchTree] = None
    disclaimer: str = (
        "Outputs are computational design candidates only — not experimental validation "
        "or a legal novelty / patentability opinion."
    )


class ModelMetadata(BaseModel):
    name: str
    version: str
    property: Optional[str] = None
    descriptor_dim: Optional[int] = None
    units: Optional[str] = None
    path: str
    notes: str = ""
