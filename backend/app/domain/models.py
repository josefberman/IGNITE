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
    ANALOGUE = "analogue"
    XGBOOST = "xgboost"
    DELTA_MODEL = "delta_model"
    UNKNOWN = "unknown"


class NoveltyStatus(str, Enum):
    KNOWN = "known"
    CLOSE_ANALOGUE = "close_analogue"
    NO_MATCH_FOUND = "no_match_found"
    INCOMPLETE_SEARCH = "incomplete_search"


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
    depth: int = Field(5, ge=1, le=50, description="Beam depth X")
    width: int = Field(8, ge=1, le=100, description="Expansions per node N")
    top_k: int = Field(5, ge=1, le=50, description="Survivors per iteration K")
    mape_threshold: float = Field(5.0, ge=0.0, le=100.0)
    random_seed: int = Field(42, ge=0)
    enable_literature: bool = True
    enable_patent_search: bool = True
    novelty_as_gate: bool = False


class ScoreBreakdown(BaseModel):
    mape: float
    mape_normalized: float
    sascore_penalty: float
    uncertainty_penalty: float
    safety_penalty: float
    minimal_edit_penalty: float
    novelty_bonus: float = 0.0
    total: float
    weights: Dict[str, float] = Field(default_factory=dict)


class PropertyEstimate(BaseModel):
    value: float
    source: PredictionSource = PredictionSource.UNKNOWN
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    lower: Optional[float] = None
    upper: Optional[float] = None
    citation: Optional[str] = None


class SafetyFinding(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0, description="0=safer, 1=higher risk")
    is_acceptable: bool = True
    sascore: float = 5.0
    unstable_motifs: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    disclaimer: str = (
        "Computational heuristic only — not a safety certification or hazard clearance."
    )


class NoveltyEvidence(BaseModel):
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
    safety: SafetyFinding | None = None
    novelty: NoveltyEvidence | None = None
    modification: ModificationRecord | None = None
    lineage: List[str] = Field(default_factory=list)
    rejected: bool = False
    rejection_reason: Optional[str] = None

    def property_values(self) -> Dict[str, float]:
        return {k: v.value for k, v in self.properties.items()}


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
    best_uncertainty: Optional[float] = None
    message: str = ""


class SearchResult(BaseModel):
    job_id: str
    status: JobStatus
    best: Optional[CandidateMolecule] = None
    top_k: List[CandidateMolecule] = Field(default_factory=list)
    progress: SearchProgress
    disclaimer: str = (
        "Outputs are computational candidates with uncertainty and hazard flags. "
        "They are not safety certifications, legal novelty opinions, or patentability advice."
    )


class ModelMetadata(BaseModel):
    name: str
    version: str
    property: Optional[str] = None
    descriptor_dim: Optional[int] = None
    units: Optional[str] = None
    path: str
    notes: str = ""
