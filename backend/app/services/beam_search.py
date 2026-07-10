"""Beam search controller following the high-level architecture diagram."""

from __future__ import annotations

import logging
import random
from typing import Callable, List, Optional

from backend.app.domain.models import (
    BeamParameters,
    CandidateMolecule,
    JobStatus,
    NoveltyStatus,
    SearchProgress,
    TargetProperties,
)
from backend.app.services.modification import ModificationMechanism
from backend.app.services.novelty import NoveltyService
from backend.app.services.property_estimator import PropertyEstimator
from backend.app.services.safety import estimate_safety
from backend.app.services.scoring import mean_uncertainty, score_candidate
from backend.app.services.seed_selector import SeedLibrary

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[SearchProgress, Optional[CandidateMolecule]], None]


class BeamSearchController:
    def __init__(
        self,
        seeds: SeedLibrary,
        estimator: PropertyEstimator,
        novelty: NoveltyService,
        modification: Optional[ModificationMechanism] = None,
    ):
        self.seeds = seeds
        self.estimator = estimator
        self.novelty = novelty
        self.modification = modification or ModificationMechanism()
        self._stop = False

    def request_stop(self) -> None:
        self._stop = True

    def run(
        self,
        job_id: str,
        targets: TargetProperties,
        beam: BeamParameters,
        on_progress: Optional[ProgressCallback] = None,
    ) -> tuple[CandidateMolecule, List[CandidateMolecule], SearchProgress]:
        self._stop = False
        rng = random.Random(beam.random_seed)
        self.modification.rng = rng

        target_dict = targets.as_dict()
        seed_pool = self.seeds.select_seeds(targets, mape_threshold=beam.mape_threshold, max_seeds=beam.top_k)
        if not seed_pool:
            raise ValueError("No seed molecules available")

        for s in seed_pool:
            s.safety = estimate_safety(s.canonical_smiles)
            s.novelty = self.novelty.evaluate(s.canonical_smiles)
            s.score = score_candidate(s, target_dict)

        beam_nodes = sorted(seed_pool, key=lambda c: c.score.total if c.score else 1e9)[: beam.top_k]
        best = beam_nodes[0]
        seen = {c.canonical_smiles for c in beam_nodes}
        rejected = 0
        generated = 0

        def emit(iteration: int, status: JobStatus, message: str) -> SearchProgress:
            prog = SearchProgress(
                job_id=job_id,
                status=status,
                iteration=iteration,
                max_iterations=beam.depth,
                beam_size=len(beam_nodes),
                candidates_generated=generated,
                rejected_count=rejected,
                best_mape=best.mape,
                best_score=best.score.total if best.score else None,
                best_uncertainty=mean_uncertainty(best),
                message=message,
            )
            if on_progress:
                on_progress(prog, best)
            return prog

        progress = emit(0, JobStatus.RUNNING, f"Seeded with {len(beam_nodes)} molecules")

        for iteration in range(1, beam.depth + 1):
            if self._stop:
                progress = emit(iteration - 1, JobStatus.CANCELLED, "Cancelled")
                break

            # Early stop on MAPE threshold
            if best.mape <= beam.mape_threshold:
                progress = emit(iteration - 1, JobStatus.COMPLETED, "MAPE threshold reached")
                break

            children: List[CandidateMolecule] = []
            for parent in beam_nodes:
                expansions = self.modification.expand(
                    parent.canonical_smiles,
                    parent.property_values(),
                    target_dict,
                    n=beam.width,
                )
                for child_smi, mod in expansions:
                    generated += 1
                    if child_smi in seen:
                        rejected += 1
                        continue
                    props = self.estimator.estimate(child_smi)
                    if props is None:
                        rejected += 1
                        continue
                    cand = CandidateMolecule(
                        smiles=child_smi,
                        canonical_smiles=child_smi,
                        generation=iteration,
                        parent_id=parent.id,
                        parent_smiles=parent.canonical_smiles,
                        properties=props,
                        modification=mod,
                        lineage=list(parent.lineage) + [parent.canonical_smiles],
                    )
                    cand.mape = score_candidate(cand, target_dict).mape
                    cand.safety = estimate_safety(child_smi)
                    if not cand.safety.is_acceptable:
                        cand.rejected = True
                        cand.rejection_reason = "safety_gate"
                        rejected += 1
                        continue
                    cand.novelty = self.novelty.evaluate(child_smi)
                    if beam.novelty_as_gate and cand.novelty.status == NoveltyStatus.KNOWN:
                        cand.rejected = True
                        cand.rejection_reason = "novelty_gate"
                        rejected += 1
                        continue
                    cand.score = score_candidate(cand, target_dict)
                    seen.add(child_smi)
                    children.append(cand)

            pool = beam_nodes + children
            pool.sort(key=lambda c: c.score.total if c.score else 1e9)
            beam_nodes = pool[: beam.top_k]
            if beam_nodes and (best.score is None or (beam_nodes[0].score and beam_nodes[0].score.total < best.score.total)):
                best = beam_nodes[0]
            elif beam_nodes and beam_nodes[0].mape < best.mape:
                best = beam_nodes[0]

            progress = emit(iteration, JobStatus.RUNNING, f"Iteration {iteration} complete")

        else:
            progress = emit(beam.depth, JobStatus.COMPLETED, "Depth limit reached")

        if progress.status == JobStatus.RUNNING:
            progress.status = JobStatus.COMPLETED

        top = sorted(beam_nodes, key=lambda c: c.score.total if c.score else 1e9)
        return best, top, progress
