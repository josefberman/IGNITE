"""Beam search controller following the high-level architecture diagram."""

from __future__ import annotations

import logging
import math
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
from backend.app.services.scoring import score_candidate
from backend.app.services.seed_selector import SeedLibrary

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[SearchProgress, Optional[CandidateMolecule]], None]

# Hard-reject only truly critical hazards; energetic azides are scored, not gated.
CRITICAL_MOTIFS = {"peroxide", "invalid", "long_n_chain"}


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

    @staticmethod
    def _select_survivors(
        pool: List[CandidateMolecule],
        top_k: int,
        *,
        explore: bool,
    ) -> List[CandidateMolecule]:
        """Keep top_k by score, reserving slots for generated children while exploring."""
        ranked = sorted(pool, key=lambda c: c.score.total if c.score else 1e9)
        if not explore or top_k <= 1:
            return ranked[:top_k]

        children = [c for c in ranked if c.generation > 0]
        seeds = [c for c in ranked if c.generation == 0]
        reserve = min(len(children), max(1, math.ceil(top_k / 2)))
        chosen: List[CandidateMolecule] = []
        seen_ids = set()

        for c in children:
            if len(chosen) >= reserve:
                break
            chosen.append(c)
            seen_ids.add(c.id)

        for c in ranked:
            if len(chosen) >= top_k:
                break
            if c.id in seen_ids:
                continue
            chosen.append(c)
            seen_ids.add(c.id)

        # Prefer filling with remaining seeds if somehow short
        for c in seeds:
            if len(chosen) >= top_k:
                break
            if c.id in seen_ids:
                continue
            chosen.append(c)

        return sorted(chosen, key=lambda c: c.score.total if c.score else 1e9)[:top_k]

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

        # Honor GUI toggles
        self.estimator.literature.enabled = bool(beam.enable_literature)
        self.novelty.pubchem_enabled = bool(beam.enable_literature and self.novelty.pubchem_enabled)
        self.novelty.openalex_enabled = bool(beam.enable_literature and self.novelty.openalex_enabled)
        if not beam.enable_patent_search:
            self.novelty.epo_ops_key = ""
            self.novelty.lens_api_key = ""

        target_dict = targets.as_dict()
        seed_pool = self.seeds.select_seeds(
            targets, mape_threshold=max(beam.mape_threshold, 25.0), max_seeds=max(beam.top_k, 3)
        )
        if not seed_pool:
            raise ValueError("No seed molecules available")

        for s in seed_pool:
            s.safety = estimate_safety(s.canonical_smiles)
            if beam.novelty_as_gate:
                s.novelty = self.novelty.evaluate(s.canonical_smiles)
                if s.novelty.status == NoveltyStatus.KNOWN:
                    s.rejected = True
                    s.rejection_reason = "novelty_gate"
            s.score = score_candidate(s, target_dict, explore=True)
            s.mape = s.score.mape
        seed_pool = [s for s in seed_pool if not s.rejected]
        if not seed_pool:
            raise ValueError("No seed molecules available after novelty gate")

        beam_nodes = sorted(seed_pool, key=lambda c: c.score.total if c.score else 1e9)[: beam.top_k]
        best = beam_nodes[0]
        seen = {c.canonical_smiles for c in beam_nodes}
        rejected = 0
        generated = 0
        source_counts = {"dataset": 0, "literature": 0, "xgboost": 0, "other": 0}

        def _tally(cand: CandidateMolecule) -> None:
            for est in cand.properties.values():
                key = est.source.value if hasattr(est.source, "value") else str(est.source)
                if key in source_counts:
                    source_counts[key] += 1
                else:
                    source_counts["other"] += 1
                break  # one count per molecule

        for s in beam_nodes:
            _tally(s)

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
                message=message,
            )
            if on_progress:
                on_progress(prog, best)
            return prog

        progress = emit(
            0,
            JobStatus.RUNNING,
            f"Seeded with {len(beam_nodes)} molecules (lit={beam.enable_literature})",
        )

        for iteration in range(1, beam.depth + 1):
            if self._stop:
                progress = emit(iteration - 1, JobStatus.CANCELLED, "Cancelled")
                break

            exploring = best.mape > beam.mape_threshold

            # Early stop only when threshold met AND we have finished at least one expansion
            # if seeds already meet threshold — allow returning seed; otherwise always expand once.
            if best.mape <= beam.mape_threshold and iteration > 1:
                progress = emit(iteration - 1, JobStatus.COMPLETED, "MAPE threshold reached")
                break
            if best.mape <= beam.mape_threshold and iteration == 1 and best.generation == 0:
                # Still run one modification pass to seek novelty/improvement
                exploring = True

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
                    props = self.estimator.estimate(
                        child_smi, prefer_literature=beam.enable_literature
                    )
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
                    cand.safety = estimate_safety(child_smi)
                    critical = set(cand.safety.unstable_motifs) & CRITICAL_MOTIFS
                    if critical or (not cand.safety.is_acceptable and "peroxide" in cand.safety.unstable_motifs):
                        cand.rejected = True
                        cand.rejection_reason = f"safety_gate:{','.join(sorted(critical)) or 'peroxide'}"
                        rejected += 1
                        continue
                    if cand.safety.score >= 0.95:
                        cand.rejected = True
                        cand.rejection_reason = "safety_score_extreme"
                        rejected += 1
                        continue

                    if beam.novelty_as_gate:
                        cand.novelty = self.novelty.evaluate(child_smi)
                        if cand.novelty.status == NoveltyStatus.KNOWN:
                            cand.rejected = True
                            cand.rejection_reason = "novelty_gate"
                            rejected += 1
                            continue

                    cand.score = score_candidate(cand, target_dict, explore=exploring)
                    cand.mape = cand.score.mape
                    seen.add(child_smi)
                    children.append(cand)
                    _tally(cand)

            pool = beam_nodes + children
            for c in pool:
                # Re-score with current explore mode for fair comparison
                c.score = score_candidate(c, target_dict, explore=exploring)
                c.mape = c.score.mape

            beam_nodes = self._select_survivors(pool, beam.top_k, explore=exploring)
            if beam_nodes:
                # Final best: prefer lowest total among survivors; break seed ties by MAPE
                candidate_best = min(
                    beam_nodes,
                    key=lambda c: (
                        c.score.total if c.score else 1e9,
                        c.mape,
                        0 if c.generation > 0 else 1,
                    ),
                )
                if (
                    best.score is None
                    or candidate_best.score.total < best.score.total - 1e-9
                    or (
                        abs(candidate_best.score.total - best.score.total) < 1e-9
                        and candidate_best.mape < best.mape
                    )
                    or (exploring and best.generation == 0 and candidate_best.generation > 0
                        and candidate_best.mape <= best.mape + 1.0)
                ):
                    best = candidate_best

            src_msg = (
                f"xgboost={source_counts['xgboost']} lit={source_counts['literature']} "
                f"dataset={source_counts['dataset']}"
            )
            progress = emit(
                iteration,
                JobStatus.RUNNING,
                f"Iteration {iteration}: +{len(children)} children, {src_msg}",
            )

        else:
            progress = emit(beam.depth, JobStatus.COMPLETED, "Depth limit reached")

        if progress.status == JobStatus.RUNNING:
            progress.status = JobStatus.COMPLETED

        # Final ranking without explore bias
        for c in beam_nodes:
            c.score = score_candidate(c, target_dict, explore=False)
            c.mape = c.score.mape
        top = sorted(beam_nodes, key=lambda c: (c.score.total if c.score else 1e9, c.mape))
        best = top[0] if top else best
        best.score = score_candidate(best, target_dict, explore=False)
        best.mape = best.score.mape
        for c in top:
            c.strip_internal_signals()
        best.strip_internal_signals()
        return best, top, progress
