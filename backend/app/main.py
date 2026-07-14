"""FastAPI application entrypoint."""

from __future__ import annotations

import csv
import io
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from sse_starlette.sse import EventSourceResponse

from backend.app.core.config import get_settings
from backend.app.domain.models import (
    BeamParameters,
    JobStatus,
    ModelMetadata,
    SearchProgress,
    SearchRequest,
    SearchResult,
)
from backend.app.db.jobs import JobStore
from backend.app.services.beam_search import BeamSearchController
from backend.app.services.modification import ModificationMechanism
from backend.app.services.molecule_identity import MoleculeIdentityResolver
from backend.app.services.novelty import NoveltyService
from backend.app.services.prediction import PropertyPredictor
from backend.app.services.property_estimator import LiteratureRetriever, PropertyEstimator
from backend.app.services.seed_selector import SeedLibrary
from backend.app.services.structure import smiles_to_png

settings = get_settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)

store = JobStore(str(settings.database_url))
_controllers: dict[str, BeamSearchController] = {}


def _build_controller(beam: BeamParameters | None = None) -> BeamSearchController:
    seeds = SeedLibrary(settings.seed_csv)
    predictor = PropertyPredictor(settings.models_dir)
    literature = LiteratureRetriever(enabled=True)
    # Exact experimental seed values only (no analogue transfer)
    literature.load_seed_library(seeds.df.to_dict(orient="records"))
    if beam is not None:
        literature.enabled = bool(beam.enable_literature)

    estimator = PropertyEstimator(predictor, literature)
    identity = MoleculeIdentityResolver()
    novelty = NoveltyService(
        seed_smiles=seeds.all_canonical_smiles(),
        pubchem_enabled=settings.pubchem_enabled and (beam.enable_literature if beam else True),
        openalex_enabled=settings.openalex_enabled and (beam.enable_literature if beam else True),
        epo_ops_key=settings.epo_ops_key if (beam is None or beam.enable_patent_search) else "",
        lens_api_key=settings.lens_api_key if (beam is None or beam.enable_patent_search) else "",
        identity_resolver=identity,
    )
    mod = ModificationMechanism(checkpoint_dir=settings.learned_checkpoint_dir)
    return BeamSearchController(seeds, estimator, novelty, mod)


def _clamp_beam(beam: BeamParameters) -> BeamParameters:
    return BeamParameters(
        depth=min(beam.depth, settings.max_beam_depth),
        width=min(beam.width, settings.max_beam_width),
        top_k=min(beam.top_k, settings.max_top_k),
        mape_threshold=beam.mape_threshold,
        random_seed=beam.random_seed,
        enable_literature=beam.enable_literature,
        enable_patent_search=beam.enable_patent_search,
        novelty_as_gate=beam.novelty_as_gate,
    )


def _run_job(job_id: str) -> None:
    try:
        req = store.get_request(job_id)
        beam = _clamp_beam(req.beam)
        controller = _build_controller(beam)
        _controllers[job_id] = controller

        def on_progress(progress: SearchProgress, _best) -> None:
            if store.is_cancelled(job_id):
                controller.request_stop()
            store.update_progress(progress)

        best, top, progress = controller.run(job_id, req.targets, beam, on_progress=on_progress)
        result = SearchResult(
            job_id=job_id,
            status=progress.status,
            best=best,
            top_k=top,
            progress=progress,
        )
        store.set_result(result)
    except Exception as e:
        progress = SearchProgress(
            job_id=job_id,
            status=JobStatus.FAILED,
            message=str(e),
        )
        store.update_progress(progress)
        store.set_result(
            SearchResult(job_id=job_id, status=JobStatus.FAILED, progress=progress)
        )
    finally:
        _controllers.pop(job_id, None)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    yield


app = FastAPI(title="IGNITE API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health")
def health():
    return {"status": "ok", "service": "ignite"}


@app.get("/api/v1/molecule/depict")
def depict_molecule(smiles: str, size: int = 320):
    """Return a 2D PNG depiction of the molecule (RDKit)."""
    size = max(64, min(int(size), 800))
    png = smiles_to_png(smiles, size=size)
    if png is None:
        raise HTTPException(400, "Invalid SMILES")
    return Response(content=png, media_type="image/png")


@app.get("/api/v1/models", response_model=List[ModelMetadata])
def list_models():
    predictor = PropertyPredictor(settings.models_dir)
    return predictor.list_metadata()


@app.post("/api/v1/search", response_model=SearchProgress)
def create_search(request: SearchRequest):
    request.beam = _clamp_beam(request.beam)
    job_id = store.create(request)
    threading.Thread(target=_run_job, args=(job_id,), daemon=True).start()
    return SearchProgress(job_id=job_id, status=JobStatus.QUEUED, message="Queued", max_iterations=request.beam.depth)


@app.get("/api/v1/search/{job_id}/progress", response_model=SearchProgress)
def get_progress(job_id: str):
    try:
        return store.get_progress(job_id)
    except KeyError:
        raise HTTPException(404, "Job not found")


@app.get("/api/v1/search/{job_id}/events")
async def search_events(job_id: str):
    async def gen():
        import asyncio

        last = None
        while True:
            try:
                prog = store.get_progress(job_id)
            except KeyError:
                yield {"event": "error", "data": "not found"}
                break
            payload = prog.model_dump_json()
            if payload != last:
                yield {"event": "progress", "data": payload}
                last = payload
            if prog.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                break
            await asyncio.sleep(0.5)

    return EventSourceResponse(gen())


@app.get("/api/v1/search/{job_id}", response_model=SearchResult)
def get_result(job_id: str):
    result = store.get_result(job_id)
    if result is None:
        try:
            prog = store.get_progress(job_id)
        except KeyError:
            raise HTTPException(404, "Job not found")
        return SearchResult(job_id=job_id, status=prog.status, progress=prog)
    return result


@app.post("/api/v1/search/{job_id}/cancel", response_model=SearchProgress)
def cancel_search(job_id: str):
    store.cancel(job_id)
    ctrl = _controllers.get(job_id)
    if ctrl:
        ctrl.request_stop()
    prog = store.get_progress(job_id)
    prog.status = JobStatus.CANCELLED
    prog.message = "Cancel requested"
    store.update_progress(prog)
    return prog


@app.get("/api/v1/search/{job_id}/export.csv")
def export_csv(job_id: str):
    result = store.get_result(job_id)
    if not result or not result.top_k:
        raise HTTPException(404, "No results")
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        ["rank", "smiles", "mape", "score", "density", "det_velocity", "det_pressure", "hf_solid"]
    )
    for i, c in enumerate(result.top_k, 1):
        props = c.property_values()
        writer.writerow(
            [
                i,
                c.canonical_smiles,
                c.mape,
                c.score.total if c.score else "",
                props.get("density", ""),
                props.get("det_velocity", ""),
                props.get("det_pressure", ""),
                props.get("hf_solid", ""),
            ]
        )
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=ignite_{job_id}.csv"},
    )


@app.get("/api/v1/search/{job_id}/export.sdf")
def export_sdf(job_id: str):
    from rdkit import Chem

    result = store.get_result(job_id)
    if not result or not result.top_k:
        raise HTTPException(404, "No results")
    blocks = []
    for c in result.top_k:
        mol = Chem.MolFromSmiles(c.canonical_smiles)
        if mol is None:
            continue
        blocks.append(Chem.MolToMolBlock(mol) + f"> <MAPE>\n{c.mape}\n\n$$$$\n")
    return StreamingResponse(
        iter(blocks),
        media_type="chemical/x-mdl-sdfile",
        headers={"Content-Disposition": f"attachment; filename=ignite_{job_id}.sdf"},
    )
