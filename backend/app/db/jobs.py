"""In-memory / SQLite job persistence for search runs."""

from __future__ import annotations

import asyncio
import json
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional
from uuid import uuid4

from sqlalchemy import DateTime, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from backend.app.domain.models import (
    CandidateMolecule,
    JobStatus,
    SearchProgress,
    SearchRequest,
    SearchResult,
)


class Base(DeclarativeBase):
    pass


class SearchJobRow(Base):
    __tablename__ = "search_jobs"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(32), default=JobStatus.QUEUED.value)
    request_json: Mapped[str] = mapped_column(Text)
    result_json: Mapped[str] = mapped_column(Text, default="")
    progress_json: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class JobStore:
    def __init__(self, database_url: str = "sqlite:///./data/ignite.db"):
        # sync engine for simplicity in worker threads
        url = database_url.replace("sqlite+aiosqlite", "sqlite")
        self.engine = create_engine(url, future=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(self.engine, expire_on_commit=False)
        self._locks: Dict[str, threading.Event] = {}
        self._cancel: Dict[str, bool] = {}
        self._progress: Dict[str, SearchProgress] = {}
        self._results: Dict[str, SearchResult] = {}

    def create(self, request: SearchRequest) -> str:
        job_id = str(uuid4())
        with self.Session() as s:
            row = SearchJobRow(
                id=job_id,
                status=JobStatus.QUEUED.value,
                request_json=request.model_dump_json(),
                progress_json=SearchProgress(
                    job_id=job_id, status=JobStatus.QUEUED, message="Queued"
                ).model_dump_json(),
            )
            s.add(row)
            s.commit()
        self._cancel[job_id] = False
        return job_id

    def get_request(self, job_id: str) -> SearchRequest:
        with self.Session() as s:
            row = s.get(SearchJobRow, job_id)
            if not row:
                raise KeyError(job_id)
            return SearchRequest.model_validate_json(row.request_json)

    def update_progress(self, progress: SearchProgress) -> None:
        self._progress[progress.job_id] = progress
        with self.Session() as s:
            row = s.get(SearchJobRow, progress.job_id)
            if row:
                row.status = progress.status.value
                row.progress_json = progress.model_dump_json()
                row.updated_at = datetime.now(timezone.utc)
                s.commit()

    def set_result(self, result: SearchResult) -> None:
        self._results[result.job_id] = result
        with self.Session() as s:
            row = s.get(SearchJobRow, result.job_id)
            if row:
                row.status = result.status.value
                row.result_json = result.model_dump_json()
                row.progress_json = result.progress.model_dump_json()
                row.updated_at = datetime.now(timezone.utc)
                s.commit()

    def get_progress(self, job_id: str) -> SearchProgress:
        if job_id in self._progress:
            return self._progress[job_id]
        with self.Session() as s:
            row = s.get(SearchJobRow, job_id)
            if not row:
                raise KeyError(job_id)
            return SearchProgress.model_validate_json(row.progress_json)

    def get_result(self, job_id: str) -> Optional[SearchResult]:
        if job_id in self._results:
            return self._results[job_id]
        with self.Session() as s:
            row = s.get(SearchJobRow, job_id)
            if not row or not row.result_json:
                return None
            return SearchResult.model_validate_json(row.result_json)

    def cancel(self, job_id: str) -> None:
        self._cancel[job_id] = True

    def is_cancelled(self, job_id: str) -> bool:
        return bool(self._cancel.get(job_id))
