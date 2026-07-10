"""API smoke tests."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure imports resolve
import backend.app.main as main_mod

client = TestClient(main_mod.app)


def test_health():
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_models():
    r = client.get("/api/v1/models")
    assert r.status_code == 200
    assert len(r.json()) >= 1


def test_search_validation_rejects_out_of_range():
    r = client.post(
        "/api/v1/search",
        json={
            "targets": {
                "density": 99,
                "det_velocity": 8500,
                "det_pressure": 30,
                "hf_solid": 100,
            },
            "beam": {"depth": 1, "width": 1, "top_k": 1},
        },
    )
    assert r.status_code == 422
