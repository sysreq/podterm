import asyncio

import pytest
from fastapi import HTTPException

from podterm.routes import runs


def test_compare_requires_two_runs():
    with pytest.raises(HTTPException) as exc:
        asyncio.run(runs.compare_runs(runs.CompareRequest(run_ids=["one"])))
    assert exc.value.status_code == 400


def test_compare_uses_typed_request(monkeypatch):
    monkeypatch.setattr(runs.db, "get_metrics_multi", lambda run_ids: {rid: [] for rid in run_ids})
    monkeypatch.setattr(runs.db, "get_run", lambda rid: {"run_id": rid})
    monkeypatch.setattr(runs.db, "get_diagnostics", lambda rid: [])

    out = asyncio.run(runs.compare_runs(runs.CompareRequest(run_ids=["a", "b"])))

    assert out["metrics"] == {"a": [], "b": []}
    assert out["runs"] == {"a": {"run_id": "a"}, "b": {"run_id": "b"}}
    assert out["diagnostics"]["base"] == "a"


def test_get_run_row(monkeypatch):
    monkeypatch.setattr(runs.db, "get_run", lambda rid: {"run_id": rid, "seed": 123})

    out = asyncio.run(runs.get_run_row("run-1"))

    assert out == {"run_id": "run-1", "seed": 123}


def test_get_run_row_404(monkeypatch):
    monkeypatch.setattr(runs.db, "get_run", lambda rid: None)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(runs.get_run_row("missing"))

    assert exc.value.status_code == 404
