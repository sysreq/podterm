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
