"""F1: the event-daemon bearer token (and any other secret) must never reach
the browser. The run-list, run-detail, and compare routes parse the stored
``config_json``, redact secret-keyed values server-side, and re-serialize, so
the token is gone before the response leaves the server.

The route handlers are exercised directly (the same pattern as
``test_routes.py``) so no extra HTTP-client dependency is needed."""

import asyncio
import json

import pytest

from podterm import db
from podterm.db import connection
from podterm.routes import runs

TOKEN = "super-secret-eventd-bearer-0xDEADBEEF"


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    db.close()
    monkeypatch.setattr(connection, "DB_DIR", str(tmp_path))
    monkeypatch.setattr(connection, "DB_PATH", str(tmp_path / "runs.db"))
    yield
    db.close()


def _seed_run(run_id: str = "run-secret") -> dict:
    config = {
        "branch": "main",
        "gpu": "RTX 5090",
        "batch_tokens": 524288,
        "tokenizer_path": "./data/tokenizers/fineweb.model",
        "eventd_token": TOKEN,
    }
    db.create_run(run_id, f"pod-{run_id}", config)
    return config


# -- pure helpers ----------------------------------------------------------


def test_redact_masks_secret_keys_recursively():
    out = runs._redact(
        {
            "eventd_token": TOKEN,
            "nested": {"api_key": "abc", "batch_tokens": 7},
            "items": [{"client_secret": "x"}, {"branch": "main"}],
            "tokenizer_path": "./tok.model",
        }
    )
    assert out["eventd_token"] == runs._REDACTED
    assert out["nested"]["api_key"] == runs._REDACTED
    assert out["nested"]["batch_tokens"] == 7  # not a secret
    assert out["items"][0]["client_secret"] == runs._REDACTED
    assert out["items"][1]["branch"] == "main"
    assert out["tokenizer_path"] == "./tok.model"  # "token" is part of "tokenizer"


def test_redact_does_not_mutate_input():
    src = {"eventd_token": TOKEN, "nested": {"api_key": "abc"}}
    runs._redact(src)
    assert src["eventd_token"] == TOKEN
    assert src["nested"]["api_key"] == "abc"


def test_sanitize_run_handles_none_and_missing_config():
    assert runs._sanitize_run(None) is None
    assert runs._sanitize_run({"run_id": "r", "seed": 1}) == {"run_id": "r", "seed": 1}


def test_sanitize_run_drops_unparseable_config():
    out = runs._sanitize_run({"run_id": "r", "config_json": "{not json"})
    assert out["config_json"] is None


# -- route integration -----------------------------------------------------


def test_run_detail_does_not_leak_token(temp_db):
    _seed_run()
    out = asyncio.run(runs.get_run_row("run-secret"))

    assert TOKEN not in json.dumps(out)
    cfg = json.loads(out["config_json"])
    assert cfg["eventd_token"] == runs._REDACTED
    # non-secret reproducibility fields survive for the frontend
    assert cfg["branch"] == "main"
    assert cfg["batch_tokens"] == 524288
    assert cfg["tokenizer_path"] == "./data/tokenizers/fineweb.model"


def test_run_list_does_not_leak_token(temp_db):
    _seed_run()
    rows = asyncio.run(runs.list_runs(limit=100))

    assert TOKEN not in json.dumps(rows)
    assert len(rows) == 1
    cfg = json.loads(rows[0]["config_json"])
    assert cfg["eventd_token"] == runs._REDACTED
    assert cfg["branch"] == "main"


def test_compare_does_not_leak_token(temp_db):
    _seed_run()
    db.create_run("run-other", "pod-other", {"branch": "dev", "eventd_token": TOKEN})

    out = asyncio.run(runs.compare_runs(runs.CompareRequest(run_ids=["run-secret", "run-other"])))

    assert TOKEN not in json.dumps(out)
    cfg = json.loads(out["runs"]["run-secret"]["config_json"])
    assert cfg["eventd_token"] == runs._REDACTED
