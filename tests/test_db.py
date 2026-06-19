import pytest

from podterm import db
from podterm.db import connection


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    db.close()
    monkeypatch.setattr(connection, "DB_DIR", str(tmp_path))
    monkeypatch.setattr(connection, "DB_PATH", str(tmp_path / "runs.db"))
    yield
    db.close()


def test_update_run_rejects_unknown_columns(temp_db):
    db.create_run("run-1", "pod")
    with pytest.raises(ValueError, match="unknown run field"):
        db.update_run("run-1", not_a_column="x")


def test_update_run_accepts_known_columns(temp_db):
    db.create_run("run-1", "pod")
    db.update_run("run-1", gpu_type="RTX 5090", total_steps=10)

    row = db.get_run("run-1")
    assert row["gpu_type"] == "RTX 5090"
    assert row["total_steps"] == 10
