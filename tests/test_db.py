import pytest

from podterm import db
from podterm.db import connection
from podterm.models import StepMetric


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


# --- F6: partial metric updates must merge, not replace --------------------


def test_partial_metric_update_preserves_prior_fields(temp_db):
    db.create_run("run-1", "pod")

    # First event records val_loss / val_bpb for this step.
    db.add_metric(
        "run-1",
        StepMetric(
            step=5,
            total_steps=100,
            train_loss=2.5,
            train_time_ms=1000,
            step_avg_ms=10.0,
            val_loss=3.0,
            val_bpb=1.2,
        ),
    )
    # A later partial event for the SAME step carries only train_loss/timing;
    # val_loss and val_bpb arrive as None and must NOT erase the prior values.
    db.add_metric(
        "run-1",
        StepMetric(
            step=5,
            total_steps=100,
            train_loss=2.4,
            train_time_ms=1100,
            step_avg_ms=11.0,
            val_loss=None,
            val_bpb=None,
        ),
    )

    rows = db.get_metrics("run-1")
    assert len(rows) == 1
    row = rows[0]
    assert row["train_loss"] == 2.4  # updated by the later event
    assert row["train_time_ms"] == 1100
    assert row["val_loss"] == 3.0  # preserved from the earlier event
    assert row["val_bpb"] == 1.2  # preserved from the earlier event


def test_partial_metric_batch_preserves_prior_fields(temp_db):
    db.create_run("run-1", "pod")

    db.add_metrics_batch(
        "run-1",
        [
            StepMetric(
                step=7,
                total_steps=100,
                train_loss=2.5,
                train_time_ms=1000,
                step_avg_ms=10.0,
                val_loss=3.0,
                val_bpb=1.2,
            )
        ],
    )
    db.add_metrics_batch(
        "run-1",
        [
            StepMetric(
                step=7,
                total_steps=100,
                train_loss=2.4,
                train_time_ms=1100,
                step_avg_ms=11.0,
                val_loss=None,
                val_bpb=None,
            )
        ],
    )

    rows = db.get_metrics("run-1")
    assert len(rows) == 1
    assert rows[0]["val_loss"] == 3.0
    assert rows[0]["val_bpb"] == 1.2
    assert rows[0]["train_loss"] == 2.4


# --- F7: a valid 0.0 train_loss must round-trip, not become NULL -----------


def test_zero_train_loss_round_trips(temp_db):
    db.create_run("run-1", "pod")
    db.add_metric(
        "run-1",
        StepMetric(
            step=1,
            total_steps=10,
            train_loss=0.0,
            train_time_ms=500,
            step_avg_ms=5.0,
        ),
    )

    rows = db.get_metrics("run-1")
    assert len(rows) == 1
    assert rows[0]["train_loss"] == 0.0
    assert rows[0]["train_loss"] is not None


# --- F5: finish_run must be idempotent -------------------------------------


def test_finish_run_idempotent_keeps_first_values(temp_db):
    db.create_run("run-1", "pod")

    db.finish_run("run-1", exit_code=0)
    first = db.get_run("run-1")
    assert first["exit_code"] == 0
    assert first["finished_at"] is not None
    first_finished_at = first["finished_at"]
    first_duration = first["duration_seconds"]

    # A second finalization (e.g. pod_gone after a clean finish) with a
    # different exit code must NOT overwrite the first non-null values.
    db.finish_run("run-1", exit_code=137)
    second = db.get_run("run-1")
    assert second["exit_code"] == 0  # first finish wins
    assert second["finished_at"] == first_finished_at
    assert second["duration_seconds"] == first_duration


def test_finish_run_fills_missing_exit_code_on_later_call(temp_db):
    db.create_run("run-1", "pod")

    # First finalization records finished_at but no exit code yet.
    db.finish_run("run-1", exit_code=None)
    first = db.get_run("run-1")
    assert first["finished_at"] is not None
    assert first["exit_code"] is None
    first_finished_at = first["finished_at"]

    # A later call supplies the exit code; finished_at stays put.
    db.finish_run("run-1", exit_code=2)
    second = db.get_run("run-1")
    assert second["exit_code"] == 2
    assert second["finished_at"] == first_finished_at
