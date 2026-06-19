"""End-to-end golden test for the diagnostics suite.

Requires gpt-golf's torch env (the suite imports the model architecture + tokenizer). It is SKIPPED
where torch/train_gpt are unavailable — i.e. in podterm's own venv, the suite only ever runs as a
subprocess under gpt-golf's env. Run it there:

    GPT_GOLF_DIR=../gpt-golf PYTHONPATH=../gpt-golf:. uv run --with pytest pytest tests/test_diagnostics_golden.py
"""
import json
import os
import sys
from pathlib import Path

import pytest

# Make the gpt-golf checkout importable (same as the real subprocess: cwd=gpt-golf, podterm on path).
_GPT_GOLF = os.environ.get("GPT_GOLF_DIR", str(Path(__file__).resolve().parents[2] / "gpt-golf"))
if _GPT_GOLF not in sys.path:
    sys.path.insert(0, _GPT_GOLF)

# Device is fixed at import (config_gpt reads DEVICE via a frozen dataclass) — default to CPU so the
# test is hermetic. Run with DEVICE=cuda to exercise the GPU path.
os.environ.setdefault("DEVICE", "cpu")

torch = pytest.importorskip("torch")
train_gpt = pytest.importorskip("train_gpt")
config_gpt = pytest.importorskip("config_gpt")

from podterm.diagnostics import health  # noqa: E402


requires_cpu = pytest.mark.skipif(config_gpt.HW.device != "cpu",
                                  reason="weights-only golden path assumes DEVICE=cpu")


@pytest.fixture(scope="module")
def cpu_doc(tmp_path_factory):
    """Run the full suite weights-only on CPU (no dataset needed) and return the written doc."""
    from podterm.diagnostics.attention import sdpa_flex
    train_gpt.flex_attention = sdpa_flex
    model = train_gpt.GPT().to("cpu")
    from podterm.diagnostics.runner import run_diagnostics
    out = tmp_path_factory.mktemp("diag") / "golden.diag.json"
    run_diagnostics(model, optimizer=None, val_tokens=None, json_path=str(out), extra_meta={"step": 7})
    return json.loads(Path(out).read_text())


@requires_cpu
def test_doc_structure_and_verdict(cpu_doc):
    assert cpu_doc["version"] == 2
    assert cpu_doc["sections"], "expected at least the weights-only sections"
    assert cpu_doc["meta"]["step"] == 7
    # runner injects a value-based verdict + headline block.
    assert cpu_doc["status"] in ("ok", "warn", "error")
    assert "health" in cpu_doc and cpu_doc["health"]["overall"] == cpu_doc["status"]


@requires_cpu
def test_health_recompute_matches_embedded(cpu_doc):
    recomputed = health.compute(cpu_doc)
    assert recomputed["overall"] == cpu_doc["health"]["overall"]
    assert [h["id"] for h in recomputed["headline"]] == [h["id"] for h in cpu_doc["health"]["headline"]]


@requires_cpu
def test_static_headline_resolves_forward_is_na(cpu_doc):
    bands = {h["id"]: h["band"] for h in cpu_doc["health"]["headline"]}
    # escape + eff_rank are weights-only → must resolve on a CPU run.
    assert bands["escape_min"] != "na"
    assert bands["eff_rank"] != "na"
    # forward-pass metrics are skipped on CPU → na, and must not poison the verdict.
    assert bands["dead_neurons"] == "na"
    assert bands["logit_saturation"] == "na"


@pytest.mark.skipif(config_gpt.HW.device != "cuda" or not torch.cuda.is_available(),
                    reason="forward-pass correctness needs the process launched with DEVICE=cuda + a GPU + val data")
def test_forward_checks_pass_on_gpu(tmp_path):
    """On GPU with the val shard present, the suite's self-checks must pass and BPB must be emitted."""
    from podterm.diagnostics.attention import sdpa_flex
    train_gpt.flex_attention = sdpa_flex
    model = train_gpt.GPT().to("cuda").bfloat16()
    val_tokens = train_gpt.load_validation_tokens()
    from podterm.diagnostics.runner import run_diagnostics
    out = tmp_path / "gpu.diag.json"
    run_diagnostics(model, optimizer=None, val_tokens=val_tokens, json_path=str(out))
    doc = json.loads(out.read_text())
    checks = doc["meta"]["checks"]
    assert checks["loss_recompute"] == "ok"
    assert all(v == "ok" for v in checks["recompose"].values())
    assert isinstance(checks.get("bpb"), float) and checks["bpb"] > 0
