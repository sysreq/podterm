"""Unit tests for the health-verdict layer (stdlib-only, runs anywhere)."""
import copy

from podterm.diagnostics import health


# A contract fixture mirroring the *actual* section/row/key names the stages emit. If a stage renames
# a section/key, the `test_every_headline_resolves` test breaks here — that's the point: it guards
# health.py's registry against producer schema drift without needing a live torch run.
def _full_doc():
    return {
        "version": 2,
        "sections": [
            {"name": "capacity: dead neurons", "status": "ok",
             "rows": {"enc1": {"dead_gate": 0.01, "dead_up": 0.02}, "global": {"dead": 0.03}}},
            {"name": "capacity: head output redundancy", "status": "ok",
             "rows": {"enc1": {"mean": 0.2, "std": 0.1, "min": 0.0, "max": 0.5}}},
            {"name": "capacity: attention entropy (bits)", "status": "ok",
             "rows": {"enc1": {"entropy": [3.0, 2.5], "oneshot": [0.1, 0.2]}}},
            {"name": "flow: logit saturation", "status": "ok",
             "rows": {"logits": {"sat": 0.05}}},
            {"name": "flow: residual norm", "status": "ok",
             "rows": {"stream": {"emb": 1.0, "enc1": 1.2, "final": 2.0}}},
            {"name": "zero-init escape", "status": "ok",
             "rows": {"enc1": {"atn_proj": 3.0, "mlp_proj": 2.5}, "lm_head": {"escape": 4.0}}},
            {"name": "gradients", "status": "ok",
             "rows": {"enc1": {"qkv_proj": 0.05}, "summary": {"max": 0.1, "min": 0.001, "ratio": 100.0}}},
            {"name": "flow: per-position loss", "status": "ok",
             "rows": {"loss": {"per_pos": [5.0, 4.0, 3.0], "0:1": 5.0}}},
            {"name": "embedding spectrum", "status": "ok",
             "rows": {"wte": {"eff_rank": 100.0, "dims_90": 50}, "lm_head": {"eff_rank": 120.0}}},
        ],
    }


def test_every_headline_resolves_on_full_doc():
    doc = _full_doc()
    for entry in health.HEADLINE:
        val = health.resolve(doc, entry)
        assert val is not None, f"{entry['id']} did not resolve — section/row/key drift?"


def test_reduce_model():
    doc = _full_doc()
    by_id = {e["id"]: e for e in health.HEADLINE}
    # value
    assert health.resolve(doc, by_id["dead_neurons"]) == 0.03
    # max across rows of key 'max'
    assert health.resolve(doc, by_id["head_redundancy"]) == 0.5
    # min over a per-head list (across rows)
    assert health.resolve(doc, by_id["attn_entropy_min"]) == 2.5
    # mean over a list
    assert health.resolve(doc, by_id["mean_loss"]) == 4.0
    # key=None collects every numeric leaf across rows, then min (3.0, 2.5, 4.0 -> 2.5)
    assert health.resolve(doc, by_id["escape_min"]) == 2.5
    # mean across rows
    assert health.resolve(doc, by_id["eff_rank"]) == 110.0


def test_classify_bands():
    hi = dict(kind="hi", warn=0.05, bad=0.20)
    assert health.classify(0.0, hi) == "good"
    assert health.classify(0.05, hi) == "warn"   # boundary is inclusive
    assert health.classify(0.20, hi) == "bad"
    lo = dict(kind="lo", warn=0.5, bad=0.1)
    assert health.classify(1.0, lo) == "good"
    assert health.classify(0.5, lo) == "warn"
    assert health.classify(0.1, lo) == "bad"
    rng = dict(kind="range", warn_lo=0.5, warn_hi=4.0, bad_lo=0.2, bad_hi=10.0)
    assert health.classify(2.0, rng) == "good"
    assert health.classify(0.3, rng) == "warn"
    assert health.classify(0.1, rng) == "bad"
    assert health.classify(11.0, rng) == "bad"
    assert health.classify(None, hi) == "na"
    assert health.classify(5.0, dict(kind="info")) == "info"


def test_overall_is_worst_of():
    # A clean doc reads ok.
    assert health.compute(_full_doc())["overall"] == "ok"

    # A single warn-band metric pulls the verdict to warn.
    doc = _full_doc()
    doc["sections"][1]["rows"]["enc1"]["max"] = 0.7  # head redundancy warn
    assert health.compute(doc)["overall"] == "warn"

    # A bad-band metric pulls it to error.
    doc = _full_doc()
    doc["sections"][0]["rows"]["global"]["dead"] = 0.5  # dead neurons bad
    assert health.compute(doc)["overall"] == "error"

    # A section execution error alone is also error, independent of values.
    doc = _full_doc()
    doc["sections"][3]["status"] = "error"
    assert health.compute(doc)["overall"] == "error"


def test_skipped_section_degrades_to_na_not_error():
    doc = _full_doc()
    # Drop the forward-pass sections (as a CPU/weights-only run would).
    forward = {"capacity: dead neurons", "capacity: head output redundancy",
               "capacity: attention entropy (bits)", "flow: logit saturation",
               "flow: residual norm", "gradients", "flow: per-position loss"}
    doc["sections"] = [s for s in doc["sections"] if s["name"] not in forward]
    out = health.compute(doc)
    assert out["overall"] == "ok"            # missing != sick
    assert out["counts"]["na"] >= len(forward) - 1
    deads = [h for h in out["headline"] if h["id"] == "dead_neurons"][0]
    assert deads["band"] == "na" and deads["value"] is None


def test_headline_order_stable():
    ids = [h["id"] for h in health.compute(_full_doc())["headline"]]
    assert ids == [e["id"] for e in health.HEADLINE]


def test_headline_carries_thresholds_and_why():
    # The UI renders threshold hints + rationale from these without hardcoding numbers.
    for h in health.compute(_full_doc())["headline"]:
        assert "thresholds" in h and isinstance(h["thresholds"], dict) and "kind" in h["thresholds"], h
        assert "why" in h and isinstance(h["why"], str) and h["why"], h
    # info-kind metrics still carry a thresholds dict of kind 'info'.
    by_id = {h["id"]: h for h in health.compute(_full_doc())["headline"]}
    assert by_id["mean_loss"]["thresholds"]["kind"] == "info"
    # serialized thresholds must not alias the registry band dict (callers may mutate).
    by_entry = {e["id"]: e for e in health.HEADLINE}
    assert by_id["dead_neurons"]["thresholds"] is not by_entry["dead_neurons"]["band"]


def test_compute_does_not_mutate_input():
    doc = _full_doc()
    before = copy.deepcopy(doc)
    health.compute(doc)
    assert doc == before


# --- Malformed-schema hardening (finding 16) ---------------------------------------------------
# A section can be present and 'ok' yet missing the exact row/key a headline entry expects (producer
# schema drift, a partial stage write). And a stage can emit NaN/inf for a degenerate metric. Neither
# may KeyError or masquerade as a passing band.

def test_missing_row_resolves_to_none_no_keyerror():
    # 'capacity: dead neurons' is present and ok, but its expected row 'global' is gone.
    doc = _full_doc()
    by_id = {e["id"]: e for e in health.HEADLINE}
    del doc["sections"][0]["rows"]["global"]
    # resolve() must return None (per its contract), not raise KeyError on rows[row].
    assert health.resolve(doc, by_id["dead_neurons"]) is None


def test_missing_key_in_present_row_resolves_to_none():
    # Row present but the expected key absent -> nothing collected -> None.
    doc = _full_doc()
    by_id = {e["id"]: e for e in health.HEADLINE}
    del doc["sections"][3]["rows"]["logits"]["sat"]
    assert health.resolve(doc, by_id["logit_saturation"]) is None


def test_nan_metric_does_not_classify_good():
    # A NaN dead-neuron fraction must not slip through to 'good' (NaN >= bad is False, NaN >= warn is
    # False, so a naive path would land in 'good'). It should be treated as unavailable -> None -> na.
    doc = _full_doc()
    by_id = {e["id"]: e for e in health.HEADLINE}
    doc["sections"][0]["rows"]["global"]["dead"] = float("nan")
    assert health.resolve(doc, by_id["dead_neurons"]) is None
    out = health.compute(doc)
    dead = [h for h in out["headline"] if h["id"] == "dead_neurons"][0]
    assert dead["band"] == "na" and dead["value"] is None


def test_inf_metric_does_not_classify_good():
    doc = _full_doc()
    by_id = {e["id"]: e for e in health.HEADLINE}
    doc["sections"][0]["rows"]["global"]["dead"] = float("inf")
    assert health.resolve(doc, by_id["dead_neurons"]) is None
    dead = [h for h in health.compute(doc)["headline"] if h["id"] == "dead_neurons"][0]
    assert dead["band"] == "na" and dead["value"] is None


def test_nan_inside_list_is_dropped():
    # Per-head lists with a NaN entry keep their finite members for the reduce.
    doc = _full_doc()
    by_id = {e["id"]: e for e in health.HEADLINE}
    doc["sections"][2]["rows"]["enc1"]["entropy"] = [float("nan"), 3.0, 2.5]
    assert health.resolve(doc, by_id["attn_entropy_min"]) == 2.5


def test_overall_unaffected_by_missing_rows():
    # Dropping rows from healthy sections degrades the affected metrics to na but never poisons the
    # verdict: missing != sick, so the overall stays 'ok'.
    doc = _full_doc()
    del doc["sections"][0]["rows"]["global"]        # dead_neurons -> na
    del doc["sections"][3]["rows"]["logits"]        # logit_saturation -> na
    out = health.compute(doc)
    assert out["overall"] == "ok"
    bands = {h["id"]: h["band"] for h in out["headline"]}
    assert bands["dead_neurons"] == "na"
    assert bands["logit_saturation"] == "na"


def test_completeness_reflects_evaluated_verdict_metrics():
    # Additive completeness block: a full doc evaluates every verdict-bearing metric; a weights-only
    # (forward-section-free) doc drops to a small fraction without changing the overall verdict.
    out_full = health.compute(_full_doc())
    comp = out_full["completeness"]
    assert comp["total"] == sum(1 for e in health.HEADLINE if e["band"].get("kind") != "info")
    assert comp["evaluated"] == comp["total"] and comp["fraction"] == 1.0

    doc = _full_doc()
    forward = {"capacity: dead neurons", "capacity: head output redundancy",
               "capacity: attention entropy (bits)", "flow: logit saturation",
               "flow: residual norm", "gradients", "flow: per-position loss"}
    doc["sections"] = [s for s in doc["sections"] if s["name"] not in forward]
    out_sparse = health.compute(doc)
    assert out_sparse["overall"] == "ok"                         # still not sick
    assert out_sparse["completeness"]["evaluated"] < out_full["completeness"]["evaluated"]
    assert out_sparse["completeness"]["fraction"] < 1.0
