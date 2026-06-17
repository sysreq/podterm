# PodTerm — Pre-Demo Code-Quality Backlog

## Context

We want to put PodTerm's best foot forward for a code-review demo where reviewers may
"look under the hood." The codebase is small and generally well-structured (good module
separation, thoughtful docstrings, a documented architecture in CLAUDE.md), so the goal is
not a rewrite — it's removing the rough edges a senior/staff reviewer would flag: scattered
duplication, a couple of oversized files/functions, a few silent-failure smells, missing
project hygiene (lint/CI/README), and thin test coverage on pure logic.

This plan is a **prioritized, validated backlog**. Every item below was confirmed by reading
the actual code. Several findings from the first-pass scan were **false positives** and are
explicitly excluded at the bottom so we don't burn demo time on non-issues.

Scope reviewed: all of `podterm/` (backend + `runpod/` + `diagnostics/`), `static/` (JS/CSS/HTML),
`tests/`, and project config (`pyproject.toml`, `.gitignore`, `.env.example`).

---

## Theme A — Project hygiene / engineering maturity (high demo signal, low risk)

- **A1 — No linter/formatter configured.** `pyproject.toml` declares only `pytest`. Add a
  `[tool.ruff]` section (lint + format) and run `ruff format` once across the tree. This is the
  single highest-signal "we have standards" change for a demo.
- **A2 — No CI.** There is no `.github/workflows/`. Add a minimal GitHub Actions workflow:
  `ruff check`, `ruff format --check`, `pytest`, and `node --test` for the JS test (A4).
- **A3 — No human-facing README.** Only `CLAUDE.md` (agent-facing) and `diagnostics/README.md`
  exist. Add a short root `README.md`: what PodTerm is, how to run it (`uv run podterm`), and a
  pointer to the architecture notes.
- **A4 — Orphaned JS test.** `static/js/derive.test.mjs` has no runner (no `package.json`); it
  is never executed. Wire `node --test static/js/derive.test.mjs` into CI (A2).
- **A5 — User-Agent version drift.** `runpod/api.py:102` sends `podterm/0.1` while
  `events.py:71` / `snapshots.py:68` send `podterm/2.0` and `pyproject.toml` says `2.0.0`.
  Collapse to one `USER_AGENT` constant (folds into B1).

## Theme B — Duplication / single source of truth (core of the request)

- **B1 — Duplicated pod-daemon HTTP client.** `events.py:_request` (66-79) and
  `snapshots.py:_request` (66-76) are near-identical (same Bearer+UA header build, same
  `HTTPError`→code / `Exception`→0 fallback). The proxy base-URL builder is also duplicated
  (`events.py:44` vs `snapshots.py:_base_url` 61-63), including the `PODTERM_EVENTS_URL`
  override. Extract one tiny helper (e.g. `podterm/podhttp.py`: `pod_base_url(pod_id)` +
  `request(url, token, timeout)`), reuse a single `USER_AGENT`. **Net LOC goes down.** Leave the
  `runpod/api.py`, `runpod/console.py`, and `helpers.py` urllib uses alone — they have different
  auth shapes and are intentionally best-effort.
- **B2 — Load-bearing phase strings as scattered literals.** `"Starting Training"` /
  `"Training finished"` are matched by substring in 5 places (`pipeline.py:110,113`,
  `events.py:222`, `state.js:198,199`) — exactly the strings CLAUDE.md calls "load-bearing,"
  yet duplicated. Define named constants once per side (a backend module, e.g. `models.py` or
  `config.py`; one JS module, e.g. `state.js`) and import them. Removes the documented
  magic-string footgun.
- **B3 — Duplicated frontend metric derivation.** `compare.js` (69-72) and `live.js` (506-509)
  both do `metrics.filter(m => m.train_loss)` → `steps`/`values`/`scaled`. Extract into
  `derive.js` (already the home for derived series) and call from both.

## Theme C — Function / file bloat & readability (the "file bloat" ask)

- **C1 — `static/js/live.js` is 553 lines** mixing KPI-card rendering, baseline/race logic,
  telemetry cards, and SSE event wiring. Split into ~2-3 focused modules (e.g. `live-kpis.js`,
  `live-baseline.js`, and a thin `live.js` orchestrator). Largest file-bloat item.
- **C2 — `pipeline.py:drain_loop` is a ~90-line `if/elif` over 11 event types** (49-138).
  Refactor to a dispatch table of small handler methods (`{"metric": self._on_metric, ...}`).
  Keep it simple — a dict of methods, not a framework — so each handler is unit-testable.
- **C3 — `diagnostics/stages_forward.py` is 218 lines** with a dense ~14-line multi-statement
  init block (21-34), a deeply nested forward loop (40-63), and a duplicated dead-neuron
  calculation (84-92). Extract an init helper + a per-block helper + `_dead_fraction()`, and
  lift magic thresholds (`REL_TOL`, `LOSS_TOL`, `0.001`, `0.95`, …) to named module constants.

## Theme D — Error handling & robustness

- **D1 — Silent `except Exception: pass` that hides real bugs.** `pipeline.py` `finalize_run`
  (44-45) and the metrics batch flush (131-134) swallow all errors with no log — a DB failure
  vanishes. At minimum `log.warning`/`log.exception`. Note `telemetry_loop` (154) already logs,
  so this is also an internal *consistency* fix. (The best-effort sentinels in `runpod/*`,
  `helpers.py`, and the pod-daemon `_request` are intentional — leave as-is, optionally a debug log.)
- **D2 — Untyped request body on `/api/compare`.** `routes/runs.py:29` takes `body: dict` and
  hand-validates `run_ids`. Replace with a Pydantic model (`CompareRequest`) for automatic
  validation + OpenAPI schema — idiomatic FastAPI and a nice thing to show in a demo.

## Theme E — Test coverage on pure logic (supports the demo story)

- **E1 — Add focused unit tests** for logic that needs no torch/network: `boot.PullTracker`
  (the layer state machine — currently 0 tests despite being intricate), the `config.py` flag
  parsers, `snapshots._overall_status`, and the new phase constants (B2). `derive.test.mjs`
  already exists for the JS side — just get it running (A4). Diagnostics stages need
  torch + the gpt-golf env, so deprioritize those.

## Theme F — Lower-priority polish (batch into one cleanup pass)

- **F1** — Inline styles in `launch.js`, `index.html`, `history.js` → CSS classes; toggle
  visibility via classes instead of `style="display:none"`.
- **F2** — A few magic numbers → named constants where it aids clarity: drain batch size `500`
  + the `5`s flush interval (`pipeline.py:56,129`), sparkline window `100`, truncation lengths.
  Don't over-constantify.
- **F3** — `logs.js:setPaused` has an `if (manual && paused) { /* comment only */ }` no-op
  branch (47-49): implement the intended "keep view in place" behavior or drop the `manual` param.
- **F4** — Quick dead-CSS/JS audit (coverage-driven, not from the scan list — several scan
  "unused" hits were wrong, e.g. `.brand-dot` is used in `index.html:16`).

---

## Explicitly NOT doing — validated false positives (mention in the demo to show rigor)

- `events.py` "file-handle leak" — the `finally` (130-135) closes the log; early returns are
  inside the `try`. Not a leak.
- `snapshots.py` `_locks`/`_pending` "thread race" — both are mutated on the single-threaded
  asyncio event loop (only `_process` is offloaded via `to_thread`). `_device_cache` is the only
  unguarded global and it's deterministic/benign.
- `server.py:_open_browser` "blocks the event loop" — `subprocess.Popen` is non-blocking
  (it's `subprocess.run` that blocks).
- `boot.py:88` "division by zero / KeyError" — guarded by `if total else 0.0`, and every
  `_LAYER_STATES` value is present in `_STATE_PROGRESS`/`_STATE_RANK`. `boot.py` is clean.
- `.brand-dot` "unused CSS" — referenced in `index.html:16`.

---

## Deliverable

**This pass is the backlog itself — no code changes now.** The team will triage and tackle
the items. The notes below are handoff guidance for whoever picks them up.

## Recommended order (when tackled)

**A (hygiene/CI/README) → B (dedup + phase constants) → D (error handling) → E (tests) →
C (bloat refactors) → F (polish).** Front-load the low-risk, high-signal wins; do the larger
structural refactors (C1/C2/C3) last, behind a smoke test, since C2 touches the event drain.

## How to verify each item (for the implementer)

1. **Lint/format:** `uv run ruff check .` and `uv run ruff format --check .` (once A1 lands).
2. **Python tests:** `uv run --with pytest pytest` (existing `tests/` + new E1 tests).
3. **JS test:** `node --test podterm/static/js/derive.test.mjs`.
4. **Smoke the app** — the risky changes touch the load-bearing phase strings (B2) and the
   shared HTTP path (B1). Launch with `uv run podterm`, open the UI, and confirm with
   `scripts/ui_shot.py` (it also reports console errors) that:
   - SSE still flows (metric/log/telemetry events render),
   - a `"Starting Training"` phase still resets the metrics buffer and `"Training finished"`
     still finalizes the run row + flips the UI to finished,
   - the snapshot→diagnostics path still downloads + runs (B1 must not change request semantics).