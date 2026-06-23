# GPT Caddy — Pre-Demo Code-Quality Backlog

## Context

We want to put GPT Caddy's best foot forward for a code-review demo where reviewers may
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

> **Status update (kept in sync with the code):** the project-hygiene gate has landed —
> ruff lint config (`[tool.ruff]` in `pyproject.toml`), a GitHub Actions workflow
> (`.github/workflows/ci.yml`) running ruff + pytest + `node --test`, a root `README.md`,
> and a `.pre-commit-config.yaml`. The `/api/compare` route is now typed (`CompareRequest`).
> Items confirmed resolved are tagged **[RESOLVED]** inline below; the rest stay open backlog.

---

## Theme A — Project hygiene / engineering maturity (high demo signal, low risk)

- **A1 — No linter configured. [RESOLVED]** `pyproject.toml` now carries a `[tool.ruff]` /
  `[tool.ruff.lint]` section (`select = ["E","F","I"]`, `line-length = 140`) tuned to pass on
  the tree as-is; ruff is in the dev group. Scope note: this is a non-disruptive *lint* gate,
  not a reformat — `ruff format` was deliberately **not** run (it would touch every module). A
  formatter pass is a separate, larger decision.
- **A2 — No CI. [RESOLVED]** `.github/workflows/ci.yml` runs on push + PR: `uv run ruff check .`,
  `uv run pytest -q`, and `node --test` for both frontend test files. (No `ruff format --check`
  — see A1.)
- **A3 — No human-facing README. [RESOLVED]** A root `README.md` now covers what GPT Caddy is,
  an architecture overview (pointing to `CLAUDE.md` for depth), setup/run, test commands, the
  security model, and limitations.
- **A4 — Orphaned JS test. [RESOLVED]** Both `podterm/static/js/derive.test.mjs` and
  `diagnostics.test.mjs` are now wired into CI (`node --test`) and mirrored in pre-commit.
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
- **D2 — Untyped request body on `/api/compare`. [RESOLVED]** `routes/runs.py` now declares
  `class CompareRequest(BaseModel)` and the handler takes `body: CompareRequest`, so validation
  + the OpenAPI schema are automatic.

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

**A (hygiene/CI/README) [done] → B (dedup + phase constants) → D (error handling) → E (tests) →
C (bloat refactors) → F (polish).** Front-load the low-risk, high-signal wins; do the larger
structural refactors (C1/C2/C3) last, behind a smoke test, since C2 touches the event drain.

## How to verify each item (for the implementer)

1. **Lint:** `uv run ruff check .` (A1 landed; lint-only, no `ruff format`). Mirrored in CI
   (A2) and `.pre-commit-config.yaml`.
2. **Python tests:** `uv run pytest -q` (existing `tests/` + new E1 tests).
3. **JS tests:** `node --test podterm/static/js/derive.test.mjs podterm/static/js/diagnostics.test.mjs`.
4. **Smoke the app** — the risky changes touch the load-bearing phase strings (B2) and the
   shared HTTP path (B1). Launch with `uv run podterm`, open the UI, and confirm with
   `scripts/ui_shot.py` (it also reports console errors) that:
   - SSE still flows (metric/log/telemetry events render),
   - a `"Starting Training"` phase still resets the metrics buffer and `"Training finished"`
     still finalizes the run row + flips the UI to finished,
   - the snapshot→diagnostics path still downloads + runs (B1 must not change request semantics).