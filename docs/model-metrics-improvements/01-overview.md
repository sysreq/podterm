# Model Metrics UX Improvements Plan



## Goal

Give model health metrics their own experience instead of forcing them into the live
training-loss dashboard.

The default experience should answer, quickly:

- Is the model healthy right now?
- What changed since the previous snapshot?
- Which few metrics need attention?
- When was this health snapshot taken?

The detailed experience should still expose the full diagnostics payload, section rows, trends,
timelines, and raw JSON, but only after an explicit user action.


## Resolved Decisions (read first)

These were open questions; they are now settled and the plan below reflects them:

1. **Deep-Dive left rail = raw section list, worst-first.** List the actual `diag.sections[]` by
   name (e.g. `capacity: dead neurons`, `flow: residual norm`, `gradients`, `zero-init escape`,
   `embedding spectrum`), sorted worst-status first — exactly what the current renderer does. The
   four-group taxonomy (capacity/flow/grad/arch) is used **only** in the summary "Health groups"
   band, which is a property of *headline metrics*, not of sections.
2. **Threshold hints come from the backend.** `health.py` will serialize each headline metric's
   threshold band plus a one-line `why` into the headline payload. No client-side duplication of
   the thresholds.
3. **Build PENDING and `vs baseline` now.** Both turn out to need *no new backend event*:
   - PENDING reuses the `snapshot` SSE the pipeline **already emits** (`pipeline.py:105`) before
     diagnostics start; the client just needs to listen for it.
   - `vs baseline` reuses `/api/runs/{id}/diagnostics` for the per-pod selected baseline
     (`state.baselineRunId`) and the renderer's existing reference-doc diff, swapping which doc is
     the reference. No `compare.py` / endpoint changes needed.


## Current State (verified)

- `podterm/static/index.html`
  - `#diagnostics-panel` (with `#diag-sub` + `#diag-content`) is embedded in the Live tab
    (`#tab-live`) below the loss chart and the `#diag-row`.
  - Tabs are `<div class="tab" data-tab="...">` in `#tab-bar`; content panes are
    `<div id="tab-{name}" class="tab-content">`.
- `podterm/static/js/app.js`
  - `switchTab(tab)` toggles `.active` on tab + `#tab-{tab}`, runs per-tab hooks
    (`history`→`loadHistory()`, `machines`→`renderMachines()`), and resizes Plotly. A `tab:switch`
    bus event already routes here (`on('tab:switch', ({tab}) => switchTab(tab))`). Adding a `health`
    tab = one nav `<div>`, one content pane, and one `if (tab === 'health')` hook.
- `podterm/static/js/live/kpis.js`
  - The **Model Health card lives in the main `#kpi-row`** (`cards.health`), *not* in `#diag-row`.
    `#diag-row` holds only `Validation BPB` + `System` (`diag.vbpb`, `diag.system`). Removing/
    relocating ValBPB is `trainloss-improvements.md`'s job — out of scope here.
  - `buildKpiRow()` wires `cards.health.el` click → `scrollIntoView(#diagnostics-panel)`. That
    handler is what changes to a tab switch.
  - `updateHealthCard(state)` reads `state.diagnostic` (`{step, status, health}`); `BAND_LABEL`
    maps `ok/warn/error`→`OK/WARN/ERROR`. Refreshed via `updateKpis`, which `live.js` calls on
    `pod:diagnostic` and `pod:health`.
- `podterm/static/js/diagnostics.js`
  - `load(runId)` fetches `/api/runs/{runId}/diagnostics`, filters to entries with `.diag`, syncs
    the live verdict card (`state.diagnostic` + `emit('pod:health')`), and `render()`s the panel:
    verdict bar, clickable timeline (pins `selectedStep`), grouped headline tiles, per-section
    detail (open by default, worst-first), and a raw-JSON `<details>`.
  - Helpers to keep: `fmtVal`, `trendArrow`, `bandClass`. Group maps already exist:
    `GROUP_ORDER = ['capacity','flow','grad','arch']`, `GROUP_LABEL = {capacity:'Capacity',
    flow:'Flow', grad:'Gradients', arch:'Architecture'}`. **Reuse these — the group keys are
    `grad`/`arch`, not `gradients`/`architecture`.**
  - `initDiagnostics()` loads on `pod:hydrated` / `pod:diagnostic` (active pod only) and hides the
    panel on `pod:reset`.
- `podterm/diagnostics/health.py`
  - `compute(diag)` → `{overall: ok|warn|error, headline: [...], counts: {good,warn,bad,info,na}}`.
  - Each headline entry today is `{id, label, value, unit, band, group, tier}` where
    `band ∈ {good,warn,bad,info,na}`, `group ∈ {capacity,flow,grad,arch}`, `tier ∈ {1,2}`.
    The good/warn/bad thresholds live in the `HEADLINE` registry's `band=dict(...)` but are **not**
    serialized. Two metrics (`mean_loss`, `eff_rank`) are `kind='info'` → band `info` (trend-only,
    no verdict).
- `podterm/db/diagnostics.py`
  - History rows: `{step, created_at, status, diag}`. `created_at` is a UTC ISO-8601 string
    (`datetime.now(timezone.utc).isoformat()`) → snapshot age = `Date.now() - Date.parse(created_at)`.
- `podterm/static/js/compare.js`
  - Reuses shared classes `diag-band-*` and `diag-row-key`. **Do not delete those** when adding the
    new `.health-*` namespace — they are cross-tab.


## UX Design

### Navigation Model

Add a top-level `Model Health` tab.

- Current tabs: `Live`, `History`, `Compare`, `Machines`
- Proposed tabs: `Live`, `Model Health`, `History`, `Compare`, `Machines`

The Live tab stays focused on training progress (loss/ValBPB chart, race banner, baseline controls,
cost/GPU/ETA/system telemetry, logs/config) plus the compact health doorway card. The Model Health
tab owns diagnostics: current health summary, headline groups, snapshot history, deep dive.

### Live Tab Changes

Keep only the compact health card in the KPI row (already `cards.health` in `#kpi-row`):

- Label: `Model Health`
- Value: `OK`, `WARN`, `ERROR`, or `PENDING`
- Subline:
  - `step 4000 · 2 warnings`
  - `no diagnostics yet`
  - `snapshot pending` (a `snapshot` SSE arrived but no `diagnostic` for that step yet)
- Caption/action: `Open Health` — clicking switches to the Model Health tab for the active run.

Remove the full `#diagnostics-panel` from `#tab-live` (it moves into the new tab — see Implementation).

### Standard Health View

Starts with a standard current-health view.

1. Header bar
   - Run name / branch / commit (from `state.runRow`)
   - Latest snapshot step
   - Snapshot age (from `created_at`)
   - Overall verdict badge
   - `Deep Dive` button

2. Current health summary band
   - Large verdict: `OK`, `WARN`, or `ERROR` (`health.overall`)
   - One-sentence interpretation derived from `counts` / failing groups, e.g.
     `Architecture and flow checks look healthy.` / `2 gradient checks need review.`
   - Counts from `health.counts`: `good`, `warn`, `bad`, plus `na`→`unavailable` and
     `info`→`info` (trend-only). Display all five; don't drop `info`.

3. Top issues
   - Show only `warn`/`bad`-band headline metrics by default (skip `info`/`na`).
   - If none, show the top 3 healthy metrics ordered by `tier` (tier 1 first).
   - Each row: metric `label`, current `value` (via `fmtVal`), `band` chip, trend vs previous
     snapshot (`trendArrow`), and the `why` one-liner + threshold hint from the new payload fields.

4. Health groups
   - Four sections keyed by `group`, labelled via `GROUP_LABEL`: Capacity, Flow, Gradients,
     Architecture.
   - Each renders as a compact row (not a tile wall): group verdict (worst band in the group),
     per-band counts, 2–3 headline values, and a sparkline/trend chip.

5. Snapshot strip
   - Horizontal compact timeline, colored by verdict, reused in summary and deep dive
     (`.health-timeline`, replacing `.diag-timeline` behavior).
   - Clicking a snapshot re-points the view to that step; a `Latest` control returns to newest.

### Deep-Dive View

A prominent `Deep Dive` button enters a mode inside the Model Health tab (not a modal) so the dense
content gets full page width.

1. Sticky toolbar
   - Back to Summary
   - Snapshot selector
   - Diff mode: `vs previous`, `vs baseline`, `absolute`
     - `vs previous`: reference = the prior snapshot in this run.
     - `vs baseline`: reference = latest snapshot of the run's selected baseline
       (`state.baselineRunId`). **Disable this toggle with a hint when no baseline is selected.**
     - `absolute`: no reference, no deltas.
   - Search/filter input
   - Toggles: only warnings/errors · include unavailable (`na`) metrics · show raw values

2. Left rail — **raw section list, worst-first** (decision 1)
   - One entry per `diag.sections[]` by name, with a status badge, sorted worst-first.
   - Clicking jumps/scrolls to that section. No group bucketing.

3. Main detail area
   - Sections collapsed by default except those with `warn`/`error`/`partial` status.
   - Worst sections first (reuse `sevRank`).
   - Consistent metric table: name · current · reference (per diff mode) · delta · status · notes.
   - Long numeric arrays summarize by min/mean/max and expand inline (reuse `fmtRowMetrics`).

4. Raw JSON — kept in a collapsed `Raw JSON` drawer, not in the default viewport.


## Information Architecture

### Standard Summary Data Contract

Use existing `diag.health` (`overall`, `counts`, `headline`) plus the two new headline fields below.

Add pure, exported selectors in `diagnostics.js` (unit-tested — the project already runs
`node --test`, see `derive.test.mjs`):

- `latestSnapshot(history)` — last entry with `.diag`.
- `snapshotByStep(history, step)` — entry for a step (or null).
- `previousSnapshot(history, step)` — entry immediately before `step` in the valid list.
- `headlineByGroup(snapshot)` — headline metrics bucketed by `group` (keys `capacity/flow/grad/arch`).
- `topHealthIssues(snapshot)` — `warn`/`bad` headline metrics; fallback to top-3 healthy by `tier`.
- `groupSummary(snapshot)` — per-group worst band, counts, and 2–3 representative values.
- `referenceDoc(history, step, diffMode, baselineHistory)` — resolves the diff reference doc for
  `vs previous` / `vs baseline` / `absolute`. Centralizes the diff-mode logic so summary trends and
  deep-dive tables share it.

### Backend Change — export thresholds (decision 2)

In `podterm/diagnostics/health.py`, `compute()` adds two fields to each serialized headline entry:

- `thresholds`: the entry's `band` spec dict (e.g. `{kind:'hi', warn:0.05, bad:0.20}` or the
  `range`/`info` variants) so the UI can render a threshold hint without hardcoding numbers.
- `why`: a short one-line rationale string. Add a `why=` to each `HEADLINE` registry entry.

`info`-kind metrics carry `thresholds={kind:'info'}` and need no `why` hint beyond a trend note.
Update `tests/test_health.py` to assert both keys are present on every headline entry (existing
`resolve`/`classify`/`compute` assertions are unaffected — they don't pin the headline key set).

### Optional Backend Enhancement (deferred)

A normalized `GET /api/runs/{run_id}/health` remains optional — only add it if the client selectors
get unwieldy. `/api/runs/{run_id}/diagnostics` stays the source for deep-dive/raw and `vs baseline`.
