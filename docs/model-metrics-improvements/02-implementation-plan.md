# Model Metrics UX Improvements Plan

## Implementation Plan

### 1. Split UI Surfaces

1. Add the `Model Health` nav tab + `#tab-health` content pane in `index.html`.
2. Move the `#diagnostics-panel` markup (`#diag-sub`, `#diag-content`) out of `#tab-live` into
   `#tab-health` — or rebuild it there under the new `.health-*` structure.
3. `app.js`: add `if (tab === 'health') ` hook in `switchTab` that ensures diagnostics are loaded/
   rendered for `app.activePod` (export a `renderHealthTab`/`loadDiagnostics` entry from
   `diagnostics.js` and call it).
4. `kpis.js`: replace the `cards.health.el` click → `scrollIntoView` with
   `emit('tab:switch', { tab: 'health' })`. Remove the scroll behavior.

### 2. Refactor Diagnostics Renderer

Split `diagnostics.js` into:

- `loadDiagnostics(runId)` — fetch + cache + sync live card (keep current `load` behavior).
- `renderHealthTab(history)` — top-level dispatcher (summary vs deep mode).
- `renderHealthSummary(valid, selectedStep)`
- `renderHealthDeepDive(valid, selectedStep)`
- `renderSnapshotTimeline(valid, selectedStep)` (the `.health-timeline`)
- `renderRawJson(snapshot)`

Keep `fmtVal`, `trendArrow`, `bandClass`, `fmtRowMetrics`, `sevRank`, `GROUP_ORDER`, `GROUP_LABEL`.
Change defaults: summary renders first; deep detail is hidden until `Deep Dive`; sections are not
all open — only problem sections expand.

### 3. Add Summary Components

DOM components: `.health-hero` (verdict + counts + latest step/age + deep-dive button),
`.health-issues` (top warn/error metrics, healthy fallback), `.health-groups` (one compact row per
group), `.health-timeline` (shared strip).

### 4. Update Live KPI Card

In `kpis.js`:

1. Keep `Model Health` in `#kpi-row` as the doorway.
2. Compact: value = verdict (incl. `PENDING`), sub = step + counts / pending / empty, caption =
   `Open Health`.
3. On click: `emit('tab:switch', { tab: 'health' })`.
4. Add PENDING: when `state.pendingSnapshotStep` is set and there's no `diagnostic` for that step
   yet, show `PENDING` / `snapshot pending` (see step 6).

### 5. CSS Design

Add a `Model Health` section in `components.css` (or a new file if it grows). Reuse the shared
`diag-band-*` classes for band colors; add new `.health-*` layout classes. **Do not remove
`diag-band-*` / `diag-row-key`** (compare.js depends on them).

Principles: rows/bands over cards for dense metrics; cards only for the hero + top issues; no
nested cards; deep dive uses full width.

Suggested classes: `#tab-health`, `.health-toolbar`, `.health-hero`, `.health-verdict`,
`.health-counts`, `.health-issues`, `.health-issue`, `.health-groups`, `.health-group-row`,
`.health-timeline`, `.health-deep-layout`, `.health-section-rail`, `.health-detail-main`,
`.health-metric-table`.

Responsive: wide → hero+issues side by side, groups below, deep dive = rail + main table; medium →
hero/issues stack, rail becomes horizontal section nav; narrow → issue rows wrap values under
labels, metric table uses compact label/value blocks. (Align with `screen-scaling.md`.)

### 6. State And Events

State in `diagnostics.js` (module-level): `selectedStep`, `mode` (`summary`|`deep`), `filterText`,
`onlyProblems`, `includeUnavailable`, `diffMode` (`previous`|`baseline`|`absolute`), and a cache of
the active run's diagnostics + a cache of the baseline run's diagnostics (for `vs baseline`).

State in `state.js` (per pod): add `pendingSnapshotStep` (init `null`; reset in `resetRunState`).

New SSE wiring in `state.js` — **the `snapshot` event is already sent by the server**
(`pipeline.py:105`), it's just unhandled today:

- `es.addEventListener('snapshot', ...)` → set `state.pendingSnapshotStep = step`, emit
  `pod:snapshot`.
- In the existing `diagnostic` handler: if `d.step >= state.pendingSnapshotStep`, clear
  `state.pendingSnapshotStep = null`.

Bus events:

- `pod:snapshot` — active run: refresh the live KPI card (PENDING) and, if the health tab is open,
  its empty/pending state.
- `pod:diagnostic` — update cached diagnostics, refresh the live card (as today), and re-render the
  health tab if open.
- `pod:health` — keep the live card in sync (as today).
- `pod:reset` — clear the health-tab caches and `pendingSnapshotStep` for that run.
- `tab:switch` (`health`) — load/render diagnostics for `app.activePod`; for `diffMode==='baseline'`
  also fetch the baseline run's diagnostics if not cached.

`vs baseline` data path (no backend change): read the active pod's `state.baselineRunId`; if set,
fetch `/api/runs/{baselineRunId}/diagnostics`, take its latest valid snapshot's `diag`, and feed it
as the reference doc to the shared diff logic (`referenceDoc(...)`). If unset, disable the toggle.

### 7. Compare Tab Follow-Up

Keep Compare's current Model Health diff table. Later: reuse `groupSummary`/`topHealthIssues` and a
`Health Diff` deep view for selected runs.


## Copy Guidelines

Short operational language: `Model Health`, `Open Health`, `Deep Dive`, `Back to Summary`,
`Only Problems`, `Diff: Previous` / `Diff: Baseline`, `Latest Snapshot`. Keep diagnostics jargon and
raw metrics in deep dive only.


## Verification Plan

Manual checks:

1. No diagnostics yet → live card `no diagnostics yet`; health tab empty state with snapshot cadence
   context (and `no run selected` when `app.activePod` is null).
2. Snapshot pending → after a `snapshot` SSE but before its `diagnostic`, live card shows `PENDING` /
   `snapshot pending`; clears when the `diagnostic` arrives.
3. Healthy run → summary `OK`; top issues shows top healthy headline metrics (tier-ordered); Deep
   Dive available.
4. Warn/error run → summary surfaces failing groups + top issues first; Deep Dive opens with problem
   sections expanded.
5. Multiple snapshots → timeline selection updates summary; trends compare vs previous selected
   snapshot; `Latest` returns to newest.
6. Diff modes → `vs previous`, `vs baseline` (with a baseline selected), and `absolute` all render;
   `vs baseline` is disabled with a hint when no baseline is selected.
7. Active run switches → health tab reloads for the new active run; live card and tab stay in sync.
8. Small screens → summary readable; deep-dive tables don't overflow incoherently.

Automated checks:

- `uv run --with pytest pytest tests/test_health.py` — extend to assert `thresholds` + `why` on
  every headline entry; existing diagnostics route/golden tests keep passing.
- Add `node --test` selector tests (new `diagnostics.test.mjs` or fold into the existing harness) for
  `topHealthIssues`, `groupSummary`, `headlineByGroup`, `previousSnapshot`, and `referenceDoc`
  against a small fixture doc.


## Relationship To Other Plans

- `trainloss-improvements.md` — Live training metrics become estimated-ValBPB focused; ValBPB/System
  relocation out of `#diag-row` is owned there, not here.
- `screen-scaling.md` — the compact Live health card and the Model Health tab follow the same
  responsive KPI/card rules.
