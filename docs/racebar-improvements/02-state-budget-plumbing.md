# Race Bar Improvements Implementation Plan

## State & Budget Plumbing

1. Active run finish budget
   - Add a small `configFinishBudgetMs(runRow)` helper (reuse `configpanel.js`'s parse pattern;
     `JSON.parse(config_json).time_budget × 1000`, null-safe).
   - Derive `state.finishBudgetMs` from `state.runRow` (already populated by
     `loadBaselineOptions`); fall back to observed last `train_time_ms` when absent.

2. Baseline run metadata (`applyBaselineSelection`)
   - In addition to `/api/runs/${runId}/metrics`, fetch `GET /api/runs/${runId}` for the
     baseline run row.
   - Store `state.baselineRunRow` and `state.baselineFinishBudgetMs =
     configFinishBudgetMs(baselineRunRow)` (fallback to `last.train_time_ms`).
   - Keep `state.baselineTotalTimeMs` for now (still used by `qualityRace`/secondary readout).
   - Reset both new fields in `resetBaselineState`.

3. Wire `finishRace` in `kpis.js`
   - Build `activeSamples`/`baselineSamples` (real-loss rows) and call `finishRace` with
     `currentFinishBudgetMs = state.finishBudgetMs` and `baselineFinishBudgetMs =
     state.baselineFinishBudgetMs`.
   - Feed the result to the hero card, `updateRaceBanner`, and `updateBaselineRow`.


## Implementation Steps

1. Pure math (`derive.js` + `derive.test.mjs`)
   - Add `lossAtTime`, `estimateFinishLoss`, `finishRace`, `TIE_LOSS`.
   - Reuse `lossDescentRatePerMs`, `timeToReachLoss`, `smoothCurrentLoss`, `throughputTps`.
   - Tests: interpolation (bracketed budget), extrapolation (budget beyond observed), flat
     trend (no win), improving trend (crossing time), tie band, missing/no-baseline data,
     and the budget fallback (no `time_budget`).

2. Budget helpers + baseline metadata
   - `configFinishBudgetMs`, `state.finishBudgetMs`, `state.baselineRunRow`,
     `state.baselineFinishBudgetMs`; extra baseline run-row fetch; reset wiring.

3. Wire `finishRace` into `kpis.js` (hero + banner + baseline row).

4. Update `race.js` rendering: headline/subline copy, marker domain/titles/labels, baseline row.

5. Verify.
   - `node --test podterm/static/js/derive.test.mjs`.
   - No Python/DB changes, so no backend tests required for this plan.
   - Manual: no baseline; baseline selected but too few samples to project; active projected
     ahead; active behind but improving (shows time-to-win); active behind with no projected
     win; baseline/active missing `time_budget` (fallback to observed time).


## Open Questions (resolved)

1. TrainBPB source — **resolved**: stay on `train_loss`; no BPB metric. (See Decision.)

2. Baseline finish estimate when the baseline stopped early or ran past budget —
   estimate at the configured budget, matching race rules. Extrapolate when the baseline's
   observed time is shorter than its budget.

3. Ties/noise — `|finishMarginLoss| < TIE_LOSS` (start at `0.001` loss; tune in manual testing).

4. Cap on "time to win" — no cap in the math; UI shows "win not currently projected" when the
   trend is flat/worsening (`timeToReachLoss` returns null).
