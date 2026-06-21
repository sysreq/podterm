# Train Loss To Estimated ValBPB Implementation Plan


> **SHELVED (2026-06-20).** Not being implemented. `train_loss` stays the proxy for finish
> quality (the race bar uses it directly — see `racebar-improvements.md`). This plan's premise
> (interpolate/extrapolate *between* periodic ValBPB evals) does not hold today: the gpt-golf
> producer runs `eval_val` **exactly once, at run end** (`train_gpt.py:376`), so `metrics.val_bpb`
> never has more than a single point and it only arrives when the run is over. A real live ValBPB
> series would need either periodic on-pod eval (gpt-golf change + image rebuild) or sourcing
> anchors from the off-pod per-snapshot diagnostics BPB (`run_diagnostics`). Revisit only if/when
> that data source exists. Blocked behind `racebar-improvements.md` regardless.


## Goal

Shift user-facing training quality displays away from `train_loss` and toward estimated validation BPB.

The UI should make ValBPB the primary quality metric everywhere a user currently sees training loss:

- Live chart
- KPI card
- Baseline race copy and math
- Compare charts and regression banner
- Sidebar pace line
- Baseline trace/target text

Lower ValBPB remains better. Raw `train_loss` can stay persisted and available internally for diagnostics or fallback math, but it should stop being the headline quality metric in the app.


## Current State

Important surfaces:

- `podterm/static/js/charts.js`
  - Main live chart is named and scaled as Training Loss.
  - Current run and baseline traces use `train_loss`.
  - Eval markers display `val_bpb` in hover text but use `val_loss` for y positioning.
- `podterm/static/index.html`
  - Chart title says `Training Loss`.
- `podterm/static/js/live/kpis.js`
  - KPI card label is `Loss (train)`.
  - Hero/race feeds `qualityRace`, which compares current `train_loss` to baseline `train_loss`.
  - Loss delta card uses `deltaVsStepsAgo(..., 'train_loss')`.
- `podterm/static/js/live/race.js`
  - Race headline and baseline row talk about loss.
- `podterm/static/js/live/baseline.js`
  - Baseline chart trace is built from baseline `train_loss`.
- `podterm/static/js/runlist.js`
  - Sidebar pace line says `at X loss`.
- `podterm/static/js/compare.js`
  - Regression banner compares final train loss.
  - First compare chart is Training Loss Comparison.
  - Second chart is sparse Validation BPB Comparison.
- `podterm/static/js/derive.js`
  - Quality race math is loss-based.
  - Existing `evalDelta` already handles `val_bpb`.
- `podterm/static/js/derive.test.mjs`
  - Race tests are train-loss based.


## Desired Semantics

1. Actual ValBPB points remain the source of truth.
   - Use `metrics.val_bpb` eval rows as anchors.
   - Do not invent better-looking numbers than the evals justify.

2. Estimated ValBPB is a derived display/race series.
   - Between two known evals, interpolate linearly by step or training time.
   - After the latest eval in a running job, extrapolate from the recent eval trend.
   - Before the first eval, show unknown rather than pretending train loss is ValBPB.
   - With only one eval, show that value as the current estimate but mark trend/finish projections as low confidence.

3. Race comparisons use estimated finish ValBPB.
   - Compare current run estimated ValBPB at its configured finish budget against baseline estimated ValBPB at the baseline finish budget.
   - Use `time_budget` / `MAX_WALLCLOCK_SECONDS` semantics from `racebar-improvements.md`.
   - If projected behind, show estimated extra training time needed to beat the baseline finish ValBPB.

4. User-facing language says ValBPB, not train loss.
   - Use `Estimated ValBPB` for interpolated/extrapolated values.
   - Use `Validation BPB` for actual eval points.
   - Avoid generic `loss` labels in live race, chart, KPI, compare regression, and sidebar.


## Data And State Plan

1. Keep backend schema as-is for this change.
   - `val_bpb` already exists in `metrics`.
   - `best_val_bpb` already exists in `runs`.
   - No new DB column is required unless we later choose to persist estimated values, which is not necessary.

2. Extend client state with derived ValBPB buffers.
   - Add fields to per-pod state:
     - `estimatedValBpbByStep`
     - `estimatedValBpbSeries`
     - `baselineEstimatedValBpbSeries`
   - These can be recomputed from `metricHistory` and `evals` rather than stored permanently.

3. Keep `train_loss` ingestion untouched.
   - Continue merging `train_loss` rows for historical compatibility and possible internal fallback.
   - Stop emitting UI events named `trainPoint` if nothing uses them after the chart update, or rename to a neutral `metricPoint`.

4. Baseline metadata.
   - Reuse the baseline run-row fetch proposed in `racebar-improvements.md` so the baseline finish budget is available.
   - Baseline estimates need both baseline metric rows and baseline eval rows.


## Derivation Math Plan

Add pure helpers in `podterm/static/js/derive.js` and cover them in `podterm/static/js/derive.test.mjs`.

1. `validValBpbSamples(metricsOrEvals)`
   - Return sorted `{ step, train_time_ms, val_bpb }` points.
   - Ignore rows without finite positive `val_bpb`.

2. `estimatedValBpbAtStep(samples, step)`
   - If `step` is bracketed by two eval points, linearly interpolate.
   - If exactly on an eval, return the actual value with method `actual`.
   - If after the latest eval:
     - Use recent eval slope when at least two evals exist.
     - Otherwise hold the latest eval with method `hold` and low confidence.
   - If before the first eval, return null.

3. `estimatedValBpbAtTime(samples, trainTimeMs)`
   - Same behavior as step estimator, but based on `train_time_ms`.
   - This is the primary helper for finish-budget race math.
   - If eval rows lack `train_time_ms`, fall back to step-based estimation.

4. `estimatedValBpbSeries(metricHistory, evals)`
   - Build a smooth line at train metric cadence:
     - Actual eval points remain visually distinct markers.
     - Estimated points form the main line.
   - Include metadata per point:
     - `method`: `actual`, `interpolated`, `projected`, or `hold`
     - `confidence`: `high`, `medium`, `low`

5. `valBpbSlopePerMs(samples)`
   - Use the last two or last few eval anchors.
   - Negative slope means improving.
   - Return null for insufficient or invalid data.

6. `timeToReachValBpb(currentEstimate, currentTimeMs, targetValBpb, slopePerMs)`
   - If current estimate is already below target, return current time.
   - If slope is null or non-improving, return null.
   - Otherwise estimate crossing time.

7. Replace loss race with ValBPB race.
   - Add `valBpbRace(...)` or evolve `qualityRace(...)`.
   - Inputs:
     - current metric/history/evals
     - baseline metrics/evals
     - current finish budget ms
     - baseline finish budget ms
     - throughput inputs
   - Return:
     - `state`: `ahead`, `behind`, `tied`, `unknown`, `no-baseline`, `no-data`
     - `currentEstimatedValBpb`
     - `currentFinishValBpb`
     - `baselineFinishValBpb`
     - `marginValBpb`
     - `estimatedWinTimeMs`
     - `extraTimeToWinMs`
     - `confidence`
     - throughput fields


## UI Plan

1. Live chart
   - Rename title from `Training Loss` to `Estimated ValBPB`.
   - Main trace: current run estimated ValBPB line.
   - Baseline trace: baseline estimated ValBPB line.
   - Eval markers: actual validation BPB points.
   - Use a BPB y-axis directly, not `scaleY(val_loss)`.
   - Hover text should show:
     - estimated ValBPB
     - actual ValBPB when applicable
     - method/confidence for projected/held points

2. KPI row
   - Replace `Loss (train)` card with `Estimated ValBPB`.
   - Value: latest estimated ValBPB.
   - Subline:
     - actual eval delta when on/after an eval
     - projected trend when between evals
     - `Waiting for first eval` before eval data exists
   - Keep the existing lower-is-better success/danger coloring.

3. Diagnostic row
   - Keep `Validation BPB` as the actual eval card.
   - This gives users both:
     - smoothed/estimated live quality in the main KPI
     - sparse actual validation anchors in the diagnostic row

4. Race banner
   - Replace loss wording with estimated ValBPB wording.
   - Example headlines:
     - `Projected to beat baseline by 0.0123 ValBPB`
     - `Projected behind baseline by 0.0087 ValBPB`
     - `Waiting for enough validation data`
   - Subline should show:
     - current estimated finish ValBPB
     - baseline finish ValBPB
     - throughput
     - time-to-win when behind

5. Baseline row
   - Replace `Target: baseline reached X loss at Y`.
   - Proposed:
     - `Target: baseline finish 1.2345 ValBPB at 10m budget`

6. Sidebar active run cards
   - Replace `at X loss` pace copy.
   - Proposed:
     - `0:32 ahead by 0.0061 ValBPB`
     - or, after racebar change, `+1:10 to beat baseline`

7. Compare tab
   - Remove Training Loss Comparison as a primary chart.
   - Replace with `Estimated ValBPB Comparison`.
   - Keep actual eval markers visible.
   - Regression banner should compare final/finish estimated ValBPB or `best_val_bpb`, not final train loss.
   - Summary table already uses `Best BPB`; keep it.
