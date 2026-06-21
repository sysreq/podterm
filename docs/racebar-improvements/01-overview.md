# Race Bar Improvements Implementation Plan



## Decision (resolved)

The race stays on **`train_loss`** as the quality axis (lower is better). We are **not**
introducing a TrainBPB metric: it isn't emitted per-step, and within a single
corpus+tokenizer `bpb = train_loss × C` for a constant `C`, so the ahead/behind verdict,
margin sign, and crossing-time geometry are identical to the loss-based ones — only the
displayed unit would change. This plan therefore has **no data-model / DB / producer /
SSE changes**. All work is in the JS derivation math and the live-race UI.

The genuinely new capability is the **finish-horizon race**: compare runs by their
*estimated finish loss at the configured max-wallclock budget*, and for a behind run,
estimate the extra training time needed to cross below the baseline's finish loss.


## Goal

Improve the race modal/banner so it compares the active run against the selected baseline
by **estimated finish quality at the budget horizon**, not only by equal-loss timing or
step progress.

The new race model answers:

- At the active run's configured max run time, what train loss is it expected to finish with?
- At the baseline's configured max run time, what train loss did/would the baseline finish with?
- If the active run is currently behind, how much extra training time is estimated before it
  would beat the baseline's finish loss?

This lets a run win either by completing more steps within the same wallclock budget or by
being more efficient per step.


## Current State

Relevant files:

- `podterm/static/js/derive.js`
  - Owns shared race math through `qualityRace`.
  - Already has loss-based, sparsity-tolerant helpers we will reuse:
    `smoothCurrentLoss` (EMA-10 of train_loss), `lossDescentRatePerMs` (positive when
    improving, paired over a ~500-step window), `timeToReachLoss` (wall-clock to reach a
    target loss at the recent rate), `baselineTimeAtLoss`, `throughputTps`, `ema`.
  - `qualityRace` currently compares at equal quality and projects time to the baseline's
    *final achieved* loss, keyed off `baselineTotalTimeMs` (the last metric's `train_time_ms`).
- `podterm/static/js/live/race.js`
  - Renders the race banner, progress markers (`.projected`, `.baseline`), and the baseline row.
  - Labels projected target time versus `state.baselineTotalTimeMs`.
- `podterm/static/js/live/kpis.js`
  - Computes ETA from remaining steps and feeds `qualityRace`; renders the hero card.
- `podterm/static/js/live/baseline.js`
  - `applyBaselineSelection` fetches baseline metrics only and sets
    `state.baselineTotalTimeMs` from the last metric row.
  - `loadBaselineOptions` already sets `state.runRow` from `/api/runs`.
- `podterm/static/js/state.js`
  - `mergeMetric` is generic (`Object.entries`) — no per-field whitelist, so no ingestion
    changes are needed for this plan.
- `podterm/static/js/configpanel.js`
  - Already parses the active run config: `state.runRow?.config_json ? JSON.parse(...) : null`.
    **`config_json` is a JSON *string*** over the API (TEXT column, returned via `dict(row)`),
    not an object — every read must `JSON.parse`.
- `podterm/static/js/derive.test.mjs`
  - Exists; run with `node --test podterm/static/js/derive.test.mjs`.
- `podterm/pods.py`
  - `LaunchConfig.time_budget` (seconds, default 600) → injected as `MAX_WALLCLOCK_SECONDS`,
    and the full config dict is persisted in `runs.config_json`.
- `podterm/routes/runs.py`
  - `GET /api/runs/{run_id}` already returns the full run row (incl. `config_json`).

Budget units are consistent: `time_budget` (seconds) bounds the same training-elapsed clock
that metric `train_time_ms` measures, so `time_budget × 1000` is directly comparable to a
run's `train_time_ms`.


## Semantics To Implement

1. Treat max run time as the finish horizon.
   - Active run finish budget: `JSON.parse(runRow.config_json).time_budget × 1000`.
   - Baseline finish budget: same, read from the baseline's run row (fetched separately),
     **not** from the final metric timestamp.
   - Fall back to the observed last `train_time_ms` only for older runs whose `config_json`
     lacks `time_budget`.

2. Estimate finish loss.
   - Active run: estimate train_loss at `finishBudgetMs` on the run's own (time, loss) series.
   - Baseline: estimate train_loss at its `baselineFinishBudgetMs` on the baseline series.
   - Interpolate when the series brackets the finish time; otherwise extrapolate from the
     latest valid sample using the recent descent rate (`lossDescentRatePerMs`).
   - Smooth the active run's *current* loss with the existing EMA-10 (`smoothCurrentLoss`) to
     avoid one-sample jitter, but keep the projection math deterministic and unit-tested.

3. Determine ahead/behind from estimated finish loss.
   - Lower loss wins.
   - `finishMarginLoss = baselineFinishLoss − currentFinishLoss` (positive ⇒ current finishes
     lower ⇒ ahead).
   - Ahead if `> tie`, behind if `< −tie`, tied/unknown within the tie band.
   - Keep wallclock lead as a secondary readout, not the primary state.

4. Estimate "time to win".
   - If the active run is projected to finish at or below the baseline finish loss by its
     budget horizon, display the configured finish ETA/time (it wins by budget).
   - If projected to lose, use `timeToReachLoss(currentLoss, currentTime, baselineFinishLoss,
     rate)` to estimate the additional training time for the run's loss curve to cross below
     the baseline finish loss.
   - This value may exceed the budget; label it "time to win" / "needs +X beyond budget".
   - If the curve is flat or worsening (`rate <= 0` ⇒ `timeToReachLoss` returns null), report
     that a win is not currently projected.


## Data Model Plan

None. No schema column, migration, producer change, or SSE field. (Resolved above.)


## Derivation Math Plan

Add pure, loss-based helpers in `podterm/static/js/derive.js` with unit tests in
`podterm/static/js/derive.test.mjs`. Reuse existing helpers where noted.

1. `lossAtTime(samples, targetTimeMs)` — **new**
   - Operate on samples with `train_loss > 0` and a valid `train_time_ms`, in time order.
   - If `targetTimeMs` is bracketed by two samples, linearly interpolate loss by time.
   - If before the first valid sample, return the first loss with low-confidence metadata.
   - If after the last valid sample, return null (the caller extrapolates) — keep this helper
     strictly an interpolator so it stays trivially testable.

2. `recentLossSlopePerMs(history, windowSteps)` — **reuse `lossDescentRatePerMs`** as-is
   (already returns the recent descent rate, positive when improving, null on insufficient
   data / non-positive dt). No new helper needed.

3. `estimateFinishLoss(samples, finishBudgetMs)` — **new**
   - If `lossAtTime` returns a value (budget bracketed), use it: `{ method: 'interp' }`.
   - Otherwise extrapolate from the latest valid sample using the recent slope:
     `latestLoss − rate × (finishBudgetMs − latestTimeMs)` (rate positive ⇒ loss decreases
     toward the horizon): `{ method: 'extrap' }`. Clamp the extrapolated loss to be ≥ 0.
   - If there is no valid latest sample or no usable slope for extrapolation, return
     `{ value: null, method: 'none' }`.
   - Return `{ value, method, confidence, atTimeMs, ratePerMs }`.

4. `timeToWinLoss(currentLoss, currentTimeMs, targetLoss, ratePerMs)` —
   **reuse `timeToReachLoss`** (identical signature and semantics: returns `currentTimeMs`
   if already at/below target, null if `rate <= 0`, else `currentTimeMs + (currentLoss −
   targetLoss) / rate`). No new helper needed; just call it with `targetLoss =
   baselineFinishLoss`.

5. `finishRace({ metric, history, baselineSamples, currentFinishBudgetMs,
   baselineFinishBudgetMs, batchTokens, emaMsPerStep, baselineSample })` — **new**
   - Mirror `qualityRace`'s guard/empty structure and reuse `smoothCurrentLoss`,
     `throughputTps`, and the `baselineSample.step_avg_ms` throughput delta.
   - Compute `currentFinishLoss = estimateFinishLoss(activeSamples, currentFinishBudgetMs)`
     where `activeSamples` is `history` filtered to real-loss rows.
   - Compute `baselineFinishLoss = estimateFinishLoss(baselineSamples, baselineFinishBudgetMs)`.
   - `finishMarginLoss = baselineFinishLoss − currentFinishLoss`.
   - `rate = lossDescentRatePerMs(history)`;
     `estimatedWinTimeMs = timeToReachLoss(currentLoss, currentTime, baselineFinishLoss, rate)`.
   - `extraTimeToWinMs = (estimatedWinTimeMs != null && estimatedWinTimeMs > currentFinishBudgetMs)
     ? estimatedWinTimeMs − currentFinishBudgetMs : null`.
   - Return:
     - `state`: `ahead`, `behind`, `tied`, `unknown`, `no-baseline`, or `no-data`
     - `currentLoss`
     - `currentFinishLoss`
     - `baselineFinishLoss`
     - `finishMarginLoss`
     - `finishBudgetMs` (= `currentFinishBudgetMs`)
     - `baselineFinishBudgetMs`
     - `estimatedWinTimeMs`
     - `extraTimeToWinMs`
     - `projectedFinishTimeMs` (the active run's budget horizon = `currentFinishBudgetMs`)
     - throughput fields already returned by `qualityRace` (`throughputTps`, `msPerStep`,
       `baselineMsPerStep`)
   - State resolution: `no-data` if no metric; `no-baseline` / `unknown` if either finish-loss
     estimate is null; `tied` if `|finishMarginLoss| < TIE_LOSS`; else `ahead`/`behind` by sign.

6. Keep `qualityRace` and its helpers — `compare.js` and any sidebar code still import them;
   do not remove until those callers are migrated (out of scope here).

Constants: add `export const TIE_LOSS = 0.001;` (loss units; ~4× a 0.00025 BPB band given the
typical corpus constant). Tune during manual testing.


## UI Plan

1. Banner headline (`race.js`)
   - Ahead: `Projected to beat baseline by 0.012 loss`
   - Behind: `Projected behind baseline by 0.009 loss`
   - Tied: `Too close to call at the budget finish`
   - Unknown: `Waiting for enough loss trend`

2. Banner subline
   - Keep throughput / step efficiency (`fmtThroughput`) as today.
   - Add finish comparison: `finish 3.412 vs baseline 3.428`.
   - Behind: `needs ~7m20s total training (+1m10s beyond budget) to win`, or
     `win not currently projected` when `estimatedWinTimeMs == null`.
   - Ahead: `wins by budget finish at 10m`.

3. Progress markers
   - The bar domain becomes the time axis `[0, max(currentFinishBudgetMs,
     baselineFinishBudgetMs, estimatedWinTimeMs?, train_time_ms)]`.
   - Marker A (`.projected`): active run's budget finish (`currentFinishBudgetMs`).
   - Marker B (`.baseline`): baseline's budget finish (`baselineFinishBudgetMs`).
   - Optional marker C: `estimatedWinTimeMs` when it exceeds the budget (reuse an existing
     marker element or add one; keep the left-to-right label ordering logic already in
     `updateRaceBanner`).
   - Retitle from "Projected finish" / "Baseline total time" to budget/win semantics.

4. Baseline row (`updateBaselineRow`)
   - Proposed: `Target: baseline finish 3.428 loss at 10m budget`.
   - Fallback when the baseline finish loss can't be estimated yet: keep
     `Select a baseline to set the target quality` / a "not enough baseline data" note.

5. Hero KPI (`kpis.js`)
   - Value = `finishMarginLoss` (formatted), unit `ahead` / `behind`.
   - Sub: `finish 3.412 vs baseline 3.428 · <throughput>`.
   - Caption: finish-quality race wording (e.g. `Better finish at budget` / `Worse finish at
     budget`) rather than equal-loss timing.
