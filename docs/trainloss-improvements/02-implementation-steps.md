# Train Loss To Estimated ValBPB Implementation Plan

## Implementation Steps

1. Add ValBPB derivation helpers.
   - Implement `validValBpbSamples`, `estimatedValBpbAtStep`, `estimatedValBpbAtTime`, `estimatedValBpbSeries`, and slope/time-to-target helpers.
   - Add unit tests for actual, interpolated, projected, hold, and no-data cases.

2. Update live chart plumbing.
   - Rename exported chart constants from `loss*` to neutral or BPB-specific names.
   - Change live traces to estimated ValBPB line, baseline estimated ValBPB line, and actual eval markers.
   - Update y-axis to direct BPB values.

3. Update baseline loading.
   - Build baseline eval samples from baseline metrics.
   - Build baseline estimated ValBPB series for the chart and race.
   - Stop using baseline `train_loss` for chart traces.

4. Update KPIs.
   - Replace `Loss (train)` card with `Estimated ValBPB`.
   - Feed it from the latest estimated ValBPB value.
   - Preserve the actual `Validation BPB` diagnostic card.

5. Update race math and race rendering.
   - Implement ValBPB-based race output.
   - Wire it into `live/kpis.js`, `live/race.js`, and `runlist.js`.
   - Coordinate this with the max-wallclock finish semantics in `racebar-improvements.md`.

6. Update compare tab.
   - Replace train-loss regression logic with ValBPB regression logic.
   - Replace Training Loss Comparison chart with Estimated ValBPB Comparison.
   - Keep or fold the existing Validation BPB chart into the new chart as actual markers.

7. Rename visible strings.
   - Search for user-facing `loss`, `train loss`, `Training Loss`, and `Loss (train)`.
   - Keep diagnostic/internal terms where they are technically correct, such as model-health `loss_recompute`.

8. Verify.
   - Run JS unit tests.
   - Run Python tests to ensure backend metric compatibility remains intact.
   - Manually verify:
     - no eval yet
     - one eval only
     - multiple evals with improving BPB
     - worsening eval trend
     - baseline selected
     - historical compare


## Open Questions

1. Should estimated ValBPB use step-based interpolation or train-time interpolation for the live chart?
   - Proposed: step-based for chart x-axis, train-time based for finish/race math.

2. With only one eval point, should the estimated line hold flat or stay hidden until two evals exist?
   - Proposed: hold flat with low-confidence labeling so the UI has a current value but avoids pretending trend is known.

3. Should the regression banner compare `best_val_bpb` or estimated finish ValBPB?
   - Proposed: use estimated finish ValBPB when enough eval data exists; otherwise fall back to `best_val_bpb`.

4. Should `train_loss` remain available in a debug/advanced view?
   - Proposed: yes, keep it in data and diagnostics, but remove it from primary quality UI.
