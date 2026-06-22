function mergeMetric(target, m) {
  for (const [k, v] of Object.entries(m)) {
    if (v == null) continue;
    // `0.0` is the live-path "no train loss this event" sentinel: pipeline.py
    // sends `train_loss = payload.get("train_loss") or 0.0`, and the producer
    // emits a val-only metric at the SAME step as the final real train metric
    // (gpt-golf train_gpt.log_metrics). Without this skip that val-only 0.0
    // clobbers the real per-step train_loss on merge. A genuine cross-entropy
    // loss is never exactly 0, so this never drops real data. (Backend finding
    // 7 should switch the sentinel to null; until then keep the `> 0` contract
    // used by every other consumer — charts.js, derive/stats.js, finish.js, …)
    if (k === 'train_loss' && !(v > 0)) continue;
    target[k] = v;
  }
}

export function ingestMetric(state, m) {
  const existing = state.metricByStep.get(m.step);
  let merged;
  let isNew = false;
  if (existing) {
    mergeMetric(existing, m);
    merged = existing;
  } else {
    merged = { step: m.step };
    mergeMetric(merged, m);
    state.metricByStep.set(m.step, merged);
    state.metricHistory.push(merged);
    isNew = true;
  }
  if (m.val_bpb != null && !state.evalSteps.has(m.step)) {
    state.evalSteps.add(m.step);
    state.evals.push({ step: m.step, val_bpb: m.val_bpb, val_loss: m.val_loss ?? null });
  }
  if (!state.lastMetric || merged.step >= state.lastMetric.step) state.lastMetric = merged;
  return { merged, isNew };
}
