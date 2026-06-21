function mergeMetric(target, m) {
  for (const [k, v] of Object.entries(m)) {
    if (v == null) continue;
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
