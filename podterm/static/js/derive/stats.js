// ── Deltas ──
// Change vs the sample nearest to `stepsAgo` steps back. Sparsity-tolerant:
// metrics arrive every ~250 steps, so this finds real samples by step search.
export function deltaVsStepsAgo(history, key, currentStep, stepsAgo = 100) {
  if (!history || !history.length || currentStep == null) return null;
  let cur = null, past = null;
  for (let i = history.length - 1; i >= 0; i--) {
    const m = history[i];
    const v = m[key];
    if (v == null || (key === 'train_loss' && !(v > 0))) continue;
    if (cur == null) {
      if (m.step <= currentStep) cur = { step: m.step, value: v };
      continue;
    }
    if (m.step <= cur.step - stepsAgo) { past = { step: m.step, value: v }; break; }
  }
  if (!cur || !past) return null;
  return { current: cur.value, delta: cur.value - past.value, stepsSpanned: cur.step - past.step };
}

export function evalDelta(evals) {
  if (!evals || !evals.length) return null;
  const cur = evals[evals.length - 1];
  const prev = evals.length > 1 ? evals[evals.length - 2] : null;
  return { current: cur.val_bpb, step: cur.step, delta: prev ? cur.val_bpb - prev.val_bpb : null };
}

// ── Metadata parsing ──
// "NVIDIA H100 80GB HBM3" -> 80. Null when the name carries no memory size.
export function parseGpuMemGiB(gpuTypeName) {
  if (!gpuTypeName) return null;
  const m = String(gpuTypeName).match(/(\d+)\s*GB/i);
  return m ? Number(m[1]) : null;
}

// Finish horizon in ms from a run row's persisted launch config. `config_json` is
// a JSON *string* (TEXT column) and `time_budget` is in seconds. Null when absent
// or unparseable — callers fall back to observed time for pre-budget runs.
export function configFinishBudgetMs(runRow) {
  if (!runRow || !runRow.config_json) return null;
  try {
    const secs = Number(JSON.parse(runRow.config_json)?.time_budget);
    return Number.isFinite(secs) && secs > 0 ? secs * 1000 : null;
  } catch {
    return null;
  }
}
