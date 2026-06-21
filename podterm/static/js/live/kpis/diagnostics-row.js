import * as d from '../../derive.js';
import { fmtInt } from '../../format.js';
import { kpiStore } from './store.js';

export function updateDiagRow(state) {
  const { diag } = kpiStore;
  if (!diag) return;
  const ed = d.evalDelta(state.evals);
  if (!ed) {
    diag.vbpb.note('No eval completed yet…');
    if (diag.vbpb.spark) diag.vbpb.spark([]);
    return;
  }
  diag.vbpb.set({
    value: ed.current.toFixed(4),
    sub: ed.delta != null
      ? `${ed.delta <= 0 ? '▼' : '▲'} ${Math.abs(ed.delta).toFixed(4)} vs prev eval`
      : `First eval @ ${fmtInt(ed.step)}`,
    subClass: ed.delta == null ? '' : (ed.delta <= 0 ? 'success' : 'danger'),
  });
  if (diag.vbpb.spark) diag.vbpb.spark(state.evals.map((e) => e.val_bpb));
}
