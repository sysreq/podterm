import * as d from '../../derive.js';
import { fmtMoney } from '../../format.js';
import { kpiStore } from './store.js';

export function updateCostCard(state, pod, eta) {
  const { cards } = kpiStore;
  const row = state.runRow;
  if (row && row.total_cost != null && row.finished_at) {
    cards.cost.set({ value: fmtMoney(row.total_cost), sub: 'Final recorded cost' });
    return;
  }
  const rate = pod.costPerHr ?? row?.cost_per_hr ?? null;
  if (rate == null) { cards.cost.note('No hourly rate recorded for this run'); return; }
  const elapsed = d.elapsedWallMs(row?.started_at, Date.now());
  const so = d.costSoFar(elapsed, rate);
  if (so == null) { cards.cost.note(`Rate ${fmtMoney(rate)}/hr — no start time recorded yet`); return; }
  const proj = d.projectedTotalCost(elapsed, eta, rate);
  cards.cost.set({
    value: fmtMoney(so),
    sub: proj != null ? `Projected total: ${fmtMoney(proj)}` : `Rate ${fmtMoney(rate)}/hr`,
  });
}
