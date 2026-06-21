import { emit } from '../../state.js';
import { kpiCard } from '../../cards.js';
import { kpiStore } from './store.js';

export function buildKpiRow() {
  if (kpiStore.cards) return;
  const row = document.getElementById('kpi-row');
  kpiStore.cards = {
    projected: kpiCard({ label: 'Projected Finish', spark: true,
      tooltip: 'Remaining steps × EMA-50 of step time; sparkline shows recent ms/step' }),
    hero: kpiCard({ label: 'Baseline', hero: true,
      tooltip: 'Quality race vs the selected baseline: training wall-clock ahead/behind to reach the current loss, plus live throughput' }),
    loss: kpiCard({ label: 'Loss (train)',
      tooltip: 'Latest training loss, with change vs ~100 steps earlier' }),
    gpu: kpiCard({ label: 'GPU', bar: true,
      tooltip: 'GPU utilization (bar/value) and memory (sub-line) from RunPod telemetry' }),
    cost: kpiCard({ label: 'Cost So Far',
      tooltip: 'Wall-clock elapsed × hourly rate from pod metadata, plus projected total at current pace' }),
    health: kpiCard({ label: 'Model Health',
      tooltip: 'Off-pod diagnostics verdict on the latest snapshot — click to open the Model Health panel' }),
  };
  for (const key of ['projected', 'hero', 'loss', 'gpu', 'cost', 'health']) row.appendChild(kpiStore.cards[key].el);
  kpiStore.cards.health.el.classList.add('clickable');
  kpiStore.cards.health.el.querySelector('.kpi-caption').hidden = false;
  kpiStore.cards.health.el.addEventListener('click', () => emit('tab:switch', { tab: 'health' }));

  const diagRow = document.getElementById('diag-row');
  kpiStore.diag = {
    vbpb: kpiCard({ label: 'Validation BPB', variant: 'diag', spark: true,
      tooltip: 'Bits per byte on the validation set at each eval — lower is better' }),
    system: kpiCard({ label: 'System', variant: 'diag', bar: true,
      tooltip: 'CPU utilization (bar/value) and RAM (sub-line) from RunPod telemetry' }),
  };
  for (const key of ['vbpb', 'system']) diagRow.appendChild(kpiStore.diag[key].el);
}
