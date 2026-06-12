// Compare tab: overlay charts + summary table for selected runs.
import { app, emit } from './state.js';
import { scaleY, lossYAxis, lossHover, plotLayout, plotConfig, COLORS } from './charts.js';
import { renderHistoryTable } from './history.js';

let lastCompareData = null; // cached for re-render on chip removal

export async function compareSelected() {
  if (app.selectedRuns.size < 2) { alert('Select at least 2 runs.'); return; }
  emit('tab:switch', { tab: 'compare' });

  const r = await fetch('/api/compare', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ run_ids: [...app.selectedRuns] }),
  });
  lastCompareData = await r.json();
  renderComparison();
}

function removeFromComparison(runId) {
  app.selectedRuns.delete(runId);
  document.getElementById('selection-count').textContent = `${app.selectedRuns.size} selected`;
  if (app.selectedRuns.size < 2) {
    // Not enough runs left — clear comparison
    document.getElementById('compare-selection').style.display = 'none';
    document.getElementById('compare-charts').innerHTML = '';
    document.getElementById('compare-summary').innerHTML = '';
    document.getElementById('compare-placeholder').style.display = '';
    document.getElementById('compare-placeholder').textContent = 'Select at least 2 runs from the History tab.';
    lastCompareData = null;
    renderHistoryTable(); // update checkboxes
    return;
  }
  // Remove from cached data and re-render
  if (lastCompareData) {
    delete lastCompareData.metrics[runId];
    delete lastCompareData.runs[runId];
    renderComparison();
  }
  renderHistoryTable(); // sync checkboxes
}

function renderComparison() {
  const data = lastCompareData;
  if (!data) return;

  document.getElementById('compare-placeholder').style.display = 'none';

  // Selection chips
  const selEl = document.getElementById('compare-selection');
  selEl.style.display = 'flex';
  let chipsHtml = '<span>Comparing:</span>';
  for (const rid of Object.keys(data.runs)) {
    const run = data.runs[rid] || {};
    const label = `${run.branch || ''}@${(run.commit_hash || '').slice(0, 7)}`;
    chipsHtml += `<div class="run-chip"><span>${label}</span><span class="chip-x" data-run-id="${rid}">&times;</span></div>`;
  }
  selEl.innerHTML = chipsHtml;

  // Loss comparison chart
  const lossTraces = [];
  const bpbTraces = [];
  let i = 0;
  for (const [rid, metrics] of Object.entries(data.metrics)) {
    const run = data.runs[rid] || {};
    const label = `${run.branch || ''}@${(run.commit_hash || '').slice(0, 7)}`;
    const color = COLORS[i % COLORS.length];

    const cmpMetrics = metrics.filter((m) => m.train_loss);
    const trainSteps = cmpMetrics.map((m) => m.step);
    const trainLoss = cmpMetrics.map((m) => scaleY(m.train_loss));
    const trainLossRaw = cmpMetrics.map((m) => m.train_loss);
    if (trainSteps.length) {
      lossTraces.push({ x: trainSteps, y: trainLoss, customdata: trainLossRaw, name: label, type: 'scatter', line: { color }, hovertemplate: lossHover });
    }

    const valSteps = metrics.filter((m) => m.val_bpb != null).map((m) => m.step);
    const valBpb = metrics.filter((m) => m.val_bpb != null).map((m) => m.val_bpb);
    if (valSteps.length) {
      bpbTraces.push({ x: valSteps, y: valBpb, name: label, mode: 'markers+lines', marker: { color, size: 8 }, line: { color } });
    }
    i++;
  }

  const chartsEl = document.getElementById('compare-charts');
  chartsEl.innerHTML = '<div class="chart-box" id="cmp-loss"></div><div class="chart-box" id="cmp-bpb"></div>';
  Plotly.newPlot('cmp-loss', lossTraces, { ...plotLayout, yaxis: lossYAxis, title: 'Training Loss Comparison', height: 300 }, plotConfig);
  Plotly.newPlot('cmp-bpb', bpbTraces, { ...plotLayout, title: 'Validation BPB Comparison', height: 300 }, plotConfig);

  // Summary table
  let summaryHtml = '<table><thead><tr><th>Run</th><th>Branch</th><th>GPU</th><th>Steps</th><th>Best BPB</th><th>Cost</th></tr></thead><tbody>';
  for (const rid of Object.keys(data.runs)) {
    const run = data.runs[rid] || {};
    const bpb = run.best_val_bpb != null ? run.best_val_bpb.toFixed(4) : '-';
    const cost = run.total_cost != null ? `$${run.total_cost.toFixed(3)}` : '-';
    summaryHtml += `<tr><td>${(run.pod_name || rid).slice(0, 20)}</td><td>${run.branch || '-'}</td><td>${run.gpu_type || '-'}</td><td>${run.total_steps || '-'}</td><td>${bpb}</td><td>${cost}</td></tr>`;
  }
  summaryHtml += '</tbody></table>';
  document.getElementById('compare-summary').innerHTML = summaryHtml;
}

export function initCompare() {
  document.getElementById('compare-selection').addEventListener('click', (e) => {
    const x = e.target.closest('.chip-x');
    if (x) removeFromComparison(x.dataset.runId);
  });
}
