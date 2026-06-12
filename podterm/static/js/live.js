// Live view: info bar, loss chart, baseline comparison, log viewer.
import { app, getPodState, ensurePodStream, hydrateFromDb, on } from './state.js';
import { scaleY, lossYAxis, lossHover, plotLayout, plotConfig } from './charts.js';

function lossSeries(state) {
  const pts = state.metricHistory.filter((m) => m.train_loss > 0);
  return {
    x: pts.map((m) => m.step),
    y: pts.map((m) => scaleY(m.train_loss)),
    raw: pts.map((m) => m.train_loss),
  };
}

function infoStrings(state) {
  const m = state.lastMetric;
  let loss = null;
  for (let i = state.metricHistory.length - 1; i >= 0; i--) {
    if (state.metricHistory[i].train_loss > 0) { loss = state.metricHistory[i].train_loss; break; }
  }
  const lastEval = state.evals[state.evals.length - 1];
  return {
    step: m ? `Step: ${m.step}/${m.total_steps}` : 'Step: -',
    avg: m && m.step_avg_ms != null ? `Avg: ${m.step_avg_ms.toFixed(1)} ms` : 'Avg: -',
    loss: loss != null ? `Loss: ${loss.toFixed(4)}` : 'Loss: -',
    bpb: lastEval ? `BPB: ${lastEval.val_bpb.toFixed(4)}` : 'BPB: -',
    mem: state.memory ? `Mem: ${state.memory.peak_mib} MiB` : 'Mem: -',
  };
}

function updateInfoBar(state) {
  const info = infoStrings(state);
  const stepEl = document.getElementById('info-step');
  if (!stepEl) return;
  stepEl.textContent = info.step;
  document.getElementById('info-avg').textContent = info.avg;
  document.getElementById('info-loss').textContent = info.loss;
  document.getElementById('info-bpb').textContent = info.bpb;
  document.getElementById('info-mem').textContent = info.mem;
}

function updateLogViewer(state) {
  const viewer = document.getElementById('log-viewer');
  if (!viewer || viewer.style.display === 'none') return;
  viewer.textContent = state.logLines.map((l) => l.text).join('\n') + (state.logLines.length ? '\n' : '');
  viewer.scrollTop = viewer.scrollHeight;
}

function renderLossChart(state) {
  const series = lossSeries(state);
  const traces = [
    { x: series.x, y: series.y, customdata: series.raw,
      name: 'train_loss', type: 'scatter', line: { color: '#ffd166' }, hovertemplate: lossHover },
  ];
  if (state.baselineX.length) {
    traces.push({ x: state.baselineX, y: state.baselineY, customdata: state.baselineRaw,
      name: 'baseline', type: 'scatter', line: { color: 'rgba(255,255,255,0.35)', dash: 'dash' }, hovertemplate: lossHover });
  }
  Plotly.newPlot('chart-loss', traces, { ...plotLayout, yaxis: lossYAxis, height: 400 }, plotConfig);
}

// ── Live View ──
export function renderLiveView(podId) {
  ensurePodStream(podId);
  hydrateFromDb(podId);
  const state = getPodState(podId);

  document.getElementById('live-placeholder').style.display = 'none';
  document.getElementById('live-info').style.display = 'flex';
  document.getElementById('live-baseline').style.display = 'flex';
  document.getElementById('live-charts').style.display = 'grid';
  document.getElementById('log-viewer').style.display = 'block';
  updateLogViewer(state);

  // Info bar — static from pod data, dynamic from cached state
  const pod = app.pods.find((p) => p.id === podId) || {};
  const rt = pod.runtime || {};
  const gpus = rt.gpus || [];
  const gpuName = gpus.length ? gpus[0].gpuDisplayName : (pod.machine || {}).gpuDisplayName || pod.gpu || '-';
  const cost = pod.costPerHr || '?';
  const info = infoStrings(state);
  document.getElementById('live-info').innerHTML =
    `<span>GPU: ${gpuName}</span><span>$/hr: ${cost}</span><span id="info-step">${info.step}</span><span id="info-avg">${info.avg}</span><span id="info-loss">${info.loss}</span><span id="info-bpb">${info.bpb}</span><span id="info-mem">${info.mem}</span><a href="/api/runs/${podId}/log" target="_blank" style="color:var(--accent);text-decoration:none;font-size:12px;margin-left:auto">Full Log</a>`;

  renderLossChart(state);

  // Populate baseline selector and restore saved selection
  loadBaselineOptions(podId);

  // Clear stale diffs from previously viewed pod, then restore if we have data
  document.getElementById('diff-loss').innerHTML = '';
  document.getElementById('diff-time').innerHTML = '';
  if (state.lastMetric) updateBaselineDiffs(state.lastMetric);
}

// ── Baseline ──
async function loadBaselineOptions(podId) {
  const sel = document.getElementById('baseline-select');
  sel.innerHTML = '<option value="">None</option>';
  try {
    const runs = await fetch('/api/runs?limit=30').then((r) => r.json());
    if (app.activePod !== podId) return; // user switched during fetch
    for (const r of runs) {
      if (r.run_id === podId) continue; // a run can never be its own baseline
      if (!r.total_steps) continue; // skip runs with no data
      const bpb = r.best_val_bpb != null ? ` | BPB ${r.best_val_bpb.toFixed(4)}` : '';
      const label = `${r.pod_name || r.run_id} (${r.branch || '?'}${bpb})`;
      const opt = document.createElement('option');
      opt.value = r.run_id;
      opt.textContent = label;
      sel.appendChild(opt);
    }
  } catch {}
  // Restore saved selection
  const state = getPodState(podId);
  if (state.baselineRunId) sel.value = state.baselineRunId;
}

async function onBaselineChange() {
  if (!app.activePod) return;
  const state = getPodState(app.activePod);
  const podId = app.activePod;
  const runId = document.getElementById('baseline-select').value;

  state.baselineRunId = runId || null;
  state.baselineByStep = {};
  state.baselineX = [];
  state.baselineY = [];
  state.baselineRaw = [];
  document.getElementById('diff-loss').innerHTML = '';
  document.getElementById('diff-time').innerHTML = '';

  if (!runId) {
    renderLossChart(state);
    return;
  }

  const metrics = await fetch(`/api/runs/${runId}/metrics`).then((r) => r.json());
  if (app.activePod !== podId) return; // user switched during fetch

  for (const m of metrics) {
    state.baselineByStep[m.step] = m;
  }
  const baselineMetrics = metrics.filter((m) => m.train_loss);
  state.baselineX = baselineMetrics.map((m) => m.step);
  state.baselineY = baselineMetrics.map((m) => scaleY(m.train_loss));
  state.baselineRaw = baselineMetrics.map((m) => m.train_loss);

  renderLossChart(state);
  if (state.lastMetric) updateBaselineDiffs(state.lastMetric);
}

function updateBaselineDiffs(m) {
  if (!app.activePod) return;
  const state = getPodState(app.activePod);
  const diffEl = document.getElementById('diff-loss');
  const timeEl = document.getElementById('diff-time');
  if (!diffEl) return;
  if (!state.baselineRunId || !Object.keys(state.baselineByStep).length) {
    diffEl.innerHTML = '';
    timeEl.innerHTML = '';
    return;
  }

  const bSteps = Object.keys(state.baselineByStep).map(Number).sort((a, b) => a - b);
  let bStep = null;
  for (let i = bSteps.length - 1; i >= 0; i--) {
    if (bSteps[i] <= m.step) { bStep = bSteps[i]; break; }
  }
  if (bStep == null) return;

  const b = state.baselineByStep[bStep];

  // Loss diff (negative = better)
  if (m.train_loss > 0 && b.train_loss) {
    const d = m.train_loss - b.train_loss;
    const sign = d > 0 ? '+' : '';
    const color = d <= 0 ? 'var(--success)' : '#ff6b6b';
    diffEl.innerHTML = `<span style="color:${color};font-weight:600">(${sign}${d.toFixed(4)})</span>`;
  }

  // Time diffs: step_avg diff + wall-clock diff from train_time
  const parts = [];
  if (m.step_avg_ms && b.step_avg_ms) {
    const dAvg = m.step_avg_ms - b.step_avg_ms;
    const sign = dAvg > 0 ? '+' : '';
    const color = dAvg <= 0 ? 'var(--success)' : '#ff6b6b';
    parts.push(`<span style="color:${color}">${sign}${dAvg.toFixed(1)}ms</span>`);
  }
  if (m.train_time_ms && b.train_time_ms) {
    const dWall = m.train_time_ms - b.train_time_ms;
    const absSec = Math.abs(dWall / 1000);
    const timeStr = absSec >= 60 ? `${(absSec / 60).toFixed(1)}m` : `${absSec.toFixed(1)}s`;
    const color = dWall <= 0 ? 'var(--success)' : '#ff6b6b';
    const label = dWall <= 0 ? 'ahead' : 'behind';
    parts.push(`<span style="color:${color}">${timeStr} ${label}</span>`);
  }
  timeEl.innerHTML = parts.join('<span style="color:var(--text-ghost)"> / </span>');
}

// ── Bus subscriptions + static element wiring ──
export function initLive() {
  document.getElementById('baseline-select').addEventListener('change', onBaselineChange);

  on('pod:metric', ({ podId, m, trainPoint }) => {
    if (app.activePod !== podId) return;
    if (!document.getElementById('info-step')) return;
    if (trainPoint) {
      Plotly.extendTraces('chart-loss', { x: [[m.step]], y: [[scaleY(m.train_loss)]], customdata: [[m.train_loss]] }, [0]);
    }
    updateInfoBar(getPodState(podId));
    updateBaselineDiffs(m);
  });

  on('pod:log', ({ podId }) => {
    if (app.activePod !== podId) return;
    updateLogViewer(getPodState(podId));
  });

  on('pod:memory', ({ podId }) => {
    if (app.activePod !== podId) return;
    updateInfoBar(getPodState(podId));
  });

  on('pod:reset', ({ podId }) => {
    if (app.activePod !== podId) return;
    if (!document.getElementById('info-step')) return;
    renderLossChart(getPodState(podId));
    updateInfoBar(getPodState(podId));
  });

  on('pod:hydrated', ({ podId }) => {
    if (app.activePod !== podId) return;
    if (!document.getElementById('info-step')) return;
    renderLossChart(getPodState(podId));
    updateInfoBar(getPodState(podId));
  });
}
