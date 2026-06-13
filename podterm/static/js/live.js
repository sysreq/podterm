// Live view: KPI card row, loss chart, baseline comparison, log viewer.
import { app, getPodState, ensurePodStream, hydrateFromDb, on } from './state.js';
import { scaleY, initLiveCharts, updateLiveCharts, updateBaselineTrace } from './charts.js';
import * as d from './derive.js';
import { fmtInt, fmtMs, fmtDuration, fmtSecShort, fmtClock, fmtMoney, fmtDelta } from './format.js';
import { kpiCard } from './cards.js';
import { renderLogs } from './logs.js';
import { renderConfigPanel } from './configpanel.js';

let cards = null;
let diag = null;

// ── Boot panel — replaces the dashboard until the container is up ──
// Any phase/metric implies the container booted (those only originate inside it),
// so reconnects to an already-running pod never get stuck in boot mode.
function isBooting(state) {
  return !!(state.boot && !state.boot.done && !state.lastMetric && !state.phase && !state.finished);
}

const LAYER_STATE_LABELS = {
  pending: 'Queued', waiting: 'Queued', downloading: 'Downloading',
  verifying: 'Verifying', downloaded: 'Downloaded', extracting: 'Extracting', complete: 'Done',
};

function renderBootPanel(boot) {
  document.getElementById('boot-stage').textContent = boot.stage || 'Booting…';
  document.getElementById('boot-image').textContent = boot.image || '';
  document.getElementById('boot-counter').textContent =
    boot.total ? `${boot.complete}/${boot.total} layers · ${Math.round(boot.pct)}%` : '';
  document.querySelector('#boot-panel .boot-bar-fill').style.width = `${boot.pct || 0}%`;
  document.getElementById('boot-message').textContent = boot.message || 'Waiting for machine logs…';
  const grid = document.getElementById('boot-layers');
  grid.innerHTML = '';
  for (const l of boot.layers || []) {
    const label = LAYER_STATE_LABELS[l.state] || l.state;
    const chip = document.createElement('div');
    chip.className = `boot-layer st-${l.state}`;
    chip.title = `${l.id} — ${label}`;
    const id = document.createElement('span');
    id.className = 'bl-id';
    id.textContent = l.id.slice(0, 6);
    const st = document.createElement('span');
    st.className = 'bl-state';
    st.textContent = label;
    chip.append(id, st);
    grid.appendChild(chip);
  }
}

function updateBootPanel(podId) {
  const view = document.getElementById('live-view');
  const state = getPodState(podId);
  const booting = isBooting(state);
  const was = view.classList.contains('booting');
  view.classList.toggle('booting', booting);
  if (booting) {
    renderBootPanel(state.boot);
  } else if (was && window.Plotly) {
    // Charts laid out while hidden have zero width — resize on reveal.
    view.querySelectorAll('.js-plotly-plot').forEach((p) => Plotly.Plots.resize(p));
  }
}

// ── KPI row ──
function buildKpiRow() {
  if (cards) return;
  const row = document.getElementById('kpi-row');
  cards = {
    projected: kpiCard({ label: 'Projected Finish', spark: true,
      tooltip: 'Remaining steps × EMA-50 of step time; sparkline shows recent ms/step' }),
    hero: kpiCard({ label: 'Baseline', hero: true,
      tooltip: 'Race vs the selected baseline: per-step ms delta and cumulative time ahead or behind — independent quantities' }),
    avg: kpiCard({ label: 'Avg ms/step',
      tooltip: 'EMA-100 of step time; the target is the pace needed from here to beat the baseline' }),
    loss: kpiCard({ label: 'Loss (train)',
      tooltip: 'Latest training loss, with change vs ~100 steps earlier' }),
    cpu: kpiCard({ label: 'CPU Usage', bar: true,
      tooltip: 'Container CPU utilization from RunPod telemetry' }),
    ram: kpiCard({ label: 'RAM Usage', bar: true,
      tooltip: 'Container memory utilization from RunPod telemetry' }),
    gpu: kpiCard({ label: 'GPU Usage', bar: true,
      tooltip: 'GPU busy percentage from RunPod telemetry' }),
    gpumem: kpiCard({ label: 'GPU Memory', bar: true,
      tooltip: 'Live GPU memory vs device total while running; trainer-reported peak after the run' }),
    cost: kpiCard({ label: 'Cost So Far',
      tooltip: 'Wall-clock elapsed × hourly rate from pod metadata, plus projected total at current pace' }),
  };
  for (const key of ['projected', 'hero', 'avg', 'loss', 'cpu', 'ram', 'gpu', 'gpumem', 'cost']) {
    row.appendChild(cards[key].el);
  }

  // Diagnostics row — 4 metrics await emitters in gpt-golf; Validation BPB is live.
  const diagRow = document.getElementById('diag-row');
  diag = {
    attn: kpiCard({ label: 'Attention Entropy', variant: 'diag', spark: true,
      tooltip: 'Mean attention entropy across layers, in bits — how spread out attention is' }),
    grad: kpiCard({ label: 'Grad Norm', variant: 'diag', spark: true,
      tooltip: 'L2 norm of the gradient — spikes flag training instability' }),
    logit: kpiCard({ label: 'Logit Saturation', variant: 'diag', spark: true,
      tooltip: 'Share of logits near the activation ceiling — high values mean squashed outputs' }),
    loader: kpiCard({ label: 'Dataloader Wait', variant: 'diag', spark: true,
      tooltip: 'Fraction of step time spent waiting on the dataloader instead of compute' }),
    vbpb: kpiCard({ label: 'Validation BPB', variant: 'diag', spark: true,
      tooltip: 'Bits per byte on the validation set at each eval — lower is better' }),
  };
  for (const key of ['attn', 'grad', 'logit', 'loader', 'vbpb']) {
    diagRow.appendChild(diag[key].el);
  }
  for (const key of ['attn', 'grad', 'logit', 'loader']) {
    diag[key].note('Awaiting emitter', { pending: true });
  }
}

function updateDiagRow(state) {
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

function podGpuName(pod) {
  const gpus = (pod.runtime || {}).gpus || [];
  return gpus.length ? gpus[0].gpuDisplayName : (pod.machine || {}).gpuDisplayName || pod.gpu || null;
}

function lastTrainLoss(state) {
  for (let i = state.metricHistory.length - 1; i >= 0; i--) {
    const m = state.metricHistory[i];
    if (m.train_loss > 0) return m;
  }
  return null;
}

function updateCostCard(state, pod, eta) {
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

function pctCard(card, pct, sub = '') {
  card.set({ value: String(Math.round(pct)), unit: '%', sub });
  card.bar.set(pct / 100);
}

function updateTelemetryCards(state, pod) {
  const t = state.telemetry;
  const running = pod.desiredStatus === 'RUNNING';
  const live = running && t;

  if (live && t.cpu_pct != null) {
    pctCard(cards.cpu, t.cpu_pct, t.cpu_name || '');
  } else noteTelemetry(cards.cpu, running);
  if (live && t.mem_pct != null) {
    const ramSub = t.ram_total_gb
      ? `${((t.mem_pct / 100) * t.ram_total_gb).toFixed(1)} / ${t.ram_total_gb} GB`
      : '';
    pctCard(cards.ram, t.mem_pct, ramSub);
  } else noteTelemetry(cards.ram, running);
  if (live && t.gpu_util_pct != null) pctCard(cards.gpu, t.gpu_util_pct);
  else noteTelemetry(cards.gpu, running);

  // GPU memory: live used/total while running; trainer-reported peak after.
  // Device total comes from telemetry metadata; GPU-name parse is the fallback
  // for stopped pods where no telemetry flows.
  const gpuName = state.info.gpu_type || state.runRow?.gpu_type || podGpuName(pod);
  const totalGiB = (t && t.gpu_mem_total_gb) || d.parseGpuMemGiB(gpuName);
  if (live && t.gpu_mem_pct != null) {
    if (totalGiB) {
      const usedGiB = (t.gpu_mem_pct / 100) * totalGiB;
      cards.gpumem.set({
        value: `${usedGiB.toFixed(1)}/${totalGiB}`, unit: 'GB',
        sub: `${Math.round(t.gpu_mem_pct)}% of device · live`,
      });
    } else {
      cards.gpumem.set({ value: String(Math.round(t.gpu_mem_pct)), unit: '%',
        sub: 'Live · device total unknown' });
    }
    cards.gpumem.bar.set(t.gpu_mem_pct / 100);
  } else if (state.memory) {
    const usedGiB = state.memory.peak_mib / 1024;
    if (totalGiB) {
      cards.gpumem.set({
        value: `${usedGiB.toFixed(1)}/${totalGiB}`, unit: 'GB',
        sub: `Peak · reserved ${(state.memory.reserved_mib / 1024).toFixed(1)} GB`,
      });
      cards.gpumem.bar.set(usedGiB / totalGiB);
    } else {
      cards.gpumem.set({ value: usedGiB.toFixed(1), unit: 'GB peak', sub: 'Device memory total unknown' });
      cards.gpumem.bar.set(0);
    }
  } else {
    cards.gpumem.note(state.finished ? 'No memory stats recorded' : 'Reported at run end');
    cards.gpumem.bar.set(0);
  }
}

function noteTelemetry(card, running) {
  card.note(running ? 'Polling RunPod telemetry…' : 'Pod not running');
  if (card.bar) card.bar.set(0);
}

// ── Race banner + baseline row (numbers shared with the KPI cards by construction) ──
function updateRaceBanner(state, m, race, eta) {
  const banner = document.getElementById('race-banner');
  const headline = document.getElementById('race-headline');
  const sub = document.getElementById('race-sub');
  const fill = banner.querySelector('.race-bar-fill');
  const markP = banner.querySelector('.race-marker.projected');
  const markB = banner.querySelector('.race-marker.baseline');
  const labP = document.getElementById('race-label-projected');
  const labB = document.getElementById('race-label-baseline');

  banner.classList.remove('ahead', 'behind');
  markP.style.display = 'none';
  markB.style.display = 'none';
  labP.textContent = '';
  labB.textContent = '';

  if (!m) {
    headline.textContent = state.finished ? 'Run finished' : 'Waiting for metrics…';
    sub.textContent = state.finished
      ? 'No step metrics were recorded for this run'
      : 'The race banner lights up once steps start streaming';
    fill.style.width = '0%';
    return;
  }

  const stepFrac = m.total_steps ? m.step / m.total_steps : 0;

  if (race.state !== 'ahead' && race.state !== 'behind') {
    headline.textContent = 'No baseline selected';
    sub.textContent = 'Choose a baseline below to start the race';
    fill.style.width = `${(stepFrac * 100).toFixed(1)}%`;
    if (eta != null && m.train_time_ms != null) {
      labP.textContent = `Projected total ${fmtDuration(m.train_time_ms + eta)}`;
    }
    return;
  }

  const ahead = race.state === 'ahead';
  banner.classList.add(ahead ? 'ahead' : 'behind');
  headline.textContent = `You're ${ahead ? 'ahead of' : 'behind'} baseline by ${fmtSecShort(race.cumulativeMs)}`;
  if (state.finished) {
    sub.textContent = `Final margin vs baseline at step ${fmtInt(m.step)}`;
  } else if (race.requiredMs != null && race.requiredMs >= 1 && race.projectedMarginMs != null) {
    // Projected margin at the finish line — not the current cumulative lead.
    sub.textContent = race.projectedMarginMs >= 0
      ? `Keep ≤ ${fmtMs(race.requiredMs)} ms/step to finish ~${fmtSecShort(race.projectedMarginMs)} ahead`
      : `Need ≤ ${fmtMs(race.requiredMs)} ms/step — current pace finishes ~${fmtSecShort(race.projectedMarginMs)} behind`;
  } else {
    sub.textContent = 'Baseline total time already elapsed — cumulative lead is what counts';
  }

  // Progress bar on the time domain [0, max(projected total, baseline total)].
  const projTotal = eta != null && m.train_time_ms != null ? m.train_time_ms + eta : null;
  const baseTotal = state.baselineTotalTimeMs;
  const domain = Math.max(projTotal ?? 0, baseTotal ?? 0);
  if (domain > 0) {
    fill.style.width = `${Math.min(100, ((m.train_time_ms ?? 0) / domain) * 100).toFixed(1)}%`;
    if (projTotal != null) {
      markP.style.left = `${((projTotal / domain) * 100).toFixed(2)}%`;
      markP.style.display = 'block';
    }
    if (baseTotal != null) {
      markB.style.left = `${((baseTotal / domain) * 100).toFixed(2)}%`;
      markB.style.display = 'block';
    }
    // Keep the under-bar labels in the same left-to-right order as the markers.
    const projText = projTotal != null ? `Projected ${fmtDuration(projTotal)}` : '';
    const baseText = baseTotal != null ? `Baseline ${fmtDuration(baseTotal)}` : '';
    const projFirst = projTotal == null || baseTotal == null || projTotal <= baseTotal;
    labP.textContent = projFirst ? projText : baseText;
    labB.textContent = projFirst ? baseText : projText;
    labP.className = projFirst ? 'lab-projected' : 'lab-baseline';
    labB.className = projFirst ? 'lab-baseline' : 'lab-projected';
  } else {
    fill.style.width = `${(stepFrac * 100).toFixed(1)}%`;
  }
}

function updateBaselineRow(race) {
  const target = document.getElementById('baseline-target');
  if (race && race.requiredMs != null && race.requiredMs >= 1) {
    target.textContent = `To beat baseline: avg ms/step ≤ ${fmtMs(race.requiredMs)}`;
  } else if (race && (race.state === 'ahead' || race.state === 'behind')) {
    target.textContent = 'Past baseline time — cumulative lead decides';
  } else {
    target.textContent = 'Select a baseline to set the target pace';
  }
}

function updateKpis(podId) {
  if (!cards) return;
  updateBootPanel(podId);
  const state = getPodState(podId);
  const pod = app.pods.find((p) => p.id === podId) || {};
  const m = state.lastMetric;
  const running = pod.desiredStatus === 'RUNNING';

  updateTelemetryCards(state, pod);
  updateDiagRow(state);

  if (!m) {
    const waitMsg = state.finished ? 'Run finished — no metrics recorded'
      : running ? 'Waiting for first metric…' : 'No metrics recorded for this pod';
    cards.projected.note(waitMsg);
    cards.hero.note(waitMsg);
    cards.avg.note(waitMsg);
    cards.loss.note(waitMsg);
    updateCostCard(state, pod, null);
    updateRaceBanner(state, null, null, null);
    updateBaselineRow(null);
    return;
  }

  const msSeries = state.metricHistory.map((x) => x.step_avg_ms).filter((v) => v != null);
  const ema50 = d.ema(msSeries, d.ETA_EMA_N);
  const ema100 = d.ema(msSeries, d.AVG_EMA_N);
  const remaining = m.total_steps != null ? m.total_steps - m.step : null;
  const eta = state.finished ? 0 : d.etaMs(remaining, ema50);

  // 1 — Projected Finish
  if (state.finished) {
    cards.projected.set({
      value: 'Finished',
      sub: state.exitCode === 0 ? 'Completed cleanly' : `Exit code ${state.exitCode ?? '?'}`,
      subClass: state.exitCode === 0 ? 'success' : 'danger',
    });
  } else if (eta != null) {
    cards.projected.set({
      value: fmtDuration(eta),
      sub: `ETA ${fmtClock(new Date(Date.now() + eta))}`,
    });
  } else {
    cards.projected.note('Waiting for step timing…');
  }
  if (cards.projected.spark) cards.projected.spark(msSeries);

  // Shared race math (also feeds the banner and sidebar pace lines)
  const race = d.raceStatus({
    metric: m,
    baselineByStep: state.baselineByStep,
    baselineSteps: state.baselineSteps,
    baselineTotalTimeMs: state.baselineTotalTimeMs,
    emaMs: ema100,
  });

  // 2 — Ahead/Behind hero
  if (race.state === 'ahead' || race.state === 'behind') {
    const ahead = race.state === 'ahead';
    cards.hero.set({
      value: fmtDelta(race.perStepMs, 1),
      unit: 'ms/step',
      sub: `${fmtSecShort(race.cumulativeMs)} ${ahead ? 'ahead' : 'behind'} overall`,
      subClass: ahead ? 'success' : 'danger',
      caption: ahead ? 'On pace to win' : 'Falling behind',
      captionClass: ahead ? 'success' : 'danger',
      accent: ahead ? 'success' : 'danger',
    });
  } else {
    cards.hero.note('No baseline selected — pick one below to start the race');
  }

  // 3 — Avg ms/step + live required pace
  if (ema100 != null) {
    let targetSub = 'Target needs a baseline';
    let targetClass = '';
    if (race.requiredMs != null) {
      if (race.requiredMs >= 1) {
        targetSub = `Target: ≤ ${fmtMs(race.requiredMs)} ms`;
        targetClass = ema100 <= race.requiredMs ? 'success' : 'danger';
      } else {
        // Baseline's total time has already elapsed — no pace can beat it.
        targetSub = 'Past baseline time';
        targetClass = 'danger';
      }
    }
    cards.avg.set({ value: fmtMs(ema100), unit: 'ms', sub: targetSub, subClass: targetClass });
  } else {
    cards.avg.note('Waiting for step timing…');
  }

  // 4 — Loss (train)
  const lossD = d.deltaVsStepsAgo(state.metricHistory, 'train_loss', m.step, 100);
  if (lossD) {
    cards.loss.set({
      value: lossD.current.toFixed(4),
      sub: `${fmtDelta(lossD.delta, 4)} / ${fmtInt(lossD.stepsSpanned)} steps`,
      subClass: lossD.delta <= 0 ? 'success' : 'danger',
    });
  } else {
    const lt = lastTrainLoss(state);
    if (lt) cards.loss.set({ value: lt.train_loss.toFixed(4), sub: 'No earlier sample to compare yet' });
    else cards.loss.note('No training loss yet…');
  }

  // Cost (val BPB lives in the diagnostics row below the charts — it only
  // materializes at evals, not live)
  updateCostCard(state, pod, eta);

  // Race banner + baseline row read the same race object as cards 2 and 3.
  updateRaceBanner(state, m, race, eta);
  updateBaselineRow(race);
}

// ── Live View ──
export function renderLiveView(podId) {
  ensurePodStream(podId);
  hydrateFromDb(podId);
  const state = getPodState(podId);

  document.getElementById('live-placeholder').style.display = 'none';
  document.getElementById('live-view').classList.add('visible');
  document.getElementById('btn-full-log').href = `/api/runs/${podId}/log`;
  buildKpiRow();
  initLiveCharts();
  updateKpis(podId);
  updateLiveCharts(state);
  updateBaselineTrace(state);
  renderLogs(podId);
  renderConfigPanel(podId);

  // Populate baseline selector and restore saved selection
  loadBaselineOptions(podId);
}

// ── Baseline ──
async function loadBaselineOptions(podId) {
  const sel = document.getElementById('baseline-select');
  sel.innerHTML = '<option value="">None</option>';
  try {
    const runs = await fetch('/api/runs?limit=30').then((r) => r.json());
    if (app.activePod !== podId) return; // user switched during fetch
    const state = getPodState(podId);
    state.runRow = runs.find((r) => r.run_id === podId) || state.runRow;
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
    updateKpis(podId); // runRow may unlock the cost card
    renderConfigPanel(podId); // …and the config rows
  } catch {}
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
  state.baselineSteps = [];
  state.baselineTotalTimeMs = null;
  state.baselineX = [];
  state.baselineY = [];
  state.baselineRaw = [];

  if (!runId) {
    updateBaselineTrace(state);
    updateKpis(podId);
    return;
  }

  const metrics = await fetch(`/api/runs/${runId}/metrics`).then((r) => r.json());
  if (app.activePod !== podId) return; // user switched during fetch

  for (const m of metrics) {
    state.baselineByStep[m.step] = m;
  }
  state.baselineSteps = metrics.map((m) => m.step).sort((a, b) => a - b);
  const last = metrics[metrics.length - 1];
  state.baselineTotalTimeMs = last ? last.train_time_ms : null;

  const baselineMetrics = metrics.filter((m) => m.train_loss);
  state.baselineX = baselineMetrics.map((m) => m.step);
  state.baselineY = baselineMetrics.map((m) => scaleY(m.train_loss));
  state.baselineRaw = baselineMetrics.map((m) => m.train_loss);

  updateBaselineTrace(state);
  updateKpis(podId);
}

// ── Bus subscriptions + static element wiring ──
export function initLive() {
  document.getElementById('baseline-select').addEventListener('change', onBaselineChange);

  on('pod:metric', ({ podId }) => {
    if (app.activePod !== podId || !cards) return;
    updateLiveCharts(getPodState(podId));
    updateKpis(podId);
  });

  on('pod:boot', ({ podId }) => {
    if (app.activePod !== podId) return;
    updateBootPanel(podId);
  });

  for (const evt of ['pod:memory', 'pod:info', 'pod:summary', 'pod:phase', 'pod:telemetry']) {
    on(evt, ({ podId }) => {
      if (app.activePod !== podId) return;
      updateKpis(podId);
    });
  }

  for (const evt of ['pod:reset', 'pod:hydrated']) {
    on(evt, ({ podId }) => {
      if (app.activePod !== podId || !cards) return;
      const state = getPodState(podId);
      updateLiveCharts(state);
      updateBaselineTrace(state);
      updateKpis(podId);
    });
  }

  // Wall-clock cards (cost, ETA clock) tick even between metric events.
  setInterval(() => {
    if (app.activePod && cards && document.getElementById('live-view').classList.contains('visible')) {
      updateKpis(app.activePod);
    }
  }, 5000);
}
