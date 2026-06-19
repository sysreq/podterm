import { app, getPodState } from '../state.js';
import * as d from '../derive.js';
import { fmtInt, fmtMoney, fmtClock, fmtDuration, fmtDelta, fmtMs, fmtSecShort, fmtTps } from '../format.js';
import { kpiCard } from '../cards.js';
import { updateBootPanel } from './boot.js';
import { updateBaselineRow, updateRaceBanner } from './race.js';

let cards = null;
let diag = null;

export function hasKpiCards() {
  return !!cards;
}

export function buildKpiRow() {
  if (cards) return;
  const row = document.getElementById('kpi-row');
  cards = {
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
  for (const key of ['projected', 'hero', 'loss', 'gpu', 'cost', 'health']) {
    row.appendChild(cards[key].el);
  }
  cards.health.el.classList.add('clickable');
  cards.health.el.addEventListener('click', () => {
    const p = document.getElementById('diagnostics-panel');
    if (p && !p.hidden) p.scrollIntoView({ behavior: 'smooth', block: 'start' });
  });

  const diagRow = document.getElementById('diag-row');
  diag = {
    vbpb: kpiCard({ label: 'Validation BPB', variant: 'diag', spark: true,
      tooltip: 'Bits per byte on the validation set at each eval — lower is better' }),
    system: kpiCard({ label: 'System', variant: 'diag', bar: true,
      tooltip: 'CPU utilization (bar/value) and RAM (sub-line) from RunPod telemetry' }),
  };
  for (const key of ['vbpb', 'system']) {
    diagRow.appendChild(diag[key].el);
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

function noteTelemetry(card, running) {
  card.note(running ? 'Polling RunPod telemetry…' : 'Pod not running');
  if (card.bar) card.bar.set(0);
}

function updateTelemetryCards(state, pod) {
  const t = state.telemetry;
  const running = pod.desiredStatus === 'RUNNING';
  const live = running && t;

  if (live && t.cpu_pct != null) {
    const ramSub = t.mem_pct != null
      ? (t.ram_total_gb
          ? `RAM ${((t.mem_pct / 100) * t.ram_total_gb).toFixed(1)}/${t.ram_total_gb} GB`
          : `RAM ${Math.round(t.mem_pct)}%`)
      : (t.cpu_name || '');
    diag.system.set({ value: String(Math.round(t.cpu_pct)), unit: '% CPU', sub: ramSub });
    diag.system.bar.set(t.cpu_pct / 100);
  } else noteTelemetry(diag.system, running);

  const gpuName = state.info.gpu_type || state.runRow?.gpu_type || podGpuName(pod);
  const totalGiB = (t && t.gpu_mem_total_gb) || d.parseGpuMemGiB(gpuName);
  let memSub = '';
  if (live && t.gpu_mem_pct != null) {
    memSub = totalGiB
      ? `${((t.gpu_mem_pct / 100) * totalGiB).toFixed(1)}/${totalGiB} GB · live`
      : `${Math.round(t.gpu_mem_pct)}% mem · live`;
  } else if (state.memory) {
    const usedGiB = state.memory.peak_mib / 1024;
    memSub = totalGiB
      ? `peak ${usedGiB.toFixed(1)}/${totalGiB} GB`
      : `peak ${usedGiB.toFixed(1)} GB`;
  } else {
    memSub = state.finished ? 'no mem stats' : 'mem at run end';
  }
  if (live && t.gpu_util_pct != null) {
    cards.gpu.set({ value: String(Math.round(t.gpu_util_pct)), unit: '%', sub: memSub });
    cards.gpu.bar.set(t.gpu_util_pct / 100);
  } else {
    cards.gpu.note(running ? 'Polling RunPod telemetry…' : (state.memory ? memSub : 'Pod not running'));
    cards.gpu.bar.set(0);
  }
}

const BAND_LABEL = { ok: 'OK', warn: 'WARN', error: 'ERROR' };

function fmtHealthMetric(m) {
  if (m.value == null) return '—';
  const v = m.value;
  if (m.unit === 'frac') return `${(v * 100).toFixed(0)}%`;
  const a = Math.abs(v);
  const num = a !== 0 && (a < 0.01 || a >= 1000) ? v.toExponential(1) : String(Math.round(v * 100) / 100);
  return m.unit && m.unit !== 'frac' ? `${num}` : num;
}

function setVerdictClass(verdict) {
  const el = cards.health.el;
  el.classList.toggle('verdict-ok', verdict === 'ok');
  el.classList.toggle('verdict-warn', verdict === 'warn');
  el.classList.toggle('verdict-error', verdict === 'error');
}

function updateHealthCard(state) {
  if (!cards) return;
  const d0 = state.diagnostic;
  if (!d0 || !d0.health) { cards.health.note('No diagnostics yet…'); setVerdictClass(null); return; }
  const h = d0.health;
  const ok = h.overall === 'ok';
  setVerdictClass(h.overall);
  const offenders = h.headline.filter((m) => m.band === 'bad').concat(h.headline.filter((m) => m.band === 'warn'));
  const sub = offenders.length
    ? offenders.slice(0, 2).map((m) => `${m.label} ${fmtHealthMetric(m)}`).join(' · ')
    : `step ${d0.step} · ${h.counts.good || 0} ok`;
  cards.health.set({
    value: BAND_LABEL[h.overall] || '—',
    sub,
    subClass: ok ? 'success' : 'danger',
    accent: ok ? 'success' : 'danger',
  });
}

export function updateKpis(podId) {
  if (!cards) return;
  updateBootPanel(podId);
  const state = getPodState(podId);
  const pod = app.pods.find((p) => p.id === podId) || {};
  const m = state.lastMetric;
  const running = pod.desiredStatus === 'RUNNING';

  updateTelemetryCards(state, pod);
  updateDiagRow(state);
  updateHealthCard(state);

  if (!m) {
    const waitMsg = state.finished ? 'Run finished — no metrics recorded'
      : running ? 'Waiting for first metric…' : 'No metrics recorded for this pod';
    cards.projected.note(waitMsg);
    cards.hero.note(waitMsg);
    cards.loss.note(waitMsg);
    updateCostCard(state, pod, null);
    updateRaceBanner(state, null, null, null);
    updateBaselineRow(state, null);
    return;
  }

  const msSeries = state.metricHistory.map((x) => x.step_avg_ms).filter((v) => v != null);
  const ema50 = d.ema(msSeries, d.ETA_EMA_N);
  const ema100 = d.ema(msSeries, d.AVG_EMA_N);
  const remaining = m.total_steps != null ? m.total_steps - m.step : null;
  const eta = state.finished ? 0 : d.etaMs(remaining, ema50);

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

  // Shared quality-vs-time race math (also feeds the banner and sidebar pace lines)
  const batchTokens = state.info.batch_tokens ?? state.runRow?.batch_tokens ?? null;
  const race = d.qualityRace({
    metric: m,
    history: state.metricHistory,
    baselineSamples: state.baselineSteps.map((s) => state.baselineByStep[s]),
    baselineTotalTimeMs: state.baselineTotalTimeMs,
    batchTokens,
    emaMsPerStep: ema100,
    baselineSample: d.baselineAtStep(state.baselineByStep, state.baselineSteps, m.step),
  });

  // 2 — Ahead/Behind hero (quality lead + live throughput)
  if (race.state === 'ahead' || race.state === 'behind') {
    const ahead = race.state === 'ahead';
    const tput = race.throughputTps != null ? `${fmtTps(race.throughputTps)} tok/s` : `${fmtMs(race.msPerStep)} ms/step`;
    cards.hero.set({
      value: fmtSecShort(race.leadMs),
      unit: ahead ? 'ahead' : 'behind',
      sub: `at ${race.currentLoss != null ? race.currentLoss.toFixed(3) : '?'} loss · ${tput}`,
      subClass: ahead ? 'success' : 'danger',
      caption: ahead ? 'Reaching loss sooner' : 'Reaching loss slower',
      captionClass: ahead ? 'success' : 'danger',
      accent: ahead ? 'success' : 'danger',
    });
  } else {
    cards.hero.note('No baseline selected — pick one below to start the race');
  }

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

  updateCostCard(state, pod, eta);
  updateRaceBanner(state, m, race, eta);
  updateBaselineRow(state, race);
}
