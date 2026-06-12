// Sidebar ACTIVE RUNS cards: name, status badge, step/BPB stats, pace line.
import { app, getPodState, on } from './state.js';
import { baselineAtStep, paceDelta } from './derive.js';
import { fmtInt, fmtDelta, fmtSecShort } from './format.js';

// Same derivation the hero card and race banner use.
function paceInfo(state) {
  if (!state || !state.baselineRunId || !state.lastMetric || !state.baselineSteps.length) return null;
  const sample = baselineAtStep(state.baselineByStep, state.baselineSteps, state.lastMetric.step);
  const p = paceDelta(state.lastMetric, sample);
  return p ? { ...p, ahead: p.cumulativeMs <= 0 } : null;
}

function fmtPace(p) {
  return `${fmtDelta(p.perStepMs, 1)}ms (${fmtSecShort(p.cumulativeMs)} ${p.ahead ? 'ahead' : 'behind'})`;
}

export function renderRunList(onSelect) {
  const el = document.getElementById('run-list');
  el.innerHTML = '';
  if (!app.pods.length) {
    el.innerHTML = '<div class="run-empty">No active runs.<br>Launch Run to start one.</div>';
    return;
  }
  for (const p of app.pods) {
    const card = document.createElement('div');
    card.className = 'run-card' + (p.id === app.activePod ? ' active' : '');
    card.dataset.podId = p.id;
    card.innerHTML = `
      <div class="run-card-top"><span class="run-name"></span><span class="badge"></span></div>
      <div class="run-card-stats"><span class="run-step"></span><span class="run-bpb"></span></div>
      <div class="run-pace"></div>`;
    card.querySelector('.run-name').textContent = (p.name || p.id).slice(0, 26);
    card.addEventListener('click', () => onSelect(p.id));
    el.appendChild(card);
    updateRunCard(p.id);
  }
}

export function updateRunCard(podId) {
  const card = document.querySelector(`.run-card[data-pod-id="${CSS.escape(podId)}"]`);
  if (!card) return;
  const pod = app.pods.find((p) => p.id === podId) || {};
  const state = getPodState(podId);
  const running = pod.desiredStatus === 'RUNNING';
  const badge = card.querySelector('.badge');
  const paceEl = card.querySelector('.run-pace');

  const m = state.lastMetric;
  card.querySelector('.run-step').textContent =
    m ? `Step ${fmtInt(m.step)}/${fmtInt(m.total_steps)}` : (running ? 'Awaiting metrics' : 'No metrics yet');
  const lastEval = state.evals[state.evals.length - 1];
  card.querySelector('.run-bpb').textContent = lastEval ? `BPB ${lastEval.val_bpb.toFixed(4)}` : '';

  if (!running) {
    badge.textContent = 'Stopped';
    badge.className = 'badge badge-neutral';
    paceEl.textContent = '';
    paceEl.className = 'run-pace muted';
    return;
  }
  const pace = paceInfo(state);
  if (pace) {
    badge.textContent = pace.ahead ? 'Ahead' : 'Behind';
    badge.className = `badge ${pace.ahead ? 'badge-success' : 'badge-danger'}`;
    paceEl.textContent = fmtPace(pace);
    paceEl.className = `run-pace ${pace.ahead ? 'success' : 'danger'}`;
  } else {
    badge.textContent = 'Running';
    badge.className = 'badge badge-neutral';
    paceEl.textContent = 'No baseline selected';
    paceEl.className = 'run-pace muted';
  }
}

export function initRunList() {
  on('pod:metric', ({ podId }) => updateRunCard(podId));
  on('pod:hydrated', ({ podId }) => updateRunCard(podId));
  on('pod:phase', ({ podId }) => updateRunCard(podId));
}
