import { app, getPodState } from '../state.js';

function hasInfo(state) {
  return Object.values(state.info || {}).some((v) => v != null && v !== '');
}

// These signals only exist after the container/event daemon is reachable. They
// prevent reconnects to already-running pods from getting stuck behind boot UI.
export function hasContainerEvidence(state) {
  return !!(
    state.boot?.done ||
    state.lastMetric ||
    state.phase ||
    state.finished ||
    state.memory ||
    state.telemetry ||
    state.summary ||
    state.diagnostic ||
    state.containerConnected ||
    hasInfo(state)
  );
}

export function isBooting(state, pod = {}) {
  if (state.streamUnavailable) return false;
  if (hasContainerEvidence(state)) return false;
  if (state.boot && !state.boot.done) return true;
  return pod.desiredStatus === 'RUNNING';
}

const LAYER_STATE_LABELS = {
  pending: 'Queued',
  waiting: 'Queued',
  downloading: 'Downloading',
  verifying: 'Verifying',
  downloaded: 'Downloaded',
  extracting: 'Extracting',
  complete: 'Done',
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

function provisionalBoot(pod) {
  return {
    stage: 'Starting container',
    image: pod.imageName || '',
    message: 'Waiting for machine logs…',
    layers: [],
    total: 0,
    complete: 0,
    pct: 0,
  };
}

export function updateBootPanel(podId) {
  const view = document.getElementById('live-view');
  const state = getPodState(podId);
  const pod = app.pods.find((p) => p.id === podId) || {};
  const booting = isBooting(state, pod);
  const was = view.classList.contains('booting');
  view.classList.toggle('booting', booting);
  if (booting) {
    renderBootPanel(state.boot || provisionalBoot(pod));
  } else if (was && window.Plotly) {
    // Charts laid out while hidden have zero width — resize on reveal.
    view.querySelectorAll('.js-plotly-plot').forEach((p) => Plotly.Plots.resize(p));
  }
  return booting;
}
