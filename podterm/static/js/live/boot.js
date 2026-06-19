import { getPodState } from '../state.js';

// Any phase/metric implies the container booted (those only originate inside it),
// so reconnects to an already-running pod never get stuck in boot mode.
function isBooting(state) {
  return !!(state.boot && !state.boot.done && !state.lastMetric && !state.phase && !state.finished);
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

export function updateBootPanel(podId) {
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
