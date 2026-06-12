// Config / Repro panel. Every value comes from run metadata (info events,
// the /api/runs row, or the live pod object) — fields the launch pipeline
// doesn't record yet render as explicit pending text.
import { app, getPodState, on } from './state.js';

function parseConfig(state) {
  try {
    return state.runRow?.config_json ? JSON.parse(state.runRow.config_json) : null;
  } catch {
    return null;
  }
}

// eventd bearer token is persisted in config_json — never show it.
function redacted(cfg) {
  const out = {};
  for (const [k, v] of Object.entries(cfg)) {
    out[k] = /token|secret|password/i.test(k) ? '••• redacted •••' : v;
  }
  return out;
}

function row(label, value, { copy = null, pending = false } = {}) {
  const div = document.createElement('div');
  div.className = 'cfg-row';
  const lab = document.createElement('span');
  lab.className = 'cfg-label';
  lab.textContent = label;
  const val = document.createElement('span');
  val.className = 'cfg-value' + (pending ? ' pending' : '');
  val.textContent = value;
  val.title = value;
  div.appendChild(lab);
  div.appendChild(val);
  if (copy) {
    const btn = document.createElement('button');
    btn.className = 'cfg-copy';
    btn.title = `Copy ${label.toLowerCase()}`;
    btn.textContent = '⧉';
    btn.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(copy);
        btn.textContent = '✓';
        setTimeout(() => { btn.textContent = '⧉'; }, 1200);
      } catch {
        btn.textContent = '✗';
        setTimeout(() => { btn.textContent = '⧉'; }, 1200);
      }
    });
    div.appendChild(btn);
  }
  return div;
}

export function renderConfigPanel(podId) {
  const rowsEl = document.getElementById('config-rows');
  if (!rowsEl) return;
  const state = getPodState(podId);
  const pod = app.pods.find((p) => p.id === podId) || {};
  const run = state.runRow || {};
  const cfg = parseConfig(state) || {};
  rowsEl.textContent = '';

  const commit = state.info.commit_hash || run.commit_hash || null;
  const branch = run.branch || cfg.branch || null;
  if (commit) {
    rowsEl.appendChild(row('Git commit', `${commit.slice(0, 7)}${branch ? ` · ${branch}` : ''}`, { copy: commit }));
  } else if (branch) {
    rowsEl.appendChild(row('Git commit', `${branch} — hash arrives at boot`, { pending: true }));
  } else {
    rowsEl.appendChild(row('Git commit', 'Awaiting boot info', { pending: true }));
  }

  const gpu = state.info.gpu_type || run.gpu_type || null;
  const gpuCount = state.info.gpu_count || run.gpu_count || 1;
  rowsEl.appendChild(gpu
    ? row('GPU', gpuCount > 1 ? `${gpu} ×${gpuCount}` : gpu)
    : row('GPU', 'Awaiting device query', { pending: true }));

  rowsEl.appendChild(pod.imageName
    ? row('Docker image', pod.imageName, { copy: pod.imageName })
    : row('Docker image', 'Not recorded for stopped pods', { pending: true }));

  rowsEl.appendChild(row('Config hash', 'Not computed at launch', { pending: true }));

  rowsEl.appendChild(cfg.seed != null
    ? row('Seed', String(cfg.seed))
    : row('Seed', 'Not in launch config', { pending: true }));
  rowsEl.appendChild(cfg.seq_len != null
    ? row('Sequence length', String(cfg.seq_len))
    : row('Sequence length', 'Not in launch config', { pending: true }));
  rowsEl.appendChild(cfg.batch_size != null
    ? row('Batch size', String(cfg.batch_size))
    : row('Batch size', 'Not in launch config', { pending: true }));

  // Real metadata the pipeline does record — keeps the panel useful today.
  if (run.data_variant || cfg.data_variant) rowsEl.appendChild(row('Data variant', run.data_variant || cfg.data_variant));
  if (run.model_params) rowsEl.appendChild(row('Model params', Number(run.model_params).toLocaleString('en-US')));
}

function openConfigDialog() {
  if (!app.activePod) return;
  const state = getPodState(app.activePod);
  const cfg = parseConfig(state);
  document.getElementById('config-json').textContent = cfg
    ? JSON.stringify(redacted(cfg), null, 2)
    : 'No launch config recorded for this run.';
  document.getElementById('config-dialog').showModal();
}

export function initConfigPanel() {
  document.getElementById('btn-full-config').addEventListener('click', openConfigDialog);
  document.getElementById('btn-config-close').addEventListener('click', () => document.getElementById('config-dialog').close());
  on('pod:info', ({ podId }) => {
    if (app.activePod === podId) renderConfigPanel(podId);
  });
}
