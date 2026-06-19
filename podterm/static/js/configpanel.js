// Config / Repro panel. Every value comes from run metadata (info events,
// the /api/runs row, or the live pod object) — fields the launch pipeline
// doesn't record yet render as explicit pending text.
import { app, getPodState, hydrateRunRow, on } from './state.js';

function parseConfig(state) {
  try {
    return state.runRow?.config_json ? JSON.parse(state.runRow.config_json) : null;
  } catch {
    return null;
  }
}

function valueAt(obj, path) {
  return path.split('.').reduce((cur, key) => (cur && typeof cur === 'object' ? cur[key] : undefined), obj);
}

function firstPresent(...values) {
  return values.find((v) => v !== undefined && v !== null && v !== '');
}

function configValue(cfg, ...paths) {
  return firstPresent(...paths.map((path) => valueAt(cfg, path)));
}

function formatTokens(value) {
  const num = Number(value);
  return Number.isFinite(num) ? `${num.toLocaleString('en-US')} tok/step` : String(value);
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
  if (!state.runRow && !state.runRowHydrated) hydrateRunRow(podId);
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

  const driver = state.info.driver_version || run.driver_version || null;
  const cuda = state.info.cuda_version || run.cuda_version || null;
  rowsEl.appendChild(driver
    ? row('Driver version', driver)
    : row('Driver version', 'Awaiting device query', { pending: true }));
  rowsEl.appendChild(cuda
    ? row('CUDA version', cuda)
    : row('CUDA version', 'Awaiting device query', { pending: true }));

  rowsEl.appendChild(pod.imageName
    ? row('Docker image', pod.imageName, { copy: pod.imageName })
    : row('Docker image', 'Not recorded for stopped pods', { pending: true }));

  // Run config emitted by the pod at training start (config event), persisted on the run row.
  const seed = firstPresent(
    state.info.seed,
    run.seed,
    configValue(cfg, 'seed', 'train.seed', 'training.seed', 'io.seed'),
  );
  const seqLen = firstPresent(
    state.info.seq_len,
    run.seq_len,
    configValue(cfg, 'seq_len', 'sequence_length', 'train.seq_len', 'train.sequence_length', 'training.seq_len', 'training.sequence_length'),
  );
  const batchTokens = firstPresent(
    state.info.batch_tokens,
    run.batch_tokens,
    configValue(cfg, 'batch_tokens', 'batch_size', 'batch', 'train.batch_tokens', 'train.batch_size', 'train.batch', 'training.batch_tokens', 'training.batch_size', 'training.batch'),
  );
  rowsEl.appendChild(seed != null
    ? row('Seed', String(seed))
    : row('Seed', 'Awaiting boot info', { pending: true }));
  rowsEl.appendChild(seqLen != null
    ? row('Sequence length', String(seqLen))
    : row('Sequence length', 'Awaiting boot info', { pending: true }));
  rowsEl.appendChild(batchTokens != null
    ? row('Batch', formatTokens(batchTokens))
    : row('Batch', 'Awaiting boot info', { pending: true }));

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
  on('pod:run-row', ({ podId }) => {
    if (app.activePod === podId) renderConfigPanel(podId);
  });
}
