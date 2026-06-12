// Launch and log-import dialogs.
import { app, emit } from './state.js';

function setStatus(text) {
  emit('status', { text });
}

export async function openLaunchDialog() {
  const dlg = document.getElementById('launch-dialog');
  // Fetch form data in parallel
  const [branches, dcs, variants, lastCfg, redis] = await Promise.all([
    fetch('/api/branches').then((r) => r.json()),
    fetch('/api/datacenters').then((r) => r.json()),
    fetch('/api/variants').then((r) => r.json()),
    fetch('/api/last-config').then((r) => r.json()),
    fetch('/api/redis-server').then((r) => r.json()),
  ]);
  app.variantLookup = variants.lookup || {};

  fillSelect('f-branch', branches.map((b) => ({ label: b, id: b })), lastCfg.branch);
  fillSelect('f-dc', dcs.map((dc) => ({ label: `${dc.id} (${dc.location || ''})`, id: dc.id })), lastCfg.datacenter);
  fillSelect('f-variant', variants.options || [], lastCfg.data_variant);

  // Load GPUs for selected DC
  await onDcChange();

  // Populate advanced fields from last config
  if (lastCfg.name) document.getElementById('f-name').value = lastCfg.name;
  if (lastCfg.gpu_count) document.getElementById('f-gpu-count').value = lastCfg.gpu_count;
  if (lastCfg.train_script) document.getElementById('f-train-script').value = lastCfg.train_script;
  if (lastCfg.profile_steps) document.getElementById('f-profile').value = lastCfg.profile_steps;
  if (lastCfg.compile_debug) document.getElementById('f-compile-debug').checked = lastCfg.compile_debug;
  if (lastCfg.graph_logs) document.getElementById('f-graph-logs').checked = lastCfg.graph_logs;
  if (lastCfg.time_budget) document.getElementById('f-time').value = lastCfg.time_budget;
  if (lastCfg.data_repo_id) document.getElementById('f-data-repo').value = lastCfg.data_repo_id;
  if (lastCfg.data_version) document.getElementById('f-data-version').value = lastCfg.data_version;
  document.getElementById('f-redis').value = lastCfg.redis_cache_server || redis.address || '';

  dlg.showModal();
}

function fillSelect(id, options, defaultVal) {
  const sel = document.getElementById(id);
  sel.innerHTML = '';
  for (const o of options) {
    const opt = document.createElement('option');
    opt.value = o.id;
    opt.textContent = o.label;
    if (o.id === defaultVal) opt.selected = true;
    sel.appendChild(opt);
  }
}

async function onDcChange() {
  const dc = document.getElementById('f-dc').value;
  const gpus = await fetch(`/api/gpus/${dc}`).then((r) => r.json());
  fillSelect('f-gpu', gpus, null);
}

async function doLaunch() {
  const variant = document.getElementById('f-variant').value;
  const vinfo = app.variantLookup[variant] || {};
  const cfg = {
    branch: document.getElementById('f-branch').value,
    name: document.getElementById('f-name').value || null,
    datacenter: document.getElementById('f-dc').value,
    gpu: document.getElementById('f-gpu').value,
    gpu_count: parseInt(document.getElementById('f-gpu-count').value) || 1,
    train_script: document.getElementById('f-train-script').value || 'train_gpt.py',
    profile_steps: parseInt(document.getElementById('f-profile').value) || 0,
    compile_debug: document.getElementById('f-compile-debug').checked,
    graph_logs: document.getElementById('f-graph-logs').checked,
    time_budget: parseInt(document.getElementById('f-time').value) || 600,
    prep_shards: parseInt(document.getElementById('f-shards').value) || 20,
    data_repo_id: document.getElementById('f-data-repo').value || 'sysrekt/parameter-golf',
    data_version: document.getElementById('f-data-version').value || 'main',
    redis_cache_server: document.getElementById('f-redis').value || '',
    data_variant: vinfo.data_variant || variant,
    data_path: vinfo.data_path || '',
    tokenizer_path: vinfo.tokenizer_path || '',
    vocab_size: vinfo.vocab_size || '',
  };

  document.getElementById('launch-dialog').close();
  setStatus('Launching...');

  try {
    const r = await fetch('/api/pods/launch', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(cfg),
    });
    const result = await r.json();
    setStatus(`Run launched: ${result.name} ($${result.cost_per_hr}/hr)`);
    emit('pods:refresh', {});
    emit('pod:select', { podId: result.pod_id });
  } catch (e) {
    setStatus(`Launch failed: ${e}`);
  }
}

// ── Import Logs ──
export async function openImportDialog() {
  const dlg = document.getElementById('import-dialog');
  const list = document.getElementById('import-list');
  list.innerHTML = '<div class="placeholder">Scanning...</div>';
  dlg.showModal();

  const files = await fetch('/api/logs').then((r) => r.json());
  if (!files.length) {
    list.innerHTML = '<div class="placeholder">No log files found.</div>';
    return;
  }

  let html = '<table><thead><tr><th>File</th><th>Size</th><th>Status</th><th></th></tr></thead><tbody>';
  for (const f of files) {
    const size = (f.size / 1024).toFixed(1) + ' KB';
    const status = f.imported ? '<span style="color:var(--success)">imported</span>' : '<span style="color:var(--text-muted)">new</span>';
    const btn = f.imported ? '' : `<button class="btn-import-one" style="padding:4px 12px;font-size:11px">Import</button>`;
    html += `<tr data-path="${encodeURIComponent(f.path)}"><td style="font-size:11px">${f.name}</td><td>${size}</td><td>${status}</td><td>${btn}</td></tr>`;
  }
  html += '</tbody></table>';
  list.innerHTML = html;
}

async function importSingleLog(path, btn) {
  btn.disabled = true;
  btn.textContent = '...';
  try {
    const r = await fetch('/api/logs/import', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path }),
    });
    const result = await r.json();
    if (result.error) { btn.textContent = 'Error'; return; }
    btn.parentElement.previousElementSibling.innerHTML = '<span style="color:var(--success)">imported</span>';
    btn.textContent = `${result.metrics} metrics`;
    btn.style.border = '1px solid var(--success)';
    emit('history:reload', {});
  } catch { btn.textContent = 'Failed'; }
}

async function importAllLogs(btn) {
  btn.disabled = true;
  btn.textContent = 'Importing...';
  try {
    const r = await fetch('/api/logs/import-all', { method: 'POST' });
    const result = await r.json();
    btn.textContent = `Imported ${result.imported} logs`;
    btn.style.border = '1px solid var(--success)';
    openImportDialog(); // refresh the list
    emit('history:reload', {});
  } catch { btn.textContent = 'Failed'; }
}

export function initLaunch() {
  document.getElementById('f-dc').addEventListener('change', onDcChange);
  document.getElementById('btn-do-launch').addEventListener('click', doLaunch);
  document.getElementById('btn-launch-cancel').addEventListener('click', () => document.getElementById('launch-dialog').close());
  document.getElementById('btn-import-all').addEventListener('click', (e) => importAllLogs(e.currentTarget));
  document.getElementById('btn-import-close').addEventListener('click', () => document.getElementById('import-dialog').close());
  document.getElementById('import-list').addEventListener('click', (e) => {
    const btn = e.target.closest('button.btn-import-one');
    if (!btn) return;
    const row = btn.closest('tr[data-path]');
    if (row) importSingleLog(decodeURIComponent(row.dataset.path), btn);
  });
}
