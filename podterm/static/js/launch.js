// Launch dialog.
import { app, emit } from './state.js';
import { fetchJson } from './api.js';

function setStatus(text) {
  emit('status', { text });
}

export async function openLaunchDialog() {
  const dlg = document.getElementById('launch-dialog');
  let branches, dcs, variants, lastCfg, redis;
  try {
    // Fetch form data in parallel
    [branches, dcs, variants, lastCfg, redis] = await Promise.all([
      fetchJson('/api/branches'),
      fetchJson('/api/datacenters'),
      fetchJson('/api/variants'),
      fetchJson('/api/last-config'),
      fetchJson('/api/redis-server'),
    ]);
  } catch (e) {
    setStatus(`Launch metadata failed: ${e.message || e}`);
    return;
  }
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
  const gpus = await fetchJson(`/api/gpus/${dc}`);
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
    const result = await fetchJson('/api/pods/launch', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(cfg),
    });
    setStatus(`Run launched: ${result.name} ($${result.cost_per_hr}/hr)`);
    emit('pods:refresh', {});
    emit('pod:select', { podId: result.pod_id });
  } catch (e) {
    setStatus(`Launch failed: ${e.message || e}`);
  }
}

export function initLaunch() {
  document.getElementById('f-dc').addEventListener('change', onDcChange);
  document.getElementById('btn-do-launch').addEventListener('click', doLaunch);
  document.getElementById('btn-launch-cancel').addEventListener('click', () => document.getElementById('launch-dialog').close());
}
