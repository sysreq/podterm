// Entry module: init, tab routing, pod polling, action wiring.
import { app, on, dropPodState, hydrateFromDb } from './state.js';
import { reconcileStreams, markPodActive } from './state/stream.js';
import { renderRunList, renderRecentRuns, initRunList } from './runlist.js';
import { renderLiveView, initLive } from './live.js';
import { loadHistory, clearSelection, initHistory } from './history.js';
import { compareSelected, initCompare } from './compare.js';
import { openLaunchDialog, initLaunch } from './launch.js';
import { initLogs } from './logs.js';
import { initConfigPanel } from './configpanel.js';
import { initDiagnostics, openHealthTab } from './diagnostics.js';
import { escapeHtml } from './dom.js';
import { fetchJson } from './api.js';

function setStatus(text) {
  document.getElementById('status').textContent = text;
}

function setStatusSub(text) {
  document.getElementById('status-sub').textContent = text;
}

function setClusterPill(ok, text) {
  const pill = document.getElementById('cluster-pill');
  pill.classList.toggle('down', !ok);
  document.getElementById('cluster-pill-text').textContent = text;
}

function setCachePill(ok, text, title) {
  const pill = document.getElementById('cache-pill');
  pill.classList.toggle('down', !ok);
  pill.title = title || 'Shared torch.compile cache (Redis pod)';
  document.getElementById('cache-pill-text').textContent = text;
}

// Compile cache = the shared torch.compile Redis pod (detect_redis_server). Polled independently
// of /api/pods so neither pill's failure affects the other.
async function refreshCompileCache() {
  try {
    const { address } = await fetchJson('/api/redis-server');
    if (address) setCachePill(true, 'Compile Cache', `Compile cache: ${address}`);
    else setCachePill(false, 'Compile Cache — none', 'No Redis compile-cache pod running');
  } catch {
    setCachePill(false, 'Compile Cache — offline', 'Failed to query compile-cache status');
  }
}

// ── Tabs ──
function switchTab(tab) {
  document.querySelectorAll('.tab').forEach((t) => {
    const active = t.dataset.tab === tab;
    t.classList.toggle('active', active);
    t.setAttribute('aria-selected', active ? 'true' : 'false');
    // Roving tabindex: only the selected tab is in the tab order; arrow keys move between the rest.
    t.tabIndex = active ? 0 : -1;
  });
  document.querySelectorAll('.tab-content').forEach((c) => c.classList.toggle('active', c.id === 'tab-' + tab));
  if (tab === 'history') loadHistory();
  if (tab === 'machines') renderMachines();
  if (tab === 'health') openHealthTab(app.activePod);
  // Plotly renders 0-width inside display:none containers — resize on reveal.
  const shown = document.getElementById('tab-' + tab);
  if (shown && window.Plotly) {
    shown.querySelectorAll('.js-plotly-plot').forEach((p) => Plotly.Plots.resize(p));
  }
}

// ── Machines (stub view) ──
function renderMachines() {
  const area = document.getElementById('machines-table-area');
  if (!app.pods.length) {
    area.innerHTML = '<div class="placeholder">No pods. Launch Run to create one.</div>';
    return;
  }
  let html = '<table><thead><tr><th>Name</th><th>ID</th><th>Status</th><th>$/hr</th><th>Image</th></tr></thead><tbody>';
  for (const p of app.pods) {
    html += `<tr><td>${escapeHtml(p.name || '?')}</td><td>${escapeHtml(p.id)}</td><td>${escapeHtml(p.desiredStatus || '?')}</td>
      <td>${escapeHtml(p.costPerHr != null ? '$' + p.costPerHr : '?')}</td><td style="font-size:11px">${escapeHtml(p.imageName || '?')}</td></tr>`;
  }
  area.innerHTML = html + '</tbody></table>';
}

// ── Pods ──
// Reconcile EventSource streams against the current pod list + active selection
// (the stream manager reserves a slot for the active pod and LRU-evicts the rest),
// then hydrate any freshly (re)opened streams from the DB. Force the hydrate so a
// stream that was evicted and re-admitted backfills the metrics/logs that landed
// while it was closed (a no-op extra fetch on a stream's very first open).
function syncStreams() {
  for (const podId of reconcileStreams(app.pods, app.activePod)) {
    hydrateFromDb(podId, true);
  }
}

async function refreshPods() {
  try {
    app.pods = await fetchJson('/api/pods');
    renderRunList(selectPod);
    syncStreams();
    const running = app.pods.filter((p) => p.desiredStatus === 'RUNNING').length;
    const queued = app.pods.length - running;
    setStatus(`${app.pods.length} pods, ${queued} queued`);
    setStatusSub('All systems nominal');
    setClusterPill(true, 'RunPod');
    if (document.getElementById('tab-machines').classList.contains('active')) renderMachines();
  } catch {
    setStatus('Cluster unreachable');
    setStatusSub('Check runpodctl and network, then Refresh');
    setClusterPill(false, 'RunPod — offline');
  }
}

// Pull the historic run list (for the sidebar RECENT RUNS) alongside the pods.
async function refreshRecentRuns() {
  try { app.historyRuns = await fetchJson('/api/runs'); } catch { return; }
  renderRecentRuns(selectPod);
}

function selectPod(podId) {
  app.activePod = podId;
  markPodActive(podId); // protect it from LRU eviction
  syncStreams();        // ensure the active pod has a stream, evicting a background one if at cap
  renderRunList(selectPod);
  renderRecentRuns(selectPod);
  // Keep the user where they are when inspecting a run's Model Health; otherwise
  // the active-run flow lands on Live as before. Picking a past run from the
  // Health tab loads its DB-backed diagnostics without yanking to Live.
  if (document.getElementById('tab-health').classList.contains('active')) {
    switchTab('health');
  } else {
    switchTab('live');
    renderLiveView(podId);
  }
}

async function stopPod() {
  if (!app.activePod) return;
  if (!confirm(`Stop run on pod ${app.activePod}? The pod will be terminated.`)) return;
  setStatus(`Stopping run on ${app.activePod}…`);
  try {
    await fetchJson(`/api/pods/${app.activePod}/stop`, { method: 'POST' });
    setStatus(`Run stopped (${app.activePod})`);
    dropPodState(app.activePod);
    app.activePod = null;
    refreshPods();
  } catch (e) {
    setStatus(`Stop failed: ${e}`);
  }
}

// ── Init ──
function init() {
  const tabs = [...document.querySelectorAll('.tab')];
  tabs.forEach((t) => {
    t.addEventListener('click', () => switchTab(t.dataset.tab));
  });
  // Arrow-key navigation across the tablist (left/right + home/end), per WAI-ARIA.
  const tabBar = document.getElementById('tab-bar');
  if (tabBar) {
    tabBar.addEventListener('keydown', (e) => {
      const i = tabs.indexOf(document.activeElement);
      if (i === -1) return;
      let next = -1;
      if (e.key === 'ArrowRight') next = (i + 1) % tabs.length;
      else if (e.key === 'ArrowLeft') next = (i - 1 + tabs.length) % tabs.length;
      else if (e.key === 'Home') next = 0;
      else if (e.key === 'End') next = tabs.length - 1;
      if (next === -1) return;
      e.preventDefault();
      switchTab(tabs[next].dataset.tab);
      tabs[next].focus();
    });
  }
  document.getElementById('btn-launch').addEventListener('click', openLaunchDialog);
  document.getElementById('btn-stop').addEventListener('click', stopPod);
  document.getElementById('btn-refresh').addEventListener('click', refreshPods);
  document.getElementById('btn-compare-selected').addEventListener('click', compareSelected);
  document.getElementById('btn-clear-selection').addEventListener('click', clearSelection);

  on('status', ({ text }) => setStatus(text));
  on('pods:refresh', refreshPods);
  on('pod:select', ({ podId }) => selectPod(podId));
  on('tab:switch', ({ tab }) => switchTab(tab));
  on('history:reload', loadHistory);

  initRunList();
  initLive();
  initLogs();
  initConfigPanel();
  initDiagnostics();
  initHistory();
  initCompare();
  initLaunch();

  refreshPods();
  refreshRecentRuns();
  refreshCompileCache();
  setInterval(() => { refreshPods(); refreshRecentRuns(); refreshCompileCache(); }, 15000);
}

init();
