// History tab: run table with filtering and compare-selection.
import { app } from './state.js';
import { escapeHtml } from './dom.js';
import { fetchJson } from './api.js';

export async function loadHistory() {
  try {
    app.historyRuns = await fetchJson('/api/runs');
    renderHistoryTable();
  } catch {}
}

export function renderHistoryTable() {
  const ft = (document.getElementById('history-filter').value || '').toLowerCase();
  let runs = app.historyRuns;
  if (ft) {
    runs = runs.filter((r) => ((r.branch || '') + (r.gpu_type || '') + (r.pod_name || '')).toLowerCase().includes(ft));
  }

  const area = document.getElementById('history-table-area');
  if (!runs.length) { area.innerHTML = '<div class="placeholder">No runs found.</div>'; return; }

  let html = '<table><thead><tr><th></th><th>Date</th><th>Branch</th><th>Commit</th><th>GPU</th><th>Variant</th><th>Steps</th><th>Best BPB</th><th>Cost</th><th>Status</th></tr></thead><tbody>';
  for (const r of runs) {
    const checked = app.selectedRuns.has(r.run_id) ? 'checked' : '';
    const date = (r.started_at || '').slice(0, 16).replace('T', ' ');
    const commit = (r.commit_hash || '-').slice(0, 7);
    const bpb = r.best_val_bpb != null ? r.best_val_bpb.toFixed(4) : '-';
    const cost = r.total_cost != null ? `$${r.total_cost.toFixed(3)}` : '-';
    let status = '-';
    if (r.exit_code == null) status = r.finished_at ? '?' : 'running';
    else if (r.exit_code === 0) status = 'done';
    else status = `exit ${r.exit_code}`;

    html += `<tr>
      <td><input type="checkbox" ${checked} data-run-id="${escapeHtml(r.run_id)}"></td>
      <td>${escapeHtml(date)}</td><td>${escapeHtml(r.branch || '-')}</td><td>${escapeHtml(commit)}</td><td>${escapeHtml(r.gpu_type || '-')}</td>
      <td>${escapeHtml(r.data_variant || '-')}</td><td>${escapeHtml(r.total_steps || '-')}</td><td>${escapeHtml(bpb)}</td><td>${escapeHtml(cost)}</td><td>${escapeHtml(status)}</td>
    </tr>`;
  }
  html += '</tbody></table>';
  area.innerHTML = html;
}

export function toggleRun(runId, checked) {
  if (checked) app.selectedRuns.add(runId); else app.selectedRuns.delete(runId);
  document.getElementById('selection-count').textContent = `${app.selectedRuns.size} selected`;
}

export function clearSelection() {
  app.selectedRuns.clear();
  document.getElementById('selection-count').textContent = '0 selected';
  renderHistoryTable();
}

export function initHistory() {
  document.getElementById('history-filter').addEventListener('input', renderHistoryTable);
  document.getElementById('history-table-area').addEventListener('change', (e) => {
    const cb = e.target.closest('input[type=checkbox][data-run-id]');
    if (cb) toggleRun(cb.dataset.runId, cb.checked);
  });
}
