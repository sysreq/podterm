// History tab: run table with filtering and compare-selection.
import { app } from './state.js';

export async function loadHistory() {
  try {
    const r = await fetch('/api/runs');
    app.historyRuns = await r.json();
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

  let html = '<table><thead><tr><th></th><th>Date</th><th>Branch</th><th>Commit</th><th>GPU</th><th>Variant</th><th>Steps</th><th>Best BPB</th><th>Cost</th><th>Status</th><th></th></tr></thead><tbody>';
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
      <td><input type="checkbox" ${checked} data-run-id="${r.run_id}"></td>
      <td>${date}</td><td>${r.branch || '-'}</td><td>${commit}</td><td>${r.gpu_type || '-'}</td>
      <td>${r.data_variant || '-'}</td><td>${r.total_steps || '-'}</td><td>${bpb}</td><td>${cost}</td><td>${status}</td>
      <td><a href="/api/runs/${r.run_id}/log" target="_blank" style="color:var(--accent);text-decoration:none;font-size:11px">log</a></td>
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
