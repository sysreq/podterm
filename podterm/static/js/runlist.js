// Sidebar pod list.
import { app } from './state.js';

export function renderPodList(onSelect) {
  const el = document.getElementById('pod-list');
  el.innerHTML = '';
  for (const p of app.pods) {
    const btn = document.createElement('button');
    btn.className = 'pod-btn' + (p.id === app.activePod ? ' active' : '');
    const running = p.desiredStatus === 'RUNNING';
    btn.innerHTML = `<span class="pod-dot ${running ? 'dot-running' : 'dot-stopped'}"></span>${(p.name || p.id).slice(0, 22)}`;
    btn.addEventListener('click', () => onSelect(p.id));
    el.appendChild(btn);
  }
}
