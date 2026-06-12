// Logs panel: level tabs, free-text filter, pause/auto-pause autoscroll,
// level coloring, latest-step-line highlight. Renders from the same
// state.logLines buffer the SSE stream feeds — single source of truth.
import { app, getPodState, on } from './state.js';

const MAX_RENDERED = 500;

let level = 'all';
let filter = '';
let paused = false;
let newWhilePaused = 0;
let latestStepEl = null;

let linesEl, pauseBtn, resumeBtn, newCountEl;

function matches(l) {
  if (level !== 'all' && l.level !== level) return false;
  if (filter && !l.text.toLowerCase().includes(filter)) return false;
  return true;
}

function lineEl(l) {
  const div = document.createElement('div');
  div.className = `log-line lvl-${l.level}`;
  div.textContent = l.text;
  if (l.step != null) {
    if (latestStepEl) latestStepEl.classList.remove('log-latest');
    div.classList.add('log-latest');
    latestStepEl = div;
  }
  return div;
}

function scrollToBottom() {
  linesEl.scrollTop = linesEl.scrollHeight;
}

function setPaused(p, manual = false) {
  paused = p;
  pauseBtn.textContent = paused ? 'Resume' : 'Pause';
  pauseBtn.classList.toggle('paused', paused);
  if (!paused) {
    newWhilePaused = 0;
    resumeBtn.hidden = true;
    scrollToBottom();
  }
  if (manual && paused) {
    // manual pause: keep view where it is
  }
}

export function renderLogs(podId) {
  const state = getPodState(podId);
  latestStepEl = null;
  linesEl.textContent = '';
  const visible = state.logLines.filter(matches).slice(-MAX_RENDERED);
  if (!visible.length) {
    const empty = document.createElement('div');
    empty.className = 'log-empty';
    empty.textContent = state.logLines.length
      ? 'No lines match the current tab/filter'
      : 'No log output yet — lines stream in once the pod boots';
    linesEl.appendChild(empty);
    return;
  }
  const frag = document.createDocumentFragment();
  for (const l of visible) frag.appendChild(lineEl(l));
  linesEl.appendChild(frag);
  if (!paused) scrollToBottom();
}

function appendLatest(podId) {
  const state = getPodState(podId);
  const l = state.logLines[state.logLines.length - 1];
  if (!l) return;
  const empty = linesEl.querySelector('.log-empty');
  if (empty) empty.remove();
  if (!matches(l)) return;
  linesEl.appendChild(lineEl(l));
  while (linesEl.childElementCount > MAX_RENDERED) linesEl.firstElementChild.remove();
  if (paused) {
    newWhilePaused++;
    newCountEl.textContent = String(newWhilePaused);
    resumeBtn.hidden = false;
  } else {
    scrollToBottom();
  }
}

export function initLogs() {
  linesEl = document.getElementById('log-lines');
  pauseBtn = document.getElementById('log-pause');
  resumeBtn = document.getElementById('log-resume');
  newCountEl = document.getElementById('log-new-count');

  document.querySelectorAll('.log-tab').forEach((t) => {
    t.addEventListener('click', () => {
      document.querySelectorAll('.log-tab').forEach((x) => x.classList.toggle('active', x === t));
      level = t.dataset.level;
      if (app.activePod) renderLogs(app.activePod);
    });
  });

  document.getElementById('log-filter').addEventListener('input', (e) => {
    filter = e.target.value.trim().toLowerCase();
    if (app.activePod) renderLogs(app.activePod);
  });

  pauseBtn.addEventListener('click', () => setPaused(!paused, true));
  resumeBtn.addEventListener('click', () => setPaused(false));

  // Auto-pause when the user scrolls up to read; resume chip counts new lines.
  linesEl.addEventListener('scroll', () => {
    const atBottom = linesEl.scrollHeight - linesEl.scrollTop - linesEl.clientHeight < 40;
    if (!atBottom && !paused) setPaused(true);
  });

  on('pod:log', ({ podId }) => {
    if (app.activePod !== podId) return;
    appendLatest(podId);
  });
}
