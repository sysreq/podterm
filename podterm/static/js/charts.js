// Shared Plotly configuration + the Live Training Loss chart. Traces are
// created once at init with fixed indices and updated via restyle — no
// index juggling. (Throughput now lives in the race banner readout.)

export const COLORS = ['#ffd166', '#ff6b6b', '#48dbfb', '#81ecec', '#a29bfe', '#fd79a8'];

// Keep in sync with css/tokens.css (Plotly can't read CSS variables).
export const CHART_BG = '#111a33';
export const CHART_GRID = '#1f2a4d';
export const CHART_TEXT = '#8e9ab8';
const SER_CURRENT = '#4a9eff';
const SER_BASELINE = 'rgba(232,237,248,0.40)';
const SER_EVALS = '#a78bfa';

// Piecewise-linear y transform for loss: equal plot spacing between anchor
// ticks, finer anchors near the loss floor. Loss spans less than a decade,
// so a true log axis renders near-linear and unreadable — this keeps the
// late-run detail visible with clean, non-duplicated tick labels.
const ANCHORS = [2, 2.2, 2.4, 2.6, 2.8, 3, 3.25, 3.5, 4, 5, 8];

export function scaleY(v) {
  const n = ANCHORS.length - 1;
  if (v <= ANCHORS[0]) return 0;
  if (v >= ANCHORS[n]) return 1;
  let i = 0;
  while (v > ANCHORS[i + 1]) i++;
  return (i + (v - ANCHORS[i]) / (ANCHORS[i + 1] - ANCHORS[i])) / n;
}

export const lossTicks = ANCHORS;

export const lossYAxis = {
  gridcolor: CHART_GRID, zerolinecolor: CHART_GRID,
  tickvals: ANCHORS.map(scaleY),
  ticktext: ANCHORS.map((v) => (v % 1 === 0 ? String(v) : v.toFixed(2).replace(/0$/, ''))),
  range: [-0.02, 1.02],
};

export const lossHover = 'Step %{x}<br>Loss: %{customdata:.4f}<extra></extra>';

export const plotLayout = {
  paper_bgcolor: CHART_BG, plot_bgcolor: CHART_BG, font: { color: CHART_TEXT, size: 12 },
  margin: { l: 48, r: 20, t: 28, b: 36 }, legend: { orientation: 'h', y: -0.15 },
  xaxis: { gridcolor: CHART_GRID, zerolinecolor: CHART_GRID },
  yaxis: { gridcolor: CHART_GRID, zerolinecolor: CHART_GRID },
};

export const plotConfig = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d'] };

// ── Live loss chart ──
let lossEl = null;
let xDomainMax = null;

const liveLegend = { orientation: 'h', x: 1, xanchor: 'right', y: 1.12, font: { size: 11 }, bgcolor: 'transparent' };

export function initLiveCharts() {
  lossEl = document.getElementById('chart-loss');
  if (!lossEl || lossEl.data) return; // already initialized

  Plotly.newPlot(lossEl, [
    { x: [], y: [], customdata: [], name: 'current', type: 'scatter', mode: 'lines',
      line: { color: SER_CURRENT, width: 2 }, hovertemplate: lossHover },
    { x: [], y: [], customdata: [], name: 'baseline', type: 'scatter', mode: 'lines',
      line: { color: SER_BASELINE, width: 2, dash: 'dash' }, hovertemplate: lossHover },
    { x: [], y: [], customdata: [], name: 'evals', type: 'scatter', mode: 'markers',
      marker: { symbol: 'diamond', size: 7, color: SER_EVALS },
      hovertemplate: 'Eval @ step %{x}<br>val_bpb %{customdata:.4f}<extra></extra>' },
  ], {
    ...plotLayout, height: lossEl.clientHeight || 280,
    yaxis: { ...lossYAxis }, legend: liveLegend, hovermode: 'x unified',
  }, plotConfig);
}

export function chartsReady() {
  return Boolean(lossEl && lossEl.data);
}

// Redraw the loss chart from the pod's buffers (cheap restyle at metric cadence).
export function updateLiveCharts(state) {
  if (!chartsReady()) return;

  const pts = state.metricHistory.filter((m) => m.train_loss > 0);
  const evalPts = state.evals.filter((e) => e.val_loss != null);
  Plotly.restyle(lossEl, {
    x: [pts.map((m) => m.step), evalPts.map((e) => e.step)],
    y: [pts.map((m) => scaleY(m.train_loss)), evalPts.map((e) => scaleY(e.val_loss))],
    customdata: [pts.map((m) => m.train_loss), evalPts.map((e) => e.val_bpb)],
  }, [0, 2]);

  syncXDomain(state);
}

export function updateBaselineTrace(state) {
  if (!chartsReady()) return;
  Plotly.restyle(lossEl, {
    x: [state.baselineX], y: [state.baselineY], customdata: [state.baselineRaw],
  }, [1]);
  syncXDomain(state, true);
}

// X domain: [0, furthest step seen by the current or baseline series].
function syncXDomain(state, force = false) {
  const lastStep = state.lastMetric?.step ?? 0;
  const lastBaseline = state.baselineSteps.length ? state.baselineSteps[state.baselineSteps.length - 1] : 0;
  const max = Math.max(lastStep, lastBaseline);
  if (!max || (max === xDomainMax && !force)) return;
  xDomainMax = max;
  Plotly.relayout(lossEl, { 'xaxis.range': [0, max * 1.02] });
}
