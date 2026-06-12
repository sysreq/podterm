// Shared Plotly configuration and the custom loss-axis transform.

export const COLORS = ['#ffd166', '#ff6b6b', '#48dbfb', '#81ecec', '#a29bfe', '#fd79a8'];

// Custom loss scale — bottom 50% = 3.0-3.5 (fine detail), then compressed above.
// Anything above 8 is clamped.
export function scaleY(loss) {
  if (loss >= 8)   return 1;
  if (loss >= 5)   return 0.9 + (loss - 5) / 3 * 0.1;
  if (loss >= 4)   return 0.8 + (loss - 4) * 0.1;
  if (loss >= 3.5) return 0.5 + (loss - 3.5) * 0.6;
  return loss - 3;
}

export const lossTicks = [3, 3.1, 3.2, 3.3, 3.4, 3.5, 4, 5, 8];

export const lossYAxis = {
  gridcolor: '#2a2a4a', zerolinecolor: '#2a2a4a',
  tickvals: lossTicks.map(scaleY),
  ticktext: lossTicks.map(v => v % 1 === 0 ? String(v) : v.toFixed(1)),
  range: [-0.02, 1.05],
};

export const lossHover = 'Step %{x}<br>Loss: %{customdata:.4f}<extra></extra>';

export const plotLayout = {
  paper_bgcolor: '#16213e', plot_bgcolor: '#16213e', font: { color: '#aaa', size: 11 },
  margin: { l: 50, r: 20, t: 30, b: 35 }, legend: { orientation: 'h', y: -0.15 },
  xaxis: { gridcolor: '#2a2a4a', zerolinecolor: '#2a2a4a' },
  yaxis: { gridcolor: '#2a2a4a', zerolinecolor: '#2a2a4a' },
};

export const plotConfig = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d'] };
