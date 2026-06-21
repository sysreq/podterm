// Keep Plotly charts sized to their containers whenever a panel resizes — not
// only on tab reveal (app.js) or boot→live transitions (live/boot.js), which
// remain as fallbacks for hidden→visible cases a ResizeObserver can miss.
// Debounced with requestAnimationFrame so a burst of layout changes triggers a
// single resize per frame.
const observed = new WeakSet();
let pending = new Set();
let raf = 0;

function flush() {
  raf = 0;
  const els = pending;
  pending = new Set();
  if (!window.Plotly) return;
  els.forEach((el) => {
    if (el.isConnected && el.data) Plotly.Plots.resize(el);
  });
}

// Observe a Plotly plot element (the div passed to Plotly.newPlot). Resizing the
// plot fills the element but doesn't change the element's own CSS-driven box, so
// this won't feedback-loop. No-op on repeat calls or where ResizeObserver is absent.
export function observePlotlyResize(el) {
  if (!el || observed.has(el) || typeof ResizeObserver === 'undefined') return;
  observed.add(el);
  const ro = new ResizeObserver(() => {
    pending.add(el);
    if (!raf) raf = requestAnimationFrame(flush);
  });
  ro.observe(el);
}
