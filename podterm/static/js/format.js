// Pure display formatters. No DOM.

export function fmtInt(n) {
  return n != null ? n.toLocaleString('en-US') : '?';
}

export function fmtMs(ms, digits = 1) {
  return ms != null ? ms.toFixed(digits) : '?';
}

// 95000 -> "1m 35s", 5_400_000 -> "1h 30m", 12000 -> "12s"
export function fmtDuration(ms) {
  if (ms == null) return '?';
  const s = Math.max(0, Math.round(ms / 1000));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${String(s % 60).padStart(2, '0')}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${String(m % 60).padStart(2, '0')}m`;
}

// Short seconds form for pace lines: "0.2s", "1.5m"
export function fmtSecShort(ms) {
  if (ms == null) return '?';
  const sec = Math.abs(ms) / 1000;
  return sec >= 60 ? `${(sec / 60).toFixed(1)}m` : `${sec.toFixed(1)}s`;
}

export function fmtClock(date) {
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

export function fmtMoney(x) {
  return x != null ? `$${x.toFixed(2)}` : '?';
}

// Signed delta: fmtDelta(-0.012, 4) -> "-0.0120", fmtDelta(3.1, 1) -> "+3.1"
export function fmtDelta(x, digits = 1) {
  if (x == null) return '?';
  return `${x > 0 ? '+' : ''}${x.toFixed(digits)}`;
}

export function fmtSteps(step, total) {
  return `${fmtInt(step)} / ${fmtInt(total)}`;
}
