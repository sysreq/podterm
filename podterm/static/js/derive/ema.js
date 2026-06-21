// Spec-mandated EMA windows live here, not in views.
export const ETA_EMA_N = 50;   // Projected Finish uses EMA-50 of ms/step
export const AVG_EMA_N = 100;  // Avg ms/step card uses EMA-100

export function ema(values, n) {
  if (!values || !values.length) return null;
  const tail = values.slice(-n);
  const alpha = 2 / (tail.length + 1);
  let e = tail[0];
  for (let i = 1; i < tail.length; i++) e = alpha * tail[i] + (1 - alpha) * e;
  return e;
}

