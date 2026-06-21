export function latestSnapshot(hist) {
  if (!Array.isArray(hist)) return null;
  for (let i = hist.length - 1; i >= 0; i--) if (hist[i] && hist[i].diag) return hist[i];
  return null;
}

export function snapshotByStep(hist, step) {
  if (!Array.isArray(hist)) return null;
  return hist.find((h) => h && h.diag && h.step === step) || null;
}

export function previousSnapshot(hist, step) {
  if (!Array.isArray(hist)) return null;
  const valid = hist.filter((h) => h && h.diag);
  const i = valid.findIndex((h) => h.step === step);
  return i > 0 ? valid[i - 1] : null;
}

const GROUP_ORDER = ['capacity', 'flow', 'grad', 'arch'];
const GROUP_LABEL = { capacity: 'Capacity', flow: 'Flow', grad: 'Gradients', arch: 'Architecture' };

export function headlineOf(snapshot) {
  return (snapshot && snapshot.diag && snapshot.diag.health && snapshot.diag.health.headline) || [];
}

export function headlineByGroup(snapshot) {
  const out = {};
  const headline = headlineOf(snapshot);
  // Fixed group order first, then any unexpected groups appended so nothing silently vanishes.
  const extra = [...new Set(headline.map((m) => m.group))].filter((g) => !GROUP_ORDER.includes(g));
  for (const g of GROUP_ORDER.concat(extra)) {
    const metrics = headline.filter((m) => m.group === g);
    if (metrics.length) out[g] = metrics;
  }
  return out;
}

// warn/bad headline metrics (bad first), else the top-3 healthy by tier (tier 1 first).
export function topHealthIssues(snapshot) {
  const headline = headlineOf(snapshot);
  const issues = headline.filter((m) => m.band === 'bad')
    .concat(headline.filter((m) => m.band === 'warn'));
  if (issues.length) return issues;
  return headline
    .filter((m) => m.band === 'good')
    .sort((a, b) => (a.tier || 9) - (b.tier || 9))
    .slice(0, 3);
}

export const BAND_ORDER = ['good', 'warn', 'bad', 'info', 'na'];
const _worse = (a, b) => (BAND_ORDER.indexOf(a) >= BAND_ORDER.indexOf(b) ? a : b);

// Per-group: worst band, per-band counts, and up to 3 representative metric values.
export function groupSummary(snapshot) {
  const byGroup = headlineByGroup(snapshot);
  return Object.entries(byGroup).map(([group, metrics]) => {
    const counts = { good: 0, warn: 0, bad: 0, info: 0, na: 0 };
    let worst = 'good';
    for (const m of metrics) {
      const band = m.band || 'na';
      if (counts[band] != null) counts[band] += 1;
      if (band !== 'info' && band !== 'na') worst = _worse(worst, band);
    }
    // Prefer surfacing problem metrics, then by tier.
    const values = [...metrics]
      .sort((a, b) => (BAND_ORDER.indexOf(b.band) - BAND_ORDER.indexOf(a.band)) || (a.tier || 9) - (b.tier || 9))
      .slice(0, 3);
    return { group, label: GROUP_LABEL[group] || group, worst, counts, values, metrics };
  });
}

// The reference diag doc for a diff mode (or null for 'absolute' / when unavailable).
export function referenceDoc(hist, step, diffMode, baselineHist) {
  if (diffMode === 'absolute') return null;
  if (diffMode === 'baseline') return latestSnapshot(baselineHist)?.diag || null;
  return previousSnapshot(hist, step)?.diag || null; // 'previous'
}

