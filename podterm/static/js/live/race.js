import { fmtDelta, fmtDuration, fmtInt, fmtMs, fmtSecShort, fmtTps } from '../format.js';

// Throughput readout shared by the banner sub-line and the hero card.
// "523k tok/s · 192.6 ms/step (+4.1 vs base)" — falls back to ms/step when no token budget.
function fmtThroughput(race) {
  const parts = [];
  if (race.throughputTps != null) parts.push(`${fmtTps(race.throughputTps)} tok/s`);
  if (race.msPerStep != null) {
    let s = `${fmtMs(race.msPerStep)} ms/step`;
    if (race.baselineMsPerStep != null) s += ` (${fmtDelta(race.msPerStep - race.baselineMsPerStep, 1)} vs base)`;
    parts.push(s);
  }
  return parts.join(' · ');
}

// ── Race banner + baseline row (numbers shared with the KPI cards by construction) ──
export function updateRaceBanner(state, m, race, eta) {
  const banner = document.getElementById('race-banner');
  const headline = document.getElementById('race-headline');
  const sub = document.getElementById('race-sub');
  const fill = banner.querySelector('.race-bar-fill');
  const markP = banner.querySelector('.race-marker.projected');
  const markB = banner.querySelector('.race-marker.baseline');
  const labP = document.getElementById('race-label-projected');
  const labB = document.getElementById('race-label-baseline');

  banner.classList.remove('ahead', 'behind');
  markP.style.display = 'none';
  markB.style.display = 'none';
  labP.textContent = '';
  labB.textContent = '';

  if (!m) {
    headline.textContent = state.finished ? 'Run finished' : 'Waiting for metrics…';
    sub.textContent = state.finished
      ? 'No step metrics were recorded for this run'
      : 'The race banner lights up once steps start streaming';
    fill.style.width = '0%';
    return;
  }

  const stepFrac = m.total_steps ? m.step / m.total_steps : 0;
  const throughput = race ? fmtThroughput(race) : '';

  if (!race || (race.state !== 'ahead' && race.state !== 'behind')) {
    headline.textContent = 'No baseline selected';
    sub.textContent = throughput || 'Choose a baseline below to start the race';
    fill.style.width = `${(stepFrac * 100).toFixed(1)}%`;
    if (eta != null && m.train_time_ms != null) {
      labP.textContent = `Projected total ${fmtDuration(m.train_time_ms + eta)}`;
    }
    return;
  }

  const ahead = race.state === 'ahead';
  banner.classList.add(ahead ? 'ahead' : 'behind');
  const lossTxt = race.currentLoss != null ? race.currentLoss.toFixed(3) : '?';
  headline.textContent = `You're ${ahead ? 'ahead of' : 'behind'} baseline by ${fmtSecShort(race.leadMs)} at ${lossTxt} loss`;

  const subParts = [];
  if (throughput) subParts.push(throughput);
  if (state.finished) {
    subParts.push(`final margin at step ${fmtInt(m.step)}`);
  } else if (race.projectedMarginMs != null && race.targetLoss != null) {
    // Projected margin reaching the baseline's final loss — distinct from the current lead.
    const pm = race.projectedMarginMs;
    subParts.push(`projected to reach ${race.targetLoss.toFixed(3)} ~${fmtSecShort(pm)} ${pm >= 0 ? 'ahead' : 'behind'}`);
  }
  sub.textContent = subParts.join(' · ');

  // Progress bar on the time-to-target domain: where elapsed sits between
  // [0, max(my projected target time, baseline's target time)].
  const projTarget = race.projectedTargetTimeMs;
  const baseTotal = state.baselineTotalTimeMs;
  const domain = Math.max(projTarget ?? 0, baseTotal ?? 0, m.train_time_ms ?? 0);
  if (domain > 0) {
    fill.style.width = `${Math.min(100, ((m.train_time_ms ?? 0) / domain) * 100).toFixed(1)}%`;
    if (projTarget != null) {
      markP.style.left = `${((projTarget / domain) * 100).toFixed(2)}%`;
      markP.style.display = 'block';
    }
    if (baseTotal != null) {
      markB.style.left = `${((baseTotal / domain) * 100).toFixed(2)}%`;
      markB.style.display = 'block';
    }
    // Keep the under-bar labels in the same left-to-right order as the markers.
    const projText = projTarget != null ? `Projected ${fmtDuration(projTarget)}` : '';
    const baseText = baseTotal != null ? `Baseline ${fmtDuration(baseTotal)}` : '';
    const projFirst = projTarget == null || baseTotal == null || projTarget <= baseTotal;
    labP.textContent = projFirst ? projText : baseText;
    labB.textContent = projFirst ? baseText : projText;
    labP.className = projFirst ? 'lab-projected' : 'lab-baseline';
    labB.className = projFirst ? 'lab-baseline' : 'lab-projected';
  } else {
    fill.style.width = `${(stepFrac * 100).toFixed(1)}%`;
  }
}

export function updateBaselineRow(state, race) {
  const target = document.getElementById('baseline-target');
  if (race && (race.state === 'ahead' || race.state === 'behind') && race.targetLoss != null) {
    const at = state.baselineTotalTimeMs != null ? ` at ${fmtDuration(state.baselineTotalTimeMs)}` : '';
    target.textContent = `Target: baseline reached ${race.targetLoss.toFixed(3)} loss${at}`;
  } else {
    target.textContent = 'Select a baseline to set the target quality';
  }
}
