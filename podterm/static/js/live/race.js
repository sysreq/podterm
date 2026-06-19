import { fmtDuration, fmtInt, fmtMs, fmtSecShort } from '../format.js';

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

  if (race.state !== 'ahead' && race.state !== 'behind') {
    headline.textContent = 'No baseline selected';
    sub.textContent = 'Choose a baseline below to start the race';
    fill.style.width = `${(stepFrac * 100).toFixed(1)}%`;
    if (eta != null && m.train_time_ms != null) {
      labP.textContent = `Projected total ${fmtDuration(m.train_time_ms + eta)}`;
    }
    return;
  }

  const ahead = race.state === 'ahead';
  banner.classList.add(ahead ? 'ahead' : 'behind');
  headline.textContent = `You're ${ahead ? 'ahead of' : 'behind'} baseline by ${fmtSecShort(race.cumulativeMs)}`;
  if (state.finished) {
    sub.textContent = `Final margin vs baseline at step ${fmtInt(m.step)}`;
  } else if (race.requiredMs != null && race.requiredMs >= 1 && race.projectedMarginMs != null) {
    // Projected margin at the finish line — not the current cumulative lead.
    sub.textContent = race.projectedMarginMs >= 0
      ? `Keep ≤ ${fmtMs(race.requiredMs)} ms/step to finish ~${fmtSecShort(race.projectedMarginMs)} ahead`
      : `Need ≤ ${fmtMs(race.requiredMs)} ms/step — current pace finishes ~${fmtSecShort(race.projectedMarginMs)} behind`;
  } else {
    sub.textContent = 'Baseline total time already elapsed — cumulative lead is what counts';
  }

  // Progress bar on the time domain [0, max(projected total, baseline total)].
  const projTotal = eta != null && m.train_time_ms != null ? m.train_time_ms + eta : null;
  const baseTotal = state.baselineTotalTimeMs;
  const domain = Math.max(projTotal ?? 0, baseTotal ?? 0);
  if (domain > 0) {
    fill.style.width = `${Math.min(100, ((m.train_time_ms ?? 0) / domain) * 100).toFixed(1)}%`;
    if (projTotal != null) {
      markP.style.left = `${((projTotal / domain) * 100).toFixed(2)}%`;
      markP.style.display = 'block';
    }
    if (baseTotal != null) {
      markB.style.left = `${((baseTotal / domain) * 100).toFixed(2)}%`;
      markB.style.display = 'block';
    }
    // Keep the under-bar labels in the same left-to-right order as the markers.
    const projText = projTotal != null ? `Projected ${fmtDuration(projTotal)}` : '';
    const baseText = baseTotal != null ? `Baseline ${fmtDuration(baseTotal)}` : '';
    const projFirst = projTotal == null || baseTotal == null || projTotal <= baseTotal;
    labP.textContent = projFirst ? projText : baseText;
    labB.textContent = projFirst ? baseText : projText;
    labP.className = projFirst ? 'lab-projected' : 'lab-baseline';
    labB.className = projFirst ? 'lab-baseline' : 'lab-projected';
  } else {
    fill.style.width = `${(stepFrac * 100).toFixed(1)}%`;
  }
}

export function updateBaselineRow(race) {
  const target = document.getElementById('baseline-target');
  if (race && race.requiredMs != null && race.requiredMs >= 1) {
    target.textContent = `To beat baseline: avg ms/step ≤ ${fmtMs(race.requiredMs)}`;
  } else if (race && (race.state === 'ahead' || race.state === 'behind')) {
    target.textContent = 'Past baseline time — cumulative lead decides';
  } else {
    target.textContent = 'Select a baseline to set the target pace';
  }
}
