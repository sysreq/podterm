import { fmtDelta, fmtDuration, fmtInt, fmtMs, fmtTps } from '../format.js';

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
// The race compares estimated finish loss at each run's configured max-wallclock
// budget (lower wins). A behind run also gets a "time to win" estimate.
export function updateRaceBanner(state, m, race, eta) {
  const banner = document.getElementById('race-banner');
  const headline = document.getElementById('race-headline');
  const sub = document.getElementById('race-sub');
  const fill = banner.querySelector('.race-bar-fill');
  const markP = banner.querySelector('.race-marker.projected');
  const markB = banner.querySelector('.race-marker.baseline');
  const markW = banner.querySelector('.race-marker.win');
  const labP = document.getElementById('race-label-projected');
  const labB = document.getElementById('race-label-baseline');

  banner.classList.remove('ahead', 'behind', 'tied');
  markP.style.display = 'none';
  markB.style.display = 'none';
  if (markW) markW.style.display = 'none';
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
  const racing = race && (race.state === 'ahead' || race.state === 'behind' || race.state === 'tied');

  if (!racing) {
    if (race && race.state === 'unknown') {
      headline.textContent = 'Waiting for enough loss trend';
      sub.textContent = throughput || 'Projecting the finish once the loss trend settles';
    } else {
      headline.textContent = 'No baseline selected';
      sub.textContent = throughput || 'Choose a baseline below to start the race';
    }
    fill.style.width = `${(stepFrac * 100).toFixed(1)}%`;
    if (eta != null && m.train_time_ms != null) {
      labP.textContent = `Projected total ${fmtDuration(m.train_time_ms + eta)}`;
    }
    return;
  }

  const ahead = race.state === 'ahead';
  const tied = race.state === 'tied';
  banner.classList.add(tied ? 'tied' : (ahead ? 'ahead' : 'behind'));
  const marginTxt = race.finishMarginLoss != null ? Math.abs(race.finishMarginLoss).toFixed(4) : '?';
  headline.textContent = tied
    ? 'Too close to call at the budget finish'
    : ahead ? `Projected to beat baseline by ${marginTxt} loss`
            : `Projected behind baseline by ${marginTxt} loss`;

  const subParts = [];
  if (throughput) subParts.push(throughput);
  if (race.currentFinishLoss != null && race.baselineFinishLoss != null) {
    subParts.push(`finish ${race.currentFinishLoss.toFixed(3)} vs baseline ${race.baselineFinishLoss.toFixed(3)}`);
  }
  if (state.finished) {
    subParts.push(`final at step ${fmtInt(m.step)}`);
  } else if (!ahead && !tied) {
    // Behind: how much total training before the loss curve crosses the baseline finish.
    if (race.estimatedWinTimeMs != null) {
      const beyond = race.extraTimeToWinMs != null ? ` (+${fmtDuration(race.extraTimeToWinMs)} beyond budget)` : '';
      subParts.push(`needs ~${fmtDuration(race.estimatedWinTimeMs)} total training${beyond} to win`);
    } else {
      subParts.push('win not currently projected');
    }
  } else if (ahead && race.finishBudgetMs != null) {
    subParts.push(`wins by budget finish at ${fmtDuration(race.finishBudgetMs)}`);
  }
  sub.textContent = subParts.join(' · ');

  // Progress bar on the time axis: elapsed within [0, the latest of the budgets,
  // a beyond-budget win time, and elapsed].
  const budget = race.finishBudgetMs;
  const baseBudget = race.baselineFinishBudgetMs;
  const winT = race.extraTimeToWinMs != null ? race.estimatedWinTimeMs : null; // only render when past budget
  const domain = Math.max(budget ?? 0, baseBudget ?? 0, winT ?? 0, m.train_time_ms ?? 0);
  if (domain > 0) {
    fill.style.width = `${Math.min(100, ((m.train_time_ms ?? 0) / domain) * 100).toFixed(1)}%`;
    if (budget != null) {
      markP.style.left = `${((budget / domain) * 100).toFixed(2)}%`;
      markP.style.display = 'block';
    }
    if (baseBudget != null) {
      markB.style.left = `${((baseBudget / domain) * 100).toFixed(2)}%`;
      markB.style.display = 'block';
    }
    if (markW && winT != null) {
      markW.style.left = `${Math.min(100, (winT / domain) * 100).toFixed(2)}%`;
      markW.style.display = 'block';
    }
    // Keep the under-bar labels in the same left-to-right order as the budget markers.
    const projText = budget != null ? `Budget ${fmtDuration(budget)}` : '';
    const baseText = baseBudget != null ? `Baseline ${fmtDuration(baseBudget)}` : '';
    const projFirst = budget == null || baseBudget == null || budget <= baseBudget;
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
  if (race && (race.state === 'ahead' || race.state === 'behind' || race.state === 'tied') && race.baselineFinishLoss != null) {
    const at = race.baselineFinishBudgetMs != null ? ` at ${fmtDuration(race.baselineFinishBudgetMs)} budget` : '';
    target.textContent = `Target: baseline finish ${race.baselineFinishLoss.toFixed(3)} loss${at}`;
  } else if (race && race.state === 'unknown' && state.baselineRunId) {
    target.textContent = 'Baseline selected — projecting its finish loss…';
  } else {
    target.textContent = 'Select a baseline to set the target quality';
  }
}
