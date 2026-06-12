// Card factories for the KPI row and diagnostics row. Every card keeps the
// same fixed internal slots in every state so values appearing never reflow.
import { attachSparkline } from './sparkline.js';

export function kpiCard({ label, tooltip = '', hero = false, spark = false, bar = false }) {
  const el = document.createElement('div');
  el.className = 'kpi-card' + (hero ? ' hero' : '');
  el.innerHTML = `
    <div class="kpi-label"><span class="kpi-label-text"></span><span class="kpi-info" hidden>&#9432;</span></div>
    <div class="kpi-value"><span class="kpi-value-text"></span><span class="kpi-unit"></span></div>
    <div class="kpi-sub"></div>
    <div class="kpi-caption" ${hero ? '' : 'hidden'}></div>
    <div class="kpi-foot"></div>`;
  el.querySelector('.kpi-label-text').textContent = label;
  const infoEl = el.querySelector('.kpi-info');
  if (tooltip) { infoEl.hidden = false; infoEl.title = tooltip; }

  const valueEl = el.querySelector('.kpi-value-text');
  const unitEl = el.querySelector('.kpi-unit');
  const subEl = el.querySelector('.kpi-sub');
  const capEl = el.querySelector('.kpi-caption');
  const footEl = el.querySelector('.kpi-foot');

  let sparkCtl = null;
  let barCtl = null;
  if (spark) sparkCtl = attachSparkline(footEl);
  if (bar) barCtl = barMeter(footEl);

  function set({ value = '', unit = '', sub = '', subClass = '', caption = '', captionClass = '', accent = null } = {}) {
    el.classList.remove('is-note');
    valueEl.textContent = value;
    unitEl.textContent = unit;
    subEl.textContent = sub;
    subEl.className = 'kpi-sub' + (subClass ? ' ' + subClass : '');
    capEl.textContent = caption;
    capEl.className = 'kpi-caption' + (captionClass ? ' ' + captionClass : '');
    el.classList.toggle('ahead', accent === 'success');
    el.classList.toggle('behind', accent === 'danger');
  }

  // Loading / pending / empty states: message in place of the value, same heights.
  function note(msg, { pending = false } = {}) {
    set({});
    el.classList.add('is-note');
    subEl.innerHTML = pending ? '<span class="pending-chip">Pending</span>' : '';
    capEl.textContent = '';
    const msgEl = document.createElement('div');
    msgEl.className = 'kpi-note-text';
    msgEl.textContent = msg;
    subEl.appendChild(msgEl);
  }

  return {
    el,
    set,
    note,
    spark: sparkCtl ? sparkCtl.update : null,
    bar: barCtl,
  };
}

export function barMeter(slotEl) {
  const wrap = document.createElement('div');
  wrap.className = 'bar-meter';
  const fillEl = document.createElement('div');
  fillEl.className = 'bar-meter-fill';
  wrap.appendChild(fillEl);
  slotEl.appendChild(wrap);
  return {
    set(fraction) {
      const pct = fraction == null ? 0 : Math.max(0, Math.min(1, fraction)) * 100;
      fillEl.style.width = `${pct}%`;
    },
  };
}
