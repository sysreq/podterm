// Hand-rolled SVG sparklines. One <polyline> + area <path> per instance;
// update() rewrites the points attribute — far cheaper than a chart library
// for the dozen-plus sparklines on the dashboard.

const NS = 'http://www.w3.org/2000/svg';
const VW = 100, VH = 28, PAD = 2;

export function attachSparkline(slotEl, { stroke = '#4a9eff', fill = 'rgba(74,158,255,0.12)' } = {}) {
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('viewBox', `0 0 ${VW} ${VH}`);
  svg.setAttribute('preserveAspectRatio', 'none');
  svg.style.cssText = 'width:100%;height:100%;display:block';

  const area = document.createElementNS(NS, 'path');
  area.setAttribute('fill', fill);
  area.setAttribute('stroke', 'none');

  const line = document.createElementNS(NS, 'polyline');
  line.setAttribute('fill', 'none');
  line.setAttribute('stroke', stroke);
  line.setAttribute('stroke-width', '1.5');
  line.setAttribute('vector-effect', 'non-scaling-stroke');
  line.setAttribute('stroke-linejoin', 'round');

  svg.appendChild(area);
  svg.appendChild(line);
  slotEl.appendChild(svg);

  function update(values) {
    if (!values || values.length < 2) { clear(); return; }
    const vals = values.slice(-100);
    let min = Infinity, max = -Infinity;
    for (const v of vals) { if (v < min) min = v; if (v > max) max = v; }
    const span = max - min || 1; // flat series renders mid-height
    const stepX = (VW - 2 * PAD) / (vals.length - 1);
    const pts = vals.map((v, i) => {
      const x = PAD + i * stepX;
      const y = PAD + (1 - (v - min) / span) * (VH - 2 * PAD);
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    });
    line.setAttribute('points', pts.join(' '));
    area.setAttribute('d', `M${PAD},${VH - PAD} L${pts.join(' L')} L${(PAD + (vals.length - 1) * stepX).toFixed(2)},${VH - PAD} Z`);
  }

  function clear() {
    line.removeAttribute('points');
    area.removeAttribute('d');
  }

  return { update, clear };
}
