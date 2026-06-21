import * as d from '../../derive.js';
import { kpiStore } from './store.js';

function podGpuName(pod) {
  const gpus = (pod.runtime || {}).gpus || [];
  return gpus.length ? gpus[0].gpuDisplayName : (pod.machine || {}).gpuDisplayName || pod.gpu || null;
}

function noteTelemetry(card, running) {
  card.note(running ? 'Polling RunPod telemetry…' : 'Pod not running');
  if (card.bar) card.bar.set(0);
}

export function updateTelemetryCards(state, pod) {
  const { cards, diag } = kpiStore;
  const t = state.telemetry;
  const running = pod.desiredStatus === 'RUNNING';
  const live = running && t;

  if (live && t.cpu_pct != null) {
    const ramSub = t.mem_pct != null
      ? (t.ram_total_gb
          ? `RAM ${((t.mem_pct / 100) * t.ram_total_gb).toFixed(1)}/${t.ram_total_gb} GB`
          : `RAM ${Math.round(t.mem_pct)}%`)
      : (t.cpu_name || '');
    diag.system.set({ value: String(Math.round(t.cpu_pct)), unit: '% CPU', sub: ramSub });
    diag.system.bar.set(t.cpu_pct / 100);
  } else noteTelemetry(diag.system, running);

  const gpuName = state.info.gpu_type || state.runRow?.gpu_type || podGpuName(pod);
  const totalGiB = (t && t.gpu_mem_total_gb) || d.parseGpuMemGiB(gpuName);
  let memSub = '';
  if (live && t.gpu_mem_pct != null) {
    memSub = totalGiB
      ? `${((t.gpu_mem_pct / 100) * totalGiB).toFixed(1)}/${totalGiB} GB · live`
      : `${Math.round(t.gpu_mem_pct)}% mem · live`;
  } else if (state.memory) {
    const usedGiB = state.memory.peak_mib / 1024;
    memSub = totalGiB ? `peak ${usedGiB.toFixed(1)}/${totalGiB} GB` : `peak ${usedGiB.toFixed(1)} GB`;
  } else {
    memSub = state.finished ? 'no mem stats' : 'mem at run end';
  }
  if (live && t.gpu_util_pct != null) {
    cards.gpu.set({ value: String(Math.round(t.gpu_util_pct)), unit: '%', sub: memSub });
    cards.gpu.bar.set(t.gpu_util_pct / 100);
  } else {
    cards.gpu.note(running ? 'Polling RunPod telemetry…' : (state.memory ? memSub : 'Pod not running'));
    cards.gpu.bar.set(0);
  }
}
