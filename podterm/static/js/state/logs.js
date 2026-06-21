const RE_STEP_LINE = /^step:(\d+)\//;
const RE_ERROR = /\b(error|traceback|exception|fatal|failed)\b/i;
const RE_WARN = /\bwarn(ing)?\b/i;
const LOG_CAP = 1000;

function classifyLog(text) {
  const stepMatch = text.match(RE_STEP_LINE);
  if (stepMatch) return { level: 'metric', step: Number(stepMatch[1]) };
  if (text.startsWith('==>')) return { level: 'metric', step: null };
  if (RE_ERROR.test(text)) return { level: 'error', step: null };
  if (RE_WARN.test(text)) return { level: 'warn', step: null };
  return { level: 'info', step: null };
}

export function pushLog(state, text) {
  const { level, step } = classifyLog(text);
  state.logLines.push({ text, level, step });
  if (state.logLines.length > LOG_CAP) state.logLines.splice(0, state.logLines.length - 800);
}
