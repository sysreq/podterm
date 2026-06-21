// Public facade for shared app state, SSE ingestion, hydration, and the event bus.
export { app } from './state/app.js';
export { emit, on } from './state/bus.js';
export { getPodState, dropPodState, resetRunState } from './state/pods.js';
export { ensurePodStream } from './state/stream.js';
export { hydrateFromDb, hydrateRunRow } from './state/hydrate.js';
