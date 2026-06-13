# PodTerm

Local FastAPI app for launching and monitoring micro-GPT training runs on RunPod GPU pods (images built from the sibling `gpt-golf` repo).

## Event pipeline

Pod-side daemon (`gpt-golf/scripts/pod_eventd.py`) serves structured JSONL events + the raw log → `PodPoller` (podterm/events.py) pulls via the RunPod HTTP proxy (`https://{pod_id}-8765.proxy.runpod.net`, bearer token from the per-launch `EVENTD_TOKEN` env, persisted in the run's `config_json`) → the thread-safe queue owned by `EventPipeline` (podterm/pipeline.py) → its `drain_loop` → SSE fan-out via `SSEHub` (podterm/sse.py) to the web UI (static/index.html) + SQLite (podterm/db.py).

- Module layout: `server.py` is glue only (FastAPI app, lifespan, static, router mounts). Logic lives in `pipeline.py` (event drain + telemetry, `EventPipeline`), `sse.py` (fan-out, `SSEHub`), `pods.py` (pod lifecycle + poller registry, `PodManager`, plus `LaunchConfig`), `routes/` (thin `APIRouter`s), `models.py` (event dataclasses), and the `runpod/` package. Service objects are module singletons (`pipeline`, `hub`, `manager`); DAG is `SSEHub` ← `EventPipeline` ← `PodManager`.
- The live path consumes structured events only (via `EventPipeline`). Keep the event schema in sync with the producers in gpt-golf (`bootstrap.sh` `emit`, `train_gpt.py` `emit_event`) — schema changes need a gpt-golf image rebuild + push.
- The phase strings `"Starting Training"` and `"Training finished"` are load-bearing in `EventPipeline.drain_loop` (podterm/pipeline.py) and the UI.
- All RunPod interaction goes through the `runpodctl` CLI (podterm/runpod/cli.py), with two sanctioned exceptions, both read-only and confined to podterm/runpod/api.py: `api_get_telemetry()` (GraphQL `myself.pods.runtime` — runpodctl exposes no utilization fields) and `api_get_machine_logs()` (`hapi.runpod.net/v1/pod/{id}/logs` — host-level boot/image-pull lines that predate the container). Keep any future direct API use confined to that package too. The `podterm.runpod` package re-exports the public surface, so callers import from `podterm.runpod` unchanged.
- The hapi logs endpoint does NOT accept the RunPod API key (403). It needs a short-lived (~60s) Clerk *console* JWT, which `podterm/runpod/console.py` mints on demand from the user's `__client` cookie (`RUNPOD_CONSOLE_CLIENT_COOKIE` in a gitignored `.env`, loaded by `config.load_dotenv()` at startup). All best-effort: no cookie → boot panel just never appears. Full auth notes in the project memory (`hapi-logs-auth`).
- Before the event daemon is reachable, `PodPoller` polls machine logs and emits `t: "pull"` events (parsed by podterm/boot.py `PullTracker`); the UI shows only the boot panel until the container is up (`isBooting` in static/js/live.js).
