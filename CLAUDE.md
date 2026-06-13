# PodTerm

Local FastAPI app for launching and monitoring micro-GPT training runs on RunPod GPU pods (images built from the sibling `gpt-golf` repo).

## Event pipeline

Pod-side daemon (`gpt-golf/scripts/pod_eventd.py`) serves structured JSONL events + the raw log → `PodPoller` (podterm/events.py) pulls via the RunPod HTTP proxy (`https://{pod_id}-8765.proxy.runpod.net`, bearer token from the per-launch `EVENTD_TOKEN` env, persisted in the run's `config_json`) → thread-safe `log_queue` → `drain_loop` (podterm/server.py) → SSE fan-out to the web UI (static/index.html) + SQLite (podterm/db.py).

- The live path consumes structured events only — `parser.py` is import-only (legacy `/api/logs*` endpoints for local `.log` files). Keep the event schema in sync with the producers in gpt-golf (`bootstrap.sh` `emit`, `train_gpt.py` `emit_event`) — schema changes need a gpt-golf image rebuild + push.
- The phase strings `"Starting Training"` and `"Training finished"` are load-bearing in `drain_loop` and the UI.
- All RunPod interaction goes through the `runpodctl` CLI (podterm/runpod.py), with two sanctioned exceptions, both read-only and confined to podterm/runpod.py: `api_get_telemetry()` (GraphQL `myself.pods.runtime` — runpodctl exposes no utilization fields) and `api_get_machine_logs()` (`hapi.runpod.net/v1/pod/{id}/logs` — host-level boot/image-pull lines that predate the container). Keep any future direct API use confined there too.
- The hapi logs endpoint does NOT accept the RunPod API key (403). It needs a short-lived (~60s) Clerk *console* JWT, which `runpod.py` mints on demand from the user's `__client` cookie (`RUNPOD_CONSOLE_CLIENT_COOKIE` in a gitignored `.env`, loaded by `config.load_dotenv()` at startup). All best-effort: no cookie → boot panel just never appears. Full auth notes in the project memory (`hapi-logs-auth`).
- Before the event daemon is reachable, `PodPoller` polls machine logs and emits `t: "pull"` events (parsed by podterm/boot.py `PullTracker`); the UI shows only the boot panel until the container is up (`isBooting` in static/js/live.js).
