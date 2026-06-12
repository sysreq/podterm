# PodTerm

Local FastAPI app for launching and monitoring micro-GPT training runs on RunPod GPU pods (images built from the sibling `gpt-golf` repo).

## Event pipeline

Pod-side daemon (`gpt-golf/scripts/pod_eventd.py`) serves structured JSONL events + the raw log → `PodPoller` (podterm/events.py) pulls via the RunPod HTTP proxy (`https://{pod_id}-8765.proxy.runpod.net`, bearer token from the per-launch `EVENTD_TOKEN` env, persisted in the run's `config_json`) → thread-safe `log_queue` → `drain_loop` (podterm/server.py) → SSE fan-out to the web UI (static/index.html) + SQLite (podterm/db.py).

- The live path consumes structured events only — `parser.py` is import-only (legacy `/api/logs*` endpoints for local `.log` files). Keep the event schema in sync with the producers in gpt-golf (`bootstrap.sh` `emit`, `train_gpt.py` `emit_event`) — schema changes need a gpt-golf image rebuild + push.
- The phase strings `"Starting Training"` and `"Training finished"` are load-bearing in `drain_loop` and the UI.
- All RunPod interaction goes through the `runpodctl` CLI (podterm/runpod.py), never the GraphQL API directly.
