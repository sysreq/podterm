# GPT Caddy

A local, single-user FastAPI + vanilla-JS app for launching and monitoring
micro-GPT training runs on [RunPod](https://www.runpod.io/) GPU pods. Pods run
images built from the sibling [`gpt-golf`](../gpt-golf) repo; GPT Caddy provisions
them, streams live training telemetry into a dashboard, and runs off-pod
model-health diagnostics on the snapshots they produce.

> Screenshot placeholder — drop a live dashboard capture here.

## What it does

- **Launch** training runs on RunPod via the `runpodctl` CLI (GPU/datacenter/branch
  selection), tracking each run in a local SQLite DB.
- **Monitor** runs live: a pod-side event daemon serves structured JSONL events +
  the raw log; GPT Caddy pulls them over RunPod's HTTP proxy and fans them out to the
  browser over Server-Sent Events (SSE), persisting metrics as they arrive.
- **Diagnose** model health off the training GPU: each snapshot is downloaded and
  run through a diagnostics suite locally, producing a threshold-coloured health
  verdict, a status timeline, and a Compare view that diffs runs.

## Architecture overview

```
gpt-golf pod  ──(JSONL events + log)──►  PodPoller  ──►  EventPipeline.drain_loop
  pod_eventd.py        RunPod HTTP proxy   (events.py)      (pipeline.py)
                                                               │
                                          ┌────────────────────┼───────────────┐
                                          ▼                    ▼               ▼
                                       SSEHub             SQLite (db/)     snapshot → diagnostics
                                       (sse.py)                            (snapshots.py +
                                          │                                 diagnostics/)
                                          ▼
                                   browser UI (static/)
```

- `server.py` is glue only (FastAPI app, lifespan, static mounts, routers). Logic
  lives in focused modules: `pipeline.py` (event drain + telemetry), `sse.py`
  (fan-out), `pods.py` (pod lifecycle), `routes/` (thin routers), `models.py`
  (event dataclasses), and the `runpod/` package (all RunPod interaction).
- The frontend (`podterm/static/`) is plain ES modules — one module per concern
  (live dashboard, compare, diagnostics panel, history, launch dialog).
- **For the full architecture, event schema, and design notes, see
  [`CLAUDE.md`](./CLAUDE.md)** and [`podterm/diagnostics/README.md`](./podterm/diagnostics/README.md).

## Setup

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.12+. Node 22+ is needed
only to run the frontend tests.

```bash
uv sync                       # install deps (incl. dev group: pytest, ruff)
cp .env.example .env          # optional: fill in RunPod console cookie etc.
```

`runpodctl` must be installed and authenticated for launches. Diagnostics expect a
sibling `gpt-golf` checkout (`GPT_GOLF_DIR`, defaults to `../gpt-golf`). All of
this is best-effort — see `.env.example` for every knob.

## Run

```bash
uv run gpt-caddy              # serves http://127.0.0.1:8000
```

The browser opens automatically. The server binds to `127.0.0.1` only.

## Tests & quality gate

```bash
uv run pytest -q                                              # Python tests
uv run ruff check .                                          # lint (E, F, I)
node --test podterm/static/js/derive.test.mjs \
            podterm/static/js/diagnostics.test.mjs           # frontend tests
```

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs all three on push
and PR. Optional pre-commit hooks mirror CI:

```bash
uv run --with pre-commit pre-commit install
```

## Security model

- **Local-only by design.** The server binds to `127.0.0.1:8000` — it is a
  single-user developer tool, not a multi-tenant service. There is no auth layer.
- **Event-daemon token.** Each launch mints a per-run `EVENTD_TOKEN` (bearer) that
  authenticates GPT Caddy's pulls from the pod's event daemon over the RunPod proxy.
  It is passed to the pod as an env var and persisted in the run's `config_json`.
  Treat the local SQLite DB and `.env` as secrets; both are gitignored.
- **RunPod credentials** (`RUNPOD_*` API key, the console `__client` cookie used to
  mint short-lived Clerk JWTs for boot logs) live only in the gitignored `.env`,
  loaded at startup. They never reach the browser.

## Limitations / follow-ups

- **Single-user, local tool.** No authentication, no multi-user isolation, no
  hardened network exposure — intentional for the scope.
- **Static typing.** A type-checker (pyright or mypy) with a documented strictness
  target is not yet wired in — a natural next step for the quality gate.
- **Coverage thresholds.** Tests run in CI but there is no `pytest-cov` minimum yet;
  enforcing a coverage floor (and adding failure-injection tests around the
  threads/SQLite/subprocess boundaries) is future work.
- **Dependency/vuln scanning.** Not yet part of CI.
