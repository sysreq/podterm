# PodTerm — Live Demo Run-of-Show

**One-liner:** *PodTerm is a thin local control plane for renting GPUs by the minute,
running micro-GPT training on them, watching it live in the browser, and grading the
model's health off the training GPU — all from a single-user app on your laptop.*

**Format:** ~20 min · mixed/general-technical audience · driven live against a real run.
**Golden rule of this demo:** *launch the pod first, talk while it boots.*

---

## 0. Pre-demo checklist (do this before the audience is watching)

- [ ] `uv run podterm` is up; browser open at `http://127.0.0.1:8000`.
- [ ] `runpodctl` authenticated; `.env` has `RUNPOD_*` + console cookie loaded.
- [ ] **At least one completed historical run** is in the local DB (the safety net — the
      whole demo can be told from history if the live pod misbehaves).
- [ ] Optionally **pre-launch a "warm" pod 3–4 min before you start** so there is
      *guaranteed* live training to show, independent of the one you launch on stage.
- [ ] Pick a **baseline run** to pin (makes the race bar meaningful immediately).
- [ ] Know your numbers: training is time-budgeted (~600 s by default), pod boot/image-pull
      is ~3–6 min. Launching on stage at minute 0 means: boots during minutes 0–6, trains
      ~6–16, diagnostics land ~16–18 — which lines up with this script.

---

## 1. The hook + launch (0:00–2:00)

**On screen:** the Live tab, then the **Launch** dialog.

- Open with the one-liner. "GPUs are expensive and ephemeral. I wanted to rent one for ten
  minutes, train a small GPT, *watch it race*, and have it grade itself — without babysitting
  a terminal."
- **Launch a run now, live.** Walk the dialog quickly: branch, datacenter, GPU, data variant.
  Hit launch. *"That pod is now booting. We'll come back to it — let's talk about what just
  got set in motion while it pulls its image."*
- Plant the three-verb spine you'll return to: **Launch → Monitor → Diagnose.**

> Why launch first: pods take minutes to boot. Starting now means the boot panel and live
> dashboard will be *real* by the time we get back, instead of a static screenshot.

---

## 2. What just happened: the ecosystem (2:00–7:00)

This is the **infrastructure** section. Frame: *"PodTerm is small on purpose — the heavy
lifting lives in a sibling repo, `gpt-golf`, which is what actually runs on the pod. PodTerm
is the remote control."* Keep it about plumbing, not the model.

### 2a. Ephemeral GPUs as cattle (1 min)
- Pods are **stateless**: they clone the repo from GitHub, pull the dataset from HuggingFace,
  train, save snapshots, and **shut themselves down**. Nothing precious lives on the pod.
- All durable state — run config, metrics, diagnostics — lives in **local SQLite** on the
  laptop. The pod can die at any moment and you lose nothing but the GPU-seconds.

### 2b. The container story — layered Docker images (1.5 min)
*The single best infra talking point.* Three images:
- **`Dockerfile.base`** — the expensive, rarely-rebuilt layer: Ubuntu + Python, **torch +
  CUDA + flash-attention baked in** (multi-GB). Notable craft: CUDA libraries are **split
  across many layers so they pull in parallel** on cold boot, and **never-called CUDA libs
  are stubbed out** (e.g. FFT/sparse) instead of shipping hundreds of MB of dead kernels.
- **`Dockerfile`** (training image) — `FROM base`, then a **seconds-long `uv sync`** of just
  the small pure-Python deps. Code changes rebuild in seconds because torch/CUDA never move.
- **`Dockerfile.redis`** — a shared **PyTorch compile-cache** pod (see 2e).
- Takeaway: *"Bake the slow stuff once; iterate on code in seconds."*

### 2c. The pod bootstrap (1 min)
`scripts/bootstrap.sh` is the pod entrypoint and runs a tidy lifecycle:
`tee all logs → start the event daemon → sync code from git → (hot-swap the daemon if it
changed) → download data → probe the compile cache → launch training → teardown`.
- Two clevernesses to call out: the **daemon can be upgraded via `git push` (no image
  rebuild)**, and the bootstrap can **auto-stop the pod** when done so you don't pay for idle
  time.

### 2d. The data pipeline — one knob (0.5 min)
- `data/cached_challenge_fineweb.py` is a **manifest-driven** downloader. **One env var,
  `DATA_VARIANT` (e.g. `sp1024`)**, selects the dataset, the tokenizer, and the vocab size
  together — no mismatched-tokenizer footguns.
- **Two-phase download:** grab a few shards + the tokenizer/val split *blocking* so training
  can start fast, then **stream the rest in the background** while training consumes it.
- Nice reuse: PodTerm runs **this same downloader locally** to fetch the val shard + tokenizer
  it needs for off-pod diagnostics. One data path, two environments.

### 2e. The redis compile cache (0.5 min)
- `torch.compile` is slow the first time (graph → C++ → CUDA binary). A **shared redis pod
  caches compiled graphs across training pods**, so cold-start compile cost is paid **once
  per architecture, not once per run.** First pod warms it; the rest hit ~95% cache.

> **Drill-down if asked:** exact byte sizes of the CUDA layers, the NCCL stub-and-restore for
> single- vs multi-GPU, the `LD_PRELOAD` tcmalloc tweak.

---

## 3. The wire: how the pod talks to the laptop (7:00–9:30)

Bridge from infra → the live thing on screen. **Cut back to the boot panel here — by now it
should show real image-pull progress.**

### 3a. The boot panel (on screen, ~0:45)
- Point at it: *"This is the pod pulling its Docker image, layer by layer, right now."*
- How it works (one breath): before the pod's own daemon is reachable, PodTerm **polls
  RunPod's host machine logs and parses them into per-layer pull progress** — that's the
  panel. (Behind the scenes this needs a short-lived console token, not the API key — a fun
  aside if the audience is into auth.)

### 3b. The event pipeline (1.5 min, diagram on a slide)
The end-to-end path, said plainly:
```
gpt-golf pod                      PodTerm (laptop)                    browser
pod_eventd.py  ──JSONL events──►  PodPoller ──► EventPipeline ──► SSEHub ──► Live UI
(stdlib HTTP    over RunPod's      (long-poll)   .drain_loop          │
 daemon :8765)  HTTPS proxy,                          │               └► (live charts/cards)
                bearer token                          ├──► SQLite (durable)
                                                      └──► snapshot ──► off-pod diagnostics
```
- **`pod_eventd.py`** is a **stdlib-only HTTP daemon** (no deps, so it starts before anything
  is installed). It serves: structured **JSONL events**, the **raw log**, and **model
  snapshots** — plus a tiny **`/snapshot/ack`** handshake (more in §5).
- PodTerm reaches it over **RunPod's HTTPS proxy** (`{pod_id}-8765.proxy.runpod.net`),
  authenticated by a **per-run bearer token** minted at launch.
- The **event schema is a contract**: line-delimited JSON, `{t, ts, ...}`, with types like
  `phase`, `metric`, `gpu`, `snapshot`, `summary`. Two phase strings are **load-bearing** —
  `"Starting Training"` and `"Training finished"` drive run start/finalization on both ends.
- Everything fans out to the browser over **Server-Sent Events** and is persisted to SQLite
  as it arrives.

> **Drill-down if asked:** the service-singleton DAG (`SSEHub ← EventPipeline ← PodManager`),
> the ordered shutdown (stop producers → drain → close DB), idempotent run finalization,
> "pod gone" requiring 2 confirmations so a transient CLI blip never kills a live run,
> metrics batching with bounded retry, SSE backpressure handling, secret redaction.

---

## 4. The money shot: the live dashboard (9:30–14:00)

**By now the pod should be training.** This is the centerpiece — spend real time here and let
the audience watch numbers move.

### 4a. The race bar (the star — ~1.5 min)
- *"This is the feature I'm proudest of."* The horizontal **race bar** predicts your run's
  **finish quality vs a baseline run at the same time budget**, and tells you live whether
  you're **ahead / behind / too close to call** — with markers for *projected*, *baseline*,
  and (if behind) the *win* point.
- Watch it shift as the loss curve settles; the hero card **glows green/red** with the verdict.
- All of this is **derived client-side** from the streamed metrics — EMA-smoothed step times,
  ETA, throughput, projected finish loss. No server round-trips.

### 4b. The KPI cards (~1 min)
Six at-a-glance cards, each with fixed slots so streaming numbers never make the layout jump:
**Projected Finish** (+sparkline), **Baseline** (hero), **Loss (train)**, **GPU** (util+mem),
**Cost So Far** (ticking $ = the visceral one), **Model Health** (verdict, click → Health tab).
Plus a second row: **Validation BPB** and **System** (CPU/RAM).

### 4c. Charts + logs (~1 min)
- The **training-loss chart** (Plotly): your run, the baseline overlaid (dashed), and eval
  points (diamonds). Custom y-scaling so convergence detail near the floor is readable.
- The **live log panel** streaming over SSE — filter by level, pause/resume. Point out that
  metrics, logs, cost, and the race verdict are all updating *right now, live*.

### 4d. Throwaway line that lands (~0:30)
- *"Cost is ticking in real time, and when the run ends the pod stops itself — I'm only
  paying for these few minutes."*

---

## 5. Diagnose: model health, off the training GPU (14:00–17:30)

The second flagship. Frame the **why** first, then show it.

### 5a. Why off-pod (~0:45)
- Training should spend every GPU-second training. So instead of grading the model on the
  pod, **PodTerm downloads each snapshot and runs the diagnostics suite locally** — as a
  subprocess under gpt-golf's torch env (the model architecture changes constantly, so it's
  single-sourced there). The pod stays focused on training.
- The **snapshot handshake**: the pod saves a checkpoint and emits a `snapshot` event;
  PodTerm pulls it, runs diagnostics, and on the *final* one calls **`/snapshot/ack`** so the
  pod knows it's safe to shut down (bootstrap waits ~120 s for this).

### 5b. The Model Health tab (on screen, ~1.5 min)
- The **verdict card**: OK / WARN / ERROR, in plain language, with the top issues and a
  **clickable snapshot timeline** (health at each checkpoint over the run).
- **Deep dive:** per-section drill-down with **threshold bands and a one-line "why" for every
  metric**, plus diff-vs-previous trends. Emphasize: *the thresholds and rationale come from
  the backend's metric registry — the UI doesn't hardcode any numbers.*
- Plain-English tour of *what* it measures (don't go deep on ML): does the model use all its
  capacity (dead neurons, redundant heads), does signal flow cleanly through the layers, are
  gradients healthy, is attention focused or smeared.

### 5c. The credibility moment — self-verification (~1 min)
- *The wow line:* **"The diagnostics check their own arithmetic."** The off-pod forward pass
  **recomposes each layer's output from its parts** and **re-derives the loss**, cross-checking
  against the model's own numbers — anything it can't verify is flagged, never silently trusted.
- And it **reconciles its bits-per-byte against the on-pod eval** (replicating the exact byte
  accounting) so you can trust a number computed on *different* hardware. *"Most tools assume
  they're right. This one proves it."*

> **Drill-down if asked:** the runtime "anatomy probe" that discovers model structure instead
> of hardcoding it; the versioned (additive-only) JSON schema; per-pod serialization +
> coalescing of snapshot jobs.

---

## 6. Compare + the workflow it enables (17:30–18:30)

**On screen:** History tab → select two runs → Compare tab.

- *"This is what the whole thing is for: did my change help?"*
- Show the **Compare** view: overlaid loss curves (aligned by step or by % progress), a
  **final-loss regression banner**, and a **model-health diff** — verdict + headline metrics
  side by side, and which internal blocks moved.
- Tie it back: launch → monitor → diagnose → **compare → decide → iterate.** That loop is the
  product.

---

## 7. Close: the shape of it + where it's going (18:30–20:00)

- **Recap the spine in one breath:** a laptop app rents an ephemeral GPU, ships it a baked
  image + a stateless bootstrap, streams structured events back over a proxy into a live
  dashboard, and grades each snapshot off-GPU with self-verifying diagnostics — then helps you
  compare runs and decide what's next.
- **Engineering rigor (quick):** real test coverage — Python (`pytest`: health verdicts,
  compare diffs, schema, a golden end-to-end diagnostics test, db/pipeline/poller/routes,
  secret redaction) and frontend (`node --test` for the derived-metrics + diagnostics logic);
  **CI runs all three on every push.**
- **Direction / what's next:** evolving toward a polished **standalone micro-GPT training
  utility**; the dashboard has been developed design-doc-first (model-health UX, budget-aware
  race bar, responsive screen-scaling). Honest follow-ups from the README: a **static type
  checker**, a **coverage floor + failure-injection tests** around the thread/SQLite/subprocess
  seams, and **dependency/vuln scanning** in CI.
- End on the live pod: by now it has likely **finished and stopped itself** — *"and that run is
  already in my history, graded, ready to compare against the next idea. Total GPU bill:
  a few minutes."*

---

## Backup plan (if the live pod misbehaves)

- **Boot stalls / image pull is slow:** keep narrating §2 infra; the pod-pull panel doubles as
  the visual. Fall back to the pre-warmed pod from the checklist.
- **No live training at all:** pivot the entire §4–§6 to a **completed historical run** — the
  dashboard, race bar (vs baseline), model health, and compare all work from stored data. The
  only thing you lose is "watch it move," so describe that one moment instead.
- **Diagnostics not ready in time:** open a *previous* run's Model Health tab; the verdict
  timeline and deep-dive are fully populated from SQLite.

---

## Appendix A — "Wow" moments to make sure you land

1. **The race bar** predicting ahead/behind vs a baseline at the same budget, live.
2. **Self-verifying diagnostics** — the suite checks its own loss/branch math and reconciles
   BPB across hardware.
3. **Layered Docker images** — bake torch/CUDA once, parallel-pull layers, stub dead libs,
   rebuild code in seconds.
4. **Cost ticking in real time** and the pod **auto-stopping** itself.
5. **One knob (`DATA_VARIANT`)** picks dataset + tokenizer + vocab together.

## Appendix B — Key files cheat sheet (for Q&A)

| Topic | Where |
|---|---|
| Pitch / overview | `README.md`, `CLAUDE.md` |
| Event pipeline | `podterm/pipeline.py` (`EventPipeline`), `podterm/sse.py` (`SSEHub`), `podterm/eventing/` (`PodPoller`) |
| RunPod control | `podterm/runpod/` (`cli.py` CLI; `api.py`/`console.py` read-only API + JWT) |
| Boot panel | `podterm/boot.py` (`PullTracker`), `static/js/live/boot.js` |
| Live dashboard | `static/js/live.js`, `live/race.js`, `live/kpis/`, `static/js/derive/` |
| Diagnostics suite | `podterm/diagnostics/` (`runner.py`, `health.py`, `stages_forward.py`), `podterm/snapshots.py` |
| Diagnostics UI | `static/js/diagnostics/`, `static/js/compare.js` |
| Pod infra (gpt-golf) | `Dockerfile*`, `scripts/bootstrap.sh`, `scripts/pod_eventd.py`, `data/cached_challenge_fineweb.py` |
| Event/snapshot contract | gpt-golf `train_gpt.py` (`emit_event`, `save_snapshot`) ↔ `podterm/models.py` |

## Appendix C — Glossary (for the non-deep audience, say once)

- **Pod** — a rented cloud GPU machine, billed by the minute, thrown away after the run.
- **Snapshot** — a saved copy of the model partway through training.
- **BPB (bits per byte)** — the quality score; lower is better, comparable across tokenizers.
- **SSE (Server-Sent Events)** — one-way live stream from server to browser; powers the
  real-time dashboard.
- **Baseline** — a previous run you're racing the current one against.
