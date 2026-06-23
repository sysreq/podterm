# PodTerm — Interview Demo Run-of-Show v3 (Stellar Science)

> The build-ready script. Merges the detailed feature material from `live-demo.md` (v1)
> with the strategic re-aim in `live-demo-v2.md`: this is a **hiring-panel demo**, not a
> meetup talk. Structured slide-by-slide so it converts directly into a PowerPoint deck.
> **Tag legend for each beat:** `[SLIDE]` = on the deck · `[LIVE]` = driven in the app ·
> `[SAY]` = near-verbatim line · `[HOOK]` = who leans forward · `[Q&A]` = reserve for later.

---

## 1. The target — read before building slides

- **Audience:** Stellar Science panel. Mixed and unpredictable — some C++, some Python,
  possibly a physics/MATLAB PhD, some non-technical, **CEO may attend**, **some on Teams.**
- **Format:** **30 min including Q&A.** Budget **~18 min present + ~12 min Q&A.** The deck
  carries the narrative; the live app provides proof, not pacing.
- **The sentences you are earning** (write every slide to serve one of these):
  - **CEO:** "This person turns a product need into a deliverable service."
  - **Everyone:** "This is a builder who owns quality — not a paycheck-collector."
  - **The C++ / systems folks:** "He can reach into a hard systems problem and solve it."
- **Live strategy (decided):** one **pre-warmed** pod is already training by the time you
  reach the dashboard (~minute 7). A fresh launch is kicked at the top **off the critical
  path** as a "this is really running" tease. **Never narrate a 6-minute boot screen.**
- **Off the table:** no multi-GPU demo (not stress-tested — keep it a Q&A drill-down).

---

## 2. The narrative spine — open on the *need*, not the product

The core change from v1. v1 opened on "GPUs are expensive and ephemeral" — that's a feature,
and anyone could say it. The story that earns the CEO's sentence is the **origin story**:
training a small GPT was guess-and-check, you couldn't answer *why* the model did what it did,
and you couldn't iterate fast enough — so you **built the instrument.** Every feature after the
opener becomes evidence of "need → deliverable," and the **self-verifying diagnostics become
the payoff of that story**, not just a cool feature.

**The spine, one line:** `need → instrument → see why → decide faster → ship it daily.`
**The product loop, one line:** `Launch → Monitor → Diagnose → Compare → Decide.`

---

## 3. Pre-show checklist

- [ ] **Pre-warm a pod 4–5 min before you start** so the dashboard is *already training live*
      when you arrive at it (~minute 7). This is the whole live strategy — do not skip it.
- [ ] **At least one completed historical run** with diagnostics, plus a **pinned baseline**
      (safety net + the Compare beat).
- [ ] `uv run podterm` up; browser at `http://127.0.0.1:8000`. `runpodctl` authenticated;
      `.env` loaded (`RUNPOD_*` + console cookie).
- [ ] **Slides built** (see the Slide Manifest, §9). Three are net-new vs what you have:
      *The need*, *Container craft*, *Self-verifying diagnostics*.
- [ ] **Teams dry-run done:** large fonts, slow pointer, narrate every click, confirm the
      dashboard text is legible when screen-shared. Close noisy apps/notifications.
- [ ] **Q&A drill-downs loaded** (§10) — especially the SentencePiece threading and the CUDA
      symbol-stub mechanics.
- [ ] **SentencePiece fork** slide uses the real headline numbers (10B tokens: >1 hr →
      <10 min; ~50% less memory; repo: `github.com/sysreq/sentencepiece_parallel`). Be ready
      to narrate the *threading internals* (chunk sizing, contention removal) cold in Q&A —
      those specifics are yours and aren't in the README.

---

## 4. Run-of-show (~18 min present, slide-by-slide)

### Beat 1 — The need (0:00–2:00) · `[SLIDE]` *The need*
- `[SAY]` (near-verbatim opener):
  > "Most of training a small GPT is guess and check. You tweak something, wait, stare at a
  > loss curve, and still can't answer the one question that matters: *why* is it doing what
  > it's doing. So I built the instrument I wanted. PodTerm rents a GPU by the minute, runs the
  > training, gives me live X-ray vision into the run, and grades the model's health *off* the
  > training GPU — so I can see *why* and decide what to change next in minutes, not hours."
- `[LIVE, off critical path]` Quietly kick a **fresh launch** here as the "it's really
  running" tease, then move on. If it dies, nobody notices — the pre-warmed pod is the one you
  rely on later.
- `[HOOK]` CEO / non-technical: the need-first framing, "decide faster," visible ownership.

### Beat 2 — The shape (2:00–3:00) · `[SLIDE]` *The loop*
- One slide: **Launch → Monitor → Diagnose → Compare → Decide.** `[SAY]` "That loop is the
  product. Everything I show you is one lap around it."
- Set expectations: "Slides for the how; the live app for the proof. There's a pod training
  right now we'll get to."

### Beat 3 — How it's built (3:00–7:00) · systems depth, bait the drill-downs
Frame: `[SAY]` "PodTerm is deliberately small — a control plane on my laptop. The heavy lifting
runs on the pod, from a sibling repo, `gpt-golf`. Let me show you the parts I'm proud of."

**3a · `[SLIDE]` Architecture — ephemeral pods, durable laptop**
- Pods are **cattle, not pets**: stateless — clone the repo, pull the dataset, train, snapshot,
  **shut themselves down**. Nothing precious lives on the pod.
- All durable state — run config, metrics, diagnostics — lives in **local SQLite** on the
  laptop. A pod can die anytime; you lose nothing but GPU-seconds.

**3b · `[SLIDE]` Container craft — the headline systems beat** *(the C++ proof)*
- **SentencePiece fork — multithreaded tokenization** *(`github.com/sysreq/sentencepiece_parallel`,
  96.7% C++).* `[SAY]` "Tokenizing the training corpus was the bottleneck, so I forked
  SentencePiece to use **multiple cores with smart buffering — a divide-and-conquer over the
  corpus.** It takes a **10-billion-token source from over an hour to under ten minutes, with
  about half the memory.**" Then bait the drill-down: *"the interesting part is sizing the
  chunks to avoid overflow and killing the cross-thread contention — happy to go deep in Q&A."*
  `[HOOK]` C++ / systems — this is your single best hard-systems proof, now with hard numbers.
- **CUDA dead-weight stubbing — ~700 MB+ never shipped.** `[SAY]` "PyTorch declares FFT and
  structured-sparse libs as load-time dependencies, so the dynamic loader *must* map them at
  `import torch` — even though this model never issues an FFT or a 2:4-sparse matmul. You can't
  delete them (ImportError) and you can't empty-stub them (the version check fails). So I
  generate a **16 KB no-op stub that exports exactly the symbols torch references, at their
  version node** — the loader is satisfied and ~500 MB of unused kernels never ship."
  - The craft: stub symbols are **auto-derived from the installed torch libs** (`nm -D
    --undefined-only`), so a torch upgrade self-corrects — no drift. *(scripts/stub_unused_cuda_libs.sh)*
  - Same trick stubs **NCCL for the single-GPU case (~206 MB compressed)**; `bootstrap.sh`
    **restores the real wheel before `torchrun` when NPROC>1**, fail-loud — a stubbed
    collective would silently corrupt a multi-GPU run.
  `[HOOK]` C++ / systems — ELF dynamic linking, version nodes, symbol-exact stubs.
- **Two-stage image build.** `Dockerfile.base` bakes torch/CUDA/flash-attention once (rarely
  rebuilt); the training `Dockerfile` is `FROM base` + a **seconds-long `uv sync`** of small
  pure-Python deps. CUDA libs are **split across layers to pull in parallel** on cold boot.
  `[SAY]` "Bake the slow stuff once; iterate on code in seconds."
- **`[RIGOR BEAT — pulled early, on purpose]`** `[SAY]` "And none of this is guess-work — the
  stub symbols are derived from torch itself, and the multi-GPU restore is fail-loud. I'd
  rather the build break than ship something that silently computes the wrong thing." *(This is
  the antidote to the "builds flashy demos" shadow-read — land it by minute 5, not minute 18.)*

**3c · `[SLIDE]` The wire — how the pod talks to the laptop**
```
gpt-golf pod                      PodTerm (laptop)                    browser
pod_eventd.py  ──JSONL events──►  PodPoller ──► EventPipeline ──► SSEHub ──► Live UI
(stdlib HTTP    over RunPod's      (long-poll)   .drain_loop          │
 daemon :8765)  HTTPS proxy,                          │               └► (live charts/cards)
                bearer token                          ├──► SQLite (durable)
                                                      └──► snapshot ──► off-pod diagnostics
```
- `pod_eventd.py` is a **stdlib-only HTTP daemon** — no dependencies, so it boots *before*
  anything is installed. Serves structured **JSONL events**, the **raw log**, **model
  snapshots**, and a tiny **`/snapshot/ack`** handshake (Beat 5).
- Reached over **RunPod's HTTPS proxy** (`{pod_id}-8765.proxy.runpod.net`), authed by a
  **per-run bearer token** minted at launch.
- The **event schema is a contract**: line-delimited `{t, ts, ...}` — `phase`, `metric`,
  `gpu`, `snapshot`, `summary`. Two phase strings are **load-bearing**: `"Starting Training"`
  and `"Training finished"` drive run start/finalization on both ends.
- `[HOOK]` Python / platform — stdlib daemon, event pipeline, SSE fan-out, clean module DAG.
- `[Q&A]` service-singleton DAG (`SSEHub ← EventPipeline ← PodManager`); ordered shutdown
  (stop producers → drain → close DB); idempotent run finalization; "pod gone" needs 2
  confirmations so a transient CLI blip never kills a live run; metrics batching with bounded
  retry; SSE backpressure; secret redaction; the boot panel (parsing RunPod machine logs into
  per-layer image-pull progress via a short-lived console token, not the API key).

### Beat 4 — The money shot: live dashboard (7:00–12:00) · `[LIVE]` the pre-warmed pod
**Switch to the app. The pre-warmed pod is training now — numbers are moving.**
- **4a · The race bar (your proudest — the delight beat).** `[SAY]` "This is the feature I'm
  proudest of." The horizontal **race bar** predicts your run's **finish quality vs a pinned
  baseline at the same time budget** and calls it live — **ahead / behind / too close to
  call** — with markers for *projected*, *baseline*, and (if behind) the *win* point. Watch it
  shift as the loss settles; the hero card **glows green/red**. All **derived client-side**
  from streamed metrics (EMA step-time, ETA, throughput, projected finish loss) — no
  round-trips. `[HOOK]` physics/PhD: baseline racing = a controlled comparison.
- **4b · KPI cards.** Six at-a-glance, fixed slots so streaming values never jiggle the
  layout: **Projected Finish** (+sparkline), **Baseline** (hero), **Loss (train)**, **GPU**
  (util+mem), **Cost So Far** (the visceral one), **Model Health** (verdict → Health tab).
  Plus **Validation BPB** and **System** (CPU/RAM).
- **4c · Charts + logs.** Plotly **training-loss chart** — your run, baseline overlaid
  (dashed), eval points (diamonds); custom y-scaling for detail near the floor. **Live log
  panel** over SSE (filter, pause/resume). `[SAY]` "Metrics, logs, cost, the race verdict —
  all updating right now, live."
- **4d · The line that lands.** `[SAY]` "Cost is ticking in real time, and when the run ends
  the pod stops *itself* — I only pay for these few minutes." `[HOOK]` CEO.

### Beat 5 — Diagnose: the payoff of "see why" (12:00–16:00) · `[LIVE]` + `[SLIDE]`
This is the resolution of the origin story. Frame the **why**, then show it, then the keystone.
- **5a · Why off-pod.** `[SAY]` "Every GPU-second should train. So I don't grade the model on
  the pod — PodTerm **downloads each snapshot and runs the diagnostics locally**, as a
  subprocess under gpt-golf's torch env (the architecture changes constantly, so it's
  single-sourced there)." The **snapshot handshake**: pod saves a checkpoint → emits a
  `snapshot` event → PodTerm pulls it, runs diagnostics → on the final one calls
  `/snapshot/ack` so the pod knows it's safe to stop (bootstrap waits ~120 s).
- **5b · `[LIVE]` Model Health tab.** The **verdict card** (OK / WARN / ERROR) in plain
  language, top issues, and a **clickable snapshot timeline** (health at each checkpoint).
  **Deep dive:** per-section drill-down with **threshold bands and a one-line "why" for every
  metric** — `[SAY]` "and the thresholds and rationale come from a backend metric registry,
  the UI hardcodes nothing." Plain-English tour of *what* it measures (don't go deep on ML):
  is capacity used (dead neurons, redundant heads), does signal flow cleanly through layers,
  are gradients healthy, is attention focused or smeared.
- **5c · `[SLIDE]` The keystone — self-verifying diagnostics.** `[SAY]` "Here's the part I
  care about most: the diagnostics **check their own arithmetic.** The off-pod forward pass
  **recomposes each layer's output from its parts**, **re-derives the loss**, and **reconciles
  its bits-per-byte against the on-pod eval — across different hardware.** Anything it can't
  verify is flagged, never silently trusted. Most tools assume they're right. This one
  *proves* it." `[HOOK]` physics/PhD (rigor, cross-hardware reconciliation) + the whole panel
  (this is your headline rigor beat *and* the origin-story payoff in one moment).
- `[Q&A]` the runtime "anatomy probe" that discovers model structure instead of hardcoding it;
  the versioned, additive-only JSON schema; per-pod serialization + coalescing of snapshot jobs.

### Beat 6 — Compare + close (16:00–18:00) · `[LIVE]` + `[SLIDE]` *Close*
- **`[LIVE]` Compare.** History → select two runs → Compare. `[SAY]` "This is what the whole
  thing is *for*: did my change help?" Overlaid loss curves (aligned by step or % progress), a
  **final-loss regression banner**, and a **model-health diff** — verdict + headline metrics
  side by side, which internal blocks moved.
- **`[SLIDE]` Close on the CEO sentence.** `[SAY]` "This started because I had a need I
  couldn't meet. This is what it looks like when I turn that into a service I use every day."
  The fresh pod from the top has likely finished and **stopped itself** — total bill, a few
  minutes. Hand to Q&A.

### Beat 7 — Q&A (18:00–30:00)
Drive it with the **hook map** (§8) and the **drill-down reserve** (§10). When a C++ person
engages, go to SentencePiece threading or the CUDA symbol stubs; for platform folks, the
lifecycle/saga and SSE; for the PhD, the BPB reconciliation.

---

## 5. Timing cheat-sheet (tape to the monitor)

| Min | Beat | Mode | If you're behind, cut to… |
|---|---|---|---|
| 0–2 | The need | SLIDE | the opener line + "I built the instrument" |
| 2–3 | The loop | SLIDE | one breath, keep moving |
| 3–7 | How it's built | SLIDE | SentencePiece + CUDA stub headline; skip layer detail |
| 7–12 | Live dashboard | LIVE | race bar + cost ticking |
| 12–16 | Diagnose | LIVE+SLIDE | the self-verifying keystone line |
| 16–18 | Compare + close | LIVE+SLIDE | the CEO sentence |
| 18–30 | Q&A | — | — |

---

## 6. Rigor: keep it visible from minute five

At a correctness-obsessed scientific shop, "builds cool shit" has a shadow reading: "builds
flashy demos that fall over in production." The antidote can't live only at the end. Plant
rigor **three times, escalating**:
1. **Minute ~5** (Beat 3b): symbols auto-derived from torch; multi-GPU restore is fail-loud.
2. **Minute ~13** (Beat 5b): thresholds come from a backend registry; the UI hardcodes nothing.
3. **Minute ~14** (Beat 5c): the self-verifying math + cross-hardware BPB reconciliation.
Then the quick credibility coda in Q&A or the close: real tests + CI (below).

**Credibility coda (have it ready):** Python `pytest` — health verdicts, compare diffs, schema,
a **golden end-to-end diagnostics test**, db/pipeline/poller/routes, secret redaction; frontend
`node --test` for the derived-metrics + diagnostics logic; **CI runs all three on every push.**
Honest what's-next: a static type checker, a coverage floor + failure-injection tests around
the thread/SQLite/subprocess seams, and dependency/vuln scanning.

---

## 7. Backup plan (if the live element misbehaves)

- **Pre-warmed pod is slow / wedged:** don't wait on it. Pivot Beats 4–6 to a **completed
  historical run** — dashboard, race bar (vs baseline), Model Health, and Compare all work
  fully from SQLite. You only lose "watch it move," so *describe* that one moment.
- **Diagnostics not ready:** open a *previous* run's Model Health tab — verdict timeline and
  deep-dive are fully populated from stored data.
- **Teams share looks bad:** fall back to the slides for the wire/architecture; keep the app
  full-screen and the pointer slow when you do go live.
- **Fresh top-of-show launch died:** ignore it — it was always off the critical path.

---

## 8. Audience hook map (who leans forward where)

| Archetype | Their moment |
|---|---|
| C++ / systems | SentencePiece fork (threading, divide-and-conquer, contention); CUDA symbol-stub (DT_NEEDED, version nodes, ~700 MB+ saved); compile cache (graph → C++ → CUDA binary, shared across pods) |
| Python / platform | stdlib-only daemon; event pipeline + SSE; clean module DAG; lifecycle/saga + ordered shutdown |
| Physics / MATLAB PhD | self-verifying diagnostics; BPB reconciliation across hardware; baseline racing as a controlled comparison |
| Non-technical / CEO | the need-first story; cost ticking; "decide faster"; visible ownership + quality |

---

## 9. Slide Manifest (build list for the deck)

> ~9 content slides for ~18 min. ★ = net-new vs v1 material. Each slide should carry **one
> message** and at most a few words — the script (§4) is the voice-over.

| # | Slide | One message | Visual to build | Source |
|---|---|---|---|---|
| 1 ★ | **The need** | "Guess-and-check → I built the instrument." | One evocative image (a loss curve + a question mark), the opener line | §2, §4 Beat 1 |
| 2 | **The loop** | "This loop is the product." | `Launch → Monitor → Diagnose → Compare → Decide` ring | §4 Beat 2 |
| 3 | **Architecture** | "Cattle pods, durable laptop." | Pod (ephemeral) ↔ laptop (SQLite) split diagram | §4 Beat 3a |
| 4 ★ | **Container craft** | "I shave the build to the bone." | SentencePiece fork (10B tokens: >1 hr → <10 min, ~50% less memory) + CUDA symbol-stub (~700 MB+); byte callouts | §4 Beat 3b |
| 5 | **The wire** | "Structured events, stdlib daemon, live to the browser." | The ASCII pipeline diagram, cleaned up | §4 Beat 3c |
| 6 | **(transition)** | "Let's watch one, live." | Minimal — cue to switch to the app | §4 Beat 4 |
| 7 ★ | **Self-verifying diagnostics** | "It checks its own arithmetic." | Layer recompose → loss re-derive → BPB reconcile, as a 3-step visual | §4 Beat 5c |
| 8 | **Compare** | "Did my change help?" | Two runs overlaid + a regression banner mock | §4 Beat 6 |
| 9 | **Close** | "Need → a service I use every day." | The CEO sentence; optional cost/"stopped itself" stat | §4 Beat 6 |

(You already have the architecture/wire material from v1; slides 1, 4, and 7 are the real build.)

---

## 10. Q&A drill-down reserve (loaded answers)

- **SentencePiece threading** *(`github.com/sysreq/sentencepiece_parallel`):* the corpus
  divide-and-conquer (how chunks are sized to avoid overflow), how cross-thread contention was
  removed, the smart buffering, and *why* it lands at >1 hr → <10 min and ~50% memory on a 10B
  corpus. This is your strongest C++ story — know the threading model cold (the README has the
  headline numbers but not the internals; those are yours to narrate).
- **CUDA symbol stubs:** why DT_NEEDED forces the map even under lazy binding; how empty stubs
  fail the version check; deriving exact symbols + version node from torch (`nm -D`); why
  `cusparse` is left intact (sparse-gradient paths can reach it); the fail-loud NCCL restore
  for multi-GPU. *(scripts/stub_unused_cuda_libs.sh, bootstrap.sh)*
- **Backend robustness:** service-singleton DAG; ordered shutdown; idempotent finalization;
  2-confirmation "pod gone"; metrics batching + bounded retry; SSE backpressure; redaction.
- **Diagnostics internals:** the runtime anatomy probe vs hardcoded structure; the versioned
  additive-only schema; per-pod serialization + coalescing.
- **Multi-GPU (asked, not demoed):** NCCL stub-and-restore; `torchrun --standalone`; why you
  haven't stress-tested it yet (honest).
- **Tests/CI:** the credibility coda in §6.

---

## 11. Appendices

### A — "Wow" moments to make sure you land
1. **Self-verifying diagnostics** — checks its own loss/branch math, reconciles BPB across
   hardware. *(the keystone)*
2. **The race bar** — predicting ahead/behind vs a baseline at the same budget, live.
3. **Container craft** — SentencePiece fork (10B tokens >1 hr → <10 min, ~50% less memory) +
   symbol-exact CUDA stubs (~700 MB+ saved).
4. **Cost ticking in real time** and the pod **auto-stopping** itself.

### B — Key files cheat sheet (for Q&A)
| Topic | Where |
|---|---|
| Pitch / overview | `README.md`, `CLAUDE.md` |
| Event pipeline | `podterm/pipeline.py` (`EventPipeline`), `podterm/sse.py` (`SSEHub`), `podterm/eventing/` (`PodPoller`) |
| RunPod control | `podterm/runpod/` (`cli.py` CLI; `api.py`/`console.py` read-only API + JWT) |
| Boot panel | `podterm/boot.py` (`PullTracker`), `static/js/live/boot.js` |
| Live dashboard | `static/js/live.js`, `live/race.js`, `live/kpis/`, `static/js/derive/` |
| Diagnostics suite | `podterm/diagnostics/` (`runner.py`, `health.py`, `stages_forward.py`), `podterm/snapshots.py` |
| Diagnostics UI | `static/js/diagnostics/`, `static/js/compare.js` |
| Pod infra (gpt-golf) | `Dockerfile*`, `scripts/bootstrap.sh`, `scripts/pod_eventd.py`, `scripts/stub_unused_cuda_libs.sh`, `data/cached_challenge_fineweb.py` |
| Event/snapshot contract | gpt-golf `train_gpt.py` (`emit_event`, `save_snapshot`) ↔ `podterm/models.py` |
| SentencePiece fork (C++) | `github.com/sysreq/sentencepiece_parallel` — parallel tokenization (10B: >1 hr → <10 min, ~50% memory) |

### C — Glossary (say once, for the non-deep audience)
- **Pod** — a rented cloud GPU machine, billed by the minute, thrown away after the run.
- **Snapshot** — a saved copy of the model partway through training.
- **BPB (bits per byte)** — the quality score; lower is better, comparable across tokenizers.
- **SSE (Server-Sent Events)** — one-way live stream from server to browser; powers the dashboard.
- **Baseline** — a previous run you're racing the current one against.
- **DT_NEEDED** — an ELF shared-library dependency the dynamic loader must resolve at load time.

---

## 12. What changed v1 → v2 → v3

- **Audience re-aimed** from general meetup to the **Stellar Science hiring panel**; every beat
  now serves one of the three "sentences you're earning."
- **Opener flipped to need-first** (origin story), and **diagnostics elevated to the
  centerpiece** as that story's payoff.
- **Tightened to ~18 min present** to protect ~12 min of Q&A.
- **Live boot no longer narrated** — a **pre-warmed pod** carries the live moment; the fresh
  launch is off the critical path.
- **SentencePiece fork promoted** from absent (v1) to a **headline C++ beat**, backed by real
  numbers from `github.com/sysreq/sentencepiece_parallel` (10B tokens: >1 hr → <10 min, ~50%
  less memory; 96.7% C++).
- **CUDA stubbing upgraded** from a one-line aside to a full systems beat with **accurate
  mechanics and byte numbers** (~500 MB FFT/sparse + ~206 MB NCCL; symbol-exact, self-correcting).
- **Rigor pulled early** and planted three times instead of living only at the end.
- **Restructured slide-by-slide** with a **Slide Manifest** and **timing cheat-sheet** so it's
  directly PowerPoint-ready.
