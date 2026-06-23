# GPT Caddy Interview Demo — Run-of-Show v2 (Stellar Science)

> Companion to `live-demo.md`. This version re-aims the demo at a hiring panel
> instead of a general meetup, and de-risks the live element for a Teams audience.

## The target (read this before building slides)

- **Audience:** Stellar Science panel. Mixed and unpredictable: some C++, some
  Python, possibly a physics/MATLAB PhD, some non-technical, **CEO may attend**,
  **some joining on Teams.**
- **Format:** 30 minutes *including Q&A*. Budget **~18 min present + ~12 min Q&A.**
  (Your v1 was 20 min of demo *before* Q&A. That is over budget. Cut accordingly.)
- **The sentence you are earning:**
  - **CEO:** "This guy turns a product need into a deliverable service."
  - **Everyone:** "This is a builder who owns quality. Not a paycheck-collector."
  - **The C++ folks:** "He can reach into a hard systems problem and solve it."
- **Structure (decided):** slides carry the narrative; one **pre-warmed** pod
  gives a real live moment; a fresh launch is kicked at the top but kept **off the
  critical path.** No narrating a 6-minute boot screen.

## The core change from v1

**Open on the need, not the product.** v1 opens with "GPUs are expensive and
ephemeral." That is a feature, and anyone could say it. Your real story is the one
that earns the CEO's sentence: GPT Golf training was guess-and-check, you could not
answer *why* the model did what it did, and you could not iterate fast enough. So
you built the instrument. Every feature after the opener is then evidence of
"need to deliverable," and the **self-verifying diagnostics become the payoff of
the origin story**, not just a feature.

### Need-first opener (say it close to verbatim)

> "Most of training a small GPT is guess and check. You tweak something, wait,
> stare at a loss curve, and still cannot answer the one question that matters:
> why is it doing what it is doing. So I built the instrument I wanted. GPT Caddy
> rents a GPU by the minute, runs the training, gives me live X-ray vision into
> the run, and grades the model's health off the training GPU, so I can see *why*
> and decide what to change next in minutes, not hours."

## Pre-show checklist (Approach A depends on this)

- [ ] **Pre-warm a pod 4-5 min before you start** so the dashboard is already
      training live when you reach it (~minute 8). This is the whole point of A.
- [ ] At least **one completed historical run** with diagnostics, plus a **pinned
      baseline** (safety net + the Compare beat).
- [ ] `uv run podterm` up; browser at `http://127.0.0.1:8000`. `runpodctl`
      authenticated; `.env` loaded.
- [ ] **Slides built** (see "Slides to build" below).
- [ ] **Teams dry-run done:** big fonts, slow pointer, narrate what you click,
      confirm the dashboard text is legible when shared.
- [ ] Drill-down notes loaded for Q&A (see hook map).
- [ ] Do **not** demo multi-GPU (not stress-tested, by your call).

## Run-of-show (~18 min)

1. **(0:00-2:00) The need.** The need-first opener above. The guess-and-check
   pain, "why are we seeing what we are seeing," decide and iterate faster. End on
   "so I built the instrument I wanted."
   *Quietly kick a fresh live launch here as the "it is really running" tease,
   then move on. It is off the critical path. If it dies, nobody notices.*
2. **(2:00-3:00) The shape.** One slide: **Launch -> Monitor -> Diagnose ->
   Compare -> Decide.** "That loop is the product."
3. **(3:00-7:00) How it is built (systems depth, bait the drill-downs).**
   Architecture slide. Ephemeral pods as cattle; durable state lives on the laptop.
   Then the **container craft as the headline:**
   - **SentencePiece fork** for multithreaded tokenization, with a corpus
     divide-and-conquer to avoid overflow and cross-thread contention. *(This is
     your single best C++ proof. It was missing from v1. Two sentences here, then:
     "happy to go deep on the threading in Q&A.")*
   - **Stubbing the hardlinked CUDA modules PyTorch never calls, to cut ~1GB.**
   - The wire: a **stdlib-only event daemon** (boots before anything is
     installed), event schema as a contract, SSE fan-out. Keep it crisp.
4. **(7:00-12:00) Live dashboard (the pre-warmed pod, moving now).** The money
   shot. The **race bar** (your proudest, the delight beat). KPI cards. **Cost
   ticking in real time.** "All of this is updating live, right now."
5. **(12:00-16:00) Diagnose: the payoff of 'see why.'** Why off-pod (every
   GPU-second should train). Model Health verdict + deep dive. **The keystone:
   self-verifying diagnostics.** "The diagnostics check their own arithmetic" — the
   forward pass recomposes each layer from its parts, re-derives the loss, and
   reconciles bits-per-byte against the on-pod eval across different hardware. This
   is your rigor beat, your physicist hook, and the resolution of the origin story,
   all in one moment.
6. **(16:00-18:00) Compare + close.** Compare view: "did my change help?" Then
   close explicitly on the CEO sentence: *"This started because I had a need I
   could not meet. This is what it looks like when I turn that into a service I use
   every day."* The live pod has likely finished and stopped itself: total bill, a
   few minutes.
7. **(18:00-30:00) Q&A** with loaded drill-downs.

## Audience hook map (who leans forward where)

| Archetype | Their moment |
|---|---|
| C++ / systems | SentencePiece fork (threading, contention), CUDA stubbing, compile cache (graph -> C++ -> CUDA binary, shared across pods) |
| Python / platform | stdlib-only daemon, event pipeline + SSE, clean module DAG, the saga/lifecycle thinking |
| Physics / MATLAB PhD | self-verifying diagnostics, BPB reconciliation across hardware, baseline racing as controlled comparison |
| Non-technical / CEO | the need-first story, cost ticking, "decide faster," visible ownership and quality |

## Rigor watch-out (the one risk in "builds cool shit")

At a correctness-obsessed scientific shop, "builds cool shit" has a shadow
reading: "builds flashy demos that fall over in production." Your antidote (the
self-verifying math, the tests, CI, the honest what-is-next list) currently lives
at the very end of v1. **Pull at least one unmistakable rigor beat early** (the
self-verifying line, or "the thresholds come from a backend metric registry, the UI
hardcodes nothing") so the panel reads "builds cool shit AND it is solid" by minute
five, not minute eighteen.

## Slides to build (the gap)

1. **The need** (origin story, one image, your opener).
2. **The container craft:** SentencePiece fork + CUDA stubbing, with the byte
   numbers.
3. **Self-verifying diagnostics:** what "checks its own arithmetic" means, visually.

(You already have the architecture/wire diagram material in v1 §3b.)

## Deltas from v1

- Open on the **need**, not GPU cost.
- **Pre-warm** the pod; live launch off the critical path; no 6-minute boot
  narration.
- **SentencePiece fork promoted** from absent to a headline C++ beat.
- **Diagnostics elevated** to the centerpiece (payoff of the origin story).
- **Rigor beat pulled early.**
- Tightened to **~18 min** to protect Q&A.
