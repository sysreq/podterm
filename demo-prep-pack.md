# GPT Caddy Demo Prep Pack (Stellar Science)

> Supporting material for `live-demo-v2.md`: the spoken opener, the three missing
> slides, and a Q&A drill-down cheat sheet. Built to earn the CEO's sentence
> ("turns a product need into a deliverable service") in front of a mixed,
> partly-remote panel.

---

## 1. The 90-second opener (say it close to verbatim)

> "Quick bit of context on why this exists, because it's the whole point.
>
> I've been training small GPT models. And most of that work, honestly, is guess
> and check. You change a hyperparameter, or the architecture, or the data mix,
> you kick off a run, you wait, and then you stare at a loss curve. And the loss
> curve tells you *what* happened. It almost never tells you *why*. Are the
> gradients healthy? Is the model actually using its capacity, or are half the
> attention heads doing the same thing? Is signal flowing cleanly through the
> layers, or dying somewhere in the middle? You can't see any of that from a loss
> curve. So you end up changing one thing at a time and hoping.
>
> That's slow. And in any project, the thing that actually compounds is how fast
> you can see what's going on and decide what to do next.
>
> So I built the instrument I wanted. This is GPT Caddy. It rents a GPU by the
> minute, runs the training on it, and gives me live X-ray vision into the run
> while it happens. Then it grades the model's health off the training GPU, so the
> training never slows down, and it tells me *why* I'm seeing what I'm seeing.
> Decisions that used to take a day of guess and check now take a few minutes.
>
> I'll walk you through it the way I actually use it: launch a run, watch it live,
> diagnose its health, compare against a baseline, decide what's next. Let me kick
> off a real run right now so it's training by the time we get to it, and while it
> boots, I'll show you how the whole thing is put together."

**Delivery notes:**
- **Slow down on "it almost never tells you *why*."** That line is the thesis.
- The list of hidden questions (gradients, capacity, signal flow) is a deliberate
  preview of the diagnostics section. You're planting a seed you pay off later.
- "So I built the instrument I wanted" is the ownership beat. Land it plainly, no
  apology, no hedging.
- "A day... to a few minutes" is the deliverable-value beat. That's the CEO line.
- The last sentence is your **live-launch action point.** Actually click launch
  here, then move to slides. The pod is now off the critical path.
- Total: ~100 seconds at a calm pace. Do not rush it to save time. This is the
  most important 100 seconds of the talk.

---

## 2. The three slides to build

Keep slides sparse. The words below in `on-slide` are the *only* text on the
slide. Everything else is in your mouth, captured in `say`.

### Slide 1 — The Need

- **Title:** `Loss tells you *what*. Not *why*.`
- **Visual:** a single loss curve going down, with a big "?" over it. Optionally,
  to the side, four greyed labels: *gradients · capacity · signal flow · attention*.
- **on-slide:**
  - Change one thing. Wait. Squint at a curve. Repeat.
  - You never see *why*.
  - So I built the instrument.
- **say:** This is the opener slide. Deliver section 1 over it.

### Slide 2 — Engineering the runtime (the container craft)

- **Title:** `Making the GPU pay for training, not for booting`
- **Visual:** two bars side by side: `NVIDIA PyTorch ~10GB` vs `GPT Caddy <2GB`.
- **on-slide:**
  - Forked **SentencePiece** for multithreaded tokenization (corpus
    divide-and-conquer; no overflow, no cross-thread contention).
  - Stubbed hardlinked CUDA modules PyTorch never calls: **~1GB gone.**
  - CUDA libs split across layers: **parallel cold-boot pull.**
  - Bake the slow stuff once. Rebuild code in seconds.
- **say:** "The official NVIDIA PyTorch image is about 10GB, and on an ephemeral
  pod the cold-boot pull is dead time you're paying for. Two pieces I'm proud of:
  I forked SentencePiece to tokenize FineWeb in parallel so I could pre-build the
  datasets offline, and I stubbed out the CUDA kernels PyTorch never actually calls
  to cut about a gig of dead weight. I'm happy to go deep on either in Q&A." Then
  move on. This slide is **bait**, not a lecture.

### Slide 3 — The diagnostics check their own arithmetic

- **Title:** `Grading the model off the training GPU, and proving the grade`
- **Visual:** snapshot -> (off-pod) diagnostics -> health verdict, with a small
  loop arrow labeled "re-derives loss, reconciles BPB."
- **on-slide:**
  - Every GPU-second trains. Diagnostics run off-pod, on each snapshot.
  - Checks: capacity (dead neurons, redundant heads), signal flow, gradient
    health, attention focus.
  - The forward pass **recomposes each layer from its parts and re-derives the
    loss**, cross-checking the model's own numbers.
  - **Reconciles bits-per-byte against the on-pod eval** across different hardware.
  - Most tools assume they're right. This one proves it.
- **say:** This is the keystone. Slow down. The recompose-and-re-derive line is
  the one that makes a numerical-methods person sit up. End on "proves it."

---

## 3. Q&A drill-down cheat sheet (60 seconds each)

For each likely question, a crisp answer ready to go. Answer, then stop. Let them
pull the next thread.

### C++ / systems

**"Tell me about the SentencePiece fork."**
Single-threaded tokenization was the bottleneck for prepping ~20 pre-tokenized
FineWeb-10B variants. I forked SentencePiece to add multithreaded tokenization
using a corpus divide-and-conquer: partition the corpus into chunks, tokenize them
in parallel, merge the results. The two real problems were memory (naive chunking
overflows) and avoiding cross-thread contention on shared state, so the
partitioning is designed so threads don't fight over the same structures. Net: the
dataset prep stopped being a bottleneck and I could generate variants offline.

**"How did you get the image from ~10GB to under 2GB?"**
The official image ships everything. I profiled what PyTorch actually loads at
runtime. The unused CUDA libraries (FFT, sparse) are hardlinked in, so I stubbed
them: kept the symbols so nothing breaks at import, dropped the payload. That alone
was ~1GB. Then I split the remaining CUDA libraries across Docker layers so they
pull in parallel on cold boot, and layered the image so torch and CUDA are a stable
base and my code is a thin top layer. Code changes rebuild in seconds because the
heavy layer never moves.

**"What's the Redis service for?"**
`torch.compile` is slow the first time: it lowers the graph to C++ and then to a
CUDA binary. A shared Redis pod caches the compiled graphs across all training
pods, so the first pod warms the cache and the rest hit roughly 95%. You pay the
compile cost once per architecture, not once per run.

### Python / platform

**"Walk me through how the pod talks to the laptop."**
The pod runs a stdlib-only HTTP daemon, no dependencies, so it can start before
anything is installed. It serves line-delimited JSON events, the raw log, and model
snapshots. GPT Caddy reaches it over RunPod's HTTPS proxy with a per-run bearer
token, long-polls the events, drains them through a single pipeline, fans them out
to the browser over Server-Sent Events, and persists to SQLite as they arrive. The
event schema is a contract between the two repos.

**"Why stdlib-only for the daemon?"**
It has to come up on a bare pod before the environment is installed, so zero
dependencies. It also keeps the wire contract small and easy to reason about.

**"What happens when a pod dies mid-run?"**
Nothing precious lives on the pod. All durable state, config, metrics, diagnostics,
is in local SQLite on the laptop. The pod is cattle: it clones, trains, snapshots,
and shuts itself down. A "pod gone" signal takes two confirmations before I
finalize a run, so a transient CLI or network blip doesn't kill a live run.

### Physics / numerical methods

**"How can you trust a health metric computed on different hardware than training?"**
That was the exact worry, so the diagnostics don't just trust themselves. The
off-pod bits-per-byte replicates the on-pod eval's exact byte accounting, then
reconciles against the number the pod reported, within a tolerance. If they
disagree it gets flagged. And the forward pass recomposes each layer's output from
its parts and re-derives the loss, cross-checking the model's own numbers. Anything
it can't verify, it flags rather than silently trusts.

**"What does 'model health' actually measure?"**
Four families: capacity utilization (dead neurons, redundant attention heads),
signal flow through the layers, gradient health (exploding or vanishing), and
attention focus via entropy. The thresholds and the one-line rationale for each
come from a metric registry in the backend. The UI hardcodes no numbers, it just
colors what the backend says.

### General / CEO / leadership

**"What would you do differently, or what's next?"**
I know exactly where the edges are. Next steps are a static type checker with a
documented strictness target, a coverage floor plus failure-injection tests around
the thread, SQLite, and subprocess seams, and dependency and vulnerability scanning
in CI. None of it is mysterious, it's the difference between a sharp single-user
tool and something I'd hand to a team.

**"Is this production-ready?"**
It's a single-user local tool by design. It binds to localhost and has no auth
because there's no multi-tenant surface to protect. The rigor I cared about is in
data integrity and diagnostics correctness, not in hardening a network service it
was never meant to be. If the goal changed to multi-user, I can tell you exactly
what would have to change.

### The likely landmine (be ready)

The deepest engineers may probe the **concurrency and data-integrity boundary**:
the shared SQLite connection across threads, the single event-drain loop, metric
batching and retry, idempotent run finalization. If someone goes there, do **not**
get defensive. The mature answer is "good eye, that boundary is the hardest part of
this and I went back and forth on it," then walk them through the tradeoff you
landed on. Showing you *know* it's the hard part signals more seniority than
pretending it was always clean.
