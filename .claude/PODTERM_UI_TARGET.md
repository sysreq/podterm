# PODTERM Live View: UI Target Spec

Goal: iterate the PODTERM web UI until the Live view matches the target mockup at
`docs/caddy-target.png`, with every displayed number derived from real run data.
No hardcoded metric values anywhere.

## Phase 0: Discovery (do this first, report before changing code)

- Explore the repo. Identify the frontend framework, component structure, styling
  approach, build tooling, and the dev server command.
- Identify the existing live-metrics data source (websocket, polling endpoint, or
  log tail) and document the current record schema. The current backend already
  emits at least: step, loss, bpb, ms/step, samples/s, lr, grad_norm,
  attn_entropy (per layer), logit saturation, dataloader wait, per-position loss,
  val_bpb, and sample completions at eval time.
- Do not break the backend contract. If a metric below is missing, add it to the
  emitter and the UI together, keeping old fields intact.

## Mock data mode (build this early; all UI work depends on it)

- Add a `PODTERM_MOCK=1` mode that replays a recorded run log through the exact
  same code path the live UI uses, at adjustable speed (1x default).
- All screenshots and iteration below run against mock mode. No GPU pod required.
- This doubles as the demo fallback: a replayed run must be indistinguishable
  from a live one in the UI.

## Iteration protocol

- Work in vertical slices, in this order:
  1. App shell, top nav, left sidebar
  2. KPI card row
  3. Race status banner + baseline row
  4. Charts
  5. Diagnostics sparkline row
  6. Logs panel + Config/Repro panel
  7. Polish pass (spacing, type scale, states)
- After each slice: start the dev server, capture a full-page Playwright
  screenshot at 1920x1080, place it side by side with docs/caddy-target.png,
  list the concrete visual diffs, fix them, re-screenshot. Repeat until the
  slice matches the target's layout and hierarchy.
- Commit after each slice with a descriptive message.

## Layout spec (top to bottom)

### Top nav
- Left: PODTERM logo. Tabs: Live (active), History, Compare, Machines.
- Right: cluster status pill (green dot + cluster name), settings gear, avatar.

### Left sidebar
- ACTIONS: Launch Run (primary), Stop Run (danger), Refresh.
- ACTIVE RUNS list, one card per run: run name, status badge (Ahead = green,
  Behind = red, Running = neutral for runs with no baseline), Step current/total,
  BPB, pace line such as "-7.4ms (0.2s ahead)" in green or "+23.1ms (1.1s
  behind)" in red. Clicking a card switches the Live view to that run.
- Footer: "N pods, M queued" plus a systems status line. These counts MUST be
  derived from the same source as the active runs list. Three active runs can
  never coexist with "0 pods".

### KPI card row (8 cards)
1. Projected Finish: remaining time + wall-clock ETA + small sparkline of recent
   ms/step. ETA = remaining_steps * EMA_50(ms_per_step).
2. Ahead/Behind (hero card): instantaneous per-step delta vs baseline (ms) AND
   cumulative seconds ahead/behind. cumulative = baseline_elapsed_at(step) -
   current_elapsed. Green border + "on pace to WIN" when ahead; red border +
   "falling behind" when behind. These are independent quantities; show both.
3. Avg ms/step: EMA_100 of step time, plus "Target to beat: <= X ms" where
   X = (baseline_total_time - current_elapsed) / remaining_steps, recomputed
   live. This is the required pace, not a static number.
4. Loss (train): current value + delta vs 100 steps ago (decrease renders green).
5. BPB (train): same treatment as loss.
6. Cost So Far: elapsed_hours * hourly_rate, plus projected total =
   (elapsed + ETA) in hours * hourly_rate. Hourly rate comes from pod metadata.
7. GPU Utilization: percent + horizontal bar.
8. Memory (GPU): used / total GB + bar. Total comes from the device query and
   must match the GPU named in the Config panel.

### Race status banner
- One-line status ("You're ahead of baseline by Xs") + a short secondary line,
  and a run-progress bar with two markers: projected finish and baseline target.
- The projected-finish-vs-baseline statement must agree with the math in cards
  2 and 3. If current pace beats required pace, the projected margin at finish
  is (required_ms - current_ms) * remaining_steps, not the current cumulative
  lead. Do not conflate the two.

### Baseline row
- Baseline dropdown listing prior completed runs as "name (branch | BPB x.xxxx)".
  The current run is excluded from the list; a run can never be its own baseline.
- "To beat baseline: avg ms/step <= X" (same X as card 3).
- View Full Log button.

### Charts (two, side by side, shared x-axis domain)
- Training Loss: current run solid, baseline dashed, log-scale y with clean
  1-2-5 style ticks (no duplicate labels), x = steps. Vertical event annotations
  with diamond markers and labels: LR decay points, evals, checkpoint saves.
  Hover tooltip shows exact values for both series.
- Step Time / Throughput: dual y-axis (ms left, samples/s right), same event
  annotations.
- Legibility floor for projection: tick labels >= 12px, series line width >= 2px.

### Diagnostics sparkline row (5 cards)
- Attention Entropy (bits, mean across layers), Grad Norm (L2), Logit
  Saturation (%), Dataloader Wait (%), Validation BPB.
- Each card: label + info tooltip explaining the metric in one sentence, big
  current value, delta arrow vs the previous eval window, ~100-point sparkline.
- Spelling is "Dataloader". The mockup contains a typo ("Datalooder"); do not
  copy it.
- Per-metric warn thresholds live in config. A card turns amber and a WARN log
  line fires only when the threshold is actually crossed. Never emit a WARN for
  a value inside the safe range.

### Bottom row
- Logs panel (left, wider): tabs All / Warnings / Errors / Metrics, free-text
  filter, pause-autoscroll toggle, monospace font, level coloring (INFO neutral,
  METRIC cyan, WARN amber, ERROR red), most recent step line highlighted.
  Streams from the same source as the metric cards.
- Config / Repro panel (right): Git commit (short hash + branch + copy button),
  GPU name (from device query), Docker image, Config hash (+ copy), Seed,
  Sequence Length, Batch Size, "View Full Config (YAML)" link. All values come
  from run metadata.

## Cross-cutting correctness rules

- Single derivation module: ETA, cost, required pace, ahead/behind, and the race
  banner all read from one tested selector/computation module. Unit test it.
  Sanity cases the tests must cover:
  - 16,883 steps remaining at 192.6 ms/step -> ETA ~54 min.
  - $3.29/hr with ~10 min elapsed -> cost so far ~$0.55.
  - current 192.6 ms vs required 199.9 ms over 16,883 remaining steps ->
    projected finish margin ~123 s.
- GPU name, memory total, and hourly rate share one source of truth (pod
  metadata). No card may disagree with the Config panel.
- Every panel has explicit loading, empty, stopped, and pod-lost states. Errors
  state what happened and what to do next, no vague text, no apologies. Nothing
  renders as "-" or NaN on the happy path.
- Copy consistency: an action keeps the same name through its whole flow
  (a "Launch Run" button leads to "Run launched", not "Job submitted").
- Dark theme via design tokens (CSS variables), consistent 8px spacing grid,
  no horizontal scroll at 1456px or 1920px widths, no layout shift while
  metrics stream (reserve fixed card heights).

## Definition of done

- Side-by-side screenshot vs docs/caddy-target.png shows matching layout,
  hierarchy, and content for every region above.
- Mock mode replays a full run end to end with zero console errors.
- All derivation unit tests pass, including the three sanity cases.

## Stretch goals (only after DoD is met; ask before starting)

- Sample completions panel: prompt -> completion pairs from each eval, with a
  step slider to compare early-run gibberish against current output.
- X-axis toggle on both charts: steps vs wall-clock time.
- Light theme toggle (projector mode) and a presentation mode that enlarges the
  hero card, loss chart, and samples panel.

## Non-goals

- No auth, no multi-user, no restyling of History / Compare / Machines beyond
  keeping navigation functional.
