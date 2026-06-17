# podterm/diagnostics/

Model-health introspection, run **off-pod by PodTerm** on snapshots pulled from a training run
(it used to run on the pod at end of run; it now lives here). Probes whatever model structure
exists at runtime (the model in gpt-golf's `train_gpt.py` changes constantly — diagnostics adapts
instead of assuming), verifies its own numbers against the real forward pass, and emits a console
report plus versioned JSON.

It imports the model architecture (`train_gpt`/`config_gpt`) from the sibling gpt-golf checkout,
so it must run under gpt-golf's torch env with that checkout on the path. `podterm/snapshots.py`
drives this automatically; the CLIs below are for manual/offline use.

## Running

- **On a checkpoint** (`DEVICE=cpu` to run off-GPU; `--no-tokens` for weights-only, no dataset):
  `python -m podterm.diagnostics step1000.pt [--out X.json] [--step N]` → `step1000.diag.json`.
- **Comparing runs** (stdlib-only, runs anywhere):
  `python -m podterm.diagnostics.compare base.json new.json [-t 5] [--all] [--top N]`.
  Output order: meta header → config Δ (config diffs usually *explain* metric deltas) → status/structural drift → movers sorted by |Δ%|.

Env knobs (`config.py`): `DIAG_JSON_PATH`, `DIAG_MAX_BATCHES` (forward-sweep cap, 0 = full val set; the sweep costs ~2-3× a plain eval pass since branches are recomputed eagerly), `DIAG_ENTROPY_BATCHES`, `DIAG_SAMPLE_TOKENS` (completion length for the sampling stage; one full forward per token, 0 disables).

## How it stays decoupled from the model

- `anatomy.probe(model)` discovers spine attrs (`wte/first_norm/final_norm/lm_head/blocks`) and per-block **capabilities** (`_qkv`/`_attend`/`_mlp` callables, head attrs, `block_mask.mask_mod`, linear activation, `_init_std` markers, `_gate_dim` annotation on gated activations). Metrics whose requirements are missing are **skipped with a reason**, never crash.
- Forward metrics come from hooks on a **real** forward pass; per-block intermediates are regenerated from the captured block input via the block's *own* `_qkv`/`_mlp` plus a compiled flex_attention + gain + projection recompute of the attn branch (the only duplicated model lines). Two cross-checks guard everything recomputed:
  - **recompose**: `h_in + attn + mlp ≈ captured block output` (validates the branch recompute, gain placement, and block composition in one shot)
  - **loss_recompute**: softcap+CE from captured logits ≈ the model's own loss
  Failures mark rows `!` in console / `"_unverified"` in JSON and land in `meta.checks` — loud, not fatal.
- Stage exceptions become `[error: ...]` sections; JSON is written even on partial failure; the entry point never raises (a diagnostics crash must never cost a run).

## JSON schema v2

```
{ version: 2,
  meta: { timestamp, git_sha, n_params, config{model,train,optim,diag}, anatomy{blocks[]},
          checks{loss_recompute, recompose{...}, y_path{...}}, val_bpb?, val_loss?, steps?, ... },
  sections: [ { name, status: ok|skipped|error|partial, reason, rows: {key: {metric: value}}, notes[] } ] }
```

Rows are keyed by **block name** (`enc1`, `dec2`, ...) from the model's block container, so runs with different layouts diff structurally instead of silently misaligning. Stability policy: within v2, changes are additive only; breaking changes bump the version and `compare` refuses cross-version.

## Metric reference

**Scaling**
- *gains* (`attn_gain`, `mlp_gain`, `qk_gain`, ...) — learned branch/channel scales. Drift from init 1.0 shows which paths the model amplifies or mutes; a gain pinned near 0 is a silenced path.
- *norm weights* (`x_norm`, `qk_norm`, first/final) — RMSNorm scale vectors. Mean ≈ 1 healthy; collapse toward 0 silences a stream.
- *zero-init escape* — `‖W‖ / (σ_init·√numel)` per `_init_std` layer + lm_head. ≈1x means a zero-init projection never escaped init (dead branch); large = strong learned signal.
- *post-norm RMS* — RMS of block input after `x_norm`; confirms the normalizer normalizes (≈1 expected).

**Optimizer** (skipped when run without an optimizer, e.g. checkpoint CLI)
- *update/weight ratio* — per reflection-labeled group: `‖exp_avg/(√exp_avg_sq+ε)‖·base_lr / ‖w‖`, the effective relative step. `schedule_phase` = current lr / base lr (≈0 after warmdown).
- *adam second moment* — per group mean `√v` and `ε/(√v+ε)`. eps_frac → 1 means ε dominates and updates have effectively stalled.

**Capacity**
- *Q weight similarity* — pairwise cosine of per-head Q rows. High max = redundant heads at the weight level.
- *embedding spectrum* — effective rank (entropy of squared singular values) and dims-to-90% for `wte`/`lm_head`. Low = wasted width.
- *dead neurons* — fraction of MLP units firing on <0.1% of val tokens (gate/up split via `_gate_dim`). High = wasted capacity.
- *head output redundancy* — data-averaged pairwise cosine of post-gain head outputs. High mean = heads computing the same function.
- *KV head utilization* — mean `‖v‖` per kv-head, `‖y‖` per query head. Near-zero entries = dead heads under GQA.
- *attention entropy (bits) / oneshot* — per-head mean attention entropy and fraction of rows with max prob >0.99. Very low H + high oneshot = copying/induction heads; uniformly high H = unfocused attention.

**Flow**
- *residual norm* — stream norm at emb → after each block → final, normalized to emb. Reveals explosion/collapse with depth.
- *branch contribution* — `‖attn_out‖/‖resid‖`, `‖mlp_out‖/‖resid‖` per block: how much each branch writes.
- *branch alignment* — `cos(branch, resid)`. ≈0 orthogonal writes (healthy); strongly negative = erasing; strongly positive = amplifying.
- *logit saturation* — fraction of logits ≥95% of the tanh softcap. High = the cap is squashing gradients.
- *per-position loss* — mean CE bucketed by position `[0:1] [1:8] [8:32] [32:128] [128:512] [512:1024]`: context use vs early-position floor.

**Gradients**
- *gradient norms* — per-`_init_std`-module `‖∇W‖` from one val-batch backward: vanishing/exploding per layer.

**Samples**
- *prompt → completion* — fixed prompts (`stages_sample.py:PROMPTS`) continued with seeded top-k/temperature sampling; prompts ride in separate rows of one batch since the model's forward is shape-locked to `(seq_count, seq_len)`, so each generated token costs one forward. A qualitative read on fluency: this is a small base model trained briefly on web text, so expect plausible-English continuation, not instruction following. Stored in JSON, so `compare` shows how samples evolve across runs.
