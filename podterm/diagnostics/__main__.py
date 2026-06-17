import argparse

p = argparse.ArgumentParser(prog='python -m podterm.diagnostics',
                            description='Run diagnostics on a saved checkpoint. Set DEVICE=cpu to run off-GPU; '
                                        '--no-tokens skips the forward/token stages (weights-only, no dataset needed).')
p.add_argument('checkpoint'); p.add_argument('--out', default=None)
p.add_argument('--no-tokens', action='store_true', help='skip forward/token stages (weights-only)')
p.add_argument('--step', type=int, default=None, help='training step this checkpoint came from (recorded in meta)')
a = p.parse_args()

import torch, train_gpt  # noqa: E402 -- import runs config_gpt.settings(): device (DEVICE env), tokenizer, seeds

# The model is uncompiled here, so its eager flex_attention is the slow reference path (minutes/forward).
# Swap in the SDPA equivalent -- same masked attention, native fused kernel, no compile -- before any forward.
from podterm.diagnostics.attention import sdpa_flex  # noqa: E402
train_gpt.flex_attention = sdpa_flex

model = train_gpt.GPT().to(train_gpt.HW.device).bfloat16()
model.load_state_dict(torch.load(a.checkpoint, map_location=train_gpt.HW.device), strict=True)
from podterm.diagnostics.runner import run_diagnostics  # noqa: E402

val_tokens = None if a.no_tokens else train_gpt.load_validation_tokens()
out = a.out or a.checkpoint.rsplit('.', 1)[0] + '.diag.json'
meta = dict(checkpoint=a.checkpoint)
if a.step is not None: meta['step'] = a.step
run_diagnostics(model, None, val_tokens, json_path=out, extra_meta=meta)
