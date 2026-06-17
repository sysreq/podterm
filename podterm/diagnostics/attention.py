"""SDPA-backed drop-in for torch's flex_attention, used throughout diagnostics.

The diagnostics model is loaded *uncompiled* (a bare `train_gpt.GPT()`), and run that way the model's
`flex_attention` is the eager reference math decomposition -- pathologically slow (minutes per forward
over the seq_count=64, seq_len=1024 diagnostic batch). Compiling it instead costs a full Inductor/Triton
pass that a once-per-snapshot run never amortizes, and the compiled FLASH/TRITON kernels assert
sparse_block_size==tile_n, which fails on some GPU archs (e.g. Blackwell sm120 tiles KV at 64 vs the
model's 128-block mask).

`F.scaled_dot_product_attention` computes the identical masked attention with a native fused kernel in
tens of milliseconds and needs no compile. `podterm.diagnostics.__main__` monkeypatches this over
`train_gpt.flex_attention` so the model's own forward (every stage runs one) takes the fast path too.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# EFFICIENT supports a custom attn_mask and is fastest; MATH is the always-available fallback (any head
# dim / mask / GQA). Forcing this pair sidesteps backend auto-selection raising "Invalid backend".
_BACKENDS = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


def sdpa_flex(q, k, v, score_mod=None, *, block_mask=None, enable_gqa=False, scale=None, **_):
    """Drop-in for `flex_attention(q, k, v, block_mask=…, enable_gqa=…, kernel_options=…)`.

    Builds the boolean attention mask from `block_mask.mask_mod` (the block layout is only a kernel
    optimization; elementwise mask_mod is the ground truth), expands K/V to the query head count for
    GQA, and runs SDPA. `kernel_options` (the FLASH/TRITON backend hint) is irrelevant to SDPA and
    ignored. This model's attention uses no `score_mod`, so it's unsupported."""
    if score_mod is not None:
        raise NotImplementedError("sdpa_flex does not support score_mod")
    heads, seq = q.shape[1], q.shape[-2]
    dev = q.device
    attn_mask = None
    if block_mask is not None:
        h = torch.arange(heads, device=dev).view(heads, 1, 1)
        idx = torch.arange(seq, device=dev)
        z = torch.zeros((), device=dev, dtype=torch.int64)
        attn_mask = block_mask.mask_mod(z, h, idx.view(1, seq, 1), idx.view(1, 1, seq))[None]  # [1,H,S,S] bool
    if enable_gqa and k.shape[1] != heads:
        rep = heads // k.shape[1]
        k = k.repeat_interleave(rep, 1)
        v = v.repeat_interleave(rep, 1)
    with sdpa_kernel(_BACKENDS):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=scale)
