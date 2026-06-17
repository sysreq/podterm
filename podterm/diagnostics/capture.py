import torch

import config_gpt as C


def rel_err(a, b): return (a.float() - b.float()).norm() / (b.float().norm() + 1e-12)


def val_batches(val_tokens, max_batches=0):
    sc, sl = C.TRAIN.seq_count, C.TRAIN.seq_len
    total = (val_tokens.numel() - 1) // (sc * sl)
    for i in range(total if max_batches == 0 else min(max_batches, total)):
        raw = val_tokens[i * sc * sl:(i + 1) * sc * sl + 1].to(device=C.HW.device, dtype=torch.int64)
        yield raw[:-1].reshape(sc, sl), raw[1:].reshape(sc, sl)


class CaptureSession:
    """Boundary hooks on a real forward. Keys: block_in, each block name, final, logits (pre-softcap)."""

    def __init__(self, anatomy):
        self.targets = []; self.out = {}; self.handles = []
        if anatomy.blocks:
            self.targets.append(('block_in', anatomy.blocks[0].module, True))
            self.targets += [(b.name, b.module, False) for b in anatomy.blocks]
        if anatomy.final_norm is not None: self.targets.append(('final', anatomy.final_norm, False))
        if anatomy.lm_head is not None: self.targets.append(('logits', anatomy.lm_head, False))

    def __enter__(self):
        for key, mod, pre in self.targets:
            if pre: self.handles.append(mod.register_forward_pre_hook(lambda m, args, k=key: self.out.__setitem__(k, args[0])))
            else: self.handles.append(mod.register_forward_hook(lambda m, args, out, k=key: self.out.__setitem__(k, out)))
        return self

    def forward(self, model, x, y):
        self.out.clear(); return model(x, y), self.out

    def __exit__(self, *exc):
        for h in self.handles: h.remove()
        self.handles.clear(); self.out.clear()
