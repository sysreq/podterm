import torch
import torch.nn.functional as F

import config_gpt as C
from . import report as R
from .runner import Stage
from .schema import SKIPPED

PROMPTS = ("What is the meaning of life?", "Write me a haiku about yourself")
TEMP = 0.8; TOP_K = 50


def _rep_ngram(toks, n=4):
    """Fraction of repeated n-grams (1 - distinct/total) — a cheap degeneration signal. 0 = no repeats."""
    if len(toks) < n + 1: return 0.0
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


class SampleStage(Stage):
    name = 'samples'; requires = ()

    def run(self, ctx):
        s = R.Section('samples: prompt -> completion')
        if not ctx.cfg.sample_tokens: s.status = SKIPPED; s.reason = 'sample_tokens=0'; ctx.report.emit(s); return
        if ctx.anatomy.lm_head is None: s.status = SKIPPED; s.reason = 'no lm_head'; ctx.report.emit(s); return
        sp = C.IO.sp; sc, sl = C.TRAIN.seq_count, C.TRAIN.seq_len; dev = C.HW.device
        prompts = PROMPTS[:sc]; enc = [list(sp.encode(p))[:sl // 2] for p in prompts]
        steps = min(ctx.cfg.sample_tokens, sl - max(len(e) for e in enc))
        x = torch.zeros(sc, sl, dtype=torch.int64, device=dev)
        for i, ids in enumerate(enc): x[i, :len(ids)] = torch.tensor(ids, dtype=torch.int64, device=dev)
        rows = torch.arange(len(enc), device=dev); pos = torch.tensor([len(e) for e in enc], device=dev)
        gen = torch.Generator(device=dev); gen.manual_seed(C.IO.seed)
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            for _ in range(steps):
                lg = ctx.model(x)[rows, pos - 1].float() / TEMP
                lg = lg.masked_fill(lg < lg.topk(min(TOP_K, lg.size(-1)), -1).values[:, -1:], float('-inf'))
                x[rows, pos] = torch.multinomial(F.softmax(lg, -1), 1, generator=gen).squeeze(-1); pos += 1
        eos = sp.eos_id()
        for i, (prompt, ids) in enumerate(zip(prompts, enc)):
            out = x[i, len(ids):int(pos[i])].cpu().tolist()
            if eos in out: out = out[:out.index(eos)]
            text = sp.decode(out)
            cov = len(set(out)) / len(out) if out else 0.0
            s.row(f'q{i + 1}', f'{prompt!r} -> {text!r}', prompt=prompt, completion=text,
                  tokens=len(out), vocab_coverage=cov, repetition=_rep_ngram(out))
        ctx.report.emit(s)
