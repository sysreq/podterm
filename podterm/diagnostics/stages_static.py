import math

import torch
import torch.nn.functional as F

from . import anatomy as A, report as R
from .runner import Stage
from .schema import SKIPPED


class ParamStatsStage(Stage):
    name = 'params'; requires = ('blocks',)

    def run(self, ctx):
        anat = ctx.anatomy; s = R.Section('scaling: gains')
        for b in anat.blocks:
            parts = []; data = {}
            for n, p in b.module._parameters.items():
                if p is None or A.classify(n, p) == 'norm': continue
                if p.numel() <= 16: parts.append(f"{n}={R.fmt_gains(p)}"); data[n] = p.data.float().flatten().tolist()
                else:
                    t = p.data.float(); parts.append(f"{n}={R.fmt_stat(p)}")
                    data[f'{n}_mean'] = t.mean().item(); data[f'{n}_std'] = t.std().item()
            s.row(b.name, ' '.join(parts), **data)
        ctx.report.emit(s)
        s = R.Section('scaling: norm weights')
        for b in anat.blocks:
            s.row(b.name, ' '.join(f'{n}={R.fmt_norm(p)}' for n, p in b.module._parameters.items()
                                   if p is not None and A.classify(n, p) == 'norm'))
        edges = ' '.join(f'{n}={R.fmt_norm(m.weight)}' for n, m in (('first', anat.first_norm), ('final', anat.final_norm))
                         if getattr(m, 'weight', None) is not None)
        if edges: s.row('edges', edges)
        ctx.report.emit(s)


class EscapeStage(Stage):
    name = 'escape'; requires = ('blocks',)

    def run(self, ctx):
        s = R.Section('zero-init escape')
        for b in ctx.anatomy.blocks:
            parts = []; data = {}
            for n, m in b.init_std:
                w = m.weight.data.float(); r = w.norm().item() / (m._init_std * math.sqrt(w.numel()))
                parts.append(f'{n}={r:.1f}x'); data[n] = r
            s.row(b.name, ' '.join(parts), **data)
        lm = ctx.anatomy.lm_head
        if lm is not None and hasattr(lm, '_init_std'):
            w = lm.weight.data.float(); r = w.norm().item() / (lm._init_std * math.sqrt(w.numel()))
            s.row('lm_head', f'{r:.1f}x', escape=r)
        ctx.report.emit(s)


class QSimStage(Stage):
    name = 'qsim'; requires = ('blocks',)

    def run(self, ctx):
        s = R.Section('Q weight similarity')
        for b in ctx.anatomy.blocks:
            if not {'heads', 'qkv_proj', 'qkv_layout'} <= b.caps: s.row(b.name, '[skipped: nonstandard qkv]'); continue
            qw = b.module.qkv_proj.weight.data[:b.heads * b.head_dim]
            wn = F.normalize(qw.float().reshape(b.heads, b.head_dim, -1).flatten(1), dim=1)
            sim = (wn @ wn.T)[~torch.eye(b.heads, device=wn.device, dtype=torch.bool)]
            s.row(b.name, f'mean={sim.mean():.4f},max={sim.max():.4f}', mean=sim.mean().item(), max=sim.max().item())
        ctx.report.emit(s)


class OptimStage(Stage):
    name = 'optimizer'; requires = ('optimizer',)

    def run(self, ctx):
        opt = ctx.optimizer; labels = A.label_groups(opt, ctx.model)
        s = R.Section('lr: update/weight ratio')
        for gn, grp in zip(labels, opt.param_groups):
            if not grp['params']: s.row(gn, '[empty]'); continue
            blr = grp.get('base_lr', grp['lr']); us = 0.0; un = 0
            for p in grp['params']:
                st = opt.state.get(p)
                if st is None or 'exp_avg' not in st: continue
                u = (st['exp_avg'] / (st['exp_avg_sq'].sqrt() + grp.get('eps', 1e-8))).float()
                us += (u.norm() * blr / (p.data.float().norm() + 1e-12)).item(); un += 1
            avg = us / max(un, 1); s.row(gn, f'base_lr={blr:.4f} uwr={avg:.6f}', base_lr=blr, uwr=avg)
        g0 = opt.param_groups[0]; ph = g0['lr'] / max(g0.get('base_lr', 1.0), 1e-12)
        s.row('schedule_phase', f'{ph:.4f}', value=ph); ctx.report.emit(s)
        s = R.Section('adam: second moment')
        for gn, grp in zip(labels, opt.param_groups):
            if not grp['params']: s.row(gn, '[empty]'); continue
            eps = grp.get('eps', 1e-8); sv_sum = ef_sum = 0.0; n = 0
            for p in grp['params']:
                st = opt.state.get(p)
                if st is None or 'exp_avg_sq' not in st: continue
                sv = st['exp_avg_sq'].float().sqrt(); sv_sum += sv.mean().item(); ef_sum += (eps / (sv + eps)).mean().item(); n += 1
            if n == 0: s.row(gn, '[no state]'); continue
            s.row(gn, f'sqrt_v_mean={sv_sum/n:.2e} eps_frac={ef_sum/n:.4f} (eps={eps:.1e})', sqrt_v=sv_sum/n, eps_frac=ef_sum/n)
        ctx.report.emit(s)


class SpectrumStage(Stage):
    name = 'spectrum'; requires = ()

    def run(self, ctx):
        anat = ctx.anatomy
        targets = [(n, m) for n, m in (('wte', anat.embedding), ('lm_head', anat.lm_head))
                   if getattr(m, 'weight', None) is not None]
        if not targets: ctx.report.emit(R.Section('embedding spectrum', status=SKIPPED, reason='no embedding/lm_head')); return
        s = R.Section('embedding spectrum')
        for n, m in targets:
            w = m.weight.data.cpu().float(); w = w - w.mean(0, keepdim=True)  # CPU SVD: cusolver dropped from slim image
            sv = torch.linalg.svdvals(w); v2 = sv.square(); cv = v2.cumsum(0) / v2.sum()
            d90 = int((cv < 0.90).sum()) + 1; pr = v2 / v2.sum()
            er = torch.exp(-(pr * (pr + 1e-12).log()).sum()).item()
            s.row(n, f'eff_rank={er:.1f}/{m.weight.size(1)} dims_90={d90}', eff_rank=er, dims_90=d90)
        ctx.report.emit(s)
