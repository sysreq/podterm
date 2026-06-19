import math, warnings

import torch
import torch.nn.functional as F

import config_gpt as C
from . import report as R
from .attention import sdpa_flex
from .capture import CaptureSession, rel_err, val_batches
from .runner import Stage
from .schema import SKIPPED

REL_TOL = 1e-2; LOSS_TOL = 1e-3; ENTROPY_CHUNK = 8


class ForwardStage(Stage):
    name = 'forward'; requires = ('val_tokens', 'blocks')

    def run(self, ctx):
        anat, model = ctx.anatomy, ctx.model; blocks = anat.blocks; nl = len(blocks)
        dev = C.HW.device; sl = C.TRAIN.seq_len; sc = C.TRAIN.seq_count; scap = C.OPTIM.logit_softcap
        f64 = dict(device=dev, dtype=torch.float64)
        can_branch = [{'qkv', 'attend', 'mlp'} <= b.caps for b in blocks]
        can_y = [can_branch[i] and {'heads', 'attn_gain', 'mask_mod', 'atn_proj'} <= b.caps for i, b in enumerate(blocks)]
        can_act = [can_branch[i] and 'act_linear' in b.caps for i, b in enumerate(blocks)]
        has_logits = anat.lm_head is not None
        rr = torch.zeros(nl + 2, **f64); errA = torch.zeros(nl, **f64); errC = torch.zeros((), **f64)
        acc = [dict(res=torch.zeros((), **f64), attn=torch.zeros((), **f64), mlp=torch.zeros((), **f64),
                    cos_a=torch.zeros((), **f64), cos_m=torch.zeros((), **f64), pnr=torch.zeros((), **f64),
                    v=torch.zeros(max(b.kv_heads, 1), **f64), y=torch.zeros(max(b.heads, 1), **f64),
                    hsi=torch.zeros(max(b.heads, 1), max(b.heads, 1), **f64),
                    up=torch.zeros(max(b.act_out, 1), **f64)) for b in blocks]
        sat = torch.zeros((), **f64); lct = 0; ploss = torch.zeros(sl, **f64); tt = 0.0; nb = 0
        # BPB reconciliation: replicate train_gpt.eval_val's exact byte accounting (SentencePiece LUTs
        # + leading-space adjustment) so the off-pod number is directly comparable to the on-pod BPB.
        loss_sum = torch.zeros((), **f64); byte_count = torch.zeros((), **f64)
        try:
            import train_gpt as _tg
            _bb, _hls, _ibt = (t.to(dev) for t in _tg.build_sentencepiece_luts())
        except Exception:
            _bb = None

        with CaptureSession(anat) as sess, torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            for x, y in val_batches(ctx.val_tokens, ctx.cfg.max_batches):
                loss, cap = sess.forward(model, x, y); tt += x.numel(); nb += 1
                loss_sum += loss.double() * y.numel()
                if _bb is not None:
                    prev_ids, tgt_ids = x.reshape(-1), y.reshape(-1)
                    tb = _bb[tgt_ids].to(torch.int16)
                    tb += (_hls[tgt_ids] & ~_ibt[prev_ids]).to(torch.int16)
                    byte_count += tb.to(torch.float64).sum()
                h_in = cap['block_in']; rr[0] += h_in.float().norm(dim=-1).sum()
                for i, b in enumerate(blocks):
                    m = b.module; a = acc[i]; h_out = cap[b.name]
                    a['res'] += h_in.float().norm(dim=-1).sum(); rr[i + 1] += h_out.float().norm(dim=-1).sum()
                    if can_branch[i]:
                        xb, q, k, v = m._qkv(h_in); mlp = m._mlp(xb)
                        if can_y[i]:
                            yv = sdpa_flex(q, k, v, block_mask=m.block_mask, enable_gqa=(b.kv_heads != b.heads))
                            yv = yv * m.attn_gain.to(yv.dtype)[None, :, None, :]
                            attn = F.linear(yv.transpose(1, 2).reshape(sc, sl, b.heads * b.head_dim), m.atn_proj.weight)
                            a['y'] += yv.float().norm(dim=-1).sum((0, 2))
                            yn = F.normalize(yv.permute(0, 2, 1, 3).reshape(-1, b.heads, b.head_dim).float(), dim=-1)
                            with torch.autocast(device_type='cuda', enabled=False): a['hsi'] += torch.einsum('nik,njk->ij', yn, yn)
                            del yv, yn
                        else: attn = m._attend(q, k, v)
                        errA[i] = torch.maximum(errA[i], rel_err(h_in + attn + mlp, h_out).double())
                        a['pnr'] += xb.float().pow(2).mean(-1).sqrt().sum()
                        a['attn'] += attn.float().norm(dim=-1).sum(); a['mlp'] += mlp.float().norm(dim=-1).sum()
                        a['cos_a'] += F.cosine_similarity(attn.float(), h_in.float(), dim=-1).sum()
                        a['cos_m'] += F.cosine_similarity(mlp.float(), h_in.float(), dim=-1).sum()
                        if 'heads' in b.caps: a['v'] += v.float().norm(dim=-1).sum((0, 2))
                        if can_act[i]: a['up'] += (F.linear(xb, m.activation.weight) > 0).to(torch.float64).sum((0, 1))
                        del xb, q, k, v, attn, mlp
                    h_in = h_out
                if 'final' in cap: rr[nl + 1] += cap['final'].float().norm(dim=-1).sum()
                if 'logits' in cap:
                    lg = cap['logits']; sat += (lg.float().abs() >= 0.95 * scap).to(torch.float64).sum(); lct += lg.numel()
                    lg_c = (scap * torch.tanh(lg / scap)).float()
                    pl = F.cross_entropy(lg_c.view(-1, lg_c.size(-1)), y.view(-1), reduction='none').reshape(sc, sl)
                    ploss += pl.sum(0).double(); errC = torch.maximum(errC, (pl.mean() - loss.float()).abs().double())

        if nb == 0: ctx.report.emit(R.Section(self.name, status=SKIPPED, reason='val set smaller than one batch')); return
        okA = [bool(v) for v in (errA < REL_TOL).tolist()]; okC = bool((errC < LOSS_TOL).item())
        ctx.report.meta['checks'] = dict(
            loss_recompute='ok' if okC else f'diff={errC.item():.1e}',
            recompose={b.name: ('ok' if okA[i] else f'rel_err={errA[i].item():.1e}') for i, b in enumerate(blocks) if can_branch[i]})
        # BPB = mean_nats/ln2 · (tokens/bytes) = loss_sum_nats / (ln2 · bytes). Only meaningful over the
        # full val set (matches on-pod); flag whether this run was capped so reconciliation can judge.
        if _bb is not None and byte_count.item() > 0:
            ctx.report.meta['checks']['bpb'] = (loss_sum / (math.log(2.0) * byte_count)).item()
            ctx.report.meta['checks']['bpb_full'] = (ctx.cfg.max_batches == 0)
        failA = [f'{b.name} recompose rel_err={errA[i].item():.1e} -- block forward diverges from _qkv/branch recompute'
                 for i, b in enumerate(blocks) if can_branch[i] and not okA[i]]
        failC = [] if okC else [f'loss recompute diff={errC.item():.1e} -- softcap/CE path diverges from model loss']

        s = R.Section('capacity: dead neurons'); s.notes += failA; td = tn = 0.0
        for i, b in enumerate(blocks):
            if not can_act[i]: s.row(b.name, '[skipped: no linear activation]'); continue
            rate = acc[i]['up'] / tt
            if b.gate_dim > 0:
                rg, ru = rate[:b.gate_dim], rate[b.gate_dim:]
                dg = (rg < 0.001).float().mean().item(); du = (ru < 0.001).float().mean().item()
                s.row(b.name, f'gate={dg:.1%} {R.fmt_full(rg)} up={du:.1%} {R.fmt_full(ru)}',
                      unverified=not okA[i], dead_gate=dg, dead_up=du)
                td += dg * b.gate_dim + du * (b.act_out - b.gate_dim); tn += b.act_out
            else:
                d = (rate < 0.001).float().mean().item()
                s.row(b.name, f'mlp={d:.1%} {R.fmt_full(rate)}', unverified=not okA[i], dead=d); td += d * b.act_out; tn += b.act_out
        gd = td / max(tn, 1); s.row('global', f'{gd:.1%}', dead=gd); ctx.report.emit(s)

        s = R.Section('capacity: head output redundancy')
        for i, b in enumerate(blocks):
            if not can_y[i]: s.row(b.name, '[skipped: no attn head capture]'); continue
            sim = (acc[i]['hsi'] / tt)[~torch.eye(b.heads, device=dev, dtype=torch.bool)].float()
            s.row(b.name, R.fmt_full(sim), unverified=not okA[i],
                  mean=sim.mean().item(), std=sim.std().item(), min=sim.min().item(), max=sim.max().item())
        ctx.report.emit(s)

        s = R.Section('capacity: KV head utilization')
        for i, b in enumerate(blocks):
            if not (can_branch[i] and 'heads' in b.caps): s.row(b.name, '[skipped: no qkv capture]'); continue
            v = acc[i]['v'] / tt; parts = [f"v=[{','.join(f'{x:.3f}' for x in v.tolist())}] {R.fmt_full(v)}"]; data = dict(v_norm=v.tolist())
            if can_y[i]:
                yh = acc[i]['y'] / tt; parts.append(f"| y=[{','.join(f'{x:.3f}' for x in yh.tolist())}] {R.fmt_full(yh)}")
                data['y_norm'] = yh.tolist()
            s.row(b.name, ' '.join(parts), unverified=not okA[i], **data)
        ctx.report.emit(s)

        s = R.Section('scaling: post-norm RMS')
        for i, b in enumerate(blocks):
            if not can_branch[i]: s.row(b.name, '[skipped: no _qkv]'); continue
            v = (acc[i]['pnr'] / tt).item(); s.row(b.name, f'xn={v:.4f}', unverified=not okA[i], rms=v)
        ctx.report.emit(s)

        rnorm = rr / tt; e_r = max(rnorm[0].item(), 1e-12)
        labels = ['emb'] + [b.name for b in blocks] + (['final'] if anat.final_norm is not None else [])
        vals = [rnorm[j].item() / e_r for j in range(len(labels))]
        s = R.Section('flow: residual norm')
        s.row('stream', ' '.join(f'{l}:{v:.3f}' for l, v in zip(labels, vals)), **dict(zip(labels, vals)))
        ctx.report.emit(s)

        s = R.Section('flow: branch contribution')
        for i, b in enumerate(blocks):
            if not can_branch[i]: s.row(b.name, '[skipped: no _qkv/_attend/_mlp]'); continue
            res = max(acc[i]['res'].item(), 1e-12); ar = acc[i]['attn'].item() / res; mr = acc[i]['mlp'].item() / res
            s.row(b.name, f'attn={ar:.3f} mlp={mr:.3f}', unverified=not okA[i], attn=ar, mlp=mr)
        ctx.report.emit(s)

        s = R.Section('flow: branch alignment')
        for i, b in enumerate(blocks):
            if not can_branch[i]: s.row(b.name, '[skipped: no _qkv/_attend/_mlp]'); continue
            ca = acc[i]['cos_a'].item() / tt; cm = acc[i]['cos_m'].item() / tt
            s.row(b.name, f'cos(attn,res)={ca:+.3f} cos(mlp,res)={cm:+.3f}', unverified=not okA[i], cos_attn=ca, cos_mlp=cm)
        ctx.report.emit(s)

        s = R.Section('flow: logit saturation')
        if has_logits and lct: sr = sat.item() / lct; s.row('logits', f'>95%cap: {sr:.2%}', unverified=not okC, sat=sr)
        else: s.status = SKIPPED; s.reason = 'no lm_head capture'
        ctx.report.emit(s)

        s = R.Section('flow: per-position loss'); s.notes += failC
        if has_logits and lct:
            pm = (ploss / (nb * sc)).cpu().tolist()
            buckets = [(lo, min(hi, sl)) for lo, hi in ((0, 1), (1, 8), (8, 32), (32, 128), (128, 512), (512, 1024)) if lo < sl]
            bdata = {f'{lo}:{hi}': sum(pm[lo:hi]) / (hi - lo) for lo, hi in buckets if lo < hi}
            s.row('loss', ' '.join(f'[{k}]={v:.3f}' for k, v in bdata.items()), unverified=not okC, per_pos=pm, **bdata)
        else: s.status = SKIPPED; s.reason = 'no lm_head capture'
        ctx.report.emit(s)


class EntropyStage(Stage):
    name = 'entropy'; requires = ('val_tokens', 'blocks')

    def run(self, ctx):
        anat, model = ctx.anatomy, ctx.model; blocks = anat.blocks
        dev = C.HW.device; sl = C.TRAIN.seq_len; sc = C.TRAIN.seq_count
        elig = [{'qkv', 'mask_mod', 'heads', 'gqa'} <= b.caps for b in blocks]
        s = R.Section('capacity: attention entropy (bits)')
        if not any(elig): s.status = SKIPPED; s.reason = 'no block exposes qkv+mask'; ctx.report.emit(s); return
        f64 = dict(device=dev, dtype=torch.float64)
        e_acc = [torch.zeros(max(b.heads, 1), **f64) for b in blocks]; o_acc = [torch.zeros(max(b.heads, 1), **f64) for b in blocks]
        qi = torch.arange(sl, device=dev); _b = torch.zeros((), device=dev, dtype=torch.int64)
        _q, _kv = qi.view(1, sl, 1), qi.view(1, 1, sl); nb = 0

        with CaptureSession(anat) as sess, torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            for x, y in val_batches(ctx.val_tokens, ctx.cfg.entropy_batches):
                _, cap = sess.forward(model, x, y); nb += 1
                h_in = cap['block_in']
                for i, b in enumerate(blocks):
                    if elig[i]:
                        m = b.module; _h = torch.arange(b.heads, device=dev).view(b.heads, 1, 1)
                        _x, q, k, _v = m._qkv(h_in)
                        ke = k.repeat_interleave(b.heads // b.kv_heads, 1) if b.kv_heads != b.heads else k
                        mask = m.block_mask.mask_mod(_b, _h, _q, _kv)
                        for c0 in range(0, sc, ENTROPY_CHUNK):
                            with torch.autocast(device_type='cuda', enabled=False):
                                scm = (q[c0:c0 + ENTROPY_CHUNK].float() @ ke[c0:c0 + ENTROPY_CHUNK].float().transpose(-2, -1)) / math.sqrt(b.head_dim)
                                scm.masked_fill_(~mask[None], float('-inf'))
                                p = F.softmax(scm, dim=-1)
                                e_acc[i] += -(p * (p + 1e-10).log2()).sum(-1).sum((0, 2)).double()
                                o_acc[i] += (p.max(-1).values > 0.99).double().sum((0, 2))
                            del scm, p
                        del _x, q, k, _v, ke
                    h_in = cap[b.name]

        if nb == 0: s.status = SKIPPED; s.reason = 'val set smaller than one batch'; ctx.report.emit(s); return
        denom = nb * sc * sl
        for i, b in enumerate(blocks):
            if not elig[i]: s.row(b.name, '[skipped: nonstandard attention]'); continue
            e = e_acc[i] / denom; o = o_acc[i] / denom
            s.row(b.name, f"H=[{','.join(f'{v:.2f}' for v in e.tolist())}] {R.fmt_full(e)} "
                          f"oneshot=[{','.join(f'{v:.1%}' for v in o.tolist())}]",
                  entropy=e.tolist(), oneshot=o.tolist())
        ctx.report.emit(s)


class GradStage(Stage):
    name = 'gradients'; requires = ('val_tokens', 'blocks')

    def run(self, ctx):
        model = ctx.model; s = R.Section('gradients')
        batch = next(iter(val_batches(ctx.val_tokens, 1)), None)
        if batch is None: s.status = SKIPPED; s.reason = 'val set smaller than one batch'; ctx.report.emit(s); return
        x, y = batch; model.zero_grad()
        with warnings.catch_warnings(), torch.enable_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            warnings.filterwarnings('ignore', message='flex_attention called without')
            model(x, y).backward()
        allg = []
        for b in ctx.anatomy.blocks:
            parts = []; data = {}
            for n, m in b.init_std:
                if getattr(m, 'weight', None) is None or m.weight.grad is None: continue
                g = m.weight.grad.float().norm().item(); parts.append(f'{n}={g:.4f}'); data[n] = g; allg.append(g)
            s.row(b.name, ' '.join(parts), **data)
        # Cross-module spread: max/min ratio flags vanishing (tiny somewhere) or exploding (huge somewhere).
        if allg:
            gmax, gmin = max(allg), min(allg); ratio = gmax / max(gmin, 1e-12)
            s.row('summary', f'max={gmax:.4f} min={gmin:.4f} ratio={ratio:.1f}', max=gmax, min=gmin, ratio=ratio)
        model.zero_grad(); ctx.report.emit(s)
