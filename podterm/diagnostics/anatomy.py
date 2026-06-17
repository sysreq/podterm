from dataclasses import dataclass, field

import torch.nn as N


def classify(name, p):
    n = name.lower()
    if 'norm' in n: return 'norm'
    if 'gain' in n: return 'gain'
    if p.ndim == 0: return 'scalar'
    return 'other'


@dataclass
class BlockInfo:
    name: str; idx: int; module: object; caps: frozenset
    heads: int = 0; kv_heads: int = 0; head_dim: int = 0; hidden: int = 0; act_out: int = 0; gate_dim: int = 0
    init_std: list = field(default_factory=list)


@dataclass
class Anatomy:
    embedding: object; first_norm: object; final_norm: object; lm_head: object
    blocks: list; n_params: int


def _block(name, idx, m):
    caps = {c for c, fn in (('qkv', '_qkv'), ('attend', '_attend'), ('mlp', '_mlp')) if callable(getattr(m, fn, None))}
    h, kv, hd = (getattr(m, a, 0) for a in ('heads', 'kv_heads', 'head_dim'))
    if all(isinstance(v, int) and v > 0 for v in (h, kv, hd)):
        caps.add('heads')
        if h % kv == 0: caps.add('gqa')
        ag = getattr(m, 'attn_gain', None)
        if ag is not None and tuple(ag.shape) == (h, hd): caps.add('attn_gain')
        if getattr(m, 'qkv_split', None) == [(h + kv) * hd, kv * hd]: caps.add('qkv_layout')
    else: h = kv = hd = 0
    if callable(getattr(getattr(m, 'block_mask', None), 'mask_mod', None)): caps.add('mask_mod')
    for cap, attr in (('qkv_proj', 'qkv_proj'), ('atn_proj', 'atn_proj')):
        w = getattr(getattr(m, attr, None), 'weight', None)
        if w is not None and w.ndim == 2: caps.add(cap)
    act = getattr(m, 'activation', None); act_out = gate = 0
    if isinstance(act, N.Linear) and act.bias is None:
        caps.add('act_linear'); act_out = act.weight.shape[0]; gate = int(getattr(act, '_gate_dim', 0))
    return BlockInfo(name, idx, m, frozenset(caps), h, kv, hd, int(getattr(m, 'hidden', 0) or 0), act_out, gate,
                     [(n, sub) for n, sub in m.named_modules() if n and hasattr(sub, '_init_std')])


def probe(model):
    spine = {n: getattr(model, n, None) for n in ('wte', 'first_norm', 'final_norm', 'lm_head', 'blocks')}
    container = spine['blocks']
    blocks = [_block(n, i, m) for i, (n, m) in enumerate(container.named_children())] if isinstance(container, N.Module) else []
    return Anatomy(spine['wte'], spine['first_norm'], spine['final_norm'], spine['lm_head'],
                   blocks, sum(p.numel() for p in model.parameters()))


def label_groups(optimizer, model):
    emb_w = getattr(getattr(model, 'wte', None), 'weight', None); head_w = getattr(getattr(model, 'lm_head', None), 'weight', None)
    labels = []
    for i, g in enumerate(optimizer.param_groups):
        ids = {id(p) for p in g['params']}; nds = {p.ndim for p in g['params']}
        if emb_w is not None and ids == {id(emb_w)}: labels.append('embed')
        elif head_w is not None and ids == {id(head_w)}: labels.append('head')
        elif nds == {3}: labels.append('mat3d')
        elif nds == {2}: labels.append('mat2d')
        elif nds and nds <= {0, 1}: labels.append('scalar')
        else: labels.append(f'g{i}')
    return labels
