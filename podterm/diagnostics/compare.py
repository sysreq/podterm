"""Diff schema-v2 diagnostics JSONs. Usage: python -m podterm.diagnostics.compare base.json run.json [...] [-t PCT] [--all] [--top N]"""
import argparse, json, sys

from .schema import SCHEMA_VERSION


def load(path):
    with open(path) as f: doc = json.load(f)
    if doc.get('version') != SCHEMA_VERSION: sys.exit(f"{path}: schema version {doc.get('version')}, expected {SCHEMA_VERSION}")
    return doc


def _num(v): return isinstance(v, (int, float)) and not isinstance(v, bool)


def flatten(doc):
    leaves, status = {}, {}
    for s in doc.get('sections', []):
        status[s['name']] = s['status'] + (f":{s['reason']}" if s.get('reason') else '')
        for key, row in s.get('rows', {}).items():
            for mk, mv in row.items():
                if mk != '_unverified': leaves[f"{s['name']}/{key}/{mk}"] = mv
    return leaves, status


def flat_meta(meta, prefix='', out=None):
    out = {} if out is None else out
    for k, v in meta.items():
        if isinstance(v, dict): flat_meta(v, f'{prefix}{k}/', out)
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            for i, d in enumerate(v): flat_meta(d, f'{prefix}{k}[{i}]/', out)
        else: out[f'{prefix}{k}'] = v
    return out


def pct(a, b):
    if abs(a) < 1e-9: return float('inf') if b != a else 0.0
    return 100.0 * (b - a) / abs(a)


def diff(a, b):
    """-> (kind, score, text) where kind is none|num|list|struct."""
    if _num(a) and _num(b):
        if a == b: return ('none', 0.0, '')
        p = pct(a, b); ptxt = 'from ~0' if p == float('inf') else f'{p:+.1f}%'
        return ('num', abs(p), f'{a:.4g} -> {b:.4g}  ({ptxt})')
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b): return ('struct', 0.0, f'len {len(a)} -> {len(b)}')
        if all(_num(x) for x in a) and all(_num(x) for x in b):
            if a == b: return ('none', 0.0, '')
            ps = [pct(x, yv) for x, yv in zip(a, b)]; wi = max(range(len(ps)), key=lambda j: abs(ps[j]))
            md = sum(yv - x for x, yv in zip(a, b)) / max(len(a), 1)
            return ('list', abs(ps[wi]), f'[{len(a)}] max {ps[wi]:+.1f}% @{wi}  meanΔ {md:+.4g}')
        return ('none', 0.0, '') if a == b else ('struct', 0.0, 'changed')
    return ('none', 0.0, '') if a == b else ('struct', 0.0, f'{a!r} -> {b!r}')


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m podterm.diagnostics.compare', description=__doc__)
    ap.add_argument('files', nargs='+'); ap.add_argument('-t', '--threshold', type=float, default=5.0)
    ap.add_argument('--all', action='store_true'); ap.add_argument('--top', type=int, default=0)
    a = ap.parse_args(argv)
    if len(a.files) < 2: ap.error('need a baseline and at least one run')
    docs = [load(p) for p in a.files]; tags = [chr(65 + i) for i in range(len(docs))]

    print('== meta ==')
    for t, p, d in zip(tags, a.files, docs):
        m = d.get('meta', {}); extra = ' '.join(f'{k}={m[k]}' for k in ('val_bpb', 'steps') if k in m)
        print(f"  {t}  {p}  {m.get('timestamp', '?')}  sha={m.get('git_sha', '?')}  {m.get('n_params', 0):,} params  {extra}".rstrip())
    metas = [flat_meta(dict(config=d.get('meta', {}).get('config', {}), anatomy=d.get('meta', {}).get('anatomy', {}))) for d in docs]
    ckeys = sorted(k for k in {k for m in metas for k in m}
                   if len({json.dumps(m.get(k), sort_keys=True, default=str) for m in metas}) > 1)
    if ckeys:
        print('  config Δ:')
        for k in ckeys: print(f'    {k}: ' + ' | '.join(f'{t}={m.get(k)}' for t, m in zip(tags, metas)))

    flats = [flatten(d) for d in docs]; base_leaves = flats[0][0]
    statuses = {n: ' -> '.join(st.get(n, 'absent') for _, st in flats)
                for n in sorted({n for _, st in flats for n in st})
                if len({st.get(n, 'absent') for _, st in flats}) > 1}
    if statuses:
        print('== status ==')
        for n, v in statuses.items(): print(f'  {n}: {v}')

    shown, structural, below = [], [], 0
    for path in sorted({p for lv, _ in flats for p in lv}):
        present = [path in lv for lv, _ in flats]
        if not all(present):
            structural.append(f"{path}: " + ', '.join(f"{'present' if pr else 'missing'} in {t}" for t, pr in zip(tags, present)))
            continue
        results = [diff(base_leaves[path], lv[path]) for lv, _ in flats[1:]]
        if any(k == 'struct' for k, _, _ in results): structural.append(f'{path}: ' + ' | '.join(t or '=' for _, _, t in results)); continue
        if all(k == 'none' for k, _, _ in results): continue
        score = max(sc for _, sc, _ in results)
        if score >= a.threshold or a.all: shown.append((score, path, [t or '=' for _, _, t in results]))
        else: below += 1
    if structural:
        print('== structural ==')
        for line in structural: print(f'  {line}')
    shown.sort(key=lambda r: -r[0])
    if a.top: shown = shown[:a.top]
    print('== leaves ==' if a.all else f'== movers (|Δ%| >= {a.threshold:g}) ==')
    w = max((len(p) for _, p, _ in shown), default=0)
    for _, path, texts in shown: print(f'  {path:<{w}}  ' + ' | '.join(texts))
    if below: print(f'== {below} changed leaves below threshold (use --all) ==')


if __name__ == '__main__': main()
