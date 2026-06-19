"""Derive a value-based health verdict from a schema-v2 diagnostics doc.

This is the single place that turns raw metrics into a judgment: a curated headline set, a
good/warn/bad band per metric, and an overall verdict. It is the producer's say in "what matters"
— replacing the fragile substring matching the UI used to do (static/js/diagnostics.js).

STDLIB-ONLY (no torch). It is imported both inside the diagnostics subprocess (gpt-golf env, via
runner.py) and on the podterm side (snapshots.py), so it must carry no heavy deps — same rule that
keeps compare.py importable in both environments.

Section/row/key IDs below are the *actual* JSON keys emitted by the stages (stages_static.py,
stages_forward.py). `optimizer`-dependent metrics (uwr, adam) are intentionally absent: the off-pod
suite runs on a bare checkpoint with optimizer=None (__main__.py), so that stage is always skipped.
"""

# kind: 'hi' = higher is worse, 'lo' = lower is worse, 'range' = outside band is worse,
#       'info' = trend-only, no verdict. bad=None means warn is the worst level.
HEADLINE = [
    dict(id='dead_neurons', label='Dead neurons', unit='frac', group='capacity', tier=1,
         section='capacity: dead neurons', row='global', key='dead', reduce='value',
         band=dict(kind='hi', warn=0.05, bad=0.20)),
    dict(id='head_redundancy', label='Head redundancy', unit='cos', group='capacity', tier=1,
         section='capacity: head output redundancy', row=None, key='max', reduce='max',
         band=dict(kind='hi', warn=0.6, bad=0.85)),
    dict(id='attn_entropy_min', label='Attn entropy (min)', unit='bits', group='capacity', tier=1,
         section='capacity: attention entropy (bits)', row=None, key='entropy', reduce='min',
         band=dict(kind='lo', warn=0.5, bad=0.1)),
    dict(id='logit_saturation', label='Logit saturation', unit='frac', group='flow', tier=1,
         section='flow: logit saturation', row='logits', key='sat', reduce='value',
         band=dict(kind='hi', warn=0.10, bad=0.30)),
    dict(id='residual_growth', label='Residual growth', unit='x', group='flow', tier=1,
         section='flow: residual norm', row='stream', key='final', reduce='value',
         band=dict(kind='range', warn_lo=0.5, warn_hi=4.0, bad_lo=0.2, bad_hi=10.0)),
    dict(id='grad_ratio', label='Grad norm ratio', unit='x', group='grad', tier=1,
         section='gradients', row='summary', key='ratio', reduce='value',
         band=dict(kind='hi', warn=1e3, bad=1e5)),
    dict(id='escape_min', label='Zero-init escape (min)', unit='x', group='arch', tier=1,
         section='zero-init escape', row=None, key=None, reduce='min',
         band=dict(kind='lo', warn=1.5, bad=None)),
    dict(id='mean_loss', label='Mean loss', unit='nats', group='flow', tier=2,
         section='flow: per-position loss', row='loss', key='per_pos', reduce='mean',
         band=dict(kind='info')),
    dict(id='eff_rank', label='Embed eff_rank', unit='dims', group='arch', tier=2,
         section='embedding spectrum', row=None, key='eff_rank', reduce='mean',
         band=dict(kind='info')),
]

# Severity lattice shared by section-execution status and metric bands. overall = worst of all.
_SEV = {'ok': 0, 'skipped': 0, 'good': 0, 'info': 0, 'na': 0,
        'partial': 1, 'warn': 1, 'error': 2, 'bad': 2}
_VERDICT = ['ok', 'warn', 'error']


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _collect(section, row, key):
    """Flatten matching numeric leaves to a list of floats. row/key=None broadens the scope:
    row given + key given -> one cell; key=None -> all numerics in the row(s); row=None -> all rows."""
    rows = section.get('rows', {}) or {}
    out = []

    def add(v):
        if _is_num(v):
            out.append(float(v))
        elif isinstance(v, list):
            out.extend(float(x) for x in v if _is_num(x))

    targets = [rows[row]] if row is not None else list(rows.values())
    for rd in targets:
        if not isinstance(rd, dict):
            continue
        if key is not None:
            if key in rd:
                add(rd[key])
        else:
            for k, v in rd.items():
                if k != '_unverified':
                    add(v)
    return out


def _reduce(vals, how):
    if not vals:
        return None
    if how == 'value':
        return vals[0]
    if how == 'min':
        return min(vals)
    if how == 'max':
        return max(vals)
    if how == 'mean':
        return sum(vals) / len(vals)
    raise ValueError(f'unknown reduce: {how}')


def classify(value, band):
    """-> 'good' | 'warn' | 'bad' | 'info' | 'na'."""
    if value is None:
        return 'na'
    kind = band['kind']
    if kind == 'info':
        return 'info'
    if kind == 'hi':
        if band.get('bad') is not None and value >= band['bad']:
            return 'bad'
        if band.get('warn') is not None and value >= band['warn']:
            return 'warn'
        return 'good'
    if kind == 'lo':
        if band.get('bad') is not None and value <= band['bad']:
            return 'bad'
        if band.get('warn') is not None and value <= band['warn']:
            return 'warn'
        return 'good'
    if kind == 'range':
        if value < band['bad_lo'] or value > band['bad_hi']:
            return 'bad'
        if value < band['warn_lo'] or value > band['warn_hi']:
            return 'warn'
        return 'good'
    return 'na'


def resolve(diag, entry):
    """Numeric value for one headline entry, or None if its section is missing/skipped/errored
    or the metric path is absent. Returned separately so tests can assert paths resolve."""
    sec = next((s for s in diag.get('sections', []) if s.get('name') == entry['section']), None)
    if sec is None or sec.get('status') in ('skipped', 'error'):
        return None
    return _reduce(_collect(sec, entry['row'], entry['key']), entry['reduce'])


def compute(diag):
    """-> {overall: ok|warn|error, headline: [...], counts: {...}}.

    overall is the worst of every section's execution status and every headline metric's band, so a
    clean-running suite with a sick metric still reads 'error'/'warn'."""
    sev = 0
    for s in diag.get('sections', []):
        sev = max(sev, _SEV.get(s.get('status', 'ok'), 0))

    headline = []
    counts = {'good': 0, 'warn': 0, 'bad': 0, 'info': 0, 'na': 0}
    for entry in HEADLINE:
        value = resolve(diag, entry)
        band = classify(value, entry['band'])
        counts[band] += 1
        sev = max(sev, _SEV.get(band, 0))
        headline.append(dict(id=entry['id'], label=entry['label'], value=value,
                             unit=entry['unit'], band=band, group=entry['group'], tier=entry['tier']))

    return dict(overall=_VERDICT[sev], headline=headline, counts=counts)
