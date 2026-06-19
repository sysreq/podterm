import json

import pytest

from podterm.diagnostics.compare import diff, diff_docs, flat_meta, flatten, load, main, pct
from podterm.diagnostics.schema import SCHEMA_VERSION


def doc(rows=None, sections=None, **meta):
    base_meta = dict(timestamp='t0', git_sha='abc', n_params=10,
                     config=dict(optim=dict(matrix_lr=0.04)), anatomy=dict(blocks=[dict(name='enc1', heads=9)]))
    base_meta.update(meta)
    secs = sections if sections is not None else [dict(
        name='flow', status='ok', reason='',
        rows=rows if rows is not None else dict(enc1=dict(attn=0.4, v=[1.0, 2.0])), notes=[])]
    return dict(version=SCHEMA_VERSION, meta=base_meta, sections=secs)


def test_diff_docs_movers_status_config():
    a = doc(rows=dict(enc1=dict(dead=0.01)))
    b = doc(rows=dict(enc1=dict(dead=0.05)), config=dict(optim=dict(matrix_lr=0.08)))
    out = diff_docs(a, b)
    mover = next(m for m in out["movers"] if m["path"].endswith("/dead"))
    assert mover["base"] == 0.01 and mover["run"] == 0.05 and mover["kind"] == "num"
    assert out["movers"] == sorted(out["movers"], key=lambda m: -m["score"])  # ranked
    assert any("matrix_lr" in k for k in out["config"])


def test_diff_docs_structural_and_status_change():
    a = doc(sections=[dict(name="cap", status="ok", reason="", rows=dict(enc1=dict(x=1.0)), notes=[])])
    b = doc(sections=[dict(name="cap", status="error", reason="boom", rows=dict(enc2=dict(x=1.0)), notes=[])])
    out = diff_docs(a, b)
    paths = {s["path"] for s in out["structural"]}
    assert any("enc1" in p for p in paths) and any("enc2" in p for p in paths)
    assert out["status"]["cap"]["base"] == "ok" and out["status"]["cap"]["run"].startswith("error")


def test_flatten_paths_and_status():
    leaves, status = flatten(doc(rows=dict(enc1=dict(attn=0.4, _unverified=True))))
    assert leaves == {'flow/enc1/attn': 0.4}  # _unverified marker excluded
    assert status == {'flow': 'ok'}
    _, status = flatten(doc(sections=[dict(name='optimizer', status='skipped', reason='no optimizer', rows={}, notes=[])]))
    assert status == {'optimizer': 'skipped:no optimizer'}


def test_numeric_diff():
    kind, score, text = diff(0.4, 0.3)
    assert kind == 'num' and score == pytest.approx(25.0) and '-25.0%' in text
    assert diff(0.4, 0.4) == ('none', 0.0, '')


def test_zero_baseline_guard():
    kind, score, text = diff(0.0, 0.5)
    assert kind == 'num' and score == float('inf') and 'from ~0' in text
    assert pct(0.0, 0.0) == 0.0


def test_list_diffs():
    kind, score, text = diff([1.0, 2.0], [1.0, 3.0])
    assert kind == 'list' and score == pytest.approx(50.0) and '@1' in text
    kind, _, text = diff([1.0] * 9, [1.0] * 4)
    assert kind == 'struct' and 'len 9 -> 4' in text
    assert diff([1.0, 2.0], [1.0, 2.0]) == ('none', 0.0, '')


def test_struct_diff_on_strings():
    kind, _, _ = diff('Encoder', 'Decoder')
    assert kind == 'struct'


def test_flat_meta_nests_dicts_and_block_lists():
    flat = flat_meta(dict(config=dict(optim=dict(lr=0.1)), anatomy=dict(blocks=[dict(name='enc1', heads=9)])))
    assert flat == {'config/optim/lr': 0.1, 'anatomy/blocks[0]/name': 'enc1', 'anatomy/blocks[0]/heads': 9}


def test_version_rejection(tmp_path):
    bad = tmp_path / 'old.json'; bad.write_text(json.dumps(dict(version=1, sections=[])))
    with pytest.raises(SystemExit): load(str(bad))


def _write(tmp_path, name, d):
    p = tmp_path / name; p.write_text(json.dumps(d)); return str(p)


def test_main_movers_structural_and_status(tmp_path, capsys):
    a = _write(tmp_path, 'a.json', doc(rows=dict(enc1=dict(attn=0.4, gone=1.0))))
    b_doc = doc(rows=dict(enc1=dict(attn=0.6)), config=dict(optim=dict(matrix_lr=0.05)))
    b_doc['sections'].append(dict(name='gradients', status='skipped', reason='no val tokens', rows={}, notes=[]))
    b = _write(tmp_path, 'b.json', b_doc)
    main([a, b, '-t', '5'])
    out = capsys.readouterr().out
    assert 'config Δ' in out and 'matrix_lr' in out
    assert 'gradients: absent -> skipped:no val tokens' in out
    assert 'flow/enc1/gone: present in A, missing in B' in out
    assert 'flow/enc1/attn' in out and '+50.0%' in out


def test_main_threshold_suppression(tmp_path, capsys):
    a = _write(tmp_path, 'a.json', doc(rows=dict(enc1=dict(attn=0.400))))
    b = _write(tmp_path, 'b.json', doc(rows=dict(enc1=dict(attn=0.401))))
    main([a, b, '-t', '5'])
    out = capsys.readouterr().out
    assert 'flow/enc1/attn' not in out and '1 changed leaves below threshold' in out
    main([a, b, '--all'])
    assert 'flow/enc1/attn' in capsys.readouterr().out


def test_main_multirun_columns(tmp_path, capsys):
    a = _write(tmp_path, 'a.json', doc(rows=dict(enc1=dict(attn=0.4))))
    b = _write(tmp_path, 'b.json', doc(rows=dict(enc1=dict(attn=0.6))))
    c = _write(tmp_path, 'c.json', doc(rows=dict(enc1=dict(attn=0.4))))
    main([a, b, c, '-t', '5'])
    out = capsys.readouterr().out
    assert '+50.0%' in out and ' | =' in out
