import json

from podterm.diagnostics import report as R
from podterm.diagnostics.schema import SCHEMA_VERSION, SKIPPED, PARTIAL


def test_roundtrip_and_statuses(tmp_path, capsys):
    rep = R.Report(meta=dict(n_params=1, checks={}))
    s = R.Section('flow: test'); s.row('enc1', 'attn=0.4', attn=0.4); s.row('enc2', 'bad', unverified=True, v=1.0)
    rep.emit(s)
    rep.emit(R.Section('optimizer', status=SKIPPED, reason='no optimizer'))
    doc = json.loads(json.dumps(rep.to_json()))
    assert doc['version'] == SCHEMA_VERSION
    assert doc['sections'][0]['status'] == PARTIAL  # auto-flip when any row is unverified
    assert doc['sections'][0]['rows']['enc2'] == {'v': 1.0, '_unverified': True}
    assert doc['sections'][0]['rows']['enc1'] == {'attn': 0.4}
    assert doc['sections'][1]['status'] == SKIPPED
    out = capsys.readouterr().out
    assert '-- flow: test --' in out and '! enc2: bad' in out
    assert '-- optimizer -- [skipped: no optimizer]' in out
    path = tmp_path / 'd.json'; rep.write(path)
    assert json.loads(path.read_text())['version'] == SCHEMA_VERSION


def test_textonly_rows_stay_out_of_json():
    rep = R.Report(); s = R.Section('scaling: norm weights'); s.row('enc1', 'x_norm=1.0')
    rep.sections.append(s)
    assert rep.to_json()['sections'][0]['rows'] == {}
