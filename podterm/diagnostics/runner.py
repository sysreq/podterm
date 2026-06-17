import subprocess
from dataclasses import dataclass, fields as dc_fields
from datetime import datetime, timezone

import torch
from torch.nn.attention import flex_attention as _flex_mod

import config_gpt as C
from . import anatomy, config, report
from .schema import SKIPPED, ERROR


class Stage:
    name = ''; requires = ()
    def run(self, ctx): raise NotImplementedError


@dataclass
class RunContext:
    model: object; anatomy: object; optimizer: object; val_tokens: object; cfg: object; report: object
    def has(self, cap):
        return dict(optimizer=self.optimizer is not None, val_tokens=self.val_tokens is not None,
                    blocks=bool(self.anatomy.blocks))[cap]


def _snapshot(cfg): return {f.name: getattr(cfg, f.name) for f in dc_fields(cfg) if f.init}


def _git_sha():
    try: return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, timeout=5).stdout.strip() or 'unknown'
    except Exception: return 'unknown'


class Diagnostics:
    def __init__(self, model, optimizer=None, val_tokens=None, json_path=None, extra_meta=None):
        self.model = model; self.optimizer = optimizer; self.val_tokens = val_tokens
        self.json_path = json_path; self.extra_meta = extra_meta or {}

    def run(self):
        from .stages_static import ParamStatsStage, EscapeStage, QSimStage, OptimStage, SpectrumStage
        from .stages_forward import ForwardStage, EntropyStage, GradStage
        from .stages_sample import SampleStage
        cfg = config.load(); anat = anatomy.probe(self.model)
        meta = dict(timestamp=datetime.now(timezone.utc).isoformat(timespec='seconds'), git_sha=_git_sha(), n_params=anat.n_params,
                    config=dict(data=_snapshot(C.IO), model=_snapshot(C.MODEL), train=_snapshot(C.TRAIN), optim=_snapshot(C.OPTIM), diag=_snapshot(cfg)),
                    anatomy=dict(blocks=[dict(name=b.name, type=type(b.module).__name__, heads=b.heads, kv_heads=b.kv_heads,
                                              head_dim=b.head_dim, hidden=b.hidden, act_out=b.act_out, gate_dim=b.gate_dim,
                                              caps=sorted(b.caps)) for b in anat.blocks]),
                    checks={}, **self.extra_meta)
        rep = report.Report(meta); ctx = RunContext(self.model, anat, self.optimizer, self.val_tokens, cfg, rep)
        _flex_mod._WARNINGS_SHOWN.add('flex_attention_performance')
        was_training = self.model.training; self.model.eval(); print('=== DIAGNOSTICS ===')
        try:
            for stage in (ParamStatsStage(), EscapeStage(), QSimStage(), OptimStage(), SpectrumStage(),
                          ForwardStage(), EntropyStage(), GradStage(), SampleStage()):
                missing = [r for r in stage.requires if not ctx.has(r)]
                if missing: rep.emit(report.Section(stage.name, status=SKIPPED, reason=f"no {missing[0].replace('_', ' ')}")); continue
                try: stage.run(ctx)
                except Exception as e: rep.emit(report.Section(stage.name, status=ERROR, reason=f'{type(e).__name__}: {e}'))
        finally:
            if was_training: self.model.train()
            path = self.json_path or cfg.json_path; rep.write(path)
            print(f'=== END DIAGNOSTICS ===\nwrote {path}')
        return rep


def run_diagnostics(model, optimizer=None, val_tokens=None, json_path=None, extra_meta=None):
    try: return Diagnostics(model, optimizer, val_tokens, json_path, extra_meta).run()
    except Exception as e: print(f'diagnostics failed: {type(e).__name__}: {e}')
