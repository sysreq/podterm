import json
import subprocess
from dataclasses import dataclass, fields as dc_fields
from datetime import datetime, timezone

import torch
from torch.nn.attention import flex_attention as _flex_mod

import config_gpt as C
from . import anatomy, config, health, report
from .schema import SKIPPED, ERROR


class Stage:
    name = ""
    requires = ()

    def run(self, ctx):
        raise NotImplementedError


@dataclass
class RunContext:
    model: object
    anatomy: object
    optimizer: object
    val_tokens: object
    cfg: object
    report: object

    def has(self, cap):
        available = {
            "optimizer": self.optimizer is not None,
            "val_tokens": self.val_tokens is not None,
            "blocks": bool(self.anatomy.blocks),
        }
        return available[cap]


def _snapshot(cfg):
    return {f.name: getattr(cfg, f.name) for f in dc_fields(cfg) if f.init}


def _git_sha():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


class Diagnostics:
    def __init__(self, model, optimizer=None, val_tokens=None, json_path=None, extra_meta=None):
        self.model = model
        self.optimizer = optimizer
        self.val_tokens = val_tokens
        self.json_path = json_path
        self.extra_meta = extra_meta or {}

    def _stages(self):
        from .stages_static import ParamStatsStage, EscapeStage, QSimStage, OptimStage, SpectrumStage
        from .stages_forward import ForwardStage, EntropyStage, GradStage
        from .stages_sample import SampleStage

        return (
            ParamStatsStage(),
            EscapeStage(),
            QSimStage(),
            OptimStage(),
            SpectrumStage(),
            ForwardStage(),
            EntropyStage(),
            GradStage(),
            SampleStage(),
        )

    def _meta(self, cfg, anat):
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_sha": _git_sha(),
            "n_params": anat.n_params,
            "config": {
                "data": _snapshot(C.IO),
                "model": _snapshot(C.MODEL),
                "train": _snapshot(C.TRAIN),
                "optim": _snapshot(C.OPTIM),
                "diag": _snapshot(cfg),
            },
            "anatomy": {
                "blocks": [
                    {
                        "name": b.name,
                        "type": type(b.module).__name__,
                        "heads": b.heads,
                        "kv_heads": b.kv_heads,
                        "head_dim": b.head_dim,
                        "hidden": b.hidden,
                        "act_out": b.act_out,
                        "gate_dim": b.gate_dim,
                        "caps": sorted(b.caps),
                    }
                    for b in anat.blocks
                ],
            },
            "checks": {},
            **self.extra_meta,
        }

    def run(self):
        cfg = config.load()
        anat = anatomy.probe(self.model)
        rep = report.Report(self._meta(cfg, anat))
        ctx = RunContext(self.model, anat, self.optimizer, self.val_tokens, cfg, rep)
        _flex_mod._WARNINGS_SHOWN.add("flex_attention_performance")
        was_training = self.model.training
        self.model.eval()
        print("=== DIAGNOSTICS ===")
        try:
            for stage in self._stages():
                missing = [r for r in stage.requires if not ctx.has(r)]
                if missing:
                    rep.emit(report.Section(
                        stage.name,
                        status=SKIPPED,
                        reason=f"no {missing[0].replace('_', ' ')}",
                    ))
                    continue
                try:
                    stage.run(ctx)
                except Exception as e:
                    rep.emit(report.Section(
                        stage.name,
                        status=ERROR,
                        reason=f"{type(e).__name__}: {e}",
                    ))
        finally:
            if was_training:
                self.model.train()
            path = self.json_path or cfg.json_path
            # Single source of truth for the verdict: compute health once, fold it into the doc.
            doc = rep.to_json()
            h = health.compute(doc)
            doc["status"] = h["overall"]
            doc["health"] = h
            with open(path, "w") as f:
                json.dump(doc, f, indent=2)
            print(f"=== END DIAGNOSTICS [{h['overall']}] ===\nwrote {path}")
        return rep


def run_diagnostics(model, optimizer=None, val_tokens=None, json_path=None, extra_meta=None):
    try:
        return Diagnostics(model, optimizer, val_tokens, json_path, extra_meta).run()
    except Exception as e:
        print(f"diagnostics failed: {type(e).__name__}: {e}")
