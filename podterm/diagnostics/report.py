import json
from dataclasses import dataclass, field

from .schema import SCHEMA_VERSION, OK, SKIPPED, ERROR, PARTIAL


def fmt_gains(t): return f"[{','.join(f'{v:.3f}' for v in t.data.float().flatten().tolist())}]"
def fmt_stat(p): f = p.data.float(); nz = (f.abs() < 0.1).float().mean().item(); return f"{f.mean():.3f}±{f.std():.3f}(~0:{nz:.0%})"
def fmt_norm(p): f = p.data.float(); return f"{f.mean():.4f}±{f.std():.4f}"
def fmt_full(t): f = t.float(); return f"mean={f.mean():.3f} std={f.std():.3f} min={f.min():.3f} max={f.max():.3f}"


@dataclass
class Row:
    key: str; text: str; data: dict; unverified: bool = False


@dataclass
class Section:
    name: str; status: str = OK; reason: str = ""
    rows: list = field(default_factory=list); notes: list = field(default_factory=list)
    def row(self, key, text, unverified=False, **data): self.rows.append(Row(key, text, data, unverified)); return self


def render(s):
    head = f"-- {s.name} --"
    if s.status == SKIPPED: return f"{head} [skipped: {s.reason}]"
    if s.status == ERROR: return f"{head} [error: {s.reason}]"
    lines = [f"  {'! ' if r.unverified else ''}{r.key + ': ' if r.key else ''}{r.text}" for r in s.rows]
    return "\n".join([head, *lines, *(f"  ! {n}" for n in s.notes)])


class Report:
    def __init__(self, meta=None): self.meta = meta if meta is not None else {}; self.sections = []

    def emit(self, section):
        if section.status == OK and any(r.unverified for r in section.rows): section.status = PARTIAL
        print(render(section), flush=True); self.sections.append(section)

    def to_json(self):
        return dict(version=SCHEMA_VERSION, meta=self.meta,
                    sections=[dict(name=s.name, status=s.status, reason=s.reason,
                                   rows={r.key: ({**r.data, "_unverified": True} if r.unverified else r.data) for r in s.rows if r.data},
                                   notes=s.notes) for s in self.sections])

    def write(self, path):
        with open(path, "w") as f: json.dump(self.to_json(), f, indent=2)
