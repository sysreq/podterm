from __future__ import annotations

import hashlib
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path

from podterm.config import EVENTD_PORT

from .state import _CACHE_DIR

log = logging.getLogger("podterm.snapshots")

# Stream the checkpoint in modest chunks so a multi-GB file never lands in RAM
# all at once.
_CHUNK_BYTES = 1 << 20  # 1 MiB

# Defaults for the env-tunable knobs below. Kept local (config.py is owned by
# another remediation unit) but follow its same os.environ idiom.
_DEFAULT_MAX_SNAPSHOT_BYTES = 10 * (1 << 30)  # 10 GiB
_DEFAULT_RETENTION = 3


def base_url(pod_id: str) -> str:
    override = os.environ.get("PODTERM_EVENTS_URL")
    return override or f"https://{pod_id}-{EVENTD_PORT}.proxy.runpod.net"


def _max_snapshot_bytes() -> int:
    """Cap on a single checkpoint download (DIAG_MAX_SNAPSHOT_BYTES, default 10 GiB).

    Guards local disk/RAM against a runaway or hostile pod. A non-positive or
    unparseable value falls back to the default rather than disabling the cap.
    """
    raw = os.environ.get("DIAG_MAX_SNAPSHOT_BYTES", "")
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_MAX_SNAPSHOT_BYTES
    return val if val > 0 else _DEFAULT_MAX_SNAPSHOT_BYTES


def _retention() -> int:
    """How many newest step*.pt checkpoints to keep per pod (DIAG_SNAPSHOT_RETENTION, default 3).

    Non-positive/unparseable disables pruning (keep everything) so retention can
    never delete the file we just wrote by accident.
    """
    raw = os.environ.get("DIAG_SNAPSHOT_RETENTION", "")
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_RETENTION
    return val if val > 0 else 0


def request(url: str, token: str, timeout: float) -> tuple[int, bytes]:
    """Small request helper that reads the whole body — for short control responses
    (e.g. the /snapshot/ack handshake). Use _stream_to_file for large payloads."""
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {token}", "User-Agent": "podterm/2.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, b""
    except Exception:
        return 0, b""


def _stream_to_file(url: str, token: str, timeout: float, dest: Path, max_bytes: int) -> str:
    """Stream the response body to a temp file beside ``dest`` while hashing it.

    Returns the hex sha256 and atomically publishes to ``dest`` via os.replace.
    Aborts (and removes the temp file) if the body exceeds ``max_bytes`` or a
    declared Content-Length disagrees. Raises RuntimeError on any HTTP/transport
    failure so the caller never publishes a partial checkpoint.
    """
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {token}", "User-Agent": "podterm/2.0"}
    )
    tmp = dest.with_name(dest.name + ".tmp")
    digest = hashlib.sha256()
    written = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = getattr(resp, "status", None)
            if status not in (200, None):
                raise RuntimeError(f"snapshot download failed (status={status})")

            declared = resp.getheader("Content-Length")
            expected: int | None = None
            if declared is not None:
                try:
                    expected = int(declared)
                except ValueError:
                    expected = None
                if expected is not None and expected > max_bytes:
                    raise RuntimeError(
                        f"snapshot too large (Content-Length={expected} > max={max_bytes})"
                    )

            with open(tmp, "wb") as fh:
                while True:
                    chunk = resp.read(_CHUNK_BYTES)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > max_bytes:
                        raise RuntimeError(
                            f"snapshot exceeded max size ({written} > {max_bytes} bytes)"
                        )
                    fh.write(chunk)
                    digest.update(chunk)

        if written == 0:
            raise RuntimeError("snapshot download failed (empty body)")
        if expected is not None and written != expected:
            raise RuntimeError(
                f"snapshot length mismatch (got {written}, Content-Length={expected})"
            )
    except urllib.error.HTTPError as e:
        _unlink_quiet(tmp)
        raise RuntimeError(f"snapshot download failed (status={e.code})") from e
    except Exception:
        _unlink_quiet(tmp)
        raise

    try:
        os.replace(tmp, dest)
    except OSError:
        _unlink_quiet(tmp)
        raise
    return digest.hexdigest()


def _unlink_quiet(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def _prune_old_snapshots(dest_dir: Path, keep: int, protect: Path) -> None:
    """Keep only the ``keep`` newest step*.pt files in ``dest_dir``.

    Never removes ``protect`` (the file just published). Best-effort: a failure
    to stat/delete an old checkpoint must not fail the download.
    """
    if keep <= 0:
        return
    try:
        files = [p for p in dest_dir.glob("step*.pt") if p.is_file()]
    except OSError:
        return
    if len(files) <= keep:
        return

    def _mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:
            return 0.0

    files.sort(key=_mtime, reverse=True)
    for stale in files[keep:]:
        if stale == protect:
            continue
        _unlink_quiet(stale)


def download_snapshot(pod_id: str, token: str, payload: dict) -> Path:
    step = int(payload.get("step", 0))
    dest_dir = _CACHE_DIR / pod_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"step{step}.pt"

    max_bytes = _max_snapshot_bytes()
    got_sha = _stream_to_file(
        f"{base_url(pod_id)}/snapshot?step={step}",
        token,
        timeout=120,
        dest=dest,
        max_bytes=max_bytes,
    )

    want = payload.get("sha256")
    if want and got_sha != want:
        # Don't keep a checkpoint we can't trust.
        _unlink_quiet(dest)
        raise RuntimeError("snapshot sha256 mismatch — refusing to diagnose a corrupt checkpoint")

    _prune_old_snapshots(dest_dir, _retention(), protect=dest)
    return dest
