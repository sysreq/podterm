"""SSH key detection, git branch enumeration, HuggingFace manifest helpers."""

from __future__ import annotations

import json
import os
import subprocess
import urllib.request

from podterm.config import DEFAULT_BRANCH, DEFAULT_DATA_REPO_ID, DEFAULT_DATA_VARIANT, DEFAULT_DATA_VERSION

# ---------------------------------------------------------------------------
# SSH key
# ---------------------------------------------------------------------------


def get_local_pubkey() -> str | None:
    """Read the user's default SSH public key, if it exists."""
    for name in ("id_ed25519.pub", "id_rsa.pub", "id_ecdsa.pub"):
        path = os.path.expanduser(f"~/.ssh/{name}")
        if os.path.exists(path):
            with open(path) as f:
                return f.read().strip()
    return None


def get_ssh_key_path() -> str | None:
    """Return the path to the user's preferred SSH private key."""
    for candidate in ("~/.ssh/id_ed25519", "~/.ssh/id_rsa", "~/.runpod/ssh/RunPod-Key-Go"):
        p = os.path.expanduser(candidate)
        if os.path.exists(p):
            return p
    return None


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def get_git_branches() -> list[str]:
    try:
        result = subprocess.run(
            ["git", "branch", "-a", "--format=%(refname:short)"],
            capture_output=True, text=True, timeout=5,
        )
        seen: set[str] = set()
        branches: list[str] = []
        for b in result.stdout.strip().splitlines():
            b = b.strip()
            if b.startswith("origin/"):
                b = b[len("origin/"):]
            if b in ("HEAD", "") or b in seen:
                continue
            seen.add(b)
            branches.append(b)
        for name in reversed(["trunk", "main", "master"]):
            if name in branches:
                branches.remove(name)
                branches.insert(0, name)
        return branches or [DEFAULT_BRANCH]
    except Exception:
        return [DEFAULT_BRANCH]


# ---------------------------------------------------------------------------
# HuggingFace manifest helpers
# ---------------------------------------------------------------------------

_FALLBACK_VARIANT_INFO = {
    "sp1024": {
        "data_variant": "sp1024",
        "data_path": "./data/datasets/fineweb10B_sp1024",
        "tokenizer_path": "./data/tokenizers/fineweb_1024_bpe.model",
        "vocab_size": "1024",
    },
}


def fetch_manifest(
    repo_id: str = DEFAULT_DATA_REPO_ID,
    revision: str = DEFAULT_DATA_VERSION,
) -> dict | None:
    """Fetch manifest.json from the HF dataset repo."""
    url = f"https://huggingface.co/datasets/{repo_id}/raw/{revision}/datasets/manifest.json"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def build_variant_choices(
    manifest: dict | None,
) -> tuple[list[tuple[str, str]], dict[str, dict]]:
    """Parse manifest into (select_options, variant_lookup).

    select_options: [(display_label, variant_suffix), ...]
    variant_lookup: {suffix: {data_variant, data_path, tokenizer_path, vocab_size}}
    """
    if not manifest:
        opts = [(f"{DEFAULT_DATA_VARIANT} (offline)", DEFAULT_DATA_VARIANT)]
        return opts, _FALLBACK_VARIANT_INFO

    tokenizers = {t["name"]: t for t in manifest.get("tokenizers", [])}
    options: list[tuple[str, str]] = []
    lookup: dict[str, dict] = {}
    for ds in manifest.get("datasets", []):
        name = ds.get("name", "")
        vocab = ds.get("vocab_size", "?")
        suffix = name.removeprefix("fineweb10B_") if name.startswith("fineweb10B_") else name
        tok = tokenizers.get(ds.get("tokenizer_name", ""), {})
        model_path = tok.get("model_path", "")
        stats = ds.get("stats", {})
        train_shards = stats.get("files_train", "?")

        max_shards = int(train_shards) if isinstance(train_shards, int) else 0
        label = f"{suffix} — vocab {vocab}, {train_shards} shards"
        lookup[suffix] = {
            "data_variant": suffix,
            "data_path": f"./data/datasets/{name}",
            "tokenizer_path": f"./data/{model_path}" if model_path else "",
            "vocab_size": str(vocab),
            "max_shards": max_shards,
        }
        options.append((label, suffix))

    if not options:
        opts = [(f"{DEFAULT_DATA_VARIANT} (offline)", DEFAULT_DATA_VARIANT)]
        return opts, _FALLBACK_VARIANT_INFO
    return options, lookup
