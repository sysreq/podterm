from dataclasses import dataclass

import config_gpt


@dataclass(frozen=True)
class DiagnosticsConfig:
    json_path: str = "diagnostics.json"  # DIAG_JSON_PATH
    max_batches: int = 0                 # DIAG_MAX_BATCHES: forward-sweep batch cap, 0 = full val set
    entropy_batches: int = 1             # DIAG_ENTROPY_BATCHES
    sample_tokens: int = 64              # DIAG_SAMPLE_TOKENS: completion length per prompt, 0 disables (one forward per token)


def load(): return config_gpt._with_env(DiagnosticsConfig, prefix="diag_")
