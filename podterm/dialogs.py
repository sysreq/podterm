"""Launch configuration dialog and confirmation popups."""

from __future__ import annotations

from typing import Callable

import dearpygui.dearpygui as dpg

from podterm.config import (
    DEFAULT_BRANCH,
    DEFAULT_DATA_REPO_ID,
    DEFAULT_DATA_VERSION,
    DEFAULT_DATA_VARIANT,
    DEFAULT_PREP_SHARDS,
    DEFAULT_TIME_BUDGET,
    default_compile_debug_enabled,
    default_graph_logs_enabled,
)
from podterm.helpers import (
    build_variant_choices,
    fetch_manifest,
    get_git_branches,
)
from podterm.runpod import get_available_gpus, get_datacenters

# ---------------------------------------------------------------------------
# Launch dialog
# ---------------------------------------------------------------------------

_FALLBACK_VARIANT_INFO = {
    "sp1024": {
        "data_variant": "sp1024",
        "data_path": "./data/datasets/fineweb10B_sp1024",
        "tokenizer_path": "./data/tokenizers/fineweb_1024_bpe.model",
        "vocab_size": "1024",
    },
}


class LaunchDialog:
    """DPG popup window for experiment configuration."""

    def __init__(self, on_launch: Callable[[dict], None], last: dict | None = None) -> None:
        self._on_launch = on_launch
        self._last = last or {}
        self._window: int | str | None = None

        # Fetch data for dropdowns
        manifest = fetch_manifest()
        self._variant_opts, self._variant_lookup = build_variant_choices(manifest)

    def show(self) -> None:
        if self._window and dpg.does_item_exist(self._window):
            dpg.focus_item(self._window)
            return

        last = self._last
        branches = get_git_branches()
        dcs = get_datacenters()
        sel_dc = last.get("datacenter", dcs[0]["id"] if dcs else "")
        gpu_opts = get_available_gpus(sel_dc)

        default_variant = last.get("data_variant",
            DEFAULT_DATA_VARIANT if DEFAULT_DATA_VARIANT in self._variant_lookup else self._variant_opts[0][1])
        default_shards = last.get("prep_shards",
            int(self._variant_lookup.get(default_variant, {}).get("max_shards", 0) * 0.4) or DEFAULT_PREP_SHARDS)
        compile_debug = default_compile_debug_enabled(last)
        graph_logs = default_graph_logs_enabled(last)

        with dpg.window(label="Launch Experiment", modal=True, width=520, height=620,
                        pos=[100, 40], on_close=self._close) as self._window:

            dpg.add_text("Branch")
            dpg.add_combo(branches, default_value=last.get("branch", DEFAULT_BRANCH),
                          tag="launch_branch", width=-1)

            dpg.add_text("Datacenter")
            dc_labels = [f"{dc['id']} ({dc['location']})" for dc in dcs]
            dc_ids = [dc["id"] for dc in dcs]
            default_dc_idx = dc_ids.index(sel_dc) if sel_dc in dc_ids else 0
            dpg.add_combo(dc_labels, default_value=dc_labels[default_dc_idx] if dc_labels else "",
                          tag="launch_dc", width=-1, callback=self._on_dc_change,
                          user_data={"dc_ids": dc_ids, "dc_labels": dc_labels})

            dpg.add_text("GPU")
            gpu_labels = [g[0] for g in gpu_opts]
            gpu_ids = [g[1] for g in gpu_opts]
            default_gpu_label = ""
            if last.get("gpu") in gpu_ids:
                default_gpu_label = gpu_labels[gpu_ids.index(last["gpu"])]
            elif gpu_labels:
                default_gpu_label = gpu_labels[0]
            dpg.add_combo(gpu_labels, default_value=default_gpu_label,
                          tag="launch_gpu", width=-1,
                          user_data={"gpu_labels": gpu_labels, "gpu_ids": gpu_ids})

            dpg.add_text("Data variant")
            var_labels = [v[0] for v in self._variant_opts]
            var_ids = [v[1] for v in self._variant_opts]
            default_var_idx = var_ids.index(default_variant) if default_variant in var_ids else 0
            dpg.add_combo(var_labels, default_value=var_labels[default_var_idx],
                          tag="launch_variant", width=-1, callback=self._on_variant_change,
                          user_data={"var_labels": var_labels, "var_ids": var_ids})

            dpg.add_spacer(height=5)

            # Advanced section (collapsible)
            with dpg.collapsing_header(label="Advanced", default_open=False):
                dpg.add_text("Name (optional)")
                dpg.add_input_text(tag="launch_name", width=-1, hint="defaults to branch name")

                dpg.add_text("GPU count")
                dpg.add_input_int(tag="launch_gpu_count", default_value=last.get("gpu_count", 1),
                                  width=-1, min_value=1, max_value=8)

                dpg.add_text("Train script")
                dpg.add_input_text(tag="launch_train_script",
                                   default_value=last.get("train_script", "train_gpt.py"), width=-1)

                dpg.add_text("Profile steps (0 = disabled)")
                dpg.add_input_int(tag="launch_profile", default_value=last.get("profile_steps", 0),
                                  width=-1, min_value=0)

                dpg.add_checkbox(label="Compile graph debug", tag="launch_compile_debug",
                                 default_value=compile_debug)
                dpg.add_checkbox(label="Graph logs", tag="launch_graph_logs",
                                 default_value=graph_logs)

                dpg.add_text("Time budget (sec)")
                dpg.add_input_int(tag="launch_time_budget",
                                  default_value=last.get("time_budget", DEFAULT_TIME_BUDGET),
                                  width=-1, min_value=60)

                dpg.add_text("Data shards")
                dpg.add_input_int(tag="launch_shards", default_value=default_shards,
                                  width=-1, min_value=1)

                dpg.add_text("Data HF repo")
                dpg.add_input_text(tag="launch_data_repo",
                                   default_value=last.get("data_repo_id", DEFAULT_DATA_REPO_ID), width=-1)

                dpg.add_text("Data version (HF revision)")
                dpg.add_input_text(tag="launch_data_version",
                                   default_value=last.get("data_version", DEFAULT_DATA_VERSION), width=-1)

            dpg.add_spacer(height=10)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Launch", callback=self._do_launch, width=120)
                dpg.add_button(label="Cancel", callback=self._close, width=120)

    def _on_dc_change(self, sender, value, user_data) -> None:
        """Refresh GPU list when datacenter changes."""
        dc_ids = user_data["dc_ids"]
        dc_labels = user_data["dc_labels"]
        idx = dc_labels.index(value) if value in dc_labels else 0
        dc_id = dc_ids[idx]

        gpu_opts = get_available_gpus(dc_id)
        gpu_labels = [g[0] for g in gpu_opts]
        gpu_ids = [g[1] for g in gpu_opts]
        dpg.configure_item("launch_gpu", items=gpu_labels, default_value=gpu_labels[0] if gpu_labels else "")
        dpg.set_item_user_data("launch_gpu", {"gpu_labels": gpu_labels, "gpu_ids": gpu_ids})

    def _on_variant_change(self, sender, value, user_data) -> None:
        """Update default shard count when variant changes."""
        var_labels = user_data["var_labels"]
        var_ids = user_data["var_ids"]
        idx = var_labels.index(value) if value in var_labels else 0
        variant_key = var_ids[idx]
        vinfo = self._variant_lookup.get(variant_key, {})
        max_shards = vinfo.get("max_shards", 0)
        if max_shards:
            dpg.set_value("launch_shards", int(max_shards * 0.4))

    def _do_launch(self) -> None:
        """Collect all form values and call the launch callback."""
        # Resolve GPU ID from label
        gpu_data = dpg.get_item_user_data("launch_gpu")
        gpu_label = dpg.get_value("launch_gpu")
        gpu_labels = gpu_data["gpu_labels"]
        gpu_ids = gpu_data["gpu_ids"]
        gpu_idx = gpu_labels.index(gpu_label) if gpu_label in gpu_labels else 0
        gpu_id = gpu_ids[gpu_idx]

        # Resolve datacenter ID
        dc_data = dpg.get_item_user_data("launch_dc")
        dc_label = dpg.get_value("launch_dc")
        dc_idx = dc_data["dc_labels"].index(dc_label) if dc_label in dc_data["dc_labels"] else 0
        dc_id = dc_data["dc_ids"][dc_idx]

        # Resolve variant
        var_data = dpg.get_item_user_data("launch_variant")
        var_label = dpg.get_value("launch_variant")
        var_idx = var_data["var_labels"].index(var_label) if var_label in var_data["var_labels"] else 0
        variant_key = var_data["var_ids"][var_idx]
        vinfo = self._variant_lookup.get(variant_key, _FALLBACK_VARIANT_INFO.get(DEFAULT_DATA_VARIANT, {}))

        cfg = {
            "branch": dpg.get_value("launch_branch"),
            "name": dpg.get_value("launch_name").strip() or None,
            "datacenter": dc_id,
            "gpu": gpu_id,
            "gpu_count": dpg.get_value("launch_gpu_count"),
            "train_script": dpg.get_value("launch_train_script").strip() or "train_gpt.py",
            "profile_steps": dpg.get_value("launch_profile"),
            "compile_debug": dpg.get_value("launch_compile_debug"),
            "graph_logs": dpg.get_value("launch_graph_logs"),
            "time_budget": dpg.get_value("launch_time_budget"),
            "prep_shards": dpg.get_value("launch_shards"),
            "data_repo_id": dpg.get_value("launch_data_repo").strip() or DEFAULT_DATA_REPO_ID,
            "data_version": dpg.get_value("launch_data_version").strip() or DEFAULT_DATA_VERSION,
            "data_variant": vinfo.get("data_variant", variant_key),
            "data_path": vinfo.get("data_path", ""),
            "tokenizer_path": vinfo.get("tokenizer_path", ""),
            "vocab_size": vinfo.get("vocab_size", ""),
        }

        self._close()
        self._on_launch(cfg)

    def _close(self, sender=None, app_data=None) -> None:
        if self._window and dpg.does_item_exist(self._window):
            dpg.delete_item(self._window)
            self._window = None


# ---------------------------------------------------------------------------
# Confirmation dialog
# ---------------------------------------------------------------------------


def confirm_dialog(title: str, message: str, on_confirm: Callable[[], None]) -> None:
    """Show a simple Yes/No confirmation dialog."""
    def _yes():
        on_confirm()
        dpg.delete_item(win)

    def _no():
        dpg.delete_item(win)

    with dpg.window(label=title, modal=True, width=350, height=120, pos=[200, 200]) as win:
        dpg.add_text(message)
        dpg.add_spacer(height=10)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Yes", callback=_yes, width=80)
            dpg.add_button(label="No", callback=_no, width=80)
