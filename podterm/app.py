"""Main Dear PyGui application — window layout, callbacks, render loop."""

from __future__ import annotations

import queue
import threading
import time

import dearpygui.dearpygui as dpg

from podterm import db
from podterm.config import (
    DEFAULT_CLOUD_TYPE,
    DEFAULT_DATA_REPO_ID,
    DEFAULT_DATA_VARIANT,
    DEFAULT_DATA_VERSION,
    DEFAULT_IMAGE,
    POD_PREFIX,
    build_optional_debug_env,
)
from podterm.dialogs import LaunchDialog, confirm_dialog
from podterm.graphs import LivePlots, create_comparison_plots
from podterm.helpers import get_local_pubkey
from podterm.parser import (
    CommitInfo,
    MemoryInfo,
    ModelInfo,
    PhaseMarker,
    RunSummary,
    StepMetric,
    parse_line,
)
from podterm.runpod import (
    api_create_pod,
    api_terminate_pod,
    create_or_update_template,
    get_gpt_golf_pods,
    get_network_volume,
)
from podterm.ssh import LogQueue, SshTailThread

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------


class App:
    def __init__(self) -> None:
        self._log_queue: LogQueue = queue.Queue()
        self._ui_queue: queue.Queue[tuple[str, ...]] = queue.Queue()  # thread-safe UI action queue
        self._ssh_threads: dict[str, SshTailThread] = {}
        self._live_plots: dict[str, LivePlots] = {}
        self._log_texts: dict[str, int | str] = {}  # pod_id → DPG text widget
        self._last_launch: dict | None = None
        self._launch_dialog: LaunchDialog | None = None
        self._pods: list[dict] = []
        self._pods_dirty = False
        self._selected_pod: str | None = None

        # Run tracking for live pods
        self._run_metrics_buffer: dict[str, list[StepMetric]] = {}
        self._run_memory: dict[str, MemoryInfo] = {}
        self._run_summary: dict[str, RunSummary] = {}
        self._run_commit: dict[str, CommitInfo] = {}
        self._run_model: dict[str, ModelInfo] = {}
        self._run_exit_code: dict[str, int] = {}

        # History comparison state
        self._selected_runs: list[str] = []
        self._history_runs: list[dict] = []

    # -----------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------

    def setup(self) -> None:
        """Create the DPG context, viewport, and all widgets."""
        dpg.create_context()
        dpg.create_viewport(title="PodTerm v2.0", width=1280, height=800)

        # Global theme
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 3)
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 8, 8)
        dpg.bind_theme(global_theme)

        # Main window
        with dpg.window(tag="primary_window"):
            with dpg.group(horizontal=True):
                # Left sidebar — pod list
                with dpg.child_window(width=220, tag="sidebar"):
                    dpg.add_text("Pods", color=(180, 180, 180))
                    dpg.add_separator()
                    dpg.add_child_window(tag="pod_list_area", height=-110, border=False)
                    dpg.add_separator()
                    dpg.add_button(label="Launch", callback=self._on_launch_click, width=-1)
                    dpg.add_button(label="Stop", callback=self._on_stop_click, width=-1)
                    dpg.add_button(label="Refresh", callback=self._on_refresh_click, width=-1)
                    dpg.add_spacer(height=5)
                    dpg.add_text("", tag="status_text", color=(140, 140, 140))

                # Right main area — tabs
                with dpg.child_window(tag="main_area"):
                    with dpg.tab_bar(tag="main_tabs"):
                        with dpg.tab(label="Live", tag="tab_live"):
                            dpg.add_text("Select a running pod or launch a new experiment.",
                                         tag="live_placeholder")
                            dpg.add_child_window(tag="live_graphs_area", height=540, show=False, border=False)
                            dpg.add_child_window(tag="live_log_area", show=False, border=False)

                        with dpg.tab(label="History", tag="tab_history"):
                            dpg.add_text("Filter")
                            dpg.add_input_text(tag="history_filter", width=300,
                                               hint="branch or GPU name...",
                                               callback=self._on_history_filter)
                            dpg.add_spacer(height=5)
                            dpg.add_child_window(tag="history_table_area", border=False)

                        with dpg.tab(label="Compare", tag="tab_compare"):
                            dpg.add_text("Select runs from History tab, then click Compare.",
                                         tag="compare_placeholder")
                            dpg.add_child_window(tag="compare_area", height=-1, border=False)

        dpg.set_primary_window("primary_window", True)
        dpg.setup_dearpygui()
        dpg.show_viewport()

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------

    def run(self) -> None:
        """Start the render loop with periodic callbacks."""
        last_refresh = 0.0
        last_history = 0.0

        # Initial refresh
        self._fetch_pods_background()
        self._refresh_history()

        while dpg.is_dearpygui_running():
            now = time.monotonic()

            # Drain log queue every frame
            self._drain_logs()

            # Drain UI action queue (thread-safe → main thread)
            self._drain_ui_queue()

            # Re-render pod list if data changed
            if self._pods_dirty:
                self._render_pod_list()
                self._pods_dirty = False

            # Update live plots
            for plots in self._live_plots.values():
                plots.update()

            # Refresh pod list every 15 seconds
            if now - last_refresh > 15:
                self._fetch_pods_background()
                last_refresh = now

            # Refresh history every 30 seconds
            if now - last_history > 30:
                self._refresh_history()
                last_history = now

            dpg.render_dearpygui_frame()

        # Cleanup
        self._cleanup()

    def _drain_ui_queue(self) -> None:
        """Execute pending UI actions queued by background threads. Runs on main thread."""
        for _ in range(50):
            try:
                action = self._ui_queue.get_nowait()
            except queue.Empty:
                break

            cmd = action[0]
            if cmd == "show_live":
                pod_id = action[1]
                self._selected_pod = pod_id
                self._show_live_view(pod_id)
            elif cmd == "set_status":
                text = action[1]
                if dpg.does_item_exist("status_text"):
                    dpg.set_value("status_text", text)
            elif cmd == "refresh_history":
                self._refresh_history()

    def _cleanup(self) -> None:
        for t in self._ssh_threads.values():
            t.stop()
        db.close()
        dpg.destroy_context()

    # -----------------------------------------------------------------------
    # Log drain
    # -----------------------------------------------------------------------

    def _drain_logs(self) -> None:
        """Drain the log queue and dispatch to parser, DB, graphs, and log viewer."""
        batch_count = 0
        metrics_to_flush: dict[str, list[StepMetric]] = {}

        while batch_count < 500:
            try:
                pod_id, line = self._log_queue.get_nowait()
            except queue.Empty:
                break
            batch_count += 1

            # Append to log viewer
            text_widget = self._log_texts.get(pod_id)
            if text_widget and dpg.does_item_exist(text_widget):
                # Strip Textual markup tags for display
                clean = line.replace("[bold]", "").replace("[/bold]", "")
                clean = clean.replace("[green]", "").replace("[/green]", "")
                clean = clean.replace("[red]", "").replace("[/red]", "")
                clean = clean.replace("[yellow]", "").replace("[/yellow]", "")
                clean = clean.replace("[cyan]", "").replace("[/cyan]", "")
                clean = clean.replace("[bold cyan]", "").replace("[/bold cyan]", "")
                current = dpg.get_value(text_widget) or ""
                # Keep last ~500 lines
                lines = current.split("\n")
                if len(lines) > 500:
                    lines = lines[-400:]
                lines.append(clean)
                dpg.set_value(text_widget, "\n".join(lines))

            # Parse
            event = parse_line(line)
            if event is None:
                continue

            if isinstance(event, StepMetric):
                # Update graph
                plots = self._live_plots.get(pod_id)
                if plots:
                    plots.push(event)
                # Buffer for DB
                metrics_to_flush.setdefault(pod_id, []).append(event)
                # Update info panel
                self._update_live_info(pod_id, event)

            elif isinstance(event, MemoryInfo):
                self._run_memory[pod_id] = event

            elif isinstance(event, RunSummary):
                self._run_summary[pod_id] = event

            elif isinstance(event, ModelInfo):
                self._run_model[pod_id] = event
                db.update_run(pod_id, model_params=event.params)

            elif isinstance(event, CommitInfo):
                self._run_commit[pod_id] = event
                db.update_run(pod_id, commit_hash=event.hash, commit_msg=event.message)

            elif isinstance(event, PhaseMarker):
                if event.exit_code is not None:
                    self._run_exit_code[pod_id] = event.exit_code
                if "Starting Training" in event.phase:
                    # Reset plots on restart
                    plots = self._live_plots.get(pod_id)
                    if plots:
                        plots.clear()
                elif "Training finished" in event.phase:
                    self._finalize_run(pod_id)

        # Batch-write metrics to DB
        for pod_id, metrics in metrics_to_flush.items():
            try:
                db.add_metrics_batch(pod_id, metrics)
            except Exception:
                pass

    def _update_live_info(self, pod_id: str, m: StepMetric) -> None:
        """Update the info panel text for a live run."""
        tag = f"info_{pod_id}"
        if dpg.does_item_exist(tag):
            lines = [
                f"Step: {m.step}/{m.total_steps}",
                f"Avg: {m.step_avg_ms:.1f} ms/step",
            ]
            if m.train_loss and m.train_loss > 0:
                lines.append(f"Loss: {m.train_loss:.4f}")
            if m.val_bpb is not None:
                lines.append(f"Val BPB: {m.val_bpb:.4f}")
            mem = self._run_memory.get(pod_id)
            if mem:
                lines.append(f"Memory: {mem.peak_mib} MiB")
            dpg.set_value(tag, "\n".join(lines))

    def _finalize_run(self, pod_id: str) -> None:
        """Write final summary to DB when training completes."""
        try:
            summary = self._run_summary.get(pod_id)
            memory = self._run_memory.get(pod_id)
            exit_code = self._run_exit_code.get(pod_id)
            db.finish_run(pod_id, summary=summary, memory=memory, exit_code=exit_code)
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Pod list
    # -----------------------------------------------------------------------

    def _fetch_pods_background(self) -> None:
        """Start a background thread to fetch pod data."""
        def _fetch():
            try:
                self._pods = get_gpt_golf_pods()
            except Exception:
                self._pods = []
            self._pods_dirty = True
        threading.Thread(target=_fetch, daemon=True).start()

    def _render_pod_list(self) -> None:
        area = "pod_list_area"
        if not dpg.does_item_exist(area):
            return

        # Clear existing pod buttons
        for child in dpg.get_item_children(area, 1) or []:
            dpg.delete_item(child)

        for p in self._pods:
            pod_id = p.get("id", "?")
            name = p.get("name", "?")
            status = p.get("desiredStatus", "?")

            # Status indicator
            indicator = "●" if status == "RUNNING" else "○"
            color = (100, 255, 100) if status == "RUNNING" else (180, 180, 180)
            label = f"{indicator} {name[:20]}"

            dpg.add_button(label=label, width=-1, parent=area,
                           callback=self._on_pod_select, user_data=pod_id)

            # Auto-connect to running pods
            if status == "RUNNING" and pod_id not in self._ssh_threads:
                self._connect_pod(pod_id, name)

        # Update status text
        running = sum(1 for p in self._pods if p.get("desiredStatus") == "RUNNING")
        if dpg.does_item_exist("status_text"):
            dpg.set_value("status_text", f"{len(self._pods)} pods, {running} running")

    def _on_pod_select(self, sender, app_data, pod_id: str) -> None:
        """Switch live view to the selected pod."""
        self._selected_pod = pod_id
        self._show_live_view(pod_id)
        dpg.set_value("main_tabs", "tab_live")

    def _connect_pod(self, pod_id: str, pod_name: str) -> None:
        """Start SSH log streaming for a pod."""
        if pod_id in self._ssh_threads:
            return

        # Create DB entry
        pod = next((p for p in self._pods if p.get("id") == pod_id), None)
        config = {"gpu": pod.get("gpu", ""), "cost_per_hr": pod.get("costPerHr")} if pod else None
        try:
            db.create_run(pod_id, pod_name, config)
        except Exception:
            pass

        thread = SshTailThread(pod_id, pod_name, self._log_queue)
        self._ssh_threads[pod_id] = thread
        thread.start()

    def _show_live_view(self, pod_id: str) -> None:
        """Set up the live tab for a specific pod."""
        # Hide placeholder
        if dpg.does_item_exist("live_placeholder"):
            dpg.configure_item("live_placeholder", show=False)

        graphs_area = "live_graphs_area"
        log_area = "live_log_area"
        dpg.configure_item(graphs_area, show=True)
        dpg.configure_item(log_area, show=True)

        # Clear and rebuild graphs
        for child in dpg.get_item_children(graphs_area, 1) or []:
            dpg.delete_item(child)

        # Info panel + graphs
        with dpg.group(parent=graphs_area):
            # Pod info at top
            pod = next((p for p in self._pods if p.get("id") == pod_id), None)
            if pod:
                runtime = pod.get("runtime") or {}
                gpus = runtime.get("gpus") or []
                machine = pod.get("machine") or {}
                gpu_name = (gpus[0].get("gpuDisplayName", "-") if gpus
                            else machine.get("gpuDisplayName", pod.get("gpu", "-")))
                cost = pod.get("costPerHr", "?")
                with dpg.group(horizontal=True):
                    dpg.add_text(f"GPU: {gpu_name}")
                    dpg.add_spacer(width=20)
                    dpg.add_text(f"$/hr: {cost}")
                    dpg.add_spacer(width=20)
                    dpg.add_text("", tag=f"info_{pod_id}")
                dpg.add_separator()

        # Create plots
        plots = LivePlots(graphs_area)
        self._live_plots[pod_id] = plots

        # Clear and rebuild log viewer
        for child in dpg.get_item_children(log_area, 1) or []:
            dpg.delete_item(child)

        dpg.add_text("Log Output", parent=log_area, color=(180, 180, 180))
        text_id = dpg.add_input_text(
            multiline=True, readonly=True, width=-1, height=-1,
            default_value="", parent=log_area, tab_input=False,
        )
        self._log_texts[pod_id] = text_id

    # -----------------------------------------------------------------------
    # Actions
    # -----------------------------------------------------------------------

    def _on_launch_click(self, sender=None, app_data=None) -> None:
        self._launch_dialog = LaunchDialog(self._exec_launch, self._last_launch)
        self._launch_dialog.show()

    def _exec_launch(self, cfg: dict) -> None:
        """Launch a pod in a background thread."""
        self._last_launch = cfg
        threading.Thread(target=self._do_launch, args=(cfg,), daemon=True).start()

    def _do_launch(self, cfg: dict) -> None:
        branch = cfg["branch"]
        suffix = cfg["name"] or branch
        pod_name = f"{POD_PREFIX}-{suffix}-{time.strftime('%m%d-%H%M%S')}"
        self._ui_queue.put(("set_status", f"Launching {pod_name}..."))

        try:
            gpu_count = cfg.get("gpu_count", 1)
            train_script = cfg.get("train_script", "train_gpt.py")
            env = {
                "BRANCH": branch,
                "PREP_SHARDS": str(cfg["prep_shards"]),
                "TRAIN_SCRIPT": train_script,
                "NPROC": str(gpu_count),
                "NCCL_IB_DISABLE": "1",
                "HF_HUB_CACHE": "/workspace/.cache/huggingface",
                "DATA_REPO_ID": cfg.get("data_repo_id", DEFAULT_DATA_REPO_ID),
                "DATA_VERSION": cfg.get("data_version", DEFAULT_DATA_VERSION),
                "DATA_PATH": cfg.get("data_path", "./data/datasets/fineweb10B_sp1024"),
                "DATA_VARIANT": cfg.get("data_variant", DEFAULT_DATA_VARIANT),
                "TOKENIZER_PATH": cfg.get("tokenizer_path", "./data/tokenizers/fineweb_1024_bpe.model"),
                "VOCAB_SIZE": cfg.get("vocab_size", "1024"),
                "MAX_WALLCLOCK_SECONDS": str(cfg["time_budget"]),
                "TRAIN_LOG_EVERY": "250",
                "VAL_LOSS_EVERY": "1000",
                "GITHUB_TOKEN": "{{ RUNPOD_SECRET_gh_gpt-golf_token }}",
                "HF_TOKEN": "{{ RUNPOD_SECRET_hf_gpt-golf_token }}",
            }
            if cfg.get("profile_steps", 0) > 0:
                env["GPT_GOLF_PROFILE"] = str(cfg["profile_steps"])
            env.update(build_optional_debug_env(cfg))

            pubkey = get_local_pubkey()
            if pubkey:
                env["PUBLIC_KEY"] = pubkey

            datacenter = cfg["datacenter"]
            network_vol = get_network_volume(datacenter)
            tpl = create_or_update_template(DEFAULT_IMAGE, env)
            pod = api_create_pod(
                pod_name, cfg["gpu"], tpl,
                DEFAULT_CLOUD_TYPE, datacenter, network_vol,
                gpu_count=gpu_count,
            )
            pod_id = pod["id"]
            cost = pod.get("costPerHr", "?")
            cfg["cost_per_hr"] = cost

            self._ui_queue.put(("set_status", f"Pod {pod_id} launched (${cost}/hr)"))

            # Create DB entry with full config
            db.create_run(pod_id, pod_name, cfg)

            # Start SSH streaming (thread-safe — just starts a thread)
            thread = SshTailThread(pod_id, pod_name, self._log_queue)
            self._ssh_threads[pod_id] = thread
            thread.start()

            # Queue UI updates for main thread
            self._pods_dirty = True
            self._fetch_pods_background()
            self._ui_queue.put(("show_live", pod_id))

        except Exception as exc:
            self._ui_queue.put(("set_status", f"Launch failed: {exc}"))

    def _on_stop_click(self, sender=None, app_data=None) -> None:
        pod_id = self._selected_pod
        if not pod_id:
            return
        confirm_dialog(
            "Stop Pod",
            f"Terminate pod {pod_id}?",
            lambda: threading.Thread(target=self._do_stop, args=(pod_id,), daemon=True).start(),
        )

    def _do_stop(self, pod_id: str) -> None:
        self._ui_queue.put(("set_status", f"Stopping {pod_id}..."))
        try:
            api_terminate_pod(pod_id)
            thread = self._ssh_threads.pop(pod_id, None)
            if thread:
                thread.stop()
            self._finalize_run(pod_id)
            self._ui_queue.put(("set_status", f"Terminated {pod_id}"))
            self._fetch_pods_background()
        except Exception as exc:
            self._ui_queue.put(("set_status", f"Stop failed: {exc}"))

    def _on_refresh_click(self, sender=None, app_data=None) -> None:
        self._fetch_pods_background()

    def _set_status(self, text: str) -> None:
        """Set status text. Only call from main thread."""
        if dpg.does_item_exist("status_text"):
            dpg.set_value("status_text", text)

    # -----------------------------------------------------------------------
    # History tab
    # -----------------------------------------------------------------------

    def _refresh_history(self) -> None:
        self._history_runs = db.list_runs(limit=100)
        self._render_history_table()

    def _on_history_filter(self, sender, filter_text) -> None:
        self._render_history_table(filter_text)

    def _render_history_table(self, filter_text: str = "") -> None:
        area = "history_table_area"
        if not dpg.does_item_exist(area):
            return

        for child in dpg.get_item_children(area, 1) or []:
            dpg.delete_item(child)

        runs = self._history_runs
        if filter_text:
            ft = filter_text.lower()
            runs = [r for r in runs if ft in (r.get("branch", "") or "").lower()
                    or ft in (r.get("gpu_type", "") or "").lower()
                    or ft in (r.get("pod_name", "") or "").lower()]

        if not runs:
            dpg.add_text("No runs found.", parent=area)
            return

        with dpg.group(horizontal=True, parent=area):
            dpg.add_button(label="Compare Selected", callback=self._on_compare_click)
            dpg.add_button(label="Clear Selection", callback=self._on_clear_selection)
            dpg.add_text(f"  ({len(self._selected_runs)} selected)", tag="selection_count")

        with dpg.table(header_row=True, parent=area, resizable=True, sortable=False,
                       borders_innerH=True, borders_outerH=True,
                       borders_innerV=True, borders_outerV=True,
                       tag="history_table"):
            dpg.add_table_column(label="Sel", width_fixed=True, init_width_or_weight=30)
            dpg.add_table_column(label="Date", init_width_or_weight=120)
            dpg.add_table_column(label="Branch", init_width_or_weight=100)
            dpg.add_table_column(label="Commit", init_width_or_weight=80)
            dpg.add_table_column(label="GPU", init_width_or_weight=120)
            dpg.add_table_column(label="Variant", init_width_or_weight=80)
            dpg.add_table_column(label="Steps", init_width_or_weight=60)
            dpg.add_table_column(label="Best BPB", init_width_or_weight=80)
            dpg.add_table_column(label="Cost", init_width_or_weight=60)
            dpg.add_table_column(label="Status", init_width_or_weight=60)

            for run in runs:
                run_id = run["run_id"]
                with dpg.table_row():
                    is_selected = run_id in self._selected_runs
                    dpg.add_checkbox(default_value=is_selected,
                                     callback=self._on_run_toggle,
                                     user_data=run_id)

                    started = (run.get("started_at") or "")[:16].replace("T", " ")
                    dpg.add_text(started)
                    dpg.add_text(run.get("branch", "-") or "-")

                    commit = run.get("commit_hash", "-") or "-"
                    dpg.add_text(commit[:7] if len(commit) > 7 else commit)

                    dpg.add_text(run.get("gpu_type", "-") or "-")
                    dpg.add_text(run.get("data_variant", "-") or "-")

                    steps = run.get("total_steps")
                    dpg.add_text(str(steps) if steps else "-")

                    bpb = run.get("best_val_bpb")
                    dpg.add_text(f"{bpb:.4f}" if bpb else "-")

                    cost = run.get("total_cost")
                    dpg.add_text(f"${cost:.3f}" if cost else "-")

                    exit_code = run.get("exit_code")
                    if exit_code is None:
                        status = "running" if run.get("finished_at") is None else "?"
                    elif exit_code == 0:
                        status = "done"
                    else:
                        status = f"exit {exit_code}"
                    dpg.add_text(status)

    def _on_run_toggle(self, sender, is_checked, run_id: str) -> None:
        if is_checked and run_id not in self._selected_runs:
            self._selected_runs.append(run_id)
        elif not is_checked and run_id in self._selected_runs:
            self._selected_runs.remove(run_id)
        if dpg.does_item_exist("selection_count"):
            dpg.set_value("selection_count", f"  ({len(self._selected_runs)} selected)")

    def _on_clear_selection(self, sender=None, app_data=None) -> None:
        self._selected_runs.clear()
        self._render_history_table(dpg.get_value("history_filter") or "")

    # -----------------------------------------------------------------------
    # Compare tab
    # -----------------------------------------------------------------------

    def _on_compare_click(self, sender=None, app_data=None) -> None:
        if len(self._selected_runs) < 2:
            self._set_status("Select at least 2 runs to compare.")
            return

        dpg.set_value("main_tabs", "tab_compare")
        if dpg.does_item_exist("compare_placeholder"):
            dpg.configure_item("compare_placeholder", show=False)

        runs_data = db.get_metrics_multi(self._selected_runs)
        run_labels = {}
        for rid in self._selected_runs:
            run = db.get_run(rid)
            if run:
                branch = run.get("branch", "")
                commit = (run.get("commit_hash", "") or "")[:7]
                run_labels[rid] = f"{branch}@{commit}" if commit else branch or rid[:8]

        create_comparison_plots("compare_area", runs_data, run_labels)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_app() -> None:
    app = App()
    app.setup()
    app.run()
