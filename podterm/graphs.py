"""DPG plot creation, real-time updates, and comparison overlays."""

from __future__ import annotations

from collections import deque

import dearpygui.dearpygui as dpg

from podterm.parser import StepMetric

# ---------------------------------------------------------------------------
# Color palette for comparison overlays
# ---------------------------------------------------------------------------

COLORS = [
    (255, 209, 102),  # yellow
    (255, 107, 107),  # red
    (72, 219, 251),   # cyan
    (129, 236, 236),  # teal
    (162, 155, 254),  # purple
    (253, 121, 168),  # pink
]


# ---------------------------------------------------------------------------
# Live plot state for a single running pod
# ---------------------------------------------------------------------------


class LivePlots:
    """Manages the three real-time plots for a single training run."""

    def __init__(self, parent: int | str) -> None:
        self._parent = parent
        self._train_steps: list[float] = []
        self._train_loss: list[float] = []
        self._time_steps: list[float] = []
        self._step_avg: list[float] = []
        self._val_steps: list[float] = []
        self._val_bpb: list[float] = []
        self._pending: deque[StepMetric] = deque()

        # -- Loss plot --
        with dpg.group(horizontal=True, parent=parent):
            with dpg.plot(label="Training Loss", width=-1, height=200):
                self._loss_x = dpg.add_plot_axis(dpg.mvXAxis, label="Step")
                with dpg.plot_axis(dpg.mvYAxis, label="Loss") as self._loss_y:
                    self._train_line = dpg.add_line_series([], [], label="train_loss")
                dpg.add_plot_legend()

        # -- Step avg plot --
        with dpg.group(horizontal=True, parent=parent):
            with dpg.plot(label="Step Time", width=-1, height=160):
                self._time_x = dpg.add_plot_axis(dpg.mvXAxis, label="Step")
                with dpg.plot_axis(dpg.mvYAxis, label="ms/step") as self._time_y:
                    self._time_line = dpg.add_line_series([], [], label="step_avg")
                dpg.add_plot_legend()

        # -- Val BPB plot --
        with dpg.group(horizontal=True, parent=parent):
            with dpg.plot(label="Validation BPB", width=-1, height=160):
                self._bpb_x = dpg.add_plot_axis(dpg.mvXAxis, label="Step")
                with dpg.plot_axis(dpg.mvYAxis, label="BPB") as self._bpb_y:
                    self._bpb_line = dpg.add_scatter_series([], [], label="val_bpb")
                dpg.add_plot_legend()

    def push(self, m: StepMetric) -> None:
        """Queue a metric for the next render update."""
        self._pending.append(m)

    def update(self) -> None:
        """Drain pending metrics and refresh plot data. Call from render loop."""
        if not self._pending:
            return

        while self._pending:
            m = self._pending.popleft()
            if m.train_loss and m.train_loss > 0:
                self._train_steps.append(float(m.step))
                self._train_loss.append(m.train_loss)
            # step_avg is reported on every metric line (both train and val)
            self._time_steps.append(float(m.step))
            self._step_avg.append(m.step_avg_ms)
            if m.val_bpb is not None:
                self._val_steps.append(float(m.step))
                self._val_bpb.append(m.val_bpb)

        # Update series data
        if self._train_steps:
            dpg.set_value(self._train_line, [list(self._train_steps), list(self._train_loss)])
            dpg.fit_axis_data(self._loss_x)
            dpg.fit_axis_data(self._loss_y)

        if self._time_steps:
            dpg.set_value(self._time_line, [list(self._time_steps), list(self._step_avg)])
            dpg.fit_axis_data(self._time_x)
            dpg.fit_axis_data(self._time_y)

        if self._val_steps:
            dpg.set_value(self._bpb_line, [list(self._val_steps), list(self._val_bpb)])
            dpg.fit_axis_data(self._bpb_x)
            dpg.fit_axis_data(self._bpb_y)

    def clear(self) -> None:
        """Reset all data (e.g. on pod restart)."""
        self._train_steps.clear()
        self._train_loss.clear()
        self._time_steps.clear()
        self._step_avg.clear()
        self._val_steps.clear()
        self._val_bpb.clear()
        self._pending.clear()


# ---------------------------------------------------------------------------
# Comparison overlay — multiple historical runs on shared axes
# ---------------------------------------------------------------------------


def create_comparison_plots(parent: int | str, runs_data: dict[str, list[dict]], run_labels: dict[str, str]) -> None:
    """Create overlay plots comparing multiple runs.

    Args:
        parent: DPG container to add plots to.
        runs_data: {run_id: [metric_dicts]} from db.get_metrics_multi().
        run_labels: {run_id: display_name} for legend entries.
    """
    # Clear existing children
    for child in dpg.get_item_children(parent, 1) or []:
        dpg.delete_item(child)

    if not runs_data:
        dpg.add_text("Select runs from the History tab to compare.", parent=parent)
        return

    # -- Loss comparison --
    with dpg.plot(label="Training Loss Comparison", width=-1, height=250, parent=parent):
        x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="Step")
        with dpg.plot_axis(dpg.mvYAxis, label="Loss") as y_axis:
            for i, (run_id, metrics) in enumerate(runs_data.items()):
                steps = [m["step"] for m in metrics if m.get("train_loss")]
                losses = [m["train_loss"] for m in metrics if m.get("train_loss")]
                if steps:
                    color = COLORS[i % len(COLORS)]
                    label = run_labels.get(run_id, run_id[:8])
                    dpg.add_line_series(steps, losses, label=label, parent=y_axis)
                    # Apply color to last added item
                    last = dpg.last_item()
                    dpg.bind_item_theme(last, _get_line_theme(color))
        dpg.add_plot_legend()
        dpg.fit_axis_data(x_axis)
        dpg.fit_axis_data(y_axis)

    # -- BPB comparison --
    with dpg.plot(label="Validation BPB Comparison", width=-1, height=250, parent=parent):
        x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="Step")
        with dpg.plot_axis(dpg.mvYAxis, label="BPB") as y_axis:
            for i, (run_id, metrics) in enumerate(runs_data.items()):
                steps = [m["step"] for m in metrics if m.get("val_bpb")]
                bpbs = [m["val_bpb"] for m in metrics if m.get("val_bpb")]
                if steps:
                    color = COLORS[i % len(COLORS)]
                    label = run_labels.get(run_id, run_id[:8])
                    dpg.add_scatter_series(steps, bpbs, label=label, parent=y_axis)
                    last = dpg.last_item()
                    dpg.bind_item_theme(last, _get_line_theme(color))
        dpg.add_plot_legend()
        dpg.fit_axis_data(x_axis)
        dpg.fit_axis_data(y_axis)

    # -- Summary table --
    dpg.add_spacer(height=10, parent=parent)
    dpg.add_text("Run Summary", parent=parent)
    with dpg.table(header_row=True, parent=parent, resizable=True,
                   borders_innerH=True, borders_outerH=True,
                   borders_innerV=True, borders_outerV=True):
        dpg.add_table_column(label="Run")
        dpg.add_table_column(label="Branch")
        dpg.add_table_column(label="GPU")
        dpg.add_table_column(label="Steps")
        dpg.add_table_column(label="Best BPB")
        dpg.add_table_column(label="Cost")

        # We need run metadata — caller should provide it via run_labels or we look it up
        from podterm import db
        for run_id in runs_data:
            run = db.get_run(run_id)
            if not run:
                continue
            with dpg.table_row():
                dpg.add_text(run_labels.get(run_id, run_id[:8]))
                dpg.add_text(run.get("branch", "-") or "-")
                dpg.add_text(run.get("gpu_type", "-") or "-")
                dpg.add_text(str(run.get("total_steps", "-") or "-"))
                bpb = run.get("best_val_bpb")
                dpg.add_text(f"{bpb:.4f}" if bpb else "-")
                cost = run.get("total_cost")
                dpg.add_text(f"${cost:.4f}" if cost else "-")


# ---------------------------------------------------------------------------
# Theme helpers
# ---------------------------------------------------------------------------

_theme_cache: dict[tuple[int, int, int], int] = {}


def _get_line_theme(color: tuple[int, int, int]) -> int:
    if color in _theme_cache:
        return _theme_cache[color]
    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvLineSeries):
            dpg.add_theme_color(dpg.mvPlotCol_Line, color, category=dpg.mvThemeCat_Plots)
        with dpg.theme_component(dpg.mvScatterSeries):
            dpg.add_theme_color(dpg.mvPlotCol_Line, color, category=dpg.mvThemeCat_Plots)
    _theme_cache[color] = theme
    return theme
