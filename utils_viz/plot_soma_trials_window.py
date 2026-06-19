#!/usr/bin/env python3
"""Plot first soma/apical voltage traces and mark a stimulation window."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils_viz.visualize_ap_ca_spike import (  # noqa: E402
    DEFAULT_AFF_LABEL_FULL,
    DEFAULT_STIM_TIME_KEY,
    DEFAULT_WINDOW_MS,
    _aff_activation_labels,
    _dt_seconds_from_trace,
    _exp_dir_from_base,
    _find_epoch_folders_two_level,
    _load_run_folder,
    _normalize_trace_shape,
    _resolve_aff_index_for_label,
    _sim_duration_ms,
    _stim_time_ms,
)

TRACE_CONFIG = {
    "soma": {
        "title": "Soma V",
        "ylabel": "Soma voltage (mV)",
        "filename_prefix": "soma_v",
    },
    "apic": {
        "title": "Apic V",
        "ylabel": "Apical voltage (mV)",
        "filename_prefix": "apic_v",
    },
}


def _tag_number(x: float) -> str:
    return f"{x:g}".replace("-", "m").replace(".", "p")


def _auto_output_path(
    *,
    output_dir: str,
    trace_type: str,
    sec_type: str,
    range_idxs: list[int],
    aff_label: int,
    n_traces: int,
    win_start_ms: float,
    win_end_ms: float,
    fig_format: str,
) -> str:
    prefix = TRACE_CONFIG[trace_type]["filename_prefix"]
    range_tag = "range" + "_".join(str(int(x)) for x in range_idxs)
    name = (
        f"{prefix}_{sec_type}_{range_tag}_aff{aff_label}_first{n_traces}_"
        f"window{_tag_number(win_start_ms)}_{_tag_number(win_end_ms)}ms.{fig_format}"
    )
    return os.path.join(output_dir, name)


def collect_voltage_traces(
    *,
    root_dir: str,
    trace_type: str,
    sec_type: str,
    range_idx: int,
    condition: str,
    suffix: str,
    aff_label: int,
    stim_idx: int,
    n_traces: int,
    stim_time_key: str,
) -> tuple[np.ndarray, list[dict]]:
    base_prefix = f"{root_dir.rstrip('/')}/{sec_type}_range{range_idx}_clus_invivo_"
    exp_dir = _exp_dir_from_base(base_prefix, condition, suffix)
    epoch_folders = _find_epoch_folders_two_level(exp_dir)
    if not epoch_folders:
        raise RuntimeError(f"No epoch folders found: {exp_dir}")

    rows: list[dict] = []
    time_ms: np.ndarray | None = None
    for epoch_idx, folder in epoch_folders:
        soma_raw, apic_raw, simu_info = _load_run_folder(folder)
        trace_raw = soma_raw if trace_type == "soma" else apic_raw
        if trace_raw is None or simu_info is None:
            continue
        trace = _normalize_trace_shape(trace_raw)
        n_stim, n_aff, n_trials = trace.shape[1], trace.shape[2], trace.shape[3]
        if stim_idx >= n_stim:
            continue
        aff_idx = _resolve_aff_index_for_label(simu_info, n_aff, aff_label)
        labels = _aff_activation_labels(simu_info, n_aff)
        dt_s = _dt_seconds_from_trace(trace, simu_info)
        dur_ms = _sim_duration_ms(simu_info)
        if time_ms is None:
            time_ms = np.arange(trace.shape[0], dtype=float) * dt_s * 1000.0
        for trial in range(n_trials):
            rows.append(
                {
                    "epoch": int(epoch_idx),
                    "trial": int(trial),
                    "trace": np.asarray(trace[:, stim_idx, aff_idx, trial], dtype=float),
                    "folder": folder,
                    "trace_type": trace_type,
                    "aff_idx": int(aff_idx),
                    "aff_label": int(labels[aff_idx]),
                    "stim_time_ms": float(_stim_time_ms(simu_info, stim_time_key=stim_time_key)),
                    "duration_ms": float(dur_ms),
                }
            )
            if len(rows) >= n_traces:
                return time_ms, rows
    if time_ms is None or not rows:
        raise RuntimeError(f"No {trace_type} traces extracted from {exp_dir}")
    return time_ms, rows


def _draw_trace_panel(
    ax: plt.Axes,
    *,
    time_ms: np.ndarray,
    rows: list[dict],
    condition: str,
    range_idx: int,
    window_ms: tuple[float, float],
    xlim_ms: tuple[float, float] | None,
    show_legend: bool,
) -> tuple[float, float]:
    stim_ms = float(rows[0]["stim_time_ms"])
    win_start_ms = stim_ms + window_ms[0]
    win_end_ms = stim_ms + window_ms[1]
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(rows))))
    for idx, row in enumerate(rows):
        ax.plot(
            time_ms,
            row["trace"],
            linewidth=1.0,
            alpha=0.9,
            color=colors[idx % len(colors)],
            label=f"epoch {row['epoch']}",
        )
    ax.axvspan(win_start_ms, win_end_ms, color="0.85", alpha=0.65, zorder=0)
    ax.axvline(stim_ms, color="black", linestyle="--", linewidth=1.0, alpha=0.75)
    ax.text(
        0.02,
        0.96,
        f"window {win_start_ms:g}-{win_end_ms:g} ms",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color="0.25",
    )
    ax.set_title(f"range{range_idx} | {condition}: first {len(rows)} traces")
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if xlim_ms is not None:
        ax.set_xlim(*xlim_ms)
    if show_legend and len(rows) <= 10:
        ax.legend(frameon=False, fontsize=7, loc="best")
    return win_start_ms, win_end_ms


def plot_range_stack(
    *,
    root_dir: str,
    trace_type: str,
    sec_type: str,
    range_idxs: list[int],
    conditions: list[str],
    suffix: str,
    aff_label: int,
    stim_idx: int,
    n_traces: int,
    window_ms: tuple[float, float],
    stim_time_key: str,
    xlim_ms: tuple[float, float] | None,
    output_dir: str,
    fig_format: str,
    output_path: str | None = None,
) -> str:
    fig, axes = plt.subplots(
        len(range_idxs),
        len(conditions),
        figsize=(5.8 * len(conditions), 3.7 * len(range_idxs)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    all_rows: list[dict] = []
    win_start_ms: float | None = None
    win_end_ms: float | None = None
    for row_idx, range_idx in enumerate(range_idxs):
        for col_idx, condition in enumerate(conditions):
            ax = axes[row_idx, col_idx]
            time_ms, rows = collect_voltage_traces(
                root_dir=root_dir,
                trace_type=trace_type,
                sec_type=sec_type,
                range_idx=range_idx,
                condition=condition,
                suffix=suffix,
                aff_label=aff_label,
                stim_idx=stim_idx,
                n_traces=n_traces,
                stim_time_key=stim_time_key,
            )
            all_rows.extend(rows)
            this_win_start_ms, this_win_end_ms = _draw_trace_panel(
                ax,
                time_ms=time_ms,
                rows=rows,
                condition=condition,
                range_idx=range_idx,
                window_ms=window_ms,
                xlim_ms=xlim_ms,
                show_legend=(row_idx == 0 and col_idx == len(conditions) - 1),
            )
            if win_start_ms is None:
                win_start_ms = this_win_start_ms
                win_end_ms = this_win_end_ms
            if col_idx == 0:
                ax.set_ylabel(TRACE_CONFIG[trace_type]["ylabel"])
            if row_idx == len(range_idxs) - 1:
                ax.set_xlabel("Time (ms)")

    fig.suptitle(
        f"{TRACE_CONFIG[trace_type]['title']} | {sec_type} ranges {','.join(str(x) for x in range_idxs)} | aff={aff_label} | {suffix}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    if output_path is None:
        if win_start_ms is None or win_end_ms is None:
            raise RuntimeError("No traces collected; cannot build output filename")
        output_path = _auto_output_path(
            output_dir=output_dir,
            trace_type=trace_type,
            sec_type=sec_type,
            range_idxs=range_idxs,
            aff_label=aff_label,
            n_traces=n_traces,
            win_start_ms=win_start_ms,
            win_end_ms=win_end_ms,
            fig_format=fig_format,
        )
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {output_path}")
    for range_idx in range_idxs:
        for condition in conditions:
            rows = [r for r in all_rows if f"/{sec_type}_range{range_idx}_{condition}_invivo_" in r["folder"]]
            epochs = [r["epoch"] for r in rows]
            aff_idxs = sorted(set(int(r["aff_idx"]) for r in rows))
            print(f"{trace_type} {sec_type} range{range_idx} {condition}: epochs={epochs}, aff_idx={aff_idxs}, n={len(rows)}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root_dir", default="/G/results/simulation_singclus_supple_May26")
    parser.add_argument("--trace_types", "--trace_type", nargs="+", choices=["soma", "apic"], default=["soma"])
    parser.add_argument("--sec_types", "--sec_type", nargs="+", choices=["basal", "apical"], default=["basal"])
    parser.add_argument("--range_idxs", "--range_idx", nargs="+", type=int, default=[1])
    parser.add_argument("--conditions", nargs="+", choices=["clus", "distr"], default=["clus", "distr"])
    parser.add_argument("--suffix", default="singclus_ap")
    parser.add_argument("--aff_label", type=int, default=DEFAULT_AFF_LABEL_FULL)
    parser.add_argument("--stim_idx", type=int, default=0)
    parser.add_argument("--n_traces", type=int, default=10)
    parser.add_argument("--window_ms", nargs=2, type=float, default=list(DEFAULT_WINDOW_MS))
    parser.add_argument("--stim_time_key", default=DEFAULT_STIM_TIME_KEY)
    parser.add_argument("--xlim_ms", nargs=2, type=float, default=[450.0, 650.0])
    parser.add_argument("--output_dir", default="results/ap_ca_spike")
    parser.add_argument("--fig_format", choices=["pdf", "png"], default="pdf")
    parser.add_argument(
        "--output_path",
        default=None,
        help="Optional explicit path, only allowed for a single trace_type/sec_type combination.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    xlim_ms = None if args.xlim_ms is None else (float(args.xlim_ms[0]), float(args.xlim_ms[1]))
    combos = [
        (trace_type, sec_type)
        for trace_type in args.trace_types
        for sec_type in args.sec_types
    ]
    if args.output_path is not None and len(combos) != 1:
        raise SystemExit("--output_path can only be used with exactly one trace_type/sec_type combo")

    range_idxs = [int(x) for x in args.range_idxs]
    for trace_type, sec_type in combos:
        plot_range_stack(
            root_dir=args.root_dir,
            trace_type=trace_type,
            sec_type=sec_type,
            range_idxs=range_idxs,
            conditions=list(args.conditions),
            suffix=args.suffix,
            aff_label=int(args.aff_label),
            stim_idx=int(args.stim_idx),
            n_traces=int(args.n_traces),
            window_ms=(float(args.window_ms[0]), float(args.window_ms[1])),
            stim_time_key=str(args.stim_time_key),
            xlim_ms=xlim_ms,
            output_dir=str(args.output_dir),
            fig_format=str(args.fig_format),
            output_path=None if args.output_path is None else str(args.output_path),
        )


if __name__ == "__main__":
    main()
