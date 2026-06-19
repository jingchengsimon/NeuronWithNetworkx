#!/usr/bin/env python3
"""Plot first soma voltage traces for a basal/apical range and mark a stim window."""

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


def collect_soma_traces(
    *,
    root_dir: str,
    sec_type: str,
    range_idx: int,
    condition: str,
    suffix: str,
    aff_label: int,
    stim_idx: int,
    n_traces: int,
) -> tuple[np.ndarray, list[dict]]:
    base_prefix = f"{root_dir.rstrip('/')}/{sec_type}_range{range_idx}_clus_invivo_"
    exp_dir = _exp_dir_from_base(base_prefix, condition, suffix)
    epoch_folders = _find_epoch_folders_two_level(exp_dir)
    if not epoch_folders:
        raise RuntimeError(f"No epoch folders found: {exp_dir}")

    rows: list[dict] = []
    time_ms: np.ndarray | None = None
    for epoch_idx, folder in epoch_folders:
        soma_raw, _, simu_info = _load_run_folder(folder)
        if soma_raw is None or simu_info is None:
            continue
        soma = _normalize_trace_shape(soma_raw)
        n_stim, n_aff, n_trials = soma.shape[1], soma.shape[2], soma.shape[3]
        if stim_idx >= n_stim:
            continue
        aff_idx = _resolve_aff_index_for_label(simu_info, n_aff, aff_label)
        labels = _aff_activation_labels(simu_info, n_aff)
        dt_s = _dt_seconds_from_trace(soma, simu_info)
        dur_ms = _sim_duration_ms(simu_info)
        if time_ms is None:
            time_ms = np.arange(soma.shape[0], dtype=float) * dt_s * 1000.0
        for trial in range(n_trials):
            rows.append(
                {
                    "epoch": int(epoch_idx),
                    "trial": int(trial),
                    "trace": np.asarray(soma[:, stim_idx, aff_idx, trial], dtype=float),
                    "folder": folder,
                    "aff_idx": int(aff_idx),
                    "aff_label": int(labels[aff_idx]),
                    "stim_time_ms": float(_stim_time_ms(simu_info)),
                    "duration_ms": float(dur_ms),
                }
            )
            if len(rows) >= n_traces:
                return time_ms, rows
    if time_ms is None or not rows:
        raise RuntimeError(f"No soma traces extracted from {exp_dir}")
    return time_ms, rows


def plot_conditions(
    *,
    root_dir: str,
    sec_type: str,
    range_idx: int,
    conditions: list[str],
    suffix: str,
    aff_label: int,
    stim_idx: int,
    n_traces: int,
    window_ms: tuple[float, float],
    stim_time_key: str,
    xlim_ms: tuple[float, float] | None,
    output_path: str,
) -> None:
    fig, axes = plt.subplots(
        1,
        len(conditions),
        figsize=(5.8 * len(conditions), 4.2),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes_1d = axes[0]

    all_rows: list[dict] = []
    for ax, condition in zip(axes_1d, conditions):
        time_ms, rows = collect_soma_traces(
            root_dir=root_dir,
            sec_type=sec_type,
            range_idx=range_idx,
            condition=condition,
            suffix=suffix,
            aff_label=aff_label,
            stim_idx=stim_idx,
            n_traces=n_traces,
        )
        all_rows.extend(rows)
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
        ax.set_title(f"{condition}: first {len(rows)} traces")
        ax.set_xlabel("Time (ms)")
        ax.grid(True, axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if xlim_ms is not None:
            ax.set_xlim(*xlim_ms)
        if len(rows) <= 10:
            ax.legend(frameon=False, fontsize=7, loc="best")

    axes_1d[0].set_ylabel("Soma voltage (mV)")
    fig.suptitle(
        f"Soma V | {sec_type} range{range_idx} | aff={aff_label} | {suffix}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {output_path}")
    for condition in conditions:
        rows = [r for r in all_rows if f"/{sec_type}_range{range_idx}_{condition}_invivo_" in r["folder"]]
        epochs = [r["epoch"] for r in rows]
        aff_idxs = sorted(set(int(r["aff_idx"]) for r in rows))
        print(f"{condition}: epochs={epochs}, aff_idx={aff_idxs}, n={len(rows)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root_dir", default="/G/results/simulation_singclus_supple_May26")
    parser.add_argument("--sec_type", choices=["basal", "apical"], default="basal")
    parser.add_argument("--range_idx", type=int, default=1)
    parser.add_argument("--conditions", nargs="+", choices=["clus", "distr"], default=["clus", "distr"])
    parser.add_argument("--suffix", default="singclus_ap")
    parser.add_argument("--aff_label", type=int, default=DEFAULT_AFF_LABEL_FULL)
    parser.add_argument("--stim_idx", type=int, default=0)
    parser.add_argument("--n_traces", type=int, default=10)
    parser.add_argument("--window_ms", nargs=2, type=float, default=list(DEFAULT_WINDOW_MS))
    parser.add_argument("--stim_time_key", default=DEFAULT_STIM_TIME_KEY)
    parser.add_argument("--xlim_ms", nargs=2, type=float, default=[450.0, 650.0])
    parser.add_argument(
        "--output_path",
        default="results/ap_ca_spike/soma_v_basal_range1_aff72_first10_window480_600ms.pdf",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    xlim_ms = None if args.xlim_ms is None else (float(args.xlim_ms[0]), float(args.xlim_ms[1]))
    plot_conditions(
        root_dir=args.root_dir,
        sec_type=args.sec_type,
        range_idx=int(args.range_idx),
        conditions=list(args.conditions),
        suffix=args.suffix,
        aff_label=int(args.aff_label),
        stim_idx=int(args.stim_idx),
        n_traces=int(args.n_traces),
        window_ms=(float(args.window_ms[0]), float(args.window_ms[1])),
        stim_time_key=str(args.stim_time_key),
        xlim_ms=xlim_ms,
        output_path=str(args.output_path),
    )


if __name__ == "__main__":
    main()
