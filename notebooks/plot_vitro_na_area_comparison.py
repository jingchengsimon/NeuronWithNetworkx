#!/usr/bin/env python3
"""Plot vitro N+A area curves from 1_sing_clus_analysis pkl output.

The figure mirrors the notebook's Fig 1 data convention but keeps only the
area metric. Rows are dendritic and somatic recordings; columns are clustered,
distributed, and the clustered-vs-distributed ratio.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SYN_NUM_LIST = np.arange(0, 72 + 1, 2)
RANGE_LABELS = {0: "Proximal", 1: "Medium", 2: "Distal"}
RANGE_COLORS_BASAL = {
    0: (8 / 255, 48 / 255, 107 / 255),
    1: (31 / 255, 119 / 255, 180 / 255),
    2: (120 / 255, 200 / 255, 255 / 255),
}
RANGE_COLORS_APICAL = {
    0: (100 / 255, 20 / 255, 20 / 255),
    1: (214 / 255, 39 / 255, 40 / 255),
    2: (255 / 255, 120 / 255, 120 / 255),
}
VAL_CTR = 0.0


def color_for_range(region: str, range_idx: int):
    color_map = RANGE_COLORS_APICAL if region == "apical" else RANGE_COLORS_BASAL
    return color_map.get(range_idx, "0.3")


def find_default_pkl() -> Path:
    """Find the newest EPSP backup in likely local locations."""
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    candidates = []
    for folder in [
        script_dir,
        repo_root / "Notebooks",
        repo_root / "results" / "epsps",
        repo_root,
    ]:
        if folder.exists():
            candidates.extend(folder.glob("epsps_backup_*.pkl"))

    if not candidates:
        raise FileNotFoundError(
            "No epsps_backup_*.pkl found. Pass one explicitly with --pkl."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_globals(pkl_path: Path) -> dict[str, object]:
    with pkl_path.open("rb") as f:
        data = pickle.load(f)
    if "globals" not in data:
        raise KeyError(f"{pkl_path} does not contain a 'globals' entry.")
    return data["globals"]


def matrix_key(prefix: str, region: str, metric: str, rec_loc: str, range_idx: int) -> str:
    return f"{prefix}_{region}_{metric}_{rec_loc}_{range_idx}_EPSP_matrix"


def array_key(prefix: str, region: str, metric: str, rec_loc: str, range_idx: int) -> str:
    return f"{prefix}_{region}_{metric}_{rec_loc}_{range_idx}_EPSP_array"


def get_matrix(
    data: dict[str, object],
    prefix: str,
    region: str,
    metric: str,
    rec_loc: str,
    range_idx: int,
) -> np.ndarray | None:
    """Return an epoch-by-synapse matrix when available.

    Older notebook backups sometimes only contain the averaged array; in that
    case this returns a single-row matrix so the plotting path still works.
    """
    key = matrix_key(prefix, region, metric, rec_loc, range_idx)
    if key in data:
        arr = np.asarray(data[key], dtype=float)
    else:
        key = array_key(prefix, region, metric, rec_loc, range_idx)
        if key not in data:
            print(f"Warning: missing {key}")
            return None
        arr = np.asarray(data[key], dtype=float)[None, :]

    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[-1] != len(SYN_NUM_LIST):
        print(f"Warning: {key} has unexpected shape {arr.shape}; expected 37 columns.")
        return None
    return arr


def mean_and_sem(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(matrix, axis=0)
    if matrix.shape[0] <= 1:
        sem = np.zeros_like(mean)
    else:
        sem = np.nanstd(matrix, axis=0, ddof=1) / np.sqrt(matrix.shape[0])
    return mean, sem


def ratio_mean_and_sem(clus_matrix: np.ndarray, distr_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    limit = min(clus_matrix.shape[0], distr_matrix.shape[0])
    clus = clus_matrix[:limit, :]
    distr = distr_matrix[:limit, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (clus - distr) / (clus + distr)
    ratio = np.where(np.isinf(ratio), VAL_CTR, ratio)
    ratio = np.where(np.isnan(ratio), VAL_CTR, ratio)
    return mean_and_sem(ratio)


def style_axis(ax: plt.Axes, title: str, ylabel: str, ylim: tuple[float, float] | None) -> None:
    ax.set_title(title)
    ax.set_xlabel("Number of Synapses")
    ax.set_ylabel(ylabel)
    ax.set_xticks([0, 24, 48, 72])
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2)


def mark_missing(ax: plt.Axes, message: str = "missing data") -> None:
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=ax.transAxes,
        color="0.4",
    )


def add_row_group_title(fig: plt.Figure, axes_row: np.ndarray, title: str) -> None:
    """Place a row title centered above the clustered/distributed subplots."""
    left = axes_row[0].get_position().x0
    right = axes_row[1].get_position().x1
    y = axes_row[0].get_position().y1 + 0.018
    fig.text((left + right) / 2, y, title, ha="center", va="bottom", fontsize=12)


def plot_condition_panel(
    ax: plt.Axes,
    data: dict[str, object],
    prefix: str,
    region: str,
    rec_loc: str,
    ranges: list[int],
) -> None:
    plotted = False
    for range_idx in ranges:
        matrix = get_matrix(data, prefix, region, "area", rec_loc, range_idx)
        if matrix is None:
            continue
        mean, sem = mean_and_sem(matrix)
        color = color_for_range(region, range_idx)
        ax.plot(SYN_NUM_LIST, mean, color=color, linewidth=2.5, label=RANGE_LABELS.get(range_idx))
        ax.fill_between(SYN_NUM_LIST, mean - sem, mean + sem, color=color, alpha=0.2)
        plotted = True
    if not plotted:
        mark_missing(ax)


def plot_ratio_panel(
    ax: plt.Axes,
    data: dict[str, object],
    region: str,
    rec_loc: str,
    ranges: list[int],
    clus_prefix: str,
    distr_prefix: str,
) -> None:
    plotted = False
    for range_idx in ranges:
        clus = get_matrix(data, clus_prefix, region, "area", rec_loc, range_idx)
        distr = get_matrix(data, distr_prefix, region, "area", rec_loc, range_idx)
        if clus is None or distr is None:
            continue
        mean, sem = ratio_mean_and_sem(clus, distr)
        color = color_for_range(region, range_idx)
        ax.plot(SYN_NUM_LIST, mean, color=color, linewidth=2.5, label=RANGE_LABELS.get(range_idx))
        ax.fill_between(SYN_NUM_LIST, mean - sem, mean + sem, color=color, alpha=0.2)
        plotted = True
    ax.axhline(VAL_CTR, color="0.5", linestyle="--", linewidth=1, alpha=0.7)
    if not plotted:
        mark_missing(ax)


def build_figure(
    data: dict[str, object],
    ranges: list[int],
    clus_prefix: str = "vitro_N+A",
    distr_prefix: str = "vitro_N+A_distr",
) -> plt.Figure:
    fig, axes = plt.subplots(4, 3, figsize=(12, 12), sharex=True)

    row_specs = [
        (
            "basal",
            "dend",
            "Dendritic EPSP integral on basal dendrites",
            "Voltage Integral (mV·ms)",
            (-0.5, 9.0),
            False,
            "",
        ),
        (
            "basal",
            "soma",
            "Soma",
            "Voltage Integral (mV·ms)",
            (-0.025, 0.45),
            True,
            "Comparison of soma EPSP integral between\nsynaptic distribution patterns on basal dendrites",
        ),
        (
            "apical",
            "dend",
            "Dendritic EPSP integral on apical dendrites",
            "Voltage Integral (mV·ms)",
            (-0.5, 9.0),
            False,
            "",
        ),
        (
            "apical",
            "soma",
            "Soma",
            "Voltage Integral (mV·ms)",
            (-0.025, 0.45),
            True,
            "Comparison of soma EPSP integral between\nsynaptic distribution patterns on apical dendrites",
        ),
    ]

    for row_idx, (region, rec_loc, row_title, ylabel, ylim, do_comparison, comparison_title) in enumerate(row_specs):
        plot_condition_panel(axes[row_idx, 0], data, clus_prefix, region, rec_loc, ranges)
        plot_condition_panel(axes[row_idx, 1], data, distr_prefix, region, rec_loc, ranges)
        style_axis(axes[row_idx, 0], "Clustered", ylabel, ylim)
        style_axis(axes[row_idx, 1], "Distributed", ylabel, ylim)

        if do_comparison:
            plot_ratio_panel(axes[row_idx, 2], data, region, rec_loc, ranges, clus_prefix, distr_prefix)
            style_axis(axes[row_idx, 2], comparison_title, "Response ratio", (-0.4, 0.3))
        else:
            axes[row_idx, 2].axis("off")

    axes[0, 0].legend(loc="upper left", frameon=False, fontsize=9)
    fig.suptitle("Vitro N+A Area Nonlinearity: Clustered vs Distributed", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.6, w_pad=2.0)
    for row_idx, (_, _, row_title, *_rest) in enumerate(row_specs):
        add_row_group_title(fig, axes[row_idx, :], row_title)
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot vitro N+A dend/soma area curves and clus-distr ratios from EPSP pkl."
    )
    parser.add_argument(
        "--pkl",
        type=Path,
        default=None,
        help="Path to epsps_backup_*.pkl. If omitted, the newest likely local file is used.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../results/vitro_na_area_comparison.pdf"),
        help="Output figure path.",
    )
    parser.add_argument(
        "--ranges",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Branch range indices to plot.",
    )
    parser.add_argument("--show", action="store_true", help="Display the figure after saving.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pkl_path = args.pkl or find_default_pkl()
    print(f"Loading {pkl_path}")

    data = load_globals(pkl_path)
    fig = build_figure(data, ranges=args.ranges)

    output = args.output
    if not output.is_absolute():
        output = Path(__file__).resolve().parent / output
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {output}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
