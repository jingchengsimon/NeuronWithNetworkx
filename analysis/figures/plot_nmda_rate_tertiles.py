#!/usr/bin/env python3
"""
NMDA spike rate by dendritic segment tertiles (basal: distance to soma; apical: distance to tuft).

From 3_trace_analysis.ipynb: segment_nmda_spike_rate.npz bar plot with proximal / medium / distal legend.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "fig1"


def aggregate_rate_per_segment(rates: np.ndarray) -> np.ndarray:
    """Average NMDA spike rate per segment over (stim, aff, trial) dimensions."""
    r = np.asarray(rates)
    if r.ndim == 5:
        r = r.mean(axis=(2, 3, 4))
    elif r.ndim == 4:
        r = r.mean(axis=(1, 2, 3))
    elif r.ndim == 2:
        r = r.mean(axis=1)
    elif r.ndim == 1:
        pass
    else:
        raise ValueError(f"Unexpected ndim={r.ndim}, shape={r.shape}")
    return np.asarray(r, dtype=float).ravel()


def tertile_means_sem(
    x: np.ndarray, y: np.ndarray, mask: np.ndarray
) -> tuple[list[str], list[float], list[float]]:
    x = np.asarray(x)[mask]
    y = np.asarray(y)[mask]
    if len(y) < 3:
        raise ValueError("Not enough segments for tertiles in this mask.")
    q1, q2 = np.quantile(x, [1 / 3, 2 / 3])
    g0 = y[x <= q1]
    g1 = y[(x > q1) & (x <= q2)]
    g2 = y[x > q2]
    groups = [g0, g1, g2]
    means = [float(np.mean(g)) for g in groups]
    sems = [float(np.std(g, ddof=1) / np.sqrt(len(g))) if len(g) > 1 else 0.0 for g in groups]
    labels = ["proximal", "medium", "distal"]
    return labels, means, sems


def plot_nmda_rate_tertiles(
    segment_nmda_npz: Path,
 *,
    figsize: tuple[float, float] = (4, 3),
) -> plt.Figure:
    blue_start = (15, 60, 90)
    blue_mid = (31, 119, 180)
    blue_end = (120, 200, 255)
    color_list_basal = [
        tuple(np.array([blue_start, blue_mid, blue_end][i]) / 255.0) for i in range(3)
    ]

    red_start = (100, 20, 20)
    red_mid = (214, 39, 40)
    red_end = (255, 120, 120)
    color_list_apical = [
        tuple(np.array([red_start, red_mid, red_end][i]) / 255.0) for i in range(3)
    ]

    d = np.load(segment_nmda_npz, allow_pickle=True)
    region = d["region"].astype(str)
    dist_soma = np.asarray(d["distance_to_soma"], dtype=float)
    dist_tuft = np.asarray(d["distance_to_tuft"], dtype=float)
    rates = d["nmda_spike_rate_hz"]
    rate_per_seg = aggregate_rate_per_segment(rates)

    valid = (
        np.isfinite(rate_per_seg)
        & np.isfinite(dist_soma)
        & np.isfinite(dist_tuft)
    )

    basal_m = valid & (region == "basal")
    apical_m = valid & (region == "apical") & (dist_tuft >= 0)

    _, mean_b, sem_b = tertile_means_sem(dist_soma, rate_per_seg, basal_m)
    _, mean_a3, sem_a3 = tertile_means_sem(dist_tuft, rate_per_seg, apical_m)

    width, col_gap = 1.0, 1.0
    x_basal = np.arange(3) * width
    x_apical = np.arange(3) * width + 3 * width + col_gap

    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(
        x_basal,
        mean_b,
        yerr=sem_b,
        width=width,
        color=color_list_basal,
        capsize=3,
        edgecolor="none",
    )
    ax.bar(
        x_apical,
        mean_a3,
        yerr=sem_a3,
        width=width,
        color=color_list_apical,
        capsize=3,
        edgecolor="none",
    )

    centers = [float(x_basal.mean()), float(x_apical.mean())]
    ax.set_xticks(centers)
    ax.set_xticklabels(["Basal", "Apical"])
    ax.set_ylabel("NMDA spike rate (Hz)")
    ax.set_title("NMDA spike rate across the dendritic tree")

    ax.set_xlim(x_basal.min() - 0.5 * width, x_apical.max() + 0.5 * width)

    handles = [
        (
            mpatches.Patch(facecolor=color_list_basal[i], edgecolor="none"),
            mpatches.Patch(facecolor=color_list_apical[i], edgecolor="none"),
        )
        for i in range(3)
    ]
    labels_leg = ["proximal", "medium", "distal"]

    ax.legend(
        handles,
        labels_leg,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        handlelength=4,
        handletextpad=0.2,
        frameon=False,
        loc="upper left",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def main():
    p = argparse.ArgumentParser(description="NMDA spike rate tertile bar plot (from3_trace_analysis.ipynb).")
    p.add_argument(
        "--segment_nmda_npz",
        type=str,
        default="/G/results/simulation_singclus_supple_Apr26/basal_range0_clus_invivo_singclus_ap_globrec/1/1/segment_nmda_spike_rate.npz",
        help="Path to segment_nmda_spike_rate.npz.",
    )
    p.add_argument(
        "--out_pdf",
        type=str,
        default=str(DEFAULT_OUT_DIR / "fig1_nmda_rate_tertiles.pdf"),
        help="Output PDF path.",
    )
    p.add_argument("--fig_width", type=float, default=4.0)
    p.add_argument("--fig_height", type=float, default=3.0)
    args = p.parse_args()

    npz_path = Path(args.segment_nmda_npz)
    if not npz_path.is_file():
        raise SystemExit(f"Missing NPZ: {npz_path}")

    out_pdf = Path(args.out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    fig = plot_nmda_rate_tertiles(npz_path, figsize=(args.fig_width, args.fig_height))
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
