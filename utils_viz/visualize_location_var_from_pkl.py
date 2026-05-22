#!/usr/bin/env python3
"""
Meta-figure plots for location-varied EPSP variability (syn_pos_seed across epochs).

Reads pre-aggregated EPSP matrices from notebooks/1_sing_clus_analysis.py pkl backups.
Each matrix row is one epoch (one synapse-layout draw); column index matches
activated preunit count on SYN_NUM_LIST.

Output is always one 2x2 meta figure per run:
  rows: peak, area
  cols: basal(soma_v), apical(apic_v)

Plot modes:
  - single syn_num: violin (clus vs distr)
  - multiple syn_nums (default): median line + error bars, clus/distr overlaid
  - multiple syn_nums with --plot_violin: grouped violin mode
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils_viz.visualize_soma_peak import _ylabel_for_anal_metric

SYN_NUM_LIST = np.arange(0, 72 + 1, 2)
DEFAULT_SYN_NUM = 72

# basal -> soma_v; apical tuft -> apic_v (notebook rec_loc 'nexus').
# Older pkl backups may store apical under rec_loc 'soma' instead of 'nexus'.
REC_LOC_CANDIDATES_BY_ANAL_LOC: dict[str, tuple[str, ...]] = {
    "basal": ("soma",),
    "apical": ("nexus", "soma"),
}

DEFAULT_CLUS_PREFIX = "vivo_N+A"
DEFAULT_DISTR_PREFIX = "vivo_N+A_distr"
LOCATION_SUFFIX_LABEL = "location"
ANAL_LOCS: tuple[str, str] = ("basal", "apical")
METRICS: tuple[str, str] = ("peak", "area")
CONDITIONS: tuple[str, str] = ("clus", "distr")


def find_default_pkl() -> Path:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    candidates: list[Path] = []
    for folder in [
        repo_root / "notebooks",
        script_dir,
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


def matrix_key(prefix: str, anal_loc: str, metric: str, rec_loc: str, range_idx: int) -> str:
    return f"{prefix}_{anal_loc}_{metric}_{rec_loc}_{range_idx}_EPSP_matrix"


def get_epoch_matrix(
    data: dict[str, object],
    prefix: str,
    anal_loc: str,
    metric: str,
    range_idx: int,
) -> tuple[np.ndarray | None, str | None]:
    for rec_loc in REC_LOC_CANDIDATES_BY_ANAL_LOC[anal_loc]:
        key = matrix_key(prefix, anal_loc, metric, rec_loc, range_idx)
        if key not in data:
            continue
        arr = np.asarray(data[key], dtype=float)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[-1] != len(SYN_NUM_LIST):
            print(f"Warning: {key} shape {arr.shape}; expected {len(SYN_NUM_LIST)} columns.")
            return None, None
        return arr, rec_loc
    return None, None


def build_location_var_dataframe(
    data: dict[str, object],
    *,
    range_idx: int = 1,
    syn_nums: tuple[int, ...] = (DEFAULT_SYN_NUM,),
    anal_locs: tuple[str, ...] = ANAL_LOCS,
    metrics: tuple[str, ...] = METRICS,
    clus_prefix: str = DEFAULT_CLUS_PREFIX,
    distr_prefix: str = DEFAULT_DISTR_PREFIX,
    conditions: tuple[str, str] = CONDITIONS,
) -> pd.DataFrame:
    rows: list[dict] = []
    prefix_by_cond = {"clus": clus_prefix, "distr": distr_prefix}

    for anal_loc in anal_locs:
        for metric in metrics:
            for cond in conditions:
                prefix = prefix_by_cond[cond]
                matrix, rec_loc = get_epoch_matrix(
                    data, prefix, anal_loc, metric, range_idx
                )
                if matrix is None:
                    continue

                for syn_num in syn_nums:
                    syn_col = int(np.where(SYN_NUM_LIST == syn_num)[0][0])
                    values = matrix[:, syn_col]
                    for epoch_idx, val in enumerate(values):
                        if not np.isfinite(val):
                            continue
                        rows.append(
                            {
                                "epoch": epoch_idx + 1,
                                "suffix": LOCATION_SUFFIX_LABEL,
                                "condition": cond,
                                "peak": float(val),
                                "anal_loc": anal_loc,
                                "metric": metric,
                                "rec_loc": rec_loc,
                                "range_idx": range_idx,
                                "syn_num": syn_num,
                            }
                        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            "No EPSP values extracted from pkl. "
            f"Expected keys like {clus_prefix}_basal_peak_soma_{range_idx}_EPSP_matrix. "
            "Re-run 1_sing_clus_analysis with attr peak/area and save_epsps_global()."
        )
    return df


def _syn_num_offsets(n_syn: int) -> np.ndarray:
    if n_syn == 1:
        return np.array([0.0], dtype=float)
    return np.linspace(-0.28, 0.28, n_syn)


def _plot_single_syn_violin(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    conditions: tuple[str, str] = CONDITIONS,
) -> None:
    cond_centers = {"clus": 1.0, "distr": 2.0}
    color_map = {"clus": "tab:red", "distr": "tab:blue"}

    for cond in conditions:
        vals = df[df["condition"] == cond]["peak"].to_numpy()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        position = cond_centers[cond]
        parts = ax.violinplot(
            dataset=[vals],
            positions=[position],
            widths=0.55,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        body = parts["bodies"][0]
        body.set_facecolor(color_map[cond])
        body.set_edgecolor("none")
        body.set_alpha(0.35)

        bp = ax.boxplot(
            [vals],
            positions=[position],
            widths=0.12,
            patch_artist=True,
            showfliers=False,
            whis=1.5,
        )
        for box in bp["boxes"]:
            box.set_facecolor("none")
            box.set_edgecolor("black")
            box.set_linewidth(1.0)
        for med in bp["medians"]:
            med.set_color("black")
            med.set_linewidth(1.0)
        for whisk in bp["whiskers"]:
            whisk.set_color("black")
            whisk.set_linewidth(1.0)
        for cap in bp["caps"]:
            cap.set_color("black")
            cap.set_linewidth(1.0)

    ax.set_xticks([cond_centers[c] for c in conditions])
    ax.set_xticklabels(list(conditions))
    ax.set_xlabel("Condition")


def _plot_multi_syn_violin(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    conditions: tuple[str, str] = CONDITIONS,
) -> None:
    syn_nums = sorted(int(x) for x in df["syn_num"].dropna().unique())
    offsets = _syn_num_offsets(len(syn_nums))
    syn_to_offset = {syn_num: offsets[idx] for idx, syn_num in enumerate(syn_nums)}
    syn_to_alpha = {
        syn_num: 0.25 + 0.45 * (idx + 1) / len(syn_nums)
        for idx, syn_num in enumerate(syn_nums)
    }
    cond_centers = {"clus": 1.0, "distr": 2.0}
    color_map = {"clus": "tab:red", "distr": "tab:blue"}
    violin_width = 0.16 if len(syn_nums) >= 3 else 0.20

    for cond in conditions:
        for syn_num in syn_nums:
            vals = df[
                (df["condition"] == cond) & (df["syn_num"] == syn_num)
            ]["peak"].to_numpy()
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue

            position = cond_centers[cond] + syn_to_offset[syn_num]
            parts = ax.violinplot(
                dataset=[vals],
                positions=[position],
                widths=violin_width,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            body = parts["bodies"][0]
            body.set_facecolor(color_map[cond])
            body.set_edgecolor("none")
            body.set_alpha(syn_to_alpha[syn_num])

            bp = ax.boxplot(
                [vals],
                positions=[position],
                widths=0.06,
                patch_artist=True,
                showfliers=False,
                whis=1.5,
            )
            for box in bp["boxes"]:
                box.set_facecolor("none")
                box.set_edgecolor("black")
                box.set_linewidth(1.0)
            for med in bp["medians"]:
                med.set_color("black")
                med.set_linewidth(1.0)
            for whisk in bp["whiskers"]:
                whisk.set_color("black")
                whisk.set_linewidth(1.0)
            for cap in bp["caps"]:
                cap.set_color("black")
                cap.set_linewidth(1.0)

    legend_handles = [
        Patch(
            facecolor="0.5",
            edgecolor="none",
            alpha=syn_to_alpha[syn_num],
            label=f"syn={syn_num}",
        )
        for syn_num in syn_nums
    ]
    ax.legend(handles=legend_handles, title="Syn num", frameon=False, loc="upper left")
    ax.set_xticks([cond_centers[c] for c in conditions])
    ax.set_xticklabels(list(conditions))
    ax.set_xlabel("Condition")


def _plot_multi_syn_line(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    conditions: tuple[str, str] = CONDITIONS,
) -> None:
    color_map = {"clus": "tab:red", "distr": "tab:blue"}
    syn_nums = sorted(int(x) for x in df["syn_num"].dropna().unique())

    for cond in conditions:
        medians: list[float] = []
        yerr_low: list[float] = []
        yerr_high: list[float] = []
        x_valid: list[int] = []
        for syn_num in syn_nums:
            vals = df[
                (df["condition"] == cond) & (df["syn_num"] == syn_num)
            ]["peak"].to_numpy()
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            q25, q50, q75 = np.percentile(vals, [25, 50, 75])
            x_valid.append(syn_num)
            medians.append(float(q50))
            yerr_low.append(float(q50 - q25))
            yerr_high.append(float(q75 - q50))

        if not x_valid:
            continue
        ax.errorbar(
            x_valid,
            medians,
            yerr=np.vstack([yerr_low, yerr_high]),
            marker="o",
            markersize=4.0,
            linewidth=1.8,
            capsize=3.0,
            color=color_map[cond],
            label=cond,
            zorder=3,
        )

    ax.set_xlabel("Syn num")
    ax.set_xticks(syn_nums)
    ax.legend(frameon=False, loc="best")


def _style_axis(ax: plt.Axes, anal_loc: str, metric: str) -> None:
    ax.set_title(f"{anal_loc} | {metric}")
    ax.set_ylabel(_ylabel_for_anal_metric(anal_loc, metric))
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    

def build_meta_figure(
    df_all: pd.DataFrame,
    *,
    syn_nums: tuple[int, ...],
    plot_violin: bool,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), sharex=False)

    for row_idx, metric in enumerate(METRICS):
        for col_idx, anal_loc in enumerate(ANAL_LOCS):
            ax = axes[row_idx, col_idx]
            sub = df_all[
                (df_all["metric"] == metric) & (df_all["anal_loc"] == anal_loc)
            ]
            if sub.empty:
                ax.text(
                    0.5,
                    0.5,
                    "missing data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color="0.4",
                )
                _style_axis(ax, anal_loc, metric)
                continue

            if len(syn_nums) == 1:
                _plot_single_syn_violin(ax, sub)
            elif plot_violin:
                _plot_multi_syn_violin(ax, sub)
            else:
                _plot_multi_syn_line(ax, sub)
            _style_axis(ax, anal_loc, metric)

    plot_mode = "violin" if (len(syn_nums) == 1 or plot_violin) else "line+errorbar"
    syn_text = ",".join(str(x) for x in syn_nums)
    fig.suptitle(
        f"Location variability ({LOCATION_SUFFIX_LABEL}) | mode={plot_mode} | syn=[{syn_text}]",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def visualize_meta_figure_from_pkl(
    pkl_path: Path,
    *,
    range_idx: int = 1,
    syn_nums: tuple[int, ...] = (DEFAULT_SYN_NUM,),
    output_dir: Path,
    fig_format: str = "pdf",
    clus_prefix: str = DEFAULT_CLUS_PREFIX,
    distr_prefix: str = DEFAULT_DISTR_PREFIX,
    plot_violin: bool = False,
    show: bool = False,
) -> tuple[pd.DataFrame, plt.Figure, Path]:
    for syn_num in syn_nums:
        if syn_num not in SYN_NUM_LIST:
            raise ValueError(f"syn_num must be one of {SYN_NUM_LIST.tolist()}, got {syn_num}")

    data = load_globals(pkl_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = build_location_var_dataframe(
        data,
        range_idx=range_idx,
        syn_nums=syn_nums,
        anal_locs=ANAL_LOCS,
        metrics=METRICS,
        clus_prefix=clus_prefix,
        distr_prefix=distr_prefix,
        conditions=CONDITIONS,
    )
    fig = build_meta_figure(df, syn_nums=syn_nums, plot_violin=plot_violin)

    syn_tag = "_".join(str(x) for x in syn_nums)
    mode_tag = "violin" if (len(syn_nums) == 1 or plot_violin) else "line"
    out_name = f"meta_range{range_idx}_syn{syn_tag}_{mode_tag}_vivo.{fig_format}"
    save_path = output_dir / out_name
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved {save_path}  (n={len(df)} points)")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return df, fig, save_path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_pkl = repo_root / "notebooks" / "epsps_backup_20260519_040744.pkl"
    default_out = repo_root / "results" / "violin_supple" / "location"

    parser = argparse.ArgumentParser(
        description=(
            "Location-varied meta figure from epsps_backup_*.pkl. "
            "Rows: peak/area; columns: basal/apical."
        )
    )
    parser.add_argument(
        "--pkl",
        type=Path,
        default=default_pkl if default_pkl.exists() else None,
        help="Path to epsps_backup_*.pkl (default: notebooks/epsps_backup_20260519_040744.pkl).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=default_out,
        help="Directory for saved figures.",
    )
    parser.add_argument("--range_idx", type=int, default=1, help="Branch range index.")
    parser.add_argument(
        "--syn_num",
        type=int,
        default=None,
        help="Single synapse count (deprecated by --syn_nums).",
    )
    parser.add_argument(
        "--syn_nums",
        type=int,
        nargs="+",
        default=None,
        help="Multiple synapse counts, e.g. --syn_nums 24 48 72.",
    )
    parser.add_argument(
        "--plot_violin",
        action="store_true",
        help="When syn_nums has multiple values, use grouped violin instead of default line+errorbar.",
    )
    parser.add_argument(
        "--clus_prefix",
        default=DEFAULT_CLUS_PREFIX,
        help="Variable name prefix for clustered condition in pkl.",
    )
    parser.add_argument(
        "--distr_prefix",
        default=DEFAULT_DISTR_PREFIX,
        help="Variable name prefix for distributed condition in pkl.",
    )
    parser.add_argument(
        "--fig_format",
        choices=["pdf", "png"],
        default="pdf",
        help="Figure file format.",
    )
    parser.add_argument("--show", action="store_true", help="Call plt.show() after saving.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pkl_path = args.pkl or find_default_pkl()
    print(f"Loading {pkl_path}")

    if args.syn_nums is not None:
        syn_nums = tuple(sorted(dict.fromkeys(args.syn_nums)))
    elif args.syn_num is not None:
        syn_nums = (int(args.syn_num),)
    else:
        syn_nums = (DEFAULT_SYN_NUM,)

    visualize_meta_figure_from_pkl(
        pkl_path,
        range_idx=args.range_idx,
        syn_nums=syn_nums,
        output_dir=args.output_dir,
        fig_format=args.fig_format,
        clus_prefix=args.clus_prefix,
        distr_prefix=args.distr_prefix,
        plot_violin=args.plot_violin,
        show=args.show,
    )


if __name__ == "__main__":
    main()
