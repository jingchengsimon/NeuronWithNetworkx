#!/usr/bin/env python3
"""
Violin plots for location-varied EPSP variability (syn_pos_seed across epochs).

Reads pre-aggregated EPSP matrices from notebooks/1_sing_clus_analysis.py pkl backups.
Each matrix row is one epoch (one synapse-layout draw); column index matches
activated preunit count on SYN_NUM_LIST (default: 72 synapses -> column 36).

Layout matches utils_viz/visualize_soma_peak.py: clus vs distr violins with
inner boxplots, one panel per figure (suffix label "location").
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils_viz.visualize_soma_peak import _ylabel_for_anal_metric, plot_peak_violins

SYN_NUM_LIST = np.arange(0, 72 + 1, 2)
DEFAULT_SYN_NUM = 72
DEFAULT_SYN_COL = int(np.where(SYN_NUM_LIST == DEFAULT_SYN_NUM)[0][0])

# basal -> soma_v; apical tuft -> apic_v (notebook rec_loc 'nexus').
# Older pkl backups may store apical under rec_loc 'soma' instead of 'nexus'.
REC_LOC_CANDIDATES_BY_ANAL_LOC: dict[str, tuple[str, ...]] = {
    "basal": ("soma",),
    "apical": ("nexus", "soma"),
}

DEFAULT_CLUS_PREFIX = "vitro_N+A"
DEFAULT_DISTR_PREFIX = "vitro_N+A_distr"
LOCATION_SUFFIX_LABEL = "location"


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
    syn_col: int = DEFAULT_SYN_COL,
    anal_locs: tuple[str, ...] = ("basal", "apical"),
    metrics: tuple[str, ...] = ("peak", "area"),
    clus_prefix: str = DEFAULT_CLUS_PREFIX,
    distr_prefix: str = DEFAULT_DISTR_PREFIX,
    conditions: tuple[str, str] = ("clus", "distr"),
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
                            "syn_num": int(SYN_NUM_LIST[syn_col]),
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


def visualize_location_var_from_pkl(
    pkl_path: Path,
    *,
    range_idx: int = 1,
    syn_num: int = DEFAULT_SYN_NUM,
    anal_locs: tuple[str, ...] = ("basal", "apical"),
    metrics: tuple[str, ...] = ("peak", "area"),
    output_dir: Path,
    fig_format: str = "pdf",
    clus_prefix: str = DEFAULT_CLUS_PREFIX,
    distr_prefix: str = DEFAULT_DISTR_PREFIX,
    show: bool = False,
) -> dict[str, tuple[pd.DataFrame, plt.Figure]]:
    if syn_num not in SYN_NUM_LIST:
        raise ValueError(f"syn_num must be one of {SYN_NUM_LIST.tolist()}, got {syn_num}")
    syn_col = int(np.where(SYN_NUM_LIST == syn_num)[0][0])

    data = load_globals(pkl_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, tuple[pd.DataFrame, plt.Figure]] = {}
    conditions = ("clus", "distr")

    for anal_loc in anal_locs:
        for metric in metrics:
            df = build_location_var_dataframe(
                data,
                range_idx=range_idx,
                syn_col=syn_col,
                anal_locs=(anal_loc,),
                metrics=(metric,),
                clus_prefix=clus_prefix,
                distr_prefix=distr_prefix,
                conditions=conditions,
            )

            fig, _axes = plot_peak_violins(df, conditions=conditions)
            fig.suptitle(
                f"Location variability ({LOCATION_SUFFIX_LABEL}), "
                f"range {range_idx}, {syn_num} synapses",
                fontsize=11,
                y=1.02,
            )

            out_name = f"{anal_loc}_range{range_idx}_{metric}_violin_median.{fig_format}"
            save_path = output_dir / out_name
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved {save_path}  (n={len(df)} points)")

            if show:
                plt.show()
            else:
                plt.close(fig)

            results[f"{anal_loc}_{metric}"] = (df, fig)

    return results


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_pkl = repo_root / "notebooks" / "epsps_backup_20260519_040744.pkl"
    default_out = repo_root / "results" / "violin_supple" / "location"

    parser = argparse.ArgumentParser(
        description=(
            "Violin plots for location-varied EPSP peak/area at fixed branch range "
            "and synapse count, from epsps_backup_*.pkl."
        )
    )
    parser.add_argument(
        "--pkl",
        type=Path,
        default=default_pkl if default_pkl.exists() else None,
        help="Path to epsps_backup_*.pkl (default: notebooks/epsps_backup_20260519_040744.pkl).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_out,
        help="Directory for saved figures.",
    )
    parser.add_argument("--range-idx", type=int, default=1, help="Branch range index.")
    parser.add_argument(
        "--syn-num",
        type=int,
        default=DEFAULT_SYN_NUM,
        help="Activated synapse / preunit count (must be on 0,2,...,72 grid).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["peak", "area"],
        default=["peak", "area"],
        help="Metrics to plot (skips missing keys in pkl with a warning).",
    )
    parser.add_argument(
        "--anal-locs",
        nargs="+",
        choices=["basal", "apical"],
        default=["basal", "apical"],
        help="Branch region (basal=soma_v, apical=apic_v via nexus rec_loc).",
    )
    parser.add_argument(
        "--clus-prefix",
        default=DEFAULT_CLUS_PREFIX,
        help="Variable name prefix for clustered condition in pkl.",
    )
    parser.add_argument(
        "--distr-prefix",
        default=DEFAULT_DISTR_PREFIX,
        help="Variable name prefix for distributed condition in pkl.",
    )
    parser.add_argument(
        "--format",
        choices=["pdf", "png"],
        default="pdf",
        dest="fig_format",
        help="Figure file format.",
    )
    parser.add_argument("--show", action="store_true", help="Call plt.show() after saving.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pkl_path = args.pkl or find_default_pkl()
    print(f"Loading {pkl_path}")

    data = load_globals(pkl_path)
    available_metrics: list[str] = []
    missing_metrics: list[str] = []
    for metric in args.metrics:
        found = False
        for anal_loc in args.anal_locs:
            for rec_loc in REC_LOC_CANDIDATES_BY_ANAL_LOC[anal_loc]:
                for prefix in (args.clus_prefix, args.distr_prefix):
                    key = matrix_key(prefix, anal_loc, metric, rec_loc, args.range_idx)
                    if key in data:
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if found:
            available_metrics.append(metric)
        else:
            missing_metrics.append(metric)

    if missing_metrics:
        print(
            f"Warning: no pkl keys for metrics {missing_metrics} "
            f"(range {args.range_idx}). Skipping those figures."
        )
    if not available_metrics:
        raise RuntimeError(
            f"No requested metrics found in {pkl_path}. "
            "Available keys sample: "
            + ", ".join(sorted(data.keys())[:5])
            + " ..."
        )

    visualize_location_var_from_pkl(
        pkl_path,
        range_idx=args.range_idx,
        syn_num=args.syn_num,
        anal_locs=tuple(args.anal_locs),
        metrics=tuple(available_metrics),
        output_dir=args.output_dir,
        fig_format=args.fig_format,
        clus_prefix=args.clus_prefix,
        distr_prefix=args.distr_prefix,
        show=args.show,
    )


if __name__ == "__main__":
    main()
