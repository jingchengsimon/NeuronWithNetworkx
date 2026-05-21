#!/usr/bin/env python3
"""
AP and Ca spike rate analysis from soma_v_array / apic_v_array.

Ca spikes (apic_v): same criterion as NMDA spikes via utils.nmda_detection_utils
  (V > -40 mV continuously for >= 26 ms).

AP spikes (soma_v): upward crossings of -10 mV with positive first derivative
  (NEURON-style threshold crossing).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.nmda_detection_utils import (
    DEFAULT_MIN_DURATION_MS,
    DEFAULT_V_THRESH_MV,
    compute_nmda_spike_rate_hz,
    count_nmda_spikes,
)

DEFAULT_AP_THRESH_MV = -10.0
DEFAULT_CA_V_THRESH_MV = DEFAULT_V_THRESH_MV
DEFAULT_CA_MIN_DURATION_MS = DEFAULT_MIN_DURATION_MS


# ---------------------------
# IO helpers (aligned with visualize_soma_peak.py)
# ---------------------------
def _load_run_folder(folder: str) -> tuple[np.ndarray | None, np.ndarray | None, dict | None]:
    soma_path = os.path.join(folder, "soma_v_array.npy")
    apic_path = os.path.join(folder, "apic_v_array.npy")
    info_path = os.path.join(folder, "simulation_params.json")

    if not os.path.exists(info_path):
        return None, None, None

    soma_v = np.load(soma_path) if os.path.exists(soma_path) else None
    apic_v = np.load(apic_path) if os.path.exists(apic_path) else None

    with open(info_path, "r") as f:
        simu_info = json.load(f)

    return soma_v, apic_v, simu_info


def _normalize_trace_shape(trace: np.ndarray) -> np.ndarray:
    """Normalize voltage array to (T, num_stim, num_aff, num_trials)."""
    if trace.ndim == 4:
        return trace
    if trace.ndim == 3:
        return trace[:, np.newaxis, :, :]
    raise ValueError(f"Unsupported trace ndim={trace.ndim}, shape={trace.shape}")


def _dt_seconds_from_trace(trace: np.ndarray, simu_info: dict) -> float:
    n_t = trace.shape[0]
    if n_t < 2:
        return 1.0 / 40000.0
    dur_ms = float(simu_info.get("SIMU DURATION", simu_info.get("SIMU_DURATION", 1000)))
    return (dur_ms / 1000.0) / (n_t - 1)


def _sim_duration_ms(simu_info: dict) -> float:
    return float(simu_info.get("SIMU DURATION", simu_info.get("SIMU_DURATION", 1000)))


def _aff_activation_labels(simu_info: dict, n_aff: int) -> list[int]:
    """
    Map aff axis index -> number of activated preunits.
    Prefer explicit aff_list saved by L5b_simulation.py.
    """
    if "aff_list" in simu_info and simu_info["aff_list"] is not None:
        labels = [int(x) for x in simu_info["aff_list"]]
        if len(labels) == n_aff:
            return labels

    aff_mode = str(simu_info.get("aff_mode", "linear"))
    iter_step = int(simu_info.get("effective_iter_step", simu_info.get("iter_step", 2)))
    num_syn = int(simu_info.get("number of synapses per cluster", 72))
    num_clusters = int(simu_info.get("number of clusters", 1))
    num_preunit = int(num_syn * np.ceil(num_clusters / 3))

    if aff_mode == "full":
        return [num_preunit]
    if aff_mode == "custom":
        raise ValueError("aff_mode=custom but aff_list missing or wrong length in simulation_params.json")
    if aff_mode == "curve":
        dense = list(range(0, iter_step + 1))
        sparse = list(range(iter_step, num_preunit + 1, iter_step))
        labels = sorted(set(dense + sparse))
    else:
        labels = list(range(0, num_preunit + 1, iter_step))

    if len(labels) != n_aff:
        raise ValueError(f"Reconstructed aff labels length {len(labels)} != n_aff={n_aff}")
    return labels


# ---------------------------
# Spike detection
# ---------------------------
def count_ap_spikes(
    v_mV: np.ndarray,
    ap_thresh_mV: float = DEFAULT_AP_THRESH_MV,
) -> int:
    """
    Count AP events as upward threshold crossings with positive first derivative.
    """
    v = np.asarray(v_mV, dtype=np.float64).reshape(-1)
    if v.size < 2:
        return 0

    count = 0
    for i in range(1, v.size):
        dv = v[i] - v[i - 1]
        if v[i - 1] <= ap_thresh_mV and v[i] > ap_thresh_mV and dv > 0:
            count += 1
    return count


def compute_ap_spike_rate_hz(
    v_mV: np.ndarray,
    sim_duration_ms: float,
    ap_thresh_mV: float = DEFAULT_AP_THRESH_MV,
) -> float:
    dur_s = sim_duration_ms / 1000.0
    if dur_s <= 0:
        return 0.0
    return float(count_ap_spikes(v_mV, ap_thresh_mV=ap_thresh_mV)) / dur_s


def compute_ca_spike_rate_hz(
    v_mV: np.ndarray,
    dt_s: float,
    sim_duration_ms: float,
    v_thresh_mV: float = DEFAULT_CA_V_THRESH_MV,
    min_duration_ms: float = DEFAULT_CA_MIN_DURATION_MS,
) -> float:
    """Ca spikes on apic_v use the NMDA-style supra-threshold run criterion."""
    return compute_nmda_spike_rate_hz(
        v_mV,
        dt_s,
        sim_duration_ms,
        v_thresh_mV=v_thresh_mV,
        min_duration_ms=min_duration_ms,
    )


def extract_spike_rates_from_folder(
    folder: str,
    stim_idx: int = 0,
    ap_thresh_mV: float = DEFAULT_AP_THRESH_MV,
    ca_v_thresh_mV: float = DEFAULT_CA_V_THRESH_MV,
    ca_min_duration_ms: float = DEFAULT_CA_MIN_DURATION_MS,
) -> pd.DataFrame | None:
    """
    Per-trial AP and Ca spike rates for one run folder.
    Returns long-form DataFrame with columns:
      aff_idx, aff_label, trial, stim, spike_type, rate_hz
    """
    soma_raw, apic_raw, simu_info = _load_run_folder(folder)
    if simu_info is None:
        return None

    rows: list[dict] = []
    dur_ms = _sim_duration_ms(simu_info)

    if soma_raw is not None:
        soma = _normalize_trace_shape(soma_raw)
        dt_s = _dt_seconds_from_trace(soma, simu_info)
        n_stim, n_aff, n_trials = soma.shape[1], soma.shape[2], soma.shape[3]
        aff_labels = _aff_activation_labels(simu_info, n_aff)
        if stim_idx < 0 or stim_idx >= n_stim:
            raise IndexError(f"stim_idx={stim_idx} out of range for n_stim={n_stim}")

        for aff_i, aff_label in enumerate(aff_labels):
            for trial in range(n_trials):
                trace = soma[:, stim_idx, aff_i, trial]
                rate = compute_ap_spike_rate_hz(trace, dur_ms, ap_thresh_mV=ap_thresh_mV)
                rows.append(
                    dict(
                        aff_idx=aff_i,
                        aff_label=int(aff_label),
                        trial=trial,
                        stim=stim_idx,
                        spike_type="ap",
                        rate_hz=float(rate),
                        n_events=int(count_ap_spikes(trace, ap_thresh_mV=ap_thresh_mV)),
                    )
                )

    if apic_raw is not None:
        apic = _normalize_trace_shape(apic_raw)
        dt_s = _dt_seconds_from_trace(apic, simu_info)
        n_stim, n_aff, n_trials = apic.shape[1], apic.shape[2], apic.shape[3]
        aff_labels = _aff_activation_labels(simu_info, n_aff)
        if stim_idx < 0 or stim_idx >= n_stim:
            raise IndexError(f"stim_idx={stim_idx} out of range for n_stim={n_stim}")

        for aff_i, aff_label in enumerate(aff_labels):
            for trial in range(n_trials):
                trace = apic[:, stim_idx, aff_i, trial]
                rate = compute_ca_spike_rate_hz(
                    trace,
                    dt_s,
                    dur_ms,
                    v_thresh_mV=ca_v_thresh_mV,
                    min_duration_ms=ca_min_duration_ms,
                )
                rows.append(
                    dict(
                        aff_idx=aff_i,
                        aff_label=int(aff_label),
                        trial=trial,
                        stim=stim_idx,
                        spike_type="ca",
                        rate_hz=float(rate),
                        n_events=int(
                            count_nmda_spikes(
                                trace,
                                dt_s,
                                v_thresh_mV=ca_v_thresh_mV,
                                min_duration_ms=ca_min_duration_ms,
                            )
                        ),
                    )
                )

    if not rows:
        return None

    return pd.DataFrame(rows)


def _infer_base_root_from_clus_prefix(base_clus_invivo_prefix: str) -> str:
    s = base_clus_invivo_prefix.rstrip("_")
    m = re.match(r"^(.*)_clus_invivo$", s)
    if not m:
        raise ValueError(
            "base path must look like '*_clus_invivo_' (ending underscore allowed). "
            f"Got: {base_clus_invivo_prefix}"
        )
    return m.group(1)


def _exp_dir_from_base(base_clus_invivo_prefix: str, condition: str, suffix: str) -> str:
    base_root = _infer_base_root_from_clus_prefix(base_clus_invivo_prefix)
    condition = condition.strip()
    suffix = suffix.strip().lstrip("_")
    return f"{base_root}_{condition}_invivo_{suffix}"


def _find_epoch_folders_two_level(exp_dir: str) -> list[tuple[int, str]]:
    out = []
    for epoch_dir in glob.glob(os.path.join(exp_dir, "*", "*")):
        if not os.path.isdir(epoch_dir):
            continue
        ep_name = os.path.basename(epoch_dir)
        if re.fullmatch(r"\d+", ep_name):
            out.append((int(ep_name), epoch_dir))
    out.sort(key=lambda x: x[0])
    return out


def build_spike_rate_dataframe(
    base_clus_invivo_prefix: str,
    suffixes: Iterable[str],
    conditions: tuple[str, str] = ("clus", "distr"),
    stim_idx: int = 0,
    ap_thresh_mV: float = DEFAULT_AP_THRESH_MV,
    ca_v_thresh_mV: float = DEFAULT_CA_V_THRESH_MV,
    ca_min_duration_ms: float = DEFAULT_CA_MIN_DURATION_MS,
) -> pd.DataFrame:
    rows = []
    for suffix in suffixes:
        for cond in conditions:
            exp_dir = _exp_dir_from_base(base_clus_invivo_prefix, cond, suffix)
            epoch_folders = _find_epoch_folders_two_level(exp_dir)
            if not epoch_folders:
                continue

            for epoch_idx, folder in epoch_folders:
                rates_df = extract_spike_rates_from_folder(
                    folder,
                    stim_idx=stim_idx,
                    ap_thresh_mV=ap_thresh_mV,
                    ca_v_thresh_mV=ca_v_thresh_mV,
                    ca_min_duration_ms=ca_min_duration_ms,
                )
                if rates_df is None or rates_df.empty:
                    continue

                for _, r in rates_df.iterrows():
                    rows.append(
                        dict(
                            epoch=epoch_idx,
                            suffix=suffix,
                            condition=cond,
                            aff_idx=int(r["aff_idx"]),
                            aff_label=int(r["aff_label"]),
                            trial=int(r["trial"]),
                            stim=int(r["stim"]),
                            spike_type=str(r["spike_type"]),
                            rate_hz=float(r["rate_hz"]),
                            n_events=int(r["n_events"]),
                            folder=folder,
                        )
                    )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No AP/Ca spike rates extracted. Check paths and recordings.")
    return df


def _summarize_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Median rate and SEM across epochs and trials, per condition / aff / spike_type."""
    grouped = (
        df.groupby(["suffix", "condition", "spike_type", "aff_label", "aff_idx"], as_index=False)
        .agg(
            rate_median=("rate_hz", "median"),
            rate_sem=(
                "rate_hz",
                lambda s: float(np.std(s, ddof=1) / np.sqrt(len(s))) if len(s) > 1 else 0.0,
            ),
            n_epochs=("epoch", "nunique"),
            n_samples=("rate_hz", "count"),
        )
    )
    return grouped


def plot_ap_ca_spike_rates_2x2(
    summary_df: pd.DataFrame,
    suffix: str,
    aff_order: list[int] | None = None,
    figsize: tuple[float, float] = (8.0, 6.0),
) -> tuple[plt.Figure, np.ndarray]:
    """
    2x2 panels: rows = AP / Ca, cols = clus / distr.
    x-axis: aff_label (activated synapse count), y-axis: spike rate (Hz).
    """
    sdf = summary_df[summary_df["suffix"] == suffix].copy()
    if sdf.empty:
        raise ValueError(f"No summary rows for suffix={suffix}")

    if aff_order is None:
        aff_order = sorted(sdf["aff_label"].unique().tolist())

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharey="row")
    panel_map = {
        ("ap", "clus"): axes[0, 0],
        ("ap", "distr"): axes[0, 1],
        ("ca", "clus"): axes[1, 0],
        ("ca", "distr"): axes[1, 1],
    }
    titles = {
        ("ap", "clus"): "AP — clustered",
        ("ap", "distr"): "AP — distributed",
        ("ca", "clus"): "Ca — clustered",
        ("ca", "distr"): "Ca — distributed",
    }
    colors = {"clus": "tab:red", "distr": "tab:blue"}

    x_pos = np.arange(len(aff_order), dtype=float)
    width = 0.35

    for spike_type in ("ap", "ca"):
        for cond in ("clus", "distr"):
            ax = panel_map[(spike_type, cond)]
            sub = sdf[(sdf["spike_type"] == spike_type) & (sdf["condition"] == cond)]

            means = []
            sems = []
            for aff in aff_order:
                row = sub[sub["aff_label"] == aff]
                if row.empty:
                    means.append(np.nan)
                    sems.append(np.nan)
                else:
                    means.append(float(row["rate_median"].iloc[0]))
                    sems.append(float(row["rate_sem"].iloc[0]))

            ax.bar(
                x_pos,
                means,
                yerr=sems,
                width=0.7,
                color=colors.get(cond, "gray"),
                alpha=0.75,
                capsize=3,
                edgecolor="none",
            )
            ax.set_xticks(x_pos)
            ax.set_xticklabels([str(a) for a in aff_order])
            ax.set_xlabel("Activated synapses")
            ax.set_title(titles[(spike_type, cond)])
            ax.grid(True, axis="y", alpha=0.25)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    axes[0, 0].set_ylabel("AP spike rate (Hz)")
    axes[1, 0].set_ylabel("Ca spike rate (Hz)")
    fig.suptitle(f"Spike rate vs activation — suffix: {suffix}", y=1.02)
    fig.tight_layout()
    return fig, axes


def visualize_ap_ca_from_base(
    base_clus_invivo_prefix: str,
    suffixes: list[str],
    conditions: tuple[str, str] = ("clus", "distr"),
    aff_order: list[int] | None = None,
    stim_idx: int = 0,
    save_dir: str | None = None,
    fig_format: str = "pdf",
    show: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[plt.Figure, str]]]:
    raw_df = build_spike_rate_dataframe(
        base_clus_invivo_prefix=base_clus_invivo_prefix,
        suffixes=suffixes,
        conditions=conditions,
        stim_idx=stim_idx,
    )
    summary_df = _summarize_rates(raw_df)

    figures = []
    for suffix in sorted(summary_df["suffix"].unique()):
        fig, _ = plot_ap_ca_spike_rates_2x2(summary_df, suffix=suffix, aff_order=aff_order)
        figures.append((fig, suffix))

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            # infer anal_loc/range from prefix if possible
            m = re.search(r"(basal|apical)_range(\d+)_clus_invivo", base_clus_invivo_prefix)
            tag = f"{m.group(1)}_range{m.group(2)}" if m else "ap_ca"
            out_name = f"{tag}_{suffix}_ap_ca_spike_rate.{fig_format}"
            fig.savefig(os.path.join(save_dir, out_name), dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        elif save_dir is not None:
            plt.close(fig)

    return raw_df, summary_df, figures


def main():
    parser = argparse.ArgumentParser(
        description="AP/Ca spike rate analysis from soma_v_array and apic_v_array."
    )
    parser.add_argument(
        "--root-dir",
        default="/G/results/simulation_singclus_supple_May26",
        help="Directory containing <sec_type>_range<N>_clus_invivo_* experiment trees.",
    )
    parser.add_argument("--range-idx", type=int, default=0, help="Range index N in path names.")
    parser.add_argument(
        "--sec-type",
        choices=["basal", "apical"],
        default="basal",
        help="Section type in experiment folder name.",
    )
    parser.add_argument(
        "--suffixes",
        nargs="+",
        default=["singclus_ap"],
        help="Channel suffix after invivo_, e.g. singclus_ap.",
    )
    parser.add_argument(
        "--aff-order",
        nargs="+",
        type=int,
        default=None,
        help="Optional x-axis order for aff labels, e.g. 4 24 48 72.",
    )
    parser.add_argument("--stim-idx", type=int, default=0, help="Stimulus index along trace array.")
    parser.add_argument(
        "--output-dir",
        default="./results/ap_ca_spike",
        help="Directory for saved figures and CSV summaries.",
    )
    parser.add_argument(
        "--format",
        choices=["pdf", "png"],
        default="pdf",
        dest="fig_format",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not call plt.show().")
    args = parser.parse_args()

    base_prefix = (
        f"{args.root_dir.rstrip('/')}/{args.sec_type}_range{args.range_idx}_clus_invivo_"
    )

    raw_df, summary_df, _ = visualize_ap_ca_from_base(
        base_clus_invivo_prefix=base_prefix,
        suffixes=list(args.suffixes),
        aff_order=args.aff_order,
        stim_idx=args.stim_idx,
        save_dir=args.output_dir,
        fig_format=args.fig_format,
        show=not args.no_show,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    raw_df.to_csv(
        os.path.join(args.output_dir, f"{args.sec_type}_range{args.range_idx}_spike_rates_raw.csv"),
        index=False,
    )
    summary_df.to_csv(
        os.path.join(
            args.output_dir,
            f"{args.sec_type}_range{args.range_idx}_spike_rates_summary.csv",
        ),
        index=False,
    )


if __name__ == "__main__":
    main()
