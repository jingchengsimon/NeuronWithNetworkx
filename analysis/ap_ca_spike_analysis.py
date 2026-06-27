#!/usr/bin/env python3
"""
AP and Ca spike rate analysis from soma_v_array / apic_v_array.

Ca spikes (apic_v): NMDA-style criterion (V > -40 mV for >= 26 ms).
AP spikes (soma_v): upward crossings of -10 mV with positive first derivative.

Default figure layout (fig1-style):
  - 2 rows x 2 columns: rows AP / Ca; cols clus / distr
  - Each panel: basal range0/1/2 bars + apical range0/1/2 bars
  - Colors by distance range (blue basal, red apical gradients)
  - Only full cluster activation (default aff_label=72)
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.nmda_spike_detection import (
    DEFAULT_MIN_DURATION_MS,
    DEFAULT_V_THRESH_MV,
    compute_nmda_spike_rate_hz,
    count_nmda_spikes,
)

DEFAULT_AP_THRESH_MV = -10.0
DEFAULT_CA_V_THRESH_MV = DEFAULT_V_THRESH_MV
DEFAULT_CA_MIN_DURATION_MS = DEFAULT_MIN_DURATION_MS
DEFAULT_AFF_LABEL_FULL = 72
DEFAULT_STIM_TIME_KEY = "time point of stimulation"
DEFAULT_WINDOW_MS = (-20.0, 100.0)


def _range_color_lists() -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    """Match figures/plot_nmda_rate_tertiles.py basal blue / apical red gradients."""
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
    return color_list_basal, color_list_apical


# ---------------------------
# IO helpers
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
        return [0, num_preunit]
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


def _resolve_aff_index_for_label(simu_info: dict, n_aff: int, aff_label: int) -> int:
    labels = _aff_activation_labels(simu_info, n_aff)
    if aff_label in labels:
        return labels.index(aff_label)
    if aff_label == labels[-1]:
        return len(labels) - 1
    raise ValueError(
        f"aff_label={aff_label} not found in activation list {labels}. "
        "Use --aff_label matching simulation_params.json aff_list."
    )


# ---------------------------
# Spike detection
# ---------------------------
def count_ap_spikes(v_mV: np.ndarray, ap_thresh_mV: float = DEFAULT_AP_THRESH_MV) -> int:
    v = np.asarray(v_mV, dtype=np.float64).reshape(-1)
    if v.size < 2:
        return 0
    count = 0
    for i in range(1, v.size):
        dv = v[i] - v[i - 1]
        if v[i - 1] <= ap_thresh_mV and v[i] > ap_thresh_mV and dv > 0:
            count += 1
    return count


def ap_spike_onset_indices(v_mV: np.ndarray, ap_thresh_mV: float = DEFAULT_AP_THRESH_MV) -> np.ndarray:
    """Return sample indices where AP upward threshold crossings start."""
    v = np.asarray(v_mV, dtype=np.float64).reshape(-1)
    if v.size < 2:
        return np.array([], dtype=int)
    prev = v[:-1]
    curr = v[1:]
    crossings = (prev <= ap_thresh_mV) & (curr > ap_thresh_mV) & ((curr - prev) > 0)
    return np.flatnonzero(crossings).astype(int) + 1


def compute_ap_spike_rate_hz(
    v_mV: np.ndarray,
    sim_duration_ms: float,
    ap_thresh_mV: float = DEFAULT_AP_THRESH_MV,
) -> float:
    dur_s = sim_duration_ms / 1000.0
    if dur_s <= 0:
        return 0.0
    return float(count_ap_spikes(v_mV, ap_thresh_mV=ap_thresh_mV)) / dur_s


def ca_spike_onset_indices(
    v_mV: np.ndarray,
    dt_s: float,
    v_thresh_mV: float = DEFAULT_CA_V_THRESH_MV,
    min_duration_ms: float = DEFAULT_CA_MIN_DURATION_MS,
) -> np.ndarray:
    """
    Return onset sample indices for Ca/NMDA-style events.

    The event criterion matches count_nmda_spikes: each connected run with
    V > threshold and duration >= min_duration_ms counts once.  The onset is
    the first sample in that qualifying run.
    """
    v = np.asarray(v_mV, dtype=np.float64).reshape(-1)
    if v.size == 0:
        return np.array([], dtype=int)
    min_samples = max(1, int(np.ceil((min_duration_ms / 1000.0) / dt_s)))
    onsets: list[int] = []
    i = 0
    while i < v.size:
        if v[i] <= v_thresh_mV:
            i += 1
            continue
        j = i + 1
        while j < v.size and v[j] > v_thresh_mV:
            j += 1
        if (j - i) >= min_samples:
            onsets.append(i)
        i = j
    return np.asarray(onsets, dtype=int)


def compute_ca_spike_rate_hz(
    v_mV: np.ndarray,
    dt_s: float,
    sim_duration_ms: float,
    v_thresh_mV: float = DEFAULT_CA_V_THRESH_MV,
    min_duration_ms: float = DEFAULT_CA_MIN_DURATION_MS,
) -> float:
    return compute_nmda_spike_rate_hz(
        v_mV,
        dt_s,
        sim_duration_ms,
        v_thresh_mV=v_thresh_mV,
        min_duration_ms=min_duration_ms,
    )


def _onset_times_ms(onset_indices: np.ndarray, dt_s: float) -> np.ndarray:
    return np.asarray(onset_indices, dtype=np.float64) * dt_s * 1000.0


def _stim_time_ms(simu_info: dict, stim_time_key: str = DEFAULT_STIM_TIME_KEY) -> float:
    if stim_time_key not in simu_info:
        raise KeyError(f"simulation_params.json missing key: {stim_time_key!r}")
    return float(simu_info[stim_time_key])


def _window_bounds_ms(
    simu_info: dict,
    window_ms: tuple[float, float],
    stim_time_key: str = DEFAULT_STIM_TIME_KEY,
) -> tuple[float, float, float]:
    stim_ms = _stim_time_ms(simu_info, stim_time_key=stim_time_key)
    start_ms = stim_ms + float(window_ms[0])
    end_ms = stim_ms + float(window_ms[1])
    if end_ms < start_ms:
        raise ValueError(f"window_ms end must be >= start, got {window_ms}")
    return stim_ms, start_ms, end_ms


def _format_onset_list_ms(onsets_ms: np.ndarray) -> str:
    return ";".join(f"{x:.6g}" for x in np.asarray(onsets_ms, dtype=float))


def extract_spike_rates_at_aff(
    folder: str,
    aff_idx: int,
    stim_idx: int = 0,
    ap_thresh_mV: float = DEFAULT_AP_THRESH_MV,
    ca_v_thresh_mV: float = DEFAULT_CA_V_THRESH_MV,
    ca_min_duration_ms: float = DEFAULT_CA_MIN_DURATION_MS,
) -> list[dict]:
    soma_raw, apic_raw, simu_info = _load_run_folder(folder)
    if simu_info is None:
        return []

    rows: list[dict] = []
    dur_ms = _sim_duration_ms(simu_info)

    if soma_raw is not None:
        soma = _normalize_trace_shape(soma_raw)
        dt_s = _dt_seconds_from_trace(soma, simu_info)
        n_stim, n_aff, n_trials = soma.shape[1], soma.shape[2], soma.shape[3]
        if stim_idx >= n_stim or aff_idx >= n_aff:
            return rows
        aff_labels = _aff_activation_labels(simu_info, n_aff)
        for trial in range(n_trials):
            trace = soma[:, stim_idx, aff_idx, trial]
            rows.append(
                dict(
                    aff_idx=aff_idx,
                    aff_label=int(aff_labels[aff_idx]),
                    trial=trial,
                    spike_type="ap",
                    rate_hz=float(compute_ap_spike_rate_hz(trace, dur_ms, ap_thresh_mV=ap_thresh_mV)),
                )
            )

    if apic_raw is not None:
        apic = _normalize_trace_shape(apic_raw)
        dt_s = _dt_seconds_from_trace(apic, simu_info)
        n_stim, n_aff, n_trials = apic.shape[1], apic.shape[2], apic.shape[3]
        if stim_idx >= n_stim or aff_idx >= n_aff:
            return rows
        aff_labels = _aff_activation_labels(simu_info, n_aff)
        for trial in range(n_trials):
            trace = apic[:, stim_idx, aff_idx, trial]
            rows.append(
                dict(
                    aff_idx=aff_idx,
                    aff_label=int(aff_labels[aff_idx]),
                    trial=trial,
                    spike_type="ca",
                    rate_hz=float(
                        compute_ca_spike_rate_hz(
                            trace,
                            dt_s,
                            dur_ms,
                            v_thresh_mV=ca_v_thresh_mV,
                            min_duration_ms=ca_min_duration_ms,
                        )
                    ),
                )
            )

    return rows


def extract_spike_occurrences_at_aff(
    folder: str,
    aff_idx: int,
    stim_idx: int = 0,
    window_ms: tuple[float, float] = DEFAULT_WINDOW_MS,
    stim_time_key: str = DEFAULT_STIM_TIME_KEY,
    ap_thresh_mV: float = DEFAULT_AP_THRESH_MV,
    ca_v_thresh_mV: float = DEFAULT_CA_V_THRESH_MV,
    ca_min_duration_ms: float = DEFAULT_CA_MIN_DURATION_MS,
) -> list[dict]:
    """
    Trial-level AP/Ca occurrence in a stimulation-aligned time window.

    occurrence is binary per trial: any number of event onsets inside the
    absolute window [stim_time + window_ms[0], stim_time + window_ms[1]]
    counts as 1.  Ca event onsets use the first sample of each qualifying
    suprathreshold run.
    """
    soma_raw, apic_raw, simu_info = _load_run_folder(folder)
    if simu_info is None:
        return []

    rows: list[dict] = []
    stim_ms, win_start_ms, win_end_ms = _window_bounds_ms(
        simu_info,
        window_ms=window_ms,
        stim_time_key=stim_time_key,
    )

    if soma_raw is not None:
        soma = _normalize_trace_shape(soma_raw)
        dt_s = _dt_seconds_from_trace(soma, simu_info)
        n_stim, n_aff, n_trials = soma.shape[1], soma.shape[2], soma.shape[3]
        if stim_idx >= n_stim or aff_idx >= n_aff or aff_idx < -n_aff:
            return rows
        aff_labels = _aff_activation_labels(simu_info, n_aff)
        for trial in range(n_trials):
            trace = soma[:, stim_idx, aff_idx, trial]
            all_onsets_ms = _onset_times_ms(
                ap_spike_onset_indices(trace, ap_thresh_mV=ap_thresh_mV),
                dt_s,
            )
            in_window = all_onsets_ms[(all_onsets_ms >= win_start_ms) & (all_onsets_ms <= win_end_ms)]
            rows.append(
                dict(
                    aff_idx=aff_idx,
                    aff_label=int(aff_labels[aff_idx]),
                    trial=trial,
                    spike_type="ap",
                    occurred=int(in_window.size > 0),
                    n_events_total=int(all_onsets_ms.size),
                    n_events_window=int(in_window.size),
                    event_onsets_ms=_format_onset_list_ms(all_onsets_ms),
                    window_event_onsets_ms=_format_onset_list_ms(in_window),
                    stim_time_ms=stim_ms,
                    window_start_ms=win_start_ms,
                    window_end_ms=win_end_ms,
                )
            )

    if apic_raw is not None:
        apic = _normalize_trace_shape(apic_raw)
        dt_s = _dt_seconds_from_trace(apic, simu_info)
        n_stim, n_aff, n_trials = apic.shape[1], apic.shape[2], apic.shape[3]
        if stim_idx >= n_stim or aff_idx >= n_aff or aff_idx < -n_aff:
            return rows
        aff_labels = _aff_activation_labels(simu_info, n_aff)
        for trial in range(n_trials):
            trace = apic[:, stim_idx, aff_idx, trial]
            all_onsets_ms = _onset_times_ms(
                ca_spike_onset_indices(
                    trace,
                    dt_s,
                    v_thresh_mV=ca_v_thresh_mV,
                    min_duration_ms=ca_min_duration_ms,
                ),
                dt_s,
            )
            in_window = all_onsets_ms[(all_onsets_ms >= win_start_ms) & (all_onsets_ms <= win_end_ms)]
            rows.append(
                dict(
                    aff_idx=aff_idx,
                    aff_label=int(aff_labels[aff_idx]),
                    trial=trial,
                    spike_type="ca",
                    occurred=int(in_window.size > 0),
                    n_events_total=int(all_onsets_ms.size),
                    n_events_window=int(in_window.size),
                    event_onsets_ms=_format_onset_list_ms(all_onsets_ms),
                    window_event_onsets_ms=_format_onset_list_ms(in_window),
                    stim_time_ms=stim_ms,
                    window_start_ms=win_start_ms,
                    window_end_ms=win_end_ms,
                )
            )

    return rows


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
    return f"{base_root}_{condition.strip()}_invivo_{suffix.strip().lstrip('_')}"


def _find_epoch_folders_two_level(
    exp_dir: str,
    folder_tag: str | int | None = None,
) -> list[tuple[int, str]]:
    out = []
    if folder_tag is not None:
        tag_name = str(int(folder_tag) % 100) if int(folder_tag) % 100 != 0 else "100"
        tag_dir = os.path.join(exp_dir, tag_name)
        if not os.path.isdir(tag_dir):
            return out
        search_glob = os.path.join(tag_dir, "*")
    else:
        search_glob = os.path.join(exp_dir, "*", "*")

    for epoch_dir in glob.glob(search_glob):
        if not os.path.isdir(epoch_dir):
            continue
        ep_name = os.path.basename(epoch_dir)
        if re.fullmatch(r"\d+", ep_name):
            out.append((int(ep_name), epoch_dir))
    out.sort(key=lambda x: x[0])
    return out


def build_spike_rate_dataframe_across_ranges(
    root_dir: str,
    range_idxs: Iterable[int],
    sec_types: Iterable[str] = ("basal", "apical"),
    conditions: Iterable[str] = ("clus", "distr"),
    suffix: str = "singclus_ap",
    aff_label: int = DEFAULT_AFF_LABEL_FULL,
    stim_idx: int = 0,
    folder_tag: str | int | None = None,
) -> pd.DataFrame:
    """
    Scan {sec_type}_range{N}_clus_invivo_{suffix} for each range_idx and spat condition.
    Only extract rates at aff_label (default 72 activated synapses).
    """
    rows = []
    root_dir = root_dir.rstrip("/")

    for condition in conditions:
        for sec_type in sec_types:
            for range_idx in range_idxs:
                base_prefix = f"{root_dir}/{sec_type}_range{range_idx}_clus_invivo_"
                exp_dir = _exp_dir_from_base(base_prefix, condition, suffix)
                epoch_folders = _find_epoch_folders_two_level(exp_dir, folder_tag=folder_tag)
                if not epoch_folders:
                    continue

                for epoch_idx, folder in epoch_folders:
                    soma_raw, apic_raw, simu_info = _load_run_folder(folder)
                    if simu_info is None:
                        continue

                    n_aff = None
                    if soma_raw is not None:
                        n_aff = _normalize_trace_shape(soma_raw).shape[2]
                    elif apic_raw is not None:
                        n_aff = _normalize_trace_shape(apic_raw).shape[2]
                    if n_aff is None:
                        continue

                    try:
                        aff_idx = _resolve_aff_index_for_label(simu_info, n_aff, aff_label)
                    except ValueError:
                        continue

                    rate_rows = extract_spike_rates_at_aff(
                        folder,
                        aff_idx=aff_idx,
                        stim_idx=stim_idx,
                    )
                    for r in rate_rows:
                        rows.append(
                            dict(
                                epoch=epoch_idx,
                                suffix=suffix,
                                condition=condition,
                                sec_type=sec_type,
                                range_idx=int(range_idx),
                                aff_idx=int(r["aff_idx"]),
                                aff_label=int(r["aff_label"]),
                                trial=int(r["trial"]),
                                spike_type=str(r["spike_type"]),
                                rate_hz=float(r["rate_hz"]),
                                folder=folder,
                            )
                        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            "No AP/Ca spike rates extracted. Check root_dir, range_idxs, "
            f"conditions={list(conditions)}, suffix={suffix}, aff_label={aff_label}."
        )
    return df


def build_spike_occurrence_dataframe_across_ranges(
    root_dir: str,
    range_idxs: Iterable[int],
    sec_types: Iterable[str] = ("basal", "apical"),
    conditions: Iterable[str] = ("clus", "distr"),
    suffix: str = "singclus_ap",
    aff_label: int = DEFAULT_AFF_LABEL_FULL,
    stim_idx: int = 0,
    window_ms: tuple[float, float] = DEFAULT_WINDOW_MS,
    stim_time_key: str = DEFAULT_STIM_TIME_KEY,
    folder_tag: str | int | None = None,
) -> pd.DataFrame:
    """
    Scan experiments and extract binary AP/Ca occurrence per trial in a window.
    """
    rows = []
    root_dir = root_dir.rstrip("/")

    for condition in conditions:
        for sec_type in sec_types:
            for range_idx in range_idxs:
                base_prefix = f"{root_dir}/{sec_type}_range{range_idx}_clus_invivo_"
                exp_dir = _exp_dir_from_base(base_prefix, condition, suffix)
                epoch_folders = _find_epoch_folders_two_level(exp_dir, folder_tag=folder_tag)
                if not epoch_folders:
                    continue

                for epoch_idx, folder in epoch_folders:
                    soma_raw, apic_raw, simu_info = _load_run_folder(folder)
                    if simu_info is None:
                        continue

                    n_aff = None
                    if soma_raw is not None:
                        n_aff = _normalize_trace_shape(soma_raw).shape[2]
                    elif apic_raw is not None:
                        n_aff = _normalize_trace_shape(apic_raw).shape[2]
                    if n_aff is None:
                        continue

                    try:
                        aff_idx = _resolve_aff_index_for_label(simu_info, n_aff, aff_label)
                    except ValueError:
                        continue

                    occurrence_rows = extract_spike_occurrences_at_aff(
                        folder,
                        aff_idx=aff_idx,
                        stim_idx=stim_idx,
                        window_ms=window_ms,
                        stim_time_key=stim_time_key,
                    )
                    for r in occurrence_rows:
                        rows.append(
                            dict(
                                epoch=epoch_idx,
                                suffix=suffix,
                                condition=condition,
                                sec_type=sec_type,
                                range_idx=int(range_idx),
                                aff_idx=int(r["aff_idx"]),
                                aff_label=int(r["aff_label"]),
                                trial=int(r["trial"]),
                                spike_type=str(r["spike_type"]),
                                occurred=int(r["occurred"]),
                                n_events_total=int(r["n_events_total"]),
                                n_events_window=int(r["n_events_window"]),
                                event_onsets_ms=str(r["event_onsets_ms"]),
                                window_event_onsets_ms=str(r["window_event_onsets_ms"]),
                                stim_time_ms=float(r["stim_time_ms"]),
                                window_start_ms=float(r["window_start_ms"]),
                                window_end_ms=float(r["window_end_ms"]),
                                folder=folder,
                            )
                        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            "No AP/Ca spike occurrences extracted. Check root_dir, range_idxs, "
            f"conditions={list(conditions)}, suffix={suffix}, aff_label={aff_label}, "
            f"window_ms={window_ms}."
        )
    return df


def summarize_rates_by_range(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and SEM across epochs and trials."""
    return (
        df.groupby(
            ["suffix", "condition", "spike_type", "sec_type", "range_idx", "aff_label"],
            as_index=False,
        )
        .agg(
            rate_mean=("rate_hz", "mean"),
            rate_sem=(
                "rate_hz",
                lambda s: float(np.std(s, ddof=1) / np.sqrt(len(s))) if len(s) > 1 else 0.0,
            ),
            n_samples=("rate_hz", "count"),
        )
    )


def summarize_occurrences_by_range(df: pd.DataFrame) -> pd.DataFrame:
    """Mean binary occurrence probability and SEM across epochs/trials."""
    return (
        df.groupby(
            ["suffix", "condition", "spike_type", "sec_type", "range_idx", "aff_label"],
            as_index=False,
        )
        .agg(
            event_prob=("occurred", "mean"),
            event_sem=(
                "occurred",
                lambda s: float(np.std(s, ddof=1) / np.sqrt(len(s))) if len(s) > 1 else 0.0,
            ),
            n_samples=("occurred", "count"),
            n_events_window=("n_events_window", "sum"),
            n_events_total=("n_events_total", "sum"),
            window_start_ms=("window_start_ms", "first"),
            window_end_ms=("window_end_ms", "first"),
        )
    )


def _lookup_value(
    summary_df: pd.DataFrame,
    spike_type: str,
    sec_type: str,
    range_idx: int,
    condition: str,
    suffix: str,
    value_col: str,
    err_col: str,
) -> tuple[float, float]:
    sub = summary_df[
        (summary_df["spike_type"] == spike_type)
        & (summary_df["sec_type"] == sec_type)
        & (summary_df["range_idx"] == range_idx)
        & (summary_df["condition"] == condition)
        & (summary_df["suffix"] == suffix)
    ]
    if sub.empty:
        return np.nan, np.nan
    return float(sub[value_col].iloc[0]), float(sub[err_col].iloc[0])


def _draw_basal_apical_range_panel(
    ax: plt.Axes,
    summary_df: pd.DataFrame,
    spike_type: str,
    condition: str,
    suffix: str,
    range_idxs: list[int],
    color_list_basal: list[tuple[float, float, float]],
    color_list_apical: list[tuple[float, float, float]],
    value_col: str = "rate_mean",
    err_col: str = "rate_sem",
) -> None:
    """One subplot: 3 basal bars (left) + 3 apical bars (right), colored by range."""
    width, col_gap = 1.0, 1.0
    x_basal = np.arange(3) * width
    x_apical = np.arange(3) * width + 3 * width + col_gap

    mean_b, sem_b, mean_a, sem_a = [], [], [], []
    for ri in range_idxs:
        mb, sb = _lookup_value(summary_df, spike_type, "basal", ri, condition, suffix, value_col, err_col)
        ma, sa = _lookup_value(summary_df, spike_type, "apical", ri, condition, suffix, value_col, err_col)
        mean_b.append(mb)
        sem_b.append(sb)
        mean_a.append(ma)
        sem_a.append(sa)

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
        mean_a,
        yerr=sem_a,
        width=width,
        color=color_list_apical,
        capsize=3,
        edgecolor="none",
    )

    centers = [float(x_basal.mean()), float(x_apical.mean())]
    ax.set_xticks(centers)
    ax.set_xticklabels(["Basal", "Apical"])
    ax.set_xlim(x_basal.min() - 0.5 * width, x_apical.max() + 0.5 * width)
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_ap_ca_by_range_fig1_style(
    summary_df: pd.DataFrame,
    suffix: str,
    conditions: tuple[str, ...] = ("clus", "distr"),
    range_idxs: list[int] | None = None,
    range_legend_labels: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    value_col: str = "rate_mean",
    err_col: str = "rate_sem",
    ylabel_map: dict[str, str] | None = None,
    title_prefix: str = "Spike rate across distance ranges",
) -> plt.Figure:
    """
    2xN panels: rows = AP / Ca; cols = spat conditions (clus and/or distr).
    Each panel: 3 basal bars + gap + 3 apical bars; bar color = distance range.
    """
    if range_idxs is None:
        range_idxs = sorted(summary_df["range_idx"].unique().tolist())
    if len(range_idxs) != 3:
        raise ValueError(f"Expected 3 range indices, got {range_idxs}")

    if range_legend_labels is None:
        range_legend_labels = [f"range {r}" for r in range_idxs]

    color_list_basal, color_list_apical = _range_color_lists()
    cond_col = {"clus": 0, "distr": 1}
    spike_row = {"ap": 0, "ca": 1}
    cond_titles = {"clus": "clustered", "distr": "distributed"}

    n_cols = len(conditions)
    if figsize is None:
        figsize = (4.5 * n_cols, 6.0)

    fig, axes = plt.subplots(2, n_cols, figsize=figsize, sharey="row")
    if n_cols == 1:
        axes = np.array(axes).reshape(2, 1)

    for spike_type, row in spike_row.items():
        for col_idx, condition in enumerate(conditions):
            if condition not in cond_col:
                raise ValueError(f"Unknown spat condition {condition!r}; expected clus or distr.")
            ax = axes[row, col_idx]
            _draw_basal_apical_range_panel(
                ax,
                summary_df,
                spike_type,
                condition,
                suffix,
                range_idxs,
                color_list_basal,
                color_list_apical,
                value_col=value_col,
                err_col=err_col,
            )
            ax.set_title(f"{spike_type.upper()} — {cond_titles.get(condition, condition)}")

    if ylabel_map is None:
        ylabel_map = {"ap": "AP spike rate (Hz)", "ca": "Ca spike rate (Hz)"}
    axes[0, 0].set_ylabel(ylabel_map.get("ap", "AP"))
    axes[1, 0].set_ylabel(ylabel_map.get("ca", "Ca"))

    handles = [
        (
            mpatches.Patch(facecolor=color_list_basal[i], edgecolor="none"),
            mpatches.Patch(facecolor=color_list_apical[i], edgecolor="none"),
        )
        for i in range(3)
    ]
    axes[0, 0].legend(
        handles,
        range_legend_labels,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        handlelength=4,
        handletextpad=0.2,
        frameon=False,
        loc="upper left",
    )
    aff_val = int(summary_df["aff_label"].iloc[0])
    fig.suptitle(
        f"{title_prefix} (aff={aff_val} synapses) — {suffix}",
        y=1.02,
    )
    fig.tight_layout()
    return fig


def visualize_ap_ca_across_ranges(
    root_dir: str,
    range_idxs: list[int],
    conditions: tuple[str, str] = ("clus", "distr"),
    suffix: str = "singclus_ap",
    aff_label: int = DEFAULT_AFF_LABEL_FULL,
    stim_idx: int = 0,
    folder_tag: str | int | None = None,
    save_path: str | None = None,
    fig_format: str = "pdf",
    show: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, plt.Figure]:
    raw_df = build_spike_rate_dataframe_across_ranges(
        root_dir=root_dir,
        range_idxs=range_idxs,
        conditions=conditions,
        suffix=suffix,
        aff_label=aff_label,
        stim_idx=stim_idx,
        folder_tag=folder_tag,
    )
    summary_df = summarize_rates_by_range(raw_df)

    fig = plot_ap_ca_by_range_fig1_style(
        summary_df,
        suffix=suffix,
        conditions=conditions,
        range_idxs=range_idxs,
    )

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return raw_df, summary_df, fig


def visualize_ap_ca_occurrence_across_ranges(
    root_dir: str,
    range_idxs: list[int],
    conditions: tuple[str, str] = ("clus", "distr"),
    suffix: str = "singclus_ap",
    aff_label: int = DEFAULT_AFF_LABEL_FULL,
    stim_idx: int = 0,
    window_ms: tuple[float, float] = DEFAULT_WINDOW_MS,
    stim_time_key: str = DEFAULT_STIM_TIME_KEY,
    folder_tag: str | int | None = None,
    save_path: str | None = None,
    fig_format: str = "pdf",
    show: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, plt.Figure]:
    raw_df = build_spike_occurrence_dataframe_across_ranges(
        root_dir=root_dir,
        range_idxs=range_idxs,
        conditions=conditions,
        suffix=suffix,
        aff_label=aff_label,
        stim_idx=stim_idx,
        window_ms=window_ms,
        stim_time_key=stim_time_key,
        folder_tag=folder_tag,
    )
    summary_df = summarize_occurrences_by_range(raw_df)

    fig = plot_ap_ca_by_range_fig1_style(
        summary_df,
        suffix=suffix,
        conditions=conditions,
        range_idxs=range_idxs,
        value_col="event_prob",
        err_col="event_sem",
        ylabel_map={"ap": "AP occurrence probability", "ca": "Ca occurrence probability"},
        title_prefix=f"Spike occurrence probability in window [{window_ms[0]:g}, {window_ms[1]:g}] ms",
    )
    for ax in fig.axes:
        ax.set_ylim(0.0, 1.0)

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return raw_df, summary_df, fig


def main():
    parser = argparse.ArgumentParser(
        description="AP/Ca spike rates at full activation, across distance ranges (fig1-style bars)."
    )
    parser.add_argument(
        "--root_dir",
        default="/G/results/simulation_singclus_supple_May26",
        help="Parent directory with <sec_type>_range<N>_clus_invivo_<suffix> trees.",
    )
    parser.add_argument(
        "--range_idxs",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Distance-to-root indices (default: 0 1 2).",
    )
    parser.add_argument(
        "--spat_conditions",
        nargs="+",
        choices=["clus", "distr"],
        default=["clus", "distr"],
        help="Spatial patterns to include (default: clus distr → 2x2 figure).",
    )
    parser.add_argument(
        "--suffix",
        default="singclus_ap",
        help="Channel suffix after invivo_, e.g. singclus_ap.",
    )
    parser.add_argument(
        "--folder_tag",
        default=None,
        help="If set, only scan runs under this folder tag (e.g. 2 for .../2/{epoch}/).",
    )
    parser.add_argument(
        "--aff_label",
        type=int,
        default=DEFAULT_AFF_LABEL_FULL,
        help="Activated synapse count to analyze (default: 72).",
    )
    parser.add_argument("--stim_idx", type=int, default=0)
    parser.add_argument(
        "--analysis_mode",
        choices=["rate", "probability", "both"],
        default="rate",
        help="rate: whole-simulation spike rate; probability: binary occurrence in window; both: save both.",
    )
    parser.add_argument(
        "--window_ms",
        nargs=2,
        type=float,
        default=list(DEFAULT_WINDOW_MS),
        metavar=("START", "END"),
        help=(
            "Window relative to stimulation time for probability mode, in ms "
            f"(default: {DEFAULT_WINDOW_MS[0]} {DEFAULT_WINDOW_MS[1]})."
        ),
    )
    parser.add_argument(
        "--stim_time_key",
        default=DEFAULT_STIM_TIME_KEY,
        help="simulation_params.json key used as stimulation time for probability mode.",
    )
    parser.add_argument(
        "--output_dir",
        default="./results/ap_ca_spike",
        help="Directory for figure and CSV output.",
    )
    parser.add_argument(
        "--format",
        choices=["pdf", "png"],
        default="pdf",
        dest="fig_format",
    )
    parser.add_argument("--show", action="store_true", help="Call plt.show().")
    args = parser.parse_args()

    spat_conds = tuple(dict.fromkeys(args.spat_conditions))
    if not spat_conds or len(spat_conds) > 2:
        raise SystemExit(
            "Provide 1 or 2 spat conditions (clus and/or distr). "
            f"Got: {spat_conds}"
        )
    spat_tag = "_".join(spat_conds)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.analysis_mode in ("rate", "both"):
        out_name = f"ap_ca_{spat_tag}_aff{args.aff_label}_by_range.{args.fig_format}"
        save_path = os.path.join(args.output_dir, out_name)

        raw_df, summary_df, _ = visualize_ap_ca_across_ranges(
            root_dir=args.root_dir,
            range_idxs=list(args.range_idxs),
            conditions=spat_conds,  # type: ignore[arg-type]
            suffix=args.suffix,
            aff_label=args.aff_label,
            stim_idx=args.stim_idx,
            folder_tag=args.folder_tag,
            save_path=save_path,
            fig_format=args.fig_format,
            show=args.show,
        )

        raw_df.to_csv(
            os.path.join(args.output_dir, f"spike_rates_raw_aff{args.aff_label}.csv"),
            index=False,
        )
        summary_df.to_csv(
            os.path.join(args.output_dir, f"spike_rates_summary_aff{args.aff_label}.csv"),
            index=False,
        )

    if args.analysis_mode in ("probability", "both"):
        window_ms = (float(args.window_ms[0]), float(args.window_ms[1]))
        win_tag = f"win{window_ms[0]:g}_{window_ms[1]:g}ms".replace("-", "m").replace(".", "p")
        out_name = f"ap_ca_{spat_tag}_aff{args.aff_label}_prob_{win_tag}_by_range.{args.fig_format}"
        save_path = os.path.join(args.output_dir, out_name)

        raw_df, summary_df, _ = visualize_ap_ca_occurrence_across_ranges(
            root_dir=args.root_dir,
            range_idxs=list(args.range_idxs),
            conditions=spat_conds,  # type: ignore[arg-type]
            suffix=args.suffix,
            aff_label=args.aff_label,
            stim_idx=args.stim_idx,
            window_ms=window_ms,
            stim_time_key=str(args.stim_time_key),
            folder_tag=args.folder_tag,
            save_path=save_path,
            fig_format=args.fig_format,
            show=args.show,
        )

        raw_df.to_csv(
            os.path.join(args.output_dir, f"spike_occurrence_raw_aff{args.aff_label}_{win_tag}.csv"),
            index=False,
        )
        summary_df.to_csv(
            os.path.join(args.output_dir, f"spike_occurrence_summary_aff{args.aff_label}_{win_tag}.csv"),
            index=False,
        )


if __name__ == "__main__":
    main()
