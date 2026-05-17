"""
Replay background (exc + inh) spike trains from a reference section_synapse_df.csv.
Experimental / removable: delete this module and call sites to drop the feature.

Matching uses structural identity, not row order:
(section_id_synapse, loc, type, region, cluster_flag).
"""
from __future__ import annotations

import ast
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

SynKey = Tuple[int, float, str, str, int, int]


def _syn_key_from_arrays(
    i: int,
    section_id: np.ndarray,
    loc: np.ndarray,
    typ: np.ndarray,
    region: np.ndarray,
    cluster_flag: np.ndarray,
    branch_idx: np.ndarray,
) -> SynKey:
    bi = branch_idx[i]
    if pd.isna(bi):
        bid = -1
    else:
        bid = int(bi)
    return (
        int(section_id[i]),
        round(float(loc[i]), 12),
        str(typ[i]),
        str(region[i]),
        int(cluster_flag[i]),
        bid,
    )


def resolve_replay_section_synapse_csv(replay_arg: Optional[str]) -> Optional[str]:
    """Resolve CLI path to section_synapse_df.csv (explicit .csv file or run directory)."""
    if replay_arg is None:
        return None
    s = str(replay_arg).strip()
    if not s or s.lower() in ("none", "false", "off"):
        return None
    path = os.path.expanduser(s)
    if os.path.isdir(path):
        return os.path.join(path, "section_synapse_df.csv")
    if path.lower().endswith(".json"):
        raise ValueError(
            "replay_bg_csv must be section_synapse_df.csv (or a directory containing it), not .json"
        )
    return path


def row_syn_key(row: pd.Series) -> SynKey:
    loc = float(row["loc"])
    if "branch_idx" in row.index and pd.notna(row["branch_idx"]):
        bid = int(row["branch_idx"])
    else:
        bid = -1
    return (
        int(row["section_id_synapse"]),
        round(loc, 12),
        str(row["type"]),
        str(row["region"]),
        int(row["cluster_flag"]),
        bid,
    )


def _parse_spike_train_bg_cell(val: Any) -> np.ndarray:
    """Normalize CSV cell to 1D spike time array (ms indices)."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.array([], dtype=float)
    if isinstance(val, str):
        if not val.strip():
            return np.array([], dtype=float)
        x = ast.literal_eval(val)
    else:
        x = val
    if isinstance(x, list):
        if len(x) == 0:
            return np.array([], dtype=float)
        if isinstance(x[0], (list, tuple)):
            return np.asarray(x[0], dtype=float)
        return np.asarray(x, dtype=float)
    return np.asarray(x, dtype=float)


def build_replay_spike_maps_from_df(ref: pd.DataFrame) -> Tuple[Dict[SynKey, np.ndarray], Dict[SynKey, np.ndarray]]:
    """Build exc/inh spike maps from an already-loaded section_synapse_df DataFrame (column-wise loop)."""
    need = {
        "section_id_synapse",
        "loc",
        "type",
        "region",
        "cluster_flag",
        "branch_idx",
        "spike_train_bg",
    }
    missing = need - set(ref.columns)
    if missing:
        raise ValueError(f"replay_bg: reference CSV missing columns: {sorted(missing)}")

    section_id = ref["section_id_synapse"].to_numpy()
    loc = ref["loc"].to_numpy()
    typ = ref["type"].astype(str).to_numpy()
    region = ref["region"].astype(str).to_numpy()
    cluster_flag = ref["cluster_flag"].to_numpy()
    branch_idx = ref["branch_idx"].to_numpy()
    spike_col = ref["spike_train_bg"]

    exc_map: Dict[SynKey, np.ndarray] = {}
    inh_map: Dict[SynKey, np.ndarray] = {}
    dup_exc: list[SynKey] = []
    dup_inh: list[SynKey] = []

    n_rows = len(ref)
    for i in tqdm(range(n_rows), desc="replay_bg: load spike maps", unit="row"):
        key = _syn_key_from_arrays(i, section_id, loc, typ, region, cluster_flag, branch_idx)
        spikes = _parse_spike_train_bg_cell(spike_col.iat[i])
        t = typ[i]
        if t == "A":
            if key in exc_map:
                dup_exc.append(key)
            exc_map[key] = spikes
        elif t == "B":
            if key in inh_map:
                dup_inh.append(key)
            inh_map[key] = spikes

    if dup_exc:
        print(f"replay_bg: warning: {len(dup_exc)} duplicate keys in ref exc rows (last wins)")
    if dup_inh:
        print(f"replay_bg: warning: {len(dup_inh)} duplicate keys in ref inh rows (last wins)")

    return exc_map, inh_map


def load_replay_csv_and_maps(csv_path: str) -> Tuple[pd.DataFrame, Dict[SynKey, np.ndarray], Dict[SynKey, np.ndarray]]:
    """Read section_synapse_df.csv once; validate layout columns; return DataFrame + spike maps."""
    from utils.replay_layout_from_csv import REQUIRED_CSV_COLUMNS

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"replay_bg: section_synapse_df not found: {csv_path}")

    ref = pd.read_csv(csv_path)
    missing = set(REQUIRED_CSV_COLUMNS) - set(ref.columns)
    if missing:
        raise ValueError(f"replay_layout/replay_bg: CSV missing columns: {sorted(missing)}")
    exc_map, inh_map = build_replay_spike_maps_from_df(ref)
    return ref, exc_map, inh_map


def load_replay_spike_maps(csv_path: str) -> Tuple[Dict[SynKey, np.ndarray], Dict[SynKey, np.ndarray]]:
    """Load exc (A) and inh (B) background spike trains keyed by syn identity (reads CSV once)."""
    _, exc_map, inh_map = load_replay_csv_and_maps(csv_path)
    return exc_map, inh_map
