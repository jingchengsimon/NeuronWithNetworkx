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


def resolve_replay_section_synapse_csv(replay_arg: Optional[str]) -> Optional[str]:
    """Resolve CLI path to section_synapse_df.csv (may pass .json path or run directory)."""
    if replay_arg is None:
        return None
    s = str(replay_arg).strip()
    if not s or s.lower() in ("none", "false", "off"):
        return None
    path = os.path.expanduser(s)
    if os.path.isdir(path):
        return os.path.join(path, "section_synapse_df.csv")
    if path.lower().endswith(".json"):
        return os.path.join(os.path.dirname(path), "section_synapse_df.csv")
    if path.lower().endswith(".csv"):
        return path
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


def load_replay_spike_maps(csv_path: str) -> Tuple[Dict[SynKey, np.ndarray], Dict[SynKey, np.ndarray]]:
    """Load exc (A) and inh (B) background spike trains keyed by syn identity."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"replay_bg: section_synapse_df not found: {csv_path}")

    ref = pd.read_csv(csv_path)
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

    exc_map: Dict[SynKey, np.ndarray] = {}
    inh_map: Dict[SynKey, np.ndarray] = {}
    dup_exc: list[SynKey] = []
    dup_inh: list[SynKey] = []

    n_rows = len(ref)
    for _, row in tqdm(
        ref.iterrows(),
        total=n_rows,
        desc="replay_bg: load spike maps",
        unit="row",
    ):
        key = row_syn_key(row)
        spikes = _parse_spike_train_bg_cell(row["spike_train_bg"])
        typ = str(row["type"])
        if typ == "A":
            if key in exc_map:
                dup_exc.append(key)
            exc_map[key] = spikes
        elif typ == "B":
            if key in inh_map:
                dup_inh.append(key)
            inh_map[key] = spikes

    if dup_exc:
        print(f"replay_bg: warning: {len(dup_exc)} duplicate keys in ref exc rows (last wins)")
    if dup_inh:
        print(f"replay_bg: warning: {len(dup_inh)} duplicate keys in ref inh rows (last wins)")

    return exc_map, inh_map
