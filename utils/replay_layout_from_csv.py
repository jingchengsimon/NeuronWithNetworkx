"""
Rebuild section_synapse_df (synapse locations + cluster assignment) from a saved CSV.
No random placement: uses section_id_synapse + loc (+ metadata columns).

Removable module: delete this file and early-return branches in simpleModelVer2.py to restore RNG layout.
"""
from __future__ import annotations

import os
from typing import Any, List

import numpy as np
import pandas as pd
from neuron import h
from tqdm import tqdm

REQUIRED_CSV_COLUMNS = (
    "section_id_synapse",
    "loc",
    "type",
    "region",
    "distance_to_soma",
    "distance_to_tuft",
    "cluster_flag",
    "cluster_center_flag",
    "cluster_id",
    "pre_unit_id",
    "branch_idx",
    "spike_train_bg",
)  # spike_train_bg read by load_replay_spike_maps; layout starts with [] then replay fills


def _safe_int(x: Any, default: int = -1) -> int:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return default
    return int(x)


def get_section_by_section_id(cell, section_id_synapse: int):
    m = cell.section_df[cell.section_df["section_id"] == section_id_synapse]
    if m.empty:
        raise ValueError(f"replay_layout: unknown section_id_synapse={section_id_synapse}")
    target_name = m["section_name"].iat[0]
    for sec in h.allsec():
        if sec.psection()["name"] == target_name:
            return sec
    raise ValueError(f"replay_layout: NEURON section not found for name={target_name!r}")


def populate_section_synapse_df_from_csv(
    cell,
    csv_path: str,
    num_syn_basal_exc: int,
    num_syn_apic_exc: int,
    num_syn_basal_inh: int,
    num_syn_apic_inh: int,
    num_syn_soma_inh: int,
) -> None:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"replay_layout: {csv_path}")

    ref = pd.read_csv(csv_path)
    missing = set(REQUIRED_CSV_COLUMNS) - set(ref.columns)
    if missing:
        raise ValueError(f"replay_layout: CSV missing columns: {sorted(missing)}")

    n_rows = len(ref)
    rows: List[dict] = []
    for _, row in tqdm(
        ref.iterrows(),
        total=n_rows,
        desc="replay_layout: synapses from CSV",
        unit="row",
    ):
        sid = _safe_int(row["section_id_synapse"], 0)
        loc = float(row["loc"])
        section = get_section_by_section_id(cell, sid)
        segment_synapse = section(loc)

        sw = row.get("syn_w")
        if sw is not None and not (isinstance(sw, float) and np.isnan(sw)):
            syn_w = float(sw)
        else:
            syn_w = None

        rows.append(
            {
                "section_id_synapse": sid,
                "section_synapse": section,
                "segment_synapse": segment_synapse,
                "loc": loc,
                "type": str(row["type"]),
                "distance_to_soma": float(row["distance_to_soma"]),
                "distance_to_tuft": float(row["distance_to_tuft"]),
                "cluster_flag": _safe_int(row["cluster_flag"], -1),
                "cluster_center_flag": _safe_int(row["cluster_center_flag"], -1),
                "cluster_id": _safe_int(row["cluster_id"], -1),
                "pre_unit_id": _safe_int(row["pre_unit_id"], -1),
                "region": str(row["region"]),
                "branch_idx": _safe_int(row.get("branch_idx"), -1),
                "syn_w": syn_w,
                "synapse": None,
                "netstim": None,
                "netcon": None,
                "spike_train": [],
                "spike_train_bg": [],
            }
        )

    cell.section_synapse_df = pd.DataFrame(rows, dtype=object)

    n_basal_a = len(cell.section_synapse_df[(cell.section_synapse_df["region"] == "basal") & (cell.section_synapse_df["type"] == "A")])
    n_apic_a = len(cell.section_synapse_df[(cell.section_synapse_df["region"] == "apical") & (cell.section_synapse_df["type"] == "A")])
    n_basal_b = len(cell.section_synapse_df[(cell.section_synapse_df["region"] == "basal") & (cell.section_synapse_df["type"] == "B")])
    n_apic_b = len(cell.section_synapse_df[(cell.section_synapse_df["region"] == "apical") & (cell.section_synapse_df["type"] == "B")])
    n_soma_b = len(cell.section_synapse_df[(cell.section_synapse_df["region"] == "soma") & (cell.section_synapse_df["type"] == "B")])

    exp = (num_syn_basal_exc, num_syn_apic_exc, num_syn_basal_inh, num_syn_apic_inh, num_syn_soma_inh)
    got = (n_basal_a, n_apic_a, n_basal_b, n_apic_b, n_soma_b)
    if exp != got:
        print(
            f"replay_layout: warning: synapse counts CSV {got} != CLI {exp}; using CSV layout."
        )

    cell.num_syn_basal_exc = n_basal_a
    cell.num_syn_apic_exc = n_apic_a
    cell.num_syn_basal_inh = n_basal_b
    cell.num_syn_apic_inh = n_apic_b
    cell.num_syn_soma_inh = n_soma_b


def replay_assign_cluster_metadata(
    cell,
    folder_path: str,
    basal_channel_type: str,
    sec_type: str,
    dis_to_root: int,
    num_clusters: int,
    cluster_radius: float,
    num_stim: int,
    stim_time: int,
    spat_condition: str,
    num_conn_per_preunit: int,
    num_syn_per_clus: int,
) -> None:
    """Set indices / unit_ids / num_preunit from existing cluster columns in section_synapse_df (no RNG)."""
    df = cell.section_synapse_df
    clus_exc = df[(df["cluster_flag"] == 1) & (df["type"] == "A")]

    if len(clus_exc) == 0:
        num_c = 0
        indices: List[List[int]] = []
        num_preunit = 1
    else:
        max_cid = int(clus_exc["cluster_id"].max())
        num_c = max_cid + 1
        indices = []
        for cid in tqdm(
            range(num_c),
            desc="replay_layout: cluster indices",
            unit="cluster",
        ):
            sub = clus_exc[clus_exc["cluster_id"] == cid]
            ctr = sub[sub["cluster_center_flag"] == 1]
            sur = sub[sub["cluster_center_flag"] == 0]
            if len(ctr) >= 1:
                pids = [int(ctr.iloc[0]["pre_unit_id"])]
                pids.extend(int(x) for x in sur["pre_unit_id"].values)
            else:
                pids = [int(sub.iloc[0]["pre_unit_id"])]
            indices.append(pids)
        pu = clus_exc["pre_unit_id"].astype(int)
        pu = pu[pu >= 0]
        num_preunit = int(pu.max()) + 1 if len(pu) else 1

    cell.basal_channel_type = basal_channel_type
    cell.sec_type = sec_type
    cell.dis_to_root = dis_to_root
    cell.num_clusters = num_c
    cell.cluster_radius = cluster_radius
    cell.num_stim = num_stim
    cell.stim_time = stim_time
    cell.num_conn_per_preunit = num_conn_per_preunit
    cell.num_syn_per_clus = num_syn_per_clus
    cell.num_preunit = num_preunit
    cell.unit_ids = np.arange(num_preunit)
    cell.indices = indices
    cell.num_clusters_sampled = num_c

    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, "preunit assignment.txt")
    with open(file_path, "w") as f:
        for i, index_list in enumerate(indices):
            f.write(f"Cluster_id: {i}, Num_preunits: {len(index_list)}, Preunit_ids: {index_list}\n")

    if num_clusters != num_c and spat_condition == "clus":
        print(
            f"replay_layout: note: --num_clusters {num_clusters} != CSV cluster count {num_c}; using CSV."
        )
