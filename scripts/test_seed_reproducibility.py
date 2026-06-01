#!/usr/bin/env python3
"""
Lightweight reproducibility checks before batch var experiments.

Verifies that (when all four seeds are explicit):
  - max_workers_epoch / max_workers_synapse do not change seed-controlled inputs
  - parallel epoch scheduling does not add extra randomness to inputs
  - aff_mode=custom is deterministic across worker settings (inputs)
  - each seed controls only its intended aspect of the simulation

Note on soma_v / voltage traces:
  add_background_exc_inputs uses ThreadPoolExecutor to attach VecStim/NetCon.
  Worker count can change NetCon registration order in NEURON, so coincident
  events may be processed in a different order. Input fingerprints (layout,
  spike trains) must still match; soma_v bit-exact match across worker counts
  is not required.

Note on layout / add_synapses:
  Synapse placement must use per-index seeds and preserve row order after
  parallel map (see cell_with_networkx.add_single_synapse). Otherwise
  max_workers_synapse changes both morphology and which spike seed binds to
  which synapse.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "tmp_seed_repro_test"

FIXED_SEEDS = {
    "bg_syn_pos_seed": 42,
    "clus_syn_pos_seed": 43,
    "bg_spike_gen_seed": 6,
    "clus_spike_gen_seed": 60,
}

# Seed-controlled quantities written to section_synapse_df (stable across worker counts).
INPUT_FINGERPRINT_KEYS = ["layout", "bg_spikes_exc", "cluster_spikes", "cluster_layout"]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_dir(results_root: Path, channel_suffix: str, epoch: int, folder_tag: str = "1") -> Path:
    folder = (
        f"basal_range1_clus_invivo_{channel_suffix}"
    )
    return results_root / folder / folder_tag / str(epoch)


def build_cmd(
    results_root: Path,
    channel_suffix: str,
    *,
    epoch: int = 1,
    num_epochs: int = 1,
    start_epoch: Optional[int] = None,
    max_workers_epoch: int = 1,
    max_workers_synapse: int = 1,
    aff_mode: str = "custom",
    aff_list: Optional[List[int]] = None,
    seeds: Optional[Dict[str, int]] = None,
    simu_duration: int = 200,
    python_bin: str = "python",
) -> List[str]:
    if aff_list is None:
        aff_list = [0, 12, 24]
    if start_epoch is None:
        start_epoch = epoch
    if seeds is None:
        seeds = FIXED_SEEDS

    cmd = [
        python_bin,
        str(REPO_ROOT / "L5b_simulation.py"),
        "--simu_cond",
        "invivo",
        "--sec_type",
        "basal",
        "--spat_cond",
        "clus",
        "--dis_to_root",
        "1",
        "--channel_suffix",
        channel_suffix,
        "--aff_mode",
        aff_mode,
        "--aff_list",
        *[str(x) for x in aff_list],
        "--num_syn_per_clus",
        "72",
        "--simu_duration",
        str(simu_duration),
        "--num_epochs",
        str(num_epochs),
        "--start_epoch",
        str(start_epoch),
        "--max_workers_epoch",
        str(max_workers_epoch),
        "--max_workers_synapse",
        str(max_workers_synapse),
        "--results_root",
        str(results_root),
        "--bg_syn_pos_seed",
        str(seeds["bg_syn_pos_seed"]),
        "--clus_syn_pos_seed",
        str(seeds["clus_syn_pos_seed"]),
        "--bg_spike_gen_seed",
        str(seeds["bg_spike_gen_seed"]),
        "--clus_spike_gen_seed",
        str(seeds["clus_spike_gen_seed"]),
    ]
    return cmd


def run_simulation(cmd: Iterable[str], label: str) -> None:
    print(f"\n>>> {label}")
    print("    ", " ".join(cmd))
    subprocess.run(list(cmd), cwd=REPO_ROOT, check=True)


def load_run_fingerprints(path: Path) -> Dict[str, str]:
    if not path.is_dir():
        raise FileNotFoundError(f"missing run directory: {path}")

    csv_path = path / "section_synapse_df.csv"
    df = pd.read_csv(csv_path)

    fps: Dict[str, str] = {}

    layout_cols = [
        "section_id_synapse",
        "loc",
        "distance_to_soma",
        "cluster_id",
        "cluster_flag",
        "cluster_center_flag",
        "pre_unit_id",
        "syn_w",
    ]
    fps["layout"] = sha256_text(df[layout_cols].astype(str).agg("|".join, axis=1).str.cat(sep="\n"))

    exc_df = df[df["type"] == "A"]
    fps["bg_spikes_exc"] = sha256_text(exc_df["spike_train_bg"].astype(str).str.cat(sep="\n"))

    clus_df = df[(df["type"] == "A") & (df["cluster_flag"] == 1)]
    fps["cluster_spikes"] = sha256_text(clus_df["spike_train"].astype(str).str.cat(sep="\n"))
    fps["cluster_layout"] = sha256_text(
        clus_df[layout_cols].astype(str).agg("|".join, axis=1).str.cat(sep="\n")
    )

    soma_path = path / "soma_v_array.npy"
    fps["soma_v"] = sha256_bytes(np.load(soma_path).tobytes())

    params_path = path / "simulation_params.json"
    with open(params_path, "r", encoding="utf-8") as f:
        params = json.load(f)
    for key in FIXED_SEEDS:
        fps[f"seed_{key}"] = sha256_text(str(params[key]))

    return fps


def assert_same(label: str, left: Dict[str, str], right: Dict[str, str], keys: List[str]) -> None:
    mismatches = [k for k in keys if left[k] != right[k]]
    if mismatches:
        details = "\n".join(f"  {k}: {left[k][:12]}... != {right[k][:12]}..." for k in mismatches)
        raise AssertionError(f"{label} mismatch on: {', '.join(mismatches)}\n{details}")


def assert_diff(label: str, left: Dict[str, str], right: Dict[str, str], keys: List[str]) -> None:
    same = [k for k in keys if left[k] == right[k]]
    if same:
        raise AssertionError(f"{label}: expected differences but these matched: {', '.join(same)}")


def report_soma_v_note(left_dir: Path, right_dir: Path, label: str) -> None:
    """Informational: soma_v may differ across worker counts even when inputs match."""
    left = np.load(left_dir / "soma_v_array.npy")
    right = np.load(right_dir / "soma_v_array.npy")
    if left.shape != right.shape:
        print(f"NOTE ({label}): soma_v shape {left.shape} vs {right.shape} (not compared)")
        return
    max_abs = float(np.max(np.abs(left - right)))
    if max_abs == 0.0:
        print(f"NOTE ({label}): soma_v identical (bit-exact)")
    else:
        print(
            f"NOTE ({label}): soma_v differs (max |Δ|={max_abs:.6g} mV). "
            "Expected when max_workers_synapse differs: NetCon attach order in NEURON, "
            "not seed leakage. Input fingerprints above are the seed reproducibility check."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed / worker reproducibility smoke tests")
    parser.add_argument(
        "--results_root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=f"Temporary output root (default: {DEFAULT_RESULTS_ROOT})",
    )
    parser.add_argument("--python", dest="python_bin", default="python", help="Python executable")
    parser.add_argument(
        "--keep_outputs",
        action="store_true",
        help="Do not delete results_root before running",
    )
    args = parser.parse_args()

    results_root = args.results_root.resolve()
    if not args.keep_outputs and results_root.exists():
        import shutil

        shutil.rmtree(results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    epoch = 99

    # 1) Synapse worker count should not affect stochastic inputs / traces.
    run_simulation(
        build_cmd(
            results_root,
            "seedtest_w1_s1",
            epoch=epoch,
            max_workers_epoch=1,
            max_workers_synapse=1,
            python_bin=args.python_bin,
        ),
        "worker test A: max_workers_epoch=1, max_workers_synapse=1",
    )
    run_simulation(
        build_cmd(
            results_root,
            "seedtest_w1_s50",
            epoch=epoch,
            max_workers_epoch=1,
            max_workers_synapse=50,
            python_bin=args.python_bin,
        ),
        "worker test B: max_workers_epoch=1, max_workers_synapse=50",
    )
    fp_a = load_run_fingerprints(run_dir(results_root, "seedtest_w1_s1", epoch))
    fp_b = load_run_fingerprints(run_dir(results_root, "seedtest_w1_s50", epoch))
    assert_same("synapse worker input reproducibility", fp_a, fp_b, INPUT_FINGERPRINT_KEYS)
    report_soma_v_note(
        run_dir(results_root, "seedtest_w1_s1", epoch),
        run_dir(results_root, "seedtest_w1_s50", epoch),
        "worker 1 vs 50",
    )
    print("PASS: synapse worker counts (1 vs 50) produce identical seed-controlled inputs")

    # 2) Epoch-level parallelism with fixed seeds: two epochs must match each other.
    parallel_epoch = 100
    run_simulation(
        build_cmd(
            results_root,
            "seedtest_parallel",
            num_epochs=2,
            start_epoch=parallel_epoch,
            max_workers_epoch=50,
            max_workers_synapse=30,
            python_bin=args.python_bin,
        ),
        "epoch parallel test: num_epochs=2, max_workers_epoch=50",
    )
    fp_ep0 = load_run_fingerprints(run_dir(results_root, "seedtest_parallel", parallel_epoch))
    fp_ep1 = load_run_fingerprints(run_dir(results_root, "seedtest_parallel", parallel_epoch + 1))
    assert_same("epoch parallel input reproducibility", fp_ep0, fp_ep1, INPUT_FINGERPRINT_KEYS)
    assert_same("fixed-seed vs epoch index (inputs)", fp_a, fp_ep0, INPUT_FINGERPRINT_KEYS)
    print("PASS: parallel epoch workers do not introduce extra input randomness")

    # 3) aff_mode=custom should be deterministic across worker settings.
    run_simulation(
        build_cmd(
            results_root,
            "seedtest_aff_custom",
            epoch=epoch,
            aff_mode="custom",
            aff_list=[0, 12, 24],
            max_workers_epoch=1,
            max_workers_synapse=1,
            python_bin=args.python_bin,
        ),
        "aff custom test A",
    )
    run_simulation(
        build_cmd(
            results_root,
            "seedtest_aff_custom2",
            epoch=epoch,
            aff_mode="custom",
            aff_list=[0, 12, 24],
            max_workers_epoch=50,
            max_workers_synapse=30,
            python_bin=args.python_bin,
        ),
        "aff custom test B (50/30 workers)",
    )
    fp_aff_a = load_run_fingerprints(run_dir(results_root, "seedtest_aff_custom", epoch))
    fp_aff_b = load_run_fingerprints(run_dir(results_root, "seedtest_aff_custom2", epoch))
    assert_same("aff_mode custom input reproducibility", fp_aff_a, fp_aff_b, INPUT_FINGERPRINT_KEYS)
    print("PASS: aff_mode=custom is deterministic across worker settings (inputs)")

    # 4) Each seed should affect only its own domain.
    # bg_syn_pos_seed  -> placement + exc syn_w (+ cluster pool); NOT bg spike times (bg_spike_gen_seed)
    # bg_spike_gen_seed -> bg spike_train_bg per row index
    # clus_syn_pos_seed  -> cluster assignment + perm
    # clus_spike_gen_seed -> cluster stimulus times (spike_train on cluster synapses)
    baseline = fp_a
    seed_checks: List[Tuple[str, Dict[str, int], Dict[str, List[str]]]] = [
        (
            "bg_syn_pos_seed",
            {**FIXED_SEEDS, "bg_syn_pos_seed": FIXED_SEEDS["bg_syn_pos_seed"] + 1},
            {
                "same": ["bg_spikes_exc"],
                "diff": ["layout", "cluster_layout", "cluster_spikes", "soma_v"],
            },
        ),
        (
            "bg_spike_gen_seed",
            {**FIXED_SEEDS, "bg_spike_gen_seed": FIXED_SEEDS["bg_spike_gen_seed"] + 1},
            {"same": ["layout", "cluster_layout"], "diff": ["bg_spikes_exc", "soma_v"]},
        ),
        (
            "clus_spike_gen_seed",
            {**FIXED_SEEDS, "clus_spike_gen_seed": FIXED_SEEDS["clus_spike_gen_seed"] + 1},
            {"same": ["layout", "cluster_layout", "bg_spikes_exc"], "diff": ["cluster_spikes", "soma_v"]},
        ),
        (
            "clus_syn_pos_seed",
            {**FIXED_SEEDS, "clus_syn_pos_seed": FIXED_SEEDS["clus_syn_pos_seed"] + 1},
            {"same": ["bg_spikes_exc"], "diff": ["layout", "cluster_layout", "cluster_spikes", "soma_v"]},
        ),
    ]

    for seed_name, seeds, expectations in seed_checks:
        suffix = f"seedtest_var_{seed_name}"
        run_simulation(
            build_cmd(
                results_root,
                suffix,
                epoch=epoch,
                seeds=seeds,
                max_workers_epoch=1,
                max_workers_synapse=1,
                python_bin=args.python_bin,
            ),
            f"seed isolation: vary {seed_name}",
        )
        fp_var = load_run_fingerprints(run_dir(results_root, suffix, epoch))
        assert_same(
            f"{seed_name} unchanged aspects",
            baseline,
            fp_var,
            expectations["same"],
        )
        assert_diff(
            f"{seed_name} changed aspects",
            baseline,
            fp_var,
            expectations["diff"],
        )
        print(f"PASS: {seed_name} controls expected outputs only")

    print("\nAll seed reproducibility checks passed.")
    print(
        "Input fingerprints (layout / spike trains) are the authoritative seed check. "
        "soma_v bit-exact match across worker counts is not required."
    )
    print(f"Outputs kept under: {results_root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"\nSimulation command failed with exit code {exc.returncode}", file=sys.stderr)
        raise SystemExit(exc.returncode)
    except Exception as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
