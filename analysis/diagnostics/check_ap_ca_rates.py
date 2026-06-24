#!/usr/bin/env python3
"""
Quick SSH-side check: print per-epoch AP / Ca spike rates (Hz) at aff_label=72.

Reuses detection logic from analysis.ap_ca_spike_analysis.
Example (50 epochs, folder_tag=1):

  cd /path/to/NeuronWithNetworkx
  python -m analysis.diagnostics.check_ap_ca_rates \\
    --root_dir /G/results/simulation_singclus_supple_May26 \\
    --start_epoch 1 --num_epochs 50 \\
    --folder_tag 1 --aff_label 72
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.ap_ca_spike_analysis import (  # noqa: E402
    _exp_dir_from_base,
    _find_epoch_folders_two_level,
    _load_run_folder,
    _normalize_trace_shape,
    _resolve_aff_index_for_label,
    compute_ap_spike_rate_hz,
    compute_ca_spike_rate_hz,
    _dt_seconds_from_trace,
    _sim_duration_ms,
)


def _epoch_folder(
    root_dir: str,
    sec_type: str,
    range_idx: int,
    condition: str,
    suffix: str,
    folder_tag: str,
    epoch: int,
) -> str | None:
    base_prefix = f"{root_dir.rstrip('/')}/{sec_type}_range{range_idx}_clus_invivo_"
    exp_dir = _exp_dir_from_base(base_prefix, condition, suffix)
    folder = os.path.join(exp_dir, str(folder_tag), str(epoch))
    if os.path.isdir(folder) and os.path.exists(os.path.join(folder, "simulation_params.json")):
        return folder
    return None


def _rates_for_folder(folder: str, aff_label: int, stim_idx: int = 0) -> dict[str, float | str]:
    soma_raw, apic_raw, simu_info = _load_run_folder(folder)
    if simu_info is None:
        return {"ap_hz": np.nan, "ca_hz": np.nan, "status": "no_json"}

    dur_ms = _sim_duration_ms(simu_info)
    out: dict[str, float | str] = {"ap_hz": np.nan, "ca_hz": np.nan, "status": "ok"}

    if soma_raw is not None:
        soma = _normalize_trace_shape(soma_raw)
        n_aff = soma.shape[2]
        try:
            aff_idx = _resolve_aff_index_for_label(simu_info, n_aff, aff_label)
        except ValueError as e:
            out["status"] = f"aff_err:{e}"
            return out
        if stim_idx < soma.shape[1] and aff_idx < n_aff:
            trace = soma[:, stim_idx, aff_idx, 0]
            out["ap_hz"] = float(compute_ap_spike_rate_hz(trace, dur_ms))

    if apic_raw is not None:
        apic = _normalize_trace_shape(apic_raw)
        dt_s = _dt_seconds_from_trace(apic, simu_info)
        n_aff = apic.shape[2]
        try:
            aff_idx = _resolve_aff_index_for_label(simu_info, n_aff, aff_label)
        except ValueError:
            return out
        if stim_idx < apic.shape[1] and aff_idx < n_aff:
            trace = apic[:, stim_idx, aff_idx, 0]
            out["ca_hz"] = float(
                compute_ca_spike_rate_hz(trace, dt_s, dur_ms)
            )

    if np.isnan(out["ap_hz"]) and np.isnan(out["ca_hz"]):
        out["status"] = "missing_arrays"
    return out


def scan_condition(
    root_dir: str,
    condition: str,
    sec_type: str,
    range_idx: int,
    suffix: str,
    folder_tag: str,
    epochs: list[int],
    aff_label: int,
    stim_idx: int,
) -> list[dict]:
    rows = []
    for ep in epochs:
        folder = _epoch_folder(
            root_dir, sec_type, range_idx, condition, suffix, folder_tag, ep
        )
        if folder is None:
            rows.append(
                dict(
                    epoch=ep,
                    condition=condition,
                    sec_type=sec_type,
                    range_idx=range_idx,
                    ap_hz=np.nan,
                    ca_hz=np.nan,
                    status="missing_folder",
                    folder="",
                )
            )
            continue
        r = _rates_for_folder(folder, aff_label, stim_idx=stim_idx)
        rows.append(
            dict(
                epoch=ep,
                condition=condition,
                sec_type=sec_type,
                range_idx=range_idx,
                ap_hz=r["ap_hz"],
                ca_hz=r["ca_hz"],
                status=r["status"],
                folder=folder,
            )
        )
    return rows


def _print_table(title: str, rows: list[dict]) -> None:
    print(f"\n{'=' * 72}")
    print(title)
    print(f"{'epoch':>5}  {'ap_hz':>10}  {'ca_hz':>10}  {'status':<16}")
    print("-" * 72)
    for r in rows:
        ap_s = f"{r['ap_hz']:.6f}" if np.isfinite(r["ap_hz"]) else "     nan"
        ca_s = f"{r['ca_hz']:.6f}" if np.isfinite(r["ca_hz"]) else "     nan"
        print(f"{r['epoch']:5d}  {ap_s:>10}  {ca_s:>10}  {r['status']:<16}")
    ap_vals = [r["ap_hz"] for r in rows if np.isfinite(r["ap_hz"])]
    ca_vals = [r["ca_hz"] for r in rows if np.isfinite(r["ca_hz"])]
    print("-" * 72)
    if ap_vals:
        print(
            f"AP  median={np.median(ap_vals):.6f}  max={np.max(ap_vals):.6f}  "
            f"nonzero_epochs={sum(v > 0 for v in ap_vals)}/{len(ap_vals)}"
        )
    else:
        print("AP  (no valid epochs)")
    if ca_vals:
        print(
            f"Ca  median={np.median(ca_vals):.6f}  max={np.max(ca_vals):.6f}  "
            f"nonzero_epochs={sum(v > 0 for v in ca_vals)}/{len(ca_vals)}"
        )
    else:
        print("Ca  (no valid epochs)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print per-epoch AP/Ca spike rates (Hz) for SSH quick checks."
    )
    parser.add_argument(
        "--root_dir",
        default="/G/results/simulation_singclus_supple_May26",
        help="Parent dir with <sec>_range<N>_clus_invivo_<suffix> trees.",
    )
    parser.add_argument("--suffix", default="singclus_ap")
    parser.add_argument("--folder_tag", default="1")
    parser.add_argument("--start_epoch", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=None,
        help="Explicit epoch list (overrides start_epoch/num_epochs).",
    )
    parser.add_argument("--aff_label", type=int, default=72)
    parser.add_argument("--stim_idx", type=int, default=0)
    parser.add_argument(
        "--range_idxs",
        nargs="+",
        type=int,
        default=[0, 1, 2],
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["clus", "distr"],
        choices=["clus", "distr"],
    )
    parser.add_argument(
        "--sec_types",
        nargs="+",
        default=["basal", "apical"],
        choices=["basal", "apical"],
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="One line per epoch (all combos) instead of separate tables.",
    )
    args = parser.parse_args()

    if args.epochs is not None:
        epochs = sorted(args.epochs)
    else:
        epochs = list(range(args.start_epoch, args.start_epoch + args.num_epochs))

    all_rows: list[dict] = []
    for condition in args.conditions:
        for sec_type in args.sec_types:
            for range_idx in args.range_idxs:
                rows = scan_condition(
                    args.root_dir,
                    condition,
                    sec_type,
                    range_idx,
                    args.suffix,
                    args.folder_tag,
                    epochs,
                    args.aff_label,
                    args.stim_idx,
                )
                all_rows.extend(rows)
                if not args.compact:
                    _print_table(
                        f"{condition} | {sec_type} | range{range_idx} | "
                        f"aff={args.aff_label} | tag={args.folder_tag}",
                        rows,
                    )

    if args.compact:
        print(
            f"\n{'epoch':>5} {'cond':>5} {'sec':>6} {'rng':>3} "
            f"{'ap_hz':>10} {'ca_hz':>10} {'status':<14}"
        )
        print("-" * 60)
        for r in sorted(
            all_rows, key=lambda x: (x["epoch"], x["condition"], x["sec_type"], x["range_idx"])
        ):
            ap_s = f"{r['ap_hz']:.4f}" if np.isfinite(r["ap_hz"]) else "   nan"
            ca_s = f"{r['ca_hz']:.4f}" if np.isfinite(r["ca_hz"]) else "   nan"
            print(
                f"{r['epoch']:5d} {r['condition']:>5} {r['sec_type']:>6} {r['range_idx']:3d} "
                f"{ap_s:>10} {ca_s:>10} {r['status']:<14}"
            )

    # Global summary across all scanned folders
    ap_all = [r["ap_hz"] for r in all_rows if np.isfinite(r["ap_hz"])]
    ca_all = [r["ca_hz"] for r in all_rows if np.isfinite(r["ca_hz"])]
    miss = sum(1 for r in all_rows if r["status"] != "ok")
    print(f"\n{'=' * 72}")
    print(
        f"Scanned {len(all_rows)} folder-epochs | missing/bad: {miss} | "
        f"aff_label={args.aff_label}"
    )
    if ap_all:
        print(
            f"ALL AP: median={np.median(ap_all):.6f} Hz  max={np.max(ap_all):.6f}  "
            f"frac>0={np.mean(np.array(ap_all) > 0):.3f}"
        )
    if ca_all:
        print(
            f"ALL Ca: median={np.median(ca_all):.6f} Hz  max={np.max(ca_all):.6f}  "
            f"frac>0={np.mean(np.array(ca_all) > 0):.3f}"
        )


if __name__ == "__main__":
    main()
