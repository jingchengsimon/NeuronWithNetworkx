#!/usr/bin/env python3
"""
Headless trace visualization from 3_trace_analysis.ipynb (visualization + load_data + process_voltage_data).

Saves figures under this repo's results/ directory (no Jupyter / display required).
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "trace_analysis"


def _trapz(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


def load_data(exp, root_folder_path: str = ""):
    folder = exp if os.path.isabs(exp) else os.path.join(root_folder_path, exp)
    dt = 1 / 40000

    npy_files = {
        "v": "dend_v_array",
        "i": "dend_i_array",
        "nmda": "dend_nmda_i_array",
        "ampa": "dend_ampa_i_array",
        "nmda_g": "dend_nmda_g_array",
        "ampa_g": "dend_ampa_g_array",
        "soma": "soma_v_array",
        "apic_v": "apic_v_array",
        "apic_ica": "apic_ica_array",
        "soma_i": "soma_i_array",
        "trunk_v": "trunk_v_array",
        "basal_v": "basal_v_array",
        "tuft_v": "tuft_v_array",
        "basal_bg_i_nmda": "basal_bg_i_nmda_array",
        "basal_bg_i_ampa": "basal_bg_i_ampa_array",
        "tuft_bg_i_nmda": "tuft_bg_i_nmda_array",
        "tuft_bg_i_ampa": "tuft_bg_i_ampa_array",
    }

    data = {}
    for var_name, file_base in npy_files.items():
        file_path = os.path.join(folder, f"{file_base}.npy")
        if os.path.exists(file_path):
            data[var_name] = np.load(file_path)
        else:
            data[var_name] = None

    with open(os.path.join(folder, "simulation_params.json")) as f:
        simu_info = json.load(f)

    sec_syn_df_path = os.path.join(folder, "section_synapse_df.csv")
    if os.path.exists(sec_syn_df_path):
        sec_syn_df = pd.read_csv(sec_syn_df_path)
    else:
        sec_syn_df = None

    data["dt"] = dt
    data["simu_info"] = simu_info
    data["sec_syn_df"] = sec_syn_df

    return data


def process_voltage_data(exp, root_folder_path: str = ""):
    data = load_data(exp, root_folder_path=root_folder_path)
    v, soma, apic_v, trunk_v, basal_v, tuft_v, sec_syn_df, dt = [
        data.get(k)
        for k in (
            "v",
            "soma",
            "apic_v",
            "trunk_v",
            "basal_v",
            "tuft_v",
            "sec_syn_df",
            "dt",
        )
    ]
    simu_info = data["simu_info"]

    if v is not None and v.ndim == 5:
        v = np.mean(v, axis=2)
        soma = np.mean(soma, axis=1)
        apic_v = np.mean(apic_v, axis=1)
        trunk_v = np.mean(trunk_v, axis=1)
        basal_v = np.mean(basal_v, axis=1)
        tuft_v = np.mean(tuft_v, axis=1)
    return v, soma, apic_v, trunk_v, basal_v, tuft_v, sec_syn_df, dt, simu_info


def visualization(
    exp,
    trial_idx=0,
    t_start=0,
    t_end=1000,
    with_background=True,
    axes_1=None,
    plot_flag=True,
    dend_flag="peak",
    root_folder_path: str = "",
    save_path: str | None = None,
    close_figure: bool = True,
):
    v, soma, apic_v, trunk_v, basal_v, tuft_v, sec_syn_df, dt, _simu_info = process_voltage_data(
        exp, root_folder_path=root_folder_path
    )

    t_start, t_end = t_start * 40, t_end * 40
    t_vals = np.arange(t_start, t_end) * dt
    x_vals = 1000 * t_vals

    x_dend = np.arange(t_end - t_start) * dt

    color_list = ["C0", "C1", "C2"]
    num_clus = np.min([v.shape[0], len(color_list)])

    s0 = soma[t_start:t_end, 0, trial_idx]
    a0 = apic_v[t_start:t_end, 0, trial_idx]
    v0 = v[:, t_start:t_end, 0, trial_idx]

    syn_num = 0
    if plot_flag:
        num_subplots = 9
        fig1, axes = plt.subplots(
            num_subplots // 3,
            3,
            figsize=(min(40, (t_end - t_start) // 1000) * 3.5, (num_subplots // 3) * 4),
            sharex=False,
            gridspec_kw={"width_ratios": [1, 1, 1]},
        )
        axes = list(axes.flat)
        plt.ioff()

        axes[6].set_title("Dend EPSPs")

        for syn_num in range(0, v.shape[2], v.shape[2] // 6):
            alpha = min(1, 0.2 + 0.8 * (syn_num + 1) / v.shape[2])
            if with_background:
                s = soma[t_start:t_end, syn_num, trial_idx]
                a = apic_v[t_start:t_end, syn_num, trial_idx]
                vd = v[:, t_start:t_end, syn_num, trial_idx]
                axes[0].plot(x_vals, s, alpha=alpha, color="k")
                axes[3].plot(x_vals, a, alpha=alpha, color="salmon")
                for clus_idx in range(num_clus):
                    axes[6].plot(
                        x_vals,
                        vd[clus_idx],
                        alpha=(alpha if num_clus == 1 else 0.3),
                        color=color_list[clus_idx],
                    )
            else:
                if "expected" not in exp:
                    s = soma[t_start:t_end, syn_num, trial_idx]
                    a = apic_v[t_start:t_end, syn_num, trial_idx]
                    vd = v[:, t_start:t_end, syn_num, trial_idx]
                    axes[0].plot(x_vals, s - s0, alpha=alpha, color="k")
                    axes[3].plot(x_vals, a - a0, alpha=alpha, color="salmon")
                    for clus_idx in range(num_clus):
                        axes[6].plot(
                            x_vals,
                            vd[clus_idx] - v0[clus_idx],
                            alpha=(alpha if num_clus == 1 else 0.3),
                            color=color_list[clus_idx],
                        )
                else:
                    soma_sum = np.zeros_like(s0)
                    apic_sum = np.zeros_like(a0)
                    v_sum = np.zeros_like(v0)
                    for idx in range(syn_num + 1):
                        soma_sum += soma[t_start:t_end, idx, trial_idx] - s0
                        apic_sum += apic_v[t_start:t_end, idx, trial_idx] - a0
                        v_sum += v[:, t_start:t_end, idx, trial_idx] - v0
                    axes[0].plot(x_vals, soma_sum, alpha=alpha, color="k")
                    axes[3].plot(x_vals, apic_sum, alpha=alpha, color="salmon")
                    for clus_idx in range(num_clus):
                        axes[6].plot(
                            x_vals,
                            v_sum[clus_idx],
                            alpha=(alpha if num_clus == 1 else 0.3),
                            color=color_list[clus_idx],
                        )

    soma_peak_list = []
    apic_peak_list = []
    dend_peak_list = []
    dend_area_list = []

    for syn_idx in range(v.shape[2]):
        s = soma[t_start:t_end, syn_idx, trial_idx]
        a = apic_v[t_start:t_end, syn_idx, trial_idx]
        vd = v[:, t_start:t_end, syn_idx, trial_idx]

        if "expected" not in exp:
            soma_peak = np.max(s - s0)
            apic_peak = np.max(a - a0)
            dend_peak = np.max(vd - v0)

            delta_dend = vd - v0
            dend_over_t = np.mean(np.clip(delta_dend, 1, None), axis=0)
            dend_area = _trapz(dend_over_t, x_dend)

        else:
            soma_sum = np.zeros_like(s0)
            apic_sum = np.zeros_like(a0)
            v_sum = np.zeros_like(v0)
            for idx in range(syn_idx + 1):
                soma_sum += soma[t_start:t_end, idx, trial_idx] - s0
                apic_sum += apic_v[t_start:t_end, idx, trial_idx] - a0
                v_sum += v[:, t_start:t_end, idx, trial_idx] - v0
            soma_peak = np.max(soma_sum)
            apic_peak = np.max(apic_sum)
            dend_peak = np.max(v_sum)

            delta_dend = v_sum
            dend_over_t = np.mean(np.clip(delta_dend, 1, None), axis=0)
            dend_area = _trapz(dend_over_t, x_dend)

        soma_peak_list.append(soma_peak)
        apic_peak_list.append(apic_peak)
        dend_peak_list.append(dend_peak)
        dend_area_list.append(dend_area)

    soma_peak_list = [x - soma_peak_list[0] for x in soma_peak_list]
    apic_peak_list = [x - apic_peak_list[0] for x in apic_peak_list]
    dend_peak_list = [x - dend_peak_list[0] for x in dend_peak_list]
    dend_area_list = [x - dend_area_list[0] for x in dend_area_list]

    dend_list = dend_peak_list if dend_flag == "peak" else dend_area_list

    if plot_flag:
        if axes_1 is not None:
            axes_tmp = axes
            axes = axes_1

        if "singclus" in exp:
            if "expected" not in exp:
                axes[2].plot(range(0, 72 + 1, 2), soma_peak_list, color="k")
                axes[5].plot(range(0, 72 + 1, 2), apic_peak_list, color="salmon")
                axes[8].plot(range(0, 72 + 1, 2), dend_list, color=color_list[0])
            else:
                axes[2].plot(range(0, 72 + 1), soma_peak_list, color="k")
                axes[5].plot(range(0, 72 + 1), apic_peak_list, color="salmon")
                axes[8].plot(range(0, 72 + 1), dend_list, color=color_list[0])

            axes[2].set_xticks([0, 24, 48, 72])
            axes[5].set_xticks([0, 24, 48, 72])
            axes[8].set_xticks([0, 24, 48, 72])
        elif "multiclus_3" in exp:
            axes[2].plot(range(0, 24 + 1, 2), soma_peak_list, color="k")
            axes[5].plot(range(0, 24 + 1, 2), apic_peak_list, color="salmon")
            axes[8].plot(range(0, 24 + 1, 2), dend_list, color=color_list[0])
        elif "multiclus" in exp:
            axes[2].plot([0, 3, 6, 9, 12, 18, 24], soma_peak_list, color="k")
            axes[5].plot([0, 3, 6, 9, 12, 18, 24], apic_peak_list, color="salmon")
            axes[8].plot([0, 3, 6, 9, 12, 18, 24], dend_list, color=color_list[0])

        if axes_1 is not None:
            axes = axes_tmp

        if sec_syn_df is not None:
            clus_sec_syn_df = (
                sec_syn_df[sec_syn_df["cluster_flag"] == 1]
                .sort_values(by="pre_unit_id", ascending=True)
                .reset_index(drop=True)
            )

            exc_bg_sec_syn_df = sec_syn_df[sec_syn_df["type"].isin(["A"])]
            inh_bg_sec_syn_df = sec_syn_df[sec_syn_df["type"].isin(["B"])]

            for i, spike_train in enumerate(clus_sec_syn_df["spike_train"]):
                try:
                    spike_train = ast.literal_eval(spike_train)
                    if "_2" in exp:
                        spike_time_list = spike_train[-1]
                    else:
                        spike_time_list = spike_train[:13][-1]

                    if "expected" in exp:
                        non_empty_index = [i for i, sublist in enumerate(spike_train) if len(sublist) > 0][0]
                        spike_time_list = spike_train[non_empty_index]

                except IndexError:
                    spike_train = []

                color = "purple"
                if len(spike_train) > 0:
                    axes[1].vlines(spike_time_list, i + 0.5, i + 1.5, color=color, linewidth=6)

            for kind, df, ax_idx in [("exc", exc_bg_sec_syn_df, 4), ("inh", inh_bg_sec_syn_df, 7)]:
                for i, (spike_train, syn_region) in enumerate(zip(df["spike_train_bg"], df["region"])):
                    try:
                        spike_time_list = ast.literal_eval(spike_train)[0 if kind == "exc" else -1]
                    except IndexError:
                        spike_time_list = []
                    if len(spike_time_list) > 0:
                        color = "blue" if syn_region == "basal" else "red" if syn_region == "apical" else "black"
                        axes[ax_idx].vlines(spike_time_list, i + 0.5, i + 1.5, color=color, linewidth=6)

        for ax, title in zip(
            [axes[i] for i in [1, 4, 7]],
            [
                "Synchronous stimulated synaptic inputs",
                "Raster Plot of Background Excitatory Spike Trains",
                "Raster Plot of Background Inhibitory Spike Trains",
            ],
        ):
            ax.set_xlim(t_start // 40, t_end // 40)
            ax.set_ylabel("Neuron Index")
            ax.set_title(title)
            ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

        for ax_idx in range(num_subplots):
            axes[ax_idx].spines["top"].set_visible(False)
            axes[ax_idx].spines["right"].set_visible(False)

        if with_background:
            axes[0].hlines(-70, t_start // 40, t_end // 40, color="k", linestyle="dashed", linewidth=1)
            axes[0].text(975 * t_vals[0], -70, "-70mV", fontsize=10, color="k")

        axes[0].set_xticks([490])
        axes[0].set_xticklabels(["10 ms"])
        axes[0].spines["bottom"].set_bounds(480, 490)

        if with_background:
            soma_v_baseline = np.mean(soma[t_start - 200 : t_start - 100, 0, trial_idx])
        else:
            soma_v_baseline = 0

        if with_background is False:
            axes[0].set_ylim(soma_v_baseline - 0.5, soma_v_baseline + soma_peak_list[-1] + 0.5)
            axes[0].set_yticks([soma_v_baseline + 0.5])
            axes[0].spines["left"].set_bounds(soma_v_baseline - 0.5, soma_v_baseline + 0.5)
        elif "vitro" in exp:
            axes[0].set_ylim(soma_v_baseline - 0.5, soma_v_baseline + 8)
            axes[0].set_yticks([soma_v_baseline + 0.5])
            axes[0].spines["left"].set_bounds(soma_v_baseline - 0.5, soma_v_baseline + 0.5)
        else:
            axes[0].set_ylim(soma_v_baseline - 1.5, soma_v_baseline + 8)
            axes[0].set_yticks([soma_v_baseline - 0.5])
            axes[0].spines["left"].set_bounds(soma_v_baseline - 1.5, soma_v_baseline - 0.5)
        axes[0].set_yticklabels(["1 mV"], rotation=90)

        axes[0].spines["bottom"].set_linewidth(1)
        axes[0].spines["left"].set_linewidth(1)

        if "expected" not in exp:
            axes[6].set_xticks([490])
            axes[6].set_xticklabels(["10 ms"])
            axes[6].set_ylim(-5, 90)
            axes[6].set_yticks([10 - 5])
            axes[6].set_yticklabels(["10 mV"], rotation=90)
            axes[6].spines["bottom"].set_bounds(480, 490)
            axes[6].spines["left"].set_bounds(-5, 5)
            axes[6].spines["bottom"].set_linewidth(1)
            axes[6].spines["left"].set_linewidth(1)

            axes[8].set_xlabel("Number of synapses")
            axes[8].set_ylabel(
                "Peak Dendritic EPSP (mV)"
                if dend_flag == "peak"
                else "Dendritic EPSP area (mV·s)"
            )

        fig1.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig1.savefig(save_path, bbox_inches="tight", dpi=150)
        if close_figure:
            plt.close(fig1)

    if plot_flag:
        return axes, soma_peak_list
    return soma_peak_list


def _parse_epoch_range(s: str) -> list[int]:
    s = s.strip()
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",") if x.strip()]


def _batch_subdir_from_exp(sim_root: Path, exp_path: Path, rel_under_root: str) -> str:
    """
    Output folder name under --out_dir: path under sim-root with the last segment (epoch dir) removed,
    path segments joined with '_' (e.g. basal_range2_.../1/21 -> basal_range2_..._1).
    """
    rel_under_root = rel_under_root.replace("\\", "/").strip("/")
    parts = Path(rel_under_root).parts
    if len(parts) > 1:
        return "_".join(parts[:-1])
    try:
        exp_r = exp_path.resolve()
        root_r = sim_root.resolve()
        rp = exp_r.relative_to(root_r).parts
        if len(rp) > 1:
            return "_".join(rp[:-1])
    except ValueError:
        pass
    return "batch"


def main():
    p = argparse.ArgumentParser(description="Batch trace visualization (from 3_trace_analysis.ipynb).")
    p.add_argument(
        "--sim_root",
        default="/G/results/simulation_singclus_supple_Apr26",
        help="Directory containing per-condition subfolders.",
    )
    p.add_argument(
        "--rel_path_template",
        default="{region}_range{range_idx}_clus_invitro_singclus_fixedW0.0004/1/{epoch}",
        help="Path under sim-root; placeholders: {region}, {range_idx}, {epoch}.",
    )
    p.add_argument("--region", default="basal")
    p.add_argument("--range_idx", type=int, default=1)
    p.add_argument("--epochs", default="8", help="e.g. 21-30 or 21,22,23")
    p.add_argument("--trial_idx", type=int, default=0)
    p.add_argument("--t_start", type=float, default=480)
    p.add_argument("--t_end", type=float, default=600)
    p.add_argument("--dend_flag", default="area", choices=("peak", "area"))
    p.add_argument(
        "--out_dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help="Figures written here (default: <repo>/results/trace_analysis).",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    epochs = _parse_epoch_range(args.epochs)
    sim_root = Path(args.sim_root)

    for epoch in epochs:
        rel = args.rel_path_template.format(region=args.region, range_idx=args.range_idx, epoch=epoch)
        exp_path = sim_root / rel
        if not exp_path.is_dir():
            print(f"Skip (missing): {exp_path}")
            continue

        run_subdir = _batch_subdir_from_exp(sim_root, exp_path, rel)
        run_subdir = f"{run_subdir}_dend_{args.dend_flag}"
        batch_dir = out_dir / run_subdir

        stem = str(epoch)
        if args.with_background:
            stem = f"{epoch}_bg"
        fname = f"{stem}.png"
        save_path = batch_dir / fname

        visualization(
            str(exp_path),
            trial_idx=args.trial_idx,
            t_start=args.t_start,
            t_end=args.t_end,
            with_background=args.with_background,
            dend_flag=args.dend_flag,
            root_folder_path="",
            save_path=str(save_path),
            close_figure=True,
        )
        print(f"Saved {save_path}")


if __name__ == "__main__":
    main()
