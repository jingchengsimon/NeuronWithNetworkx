#!/usr/bin/env python3
"""
Figure 1 style panels (dendritic + somatic) from trace_visualization / 3_trace_analysis.ipynb.

Produces two PDFs per epoch:
  - Dend: EPSP traces, num_syn vs peak, num_syn vs area
  - Soma: measured EPSPs, expected (linear sum) EPSPs, measured vs expected summary
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from trace_visualization import PROJECT_ROOT, _parse_epoch_range, _trapz, process_voltage_data


def _synapse_x_axis(n_points: int, is_expected: bool) -> np.ndarray:
    """Match trace_visualization x ticks: singclus uses 0..72 step 2 (measured) or step 1 (expected)."""
    if is_expected:
        return np.arange(0, n_points, dtype=float)
    return np.arange(0, 2 * n_points, 2, dtype=float)


def _compute_summaries(
    exp: str,
    trial_idx: int,
    t_start_ms: float,
    t_end_ms: float,
    root_folder_path: str = "",
):
    v, soma, apic_v, trunk_v, basal_v, tuft_v, sec_syn_df, dt, _simu_info = process_voltage_data(
        exp, root_folder_path=root_folder_path
    )

    t_start = int(t_start_ms * 40)
    t_end = int(t_end_ms * 40)
    t_vals = np.arange(t_start, t_end) * dt
    x_vals = 1000 * t_vals
    x_dend = np.arange(t_end - t_start) * dt

    s0 = soma[t_start:t_end, 0, trial_idx]
    a0 = apic_v[t_start:t_end, 0, trial_idx]
    v0 = v[:, t_start:t_end, 0, trial_idx]

    is_expected = "expected" in exp

    soma_peak_list: list[float] = []
    dend_peak_list: list[float] = []
    dend_area_list: list[float] = []

    for syn_idx in range(v.shape[2]):
        s = soma[t_start:t_end, syn_idx, trial_idx]
        a = apic_v[t_start:t_end, syn_idx, trial_idx]
        vd = v[:, t_start:t_end, syn_idx, trial_idx]

        if not is_expected:
            soma_peak = float(np.max(s - s0))
            dend_peak = float(np.max(vd - v0))
            delta_dend = vd - v0
            dend_over_t = np.mean(np.clip(delta_dend, 1, None), axis=0)
            dend_area = float(_trapz(dend_over_t, x_dend))
        else:
            soma_sum = np.zeros_like(s0)
            apic_sum = np.zeros_like(a0)
            v_sum = np.zeros_like(v0)
            for idx in range(syn_idx + 1):
                soma_sum += soma[t_start:t_end, idx, trial_idx] - s0
                apic_sum += apic_v[t_start:t_end, idx, trial_idx] - a0
                v_sum += v[:, t_start:t_end, idx, trial_idx] - v0
            soma_peak = float(np.max(soma_sum))
            dend_peak = float(np.max(v_sum))
            delta_dend = v_sum
            dend_over_t = np.mean(np.clip(delta_dend, 1, None), axis=0)
            dend_area = float(_trapz(dend_over_t, x_dend))

        soma_peak_list.append(soma_peak)
        dend_peak_list.append(dend_peak)
        dend_area_list.append(dend_area)

    soma_peak_list = [x - soma_peak_list[0] for x in soma_peak_list]
    dend_peak_list = [x - dend_peak_list[0] for x in dend_peak_list]
    dend_area_list = [x - dend_area_list[0] for x in dend_area_list]

    color_list = ["C0", "C1", "C2"]
    num_clus = int(np.min([v.shape[0], len(color_list)]))

    return {
        "v": v,
        "soma": soma,
        "apic_v": apic_v,
        "x_vals": x_vals,
        "t_start": t_start,
        "t_end": t_end,
        "s0": s0,
        "a0": a0,
        "v0": v0,
        "trial_idx": trial_idx,
        "num_clus": num_clus,
        "color_list": color_list,
        "soma_peak_list": soma_peak_list,
        "dend_peak_list": dend_peak_list,
        "dend_area_list": dend_area_list,
        "is_expected": is_expected,
    }


def _subplot_shape(panel_layout: str) -> tuple[int, int]:
    if panel_layout == "vertical":
        return 3, 1
    if panel_layout == "horizontal":
        return 1, 3
    raise ValueError(panel_layout)


def _plot_dend_traces(ax, pack: dict) -> None:
    v = pack["v"]
    x_vals = pack["x_vals"]
    t_start = pack["t_start"]
    t_end = pack["t_end"]
    s0 = pack["s0"]
    a0 = pack["a0"]
    v0 = pack["v0"]
    trial_idx = pack["trial_idx"]
    num_clus = pack["num_clus"]
    color_list = pack["color_list"]

    step = max(1, v.shape[2] // 6)
    for syn_num in range(0, v.shape[2], step):
        alpha = min(1.0, 0.2 + 0.8 * (syn_num + 1) / v.shape[2])
        vd = v[:, t_start:t_end, syn_num, trial_idx]
        for clus_idx in range(num_clus):
            ax.plot(
                x_vals,
                vd[clus_idx] - v0[clus_idx],
                alpha=(alpha if num_clus == 1 else 0.3),
                color=color_list[clus_idx],
            )

    ax.set_title("Dendritic EPSPs across activation levels")
    ax.set_xticks([490])
    ax.set_xticklabels(["10 ms"])
    ax.set_ylim(-5, 90)
    ax.set_yticks([10 - 5])
    ax.set_yticklabels(["10 mV"], rotation=90)
    ax.spines["bottom"].set_bounds(480, 490)
    ax.spines["left"].set_bounds(-5, 5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_soma_traces(
    ax,
    pack: dict,
    cumulative_expected: bool,
    ylim: tuple[float, float] | None = None,
) -> None:
    v = pack["v"]
    soma = pack["soma"]
    x_vals = pack["x_vals"]
    t_start = pack["t_start"]
    t_end = pack["t_end"]
    s0 = pack["s0"]
    trial_idx = pack["trial_idx"]

    step = max(1, v.shape[2] // 6)
    for syn_num in range(0, v.shape[2], step):
        alpha = min(1.0, 0.2 + 0.8 * (syn_num + 1) / v.shape[2])
        if not cumulative_expected:
            s = soma[t_start:t_end, syn_num, trial_idx]
            ax.plot(x_vals, s - s0, alpha=alpha, color="k")
        else:
            soma_sum = np.zeros_like(s0)
            for idx in range(syn_num + 1):
                soma_sum += soma[t_start:t_end, idx, trial_idx] - s0
            ax.plot(x_vals, soma_sum, alpha=alpha, color="k")

    ax.set_xticks([490])
    ax.set_xticklabels(["10 ms"])
    soma_v_baseline = 0.0
    if ylim is None:
        ylim = (
            soma_v_baseline - 0.5,
            soma_v_baseline + float(pack["soma_peak_list"][-1]) + 0.5,
        )
    ax.set_ylim(ylim)
    ax.set_yticks([soma_v_baseline + 0.5])
    ax.spines["left"].set_bounds(soma_v_baseline - 0.5, soma_v_baseline + 0.5)
    ax.set_yticklabels(["1 mV"], rotation=90)
    ax.spines["bottom"].set_bounds(480, 490)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_dend_figure(
    exp_measured: str,
    trial_idx: int,
    t_start_ms: float,
    t_end_ms: float,
    panel_layout: str,
    root_folder_path: str = "",
) -> plt.Figure:
    pack = _compute_summaries(exp_measured, trial_idx, t_start_ms, t_end_ms, root_folder_path)
    if pack["is_expected"]:
        raise ValueError("Dend figure requires measured (non-expected) experiment path.")

    nrows, ncols = _subplot_shape(panel_layout)
    fig_w, fig_h = (12, 10) if panel_layout == "vertical" else (14, 4.5)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    axes = np.atleast_1d(axes).ravel()

    _plot_dend_traces(axes[0], pack)

    sx = _synapse_x_axis(len(pack["dend_peak_list"]), is_expected=False)
    axes[1].plot(sx, pack["dend_peak_list"], color="C0")
    axes[1].set_title("Dendritic EPSP peak vs synaptic activation")
    axes[1].set_xlabel("Number of synapses")
    axes[1].set_ylabel("Peak dendritic EPSP (mV)")
    axes[1].set_xticks([0, 24, 48, 72])
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    axes[2].plot(sx, pack["dend_area_list"], color="C0")
    axes[2].set_title("Integrated depolarization vs synaptic activation")
    axes[2].set_xlabel("Number of synapses")
    axes[2].set_ylabel("Time-integrated depolarization (mV·s)")
    axes[2].set_xticks([0, 24, 48, 72])
    axes[2].spines["top"].set_visible(False)
    axes[2].spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


def plot_soma_figure(
    exp_measured: str,
    exp_expected: str,
    trial_idx: int,
    t_start_ms: float,
    t_end_ms: float,
    panel_layout: str,
    root_folder_path: str = "",
) -> plt.Figure:
    pack_m = _compute_summaries(exp_measured, trial_idx, t_start_ms, t_end_ms, root_folder_path)
    pack_e = _compute_summaries(exp_expected, trial_idx, t_start_ms, t_end_ms, root_folder_path)
    if pack_m["is_expected"]:
        raise ValueError("Soma measured path must not be an expected/linear-sum folder.")
    if not pack_e["is_expected"]:
        raise ValueError("Soma expected path should contain 'expected' (linear cumulative) logic.")

    nrows, ncols = _subplot_shape(panel_layout)
    fig_w, fig_h = (12, 10) if panel_layout == "vertical" else (14, 4.5)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    axes = np.atleast_1d(axes).ravel()

    soma_v_baseline = 0.0
    ymax = max(
        float(pack_m["soma_peak_list"][-1]),
        float(pack_e["soma_peak_list"][-1]),
    )
    soma_ylim = (soma_v_baseline - 0.5, soma_v_baseline + ymax + 0.5)

    _plot_soma_traces(axes[0], pack_m, cumulative_expected=False, ylim=soma_ylim)
    axes[0].set_title("Soma EPSPs (measured)")

    _plot_soma_traces(axes[1], pack_e, cumulative_expected=True, ylim=soma_ylim)
    axes[1].set_title("Soma EPSPs (expected)")

    y_m = np.asarray(pack_m["soma_peak_list"], dtype=float)
    y_e = np.asarray(pack_e["soma_peak_list"], dtype=float)
    if len(y_e) >= 2 * len(y_m) - 1:
        x_plot = y_e[::2][: len(y_m)]
    else:
        x_plot = y_e[: len(y_m)]

    axes[2].plot(x_plot, y_m, color="k", linestyle="-")
    min_val = float(min(x_plot.min(), y_m.min()))
    max_val = float(max(x_plot.max(), y_m.max()))
    axes[2].plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1)
    axes[2].set_title("Nonlinearity of somatic summation")
    axes[2].set_xlabel("Soma EPSP linear sum (mV)")
    axes[2].set_ylabel("Measured Soma EPSP (mV)")
    axes[2].set_xticks([0, 3, 6])
    axes[2].set_yticks([0, 3, 6])
    axes[2].spines["top"].set_visible(False)
    axes[2].spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


def main():
    p = argparse.ArgumentParser(description="Fig1 dend + soma PDFs (panels C/D style).")
    p.add_argument(
        "--sim_root",
        default="/G/results/simulation_singclus_supple_Apr26",
        help="Directory containing per-condition subfolders.",
    )
    p.add_argument(
        "--rel_path_template",
        default="{region}_range{range_idx}_clus_invitro_singclus_fixedW0.0004/1/{epoch}",
        help="Measured path under sim_root; placeholders: {region}, {range_idx}, {epoch}.",
    )
    p.add_argument(
        "--rel_path_template_expected",
        default="{region}_range{range_idx}_clus_invitro_singclus_fixedW0.0004_expected/1/{epoch}",
        help="Expected (linear sum) path under sim_root; same placeholders.",
    )
    p.add_argument("--region", default="basal")
    p.add_argument("--range_idx", type=int, default=1)
    p.add_argument("--epochs", default="8", help="e.g. 21-30 or 21,22,23")
    p.add_argument("--trial_idx", type=int, default=0)
    p.add_argument("--t_start", type=float, default=480)
    p.add_argument("--t_end", type=float, default=600)
    p.add_argument(
        "--panel_layout",
        default="vertical",
        choices=("vertical", "horizontal"),
        help="vertical: 3x1 subplots; horizontal: 1x3 subplots.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=str(PROJECT_ROOT / "results" / "fig1"),
        help="Directory for PDF outputs.",
    )
    p.add_argument(
        "--out_dend_stem",
        type=str,
        default="fig1_dend_{region}_r{range_idx}_e{epoch}",
        help="Filename stem (no extension); supports placeholders.",
    )
    p.add_argument(
        "--out_soma_stem",
        type=str,
        default="fig1_soma_{region}_r{range_idx}_e{epoch}",
        help="Filename stem (no extension); supports placeholders.",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sim_root = Path(args.sim_root)
    epochs = _parse_epoch_range(args.epochs)

    fmt_kw = {"region": args.region, "range_idx": args.range_idx}

    for epoch in epochs:
        rel_m = args.rel_path_template.format(epoch=epoch, **fmt_kw)
        rel_e = args.rel_path_template_expected.format(epoch=epoch, **fmt_kw)
        exp_m = str(sim_root / rel_m)
        exp_e = str(sim_root / rel_e)

        if not Path(exp_m).is_dir():
            print(f"Skip (missing measured): {exp_m}")
            continue
        if not Path(exp_e).is_dir():
            print(f"Skip (missing expected): {exp_e}")
            continue

        stem_kw = {**fmt_kw, "epoch": epoch}
        dend_path = out_dir / f"{args.out_dend_stem.format(**stem_kw)}.pdf"
        soma_path = out_dir / f"{args.out_soma_stem.format(**stem_kw)}.pdf"

        fig_d = plot_dend_figure(
            exp_m,
            trial_idx=args.trial_idx,
            t_start_ms=args.t_start,
            t_end_ms=args.t_end,
            panel_layout=args.panel_layout,
        )
        fig_d.savefig(dend_path, bbox_inches="tight")
        plt.close(fig_d)

        fig_s = plot_soma_figure(
            exp_m,
            exp_e,
            trial_idx=args.trial_idx,
            t_start_ms=args.t_start,
            t_end_ms=args.t_end,
            panel_layout=args.panel_layout,
        )
        fig_s.savefig(soma_path, bbox_inches="tight")
        plt.close(fig_s)

        print(f"Saved {dend_path}")
        print(f"Saved {soma_path}")


if __name__ == "__main__":
    main()
