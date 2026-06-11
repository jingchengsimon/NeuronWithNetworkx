import argparse
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

ANAL_LOCS: tuple[str, str] = ("basal", "apical")
METRICS: tuple[str, str] = ("peak", "area")
CONDITIONS: tuple[str, str] = ("clus", "distr")
VAR_SUFFIXES: tuple[str, ...] = (
    "bgtimevar_cspk60",
    "bgtimevar_cspk61",
    "bgtimevar_cspk62",
    "spktimevar",
    "bgposvar",
    "clusposvar",
)


def _load_trace_folder(folder: str, anal_loc: str):
    """
    anal_loc:
      - 'basal'  -> soma_v_array.npy
      - 'apical' -> apic_v_array.npy
    """
    trace_name = "apic_v_array.npy" if anal_loc == "apical" else "soma_v_array.npy"
    trace_path = os.path.join(folder, trace_name)
    info_path = os.path.join(folder, "simulation_params.json")
    if (not os.path.exists(trace_path)) or (not os.path.exists(info_path)):
        return None, None, trace_name

    trace = np.load(trace_path)
    with open(info_path, "r") as f:
        simu_info = json.load(f)
    return trace, simu_info, trace_name


def _normalize_trace_shape(trace: np.ndarray) -> np.ndarray:
    """
    Normalize trace to (T, A, Trials).
    """
    if trace.ndim == 3:
        return trace
    if trace.ndim == 4:
        return np.mean(trace, axis=1)
    if trace.ndim == 5:
        return np.mean(trace, axis=1)
    raise ValueError(f"Unsupported trace ndim={trace.ndim}, shape={trace.shape}")


def _samples_per_ms_from_dt(dt_seconds: float) -> int:
    return int(round(1e-3 / dt_seconds))


def _ylabel_for_anal_metric(anal_loc: str, metric: str) -> str:
    loc = (anal_loc or "").lower()
    m = (metric or "").lower()
    if m == "peak":
        if loc == "basal":
            return "Soma EPSP peak (mV)"
        if loc == "apical":
            return "Tuft EPSP peak (mV)"
    if m == "area":
        if loc == "basal":
            return "Soma EPSP area (mV·s)"
        if loc == "apical":
            return "Tuft EPSP area (mV·s)"
    return f"{anal_loc.capitalize()} {metric}"


def _infer_base_root_from_clus_prefix(base_clus_invivo_prefix: str) -> str:
    """
    base_clus_invivo_prefix example:
      /G/.../basal_range1_clus_invivo_
    We infer base_root:
      /G/.../basal_range1
    """
    s = base_clus_invivo_prefix.rstrip("_")
    m = re.match(r"^(.*)_clus_invivo$", s)
    if not m:
        raise ValueError(
            "base path must look like '*_clus_invivo_' (ending underscore allowed). "
            f"Got: {base_clus_invivo_prefix}"
        )
    return m.group(1)


def _exp_dir_from_base(base_clus_invivo_prefix: str, condition: str, suffix: str) -> str:
    """
    Build experiment dir:
      {base_root}_{condition}_invivo_{suffix}
    """
    base_root = _infer_base_root_from_clus_prefix(base_clus_invivo_prefix)
    condition = condition.strip()
    suffix = suffix.strip().lstrip("_")
    return f"{base_root}_{condition}_invivo_{suffix}"


def _exp_dir_for_anal_loc(
    base_clus_invivo_prefix: str,
    anal_loc: str,
    condition: str,
    suffix: str,
) -> str:
    """
    Convert a base prefix to location-specific directory:
      <root>/<anal_loc>_range<N>_<condition>_invivo_<suffix>
    """
    base_root = _infer_base_root_from_clus_prefix(base_clus_invivo_prefix)
    m = re.match(r"^(.*)/(basal|apical)_range(\d+)$", base_root)
    if m:
        root_parent = m.group(1)
        range_idx = m.group(3)
        suf = suffix.strip().lstrip("_")
        return f"{root_parent}/{anal_loc}_range{range_idx}_{condition}_invivo_{suf}"
    return _exp_dir_from_base(base_clus_invivo_prefix, condition, suffix)


def _find_epoch_folders_two_level(exp_dir: str):
    """
    Layout:
      exp_dir/<replicate>/<epoch>/soma_v_array.npy
    Returns list[(epoch_int, epoch_dir)].
    """
    out = []
    for epoch_dir in glob.glob(os.path.join(exp_dir, "*", "*")):
        if not os.path.isdir(epoch_dir):
            continue
        ep_name = os.path.basename(epoch_dir)
        if re.fullmatch(r"\d+", ep_name):
            out.append((int(ep_name), epoch_dir))
    out.sort(key=lambda x: x[0])
    return out


def _safe_int_list(raw):
    if raw is None:
        return None
    if isinstance(raw, np.ndarray):
        raw = raw.tolist()
    if not isinstance(raw, (list, tuple)):
        return None
    out: list[int] = []
    for x in raw:
        try:
            out.append(int(x))
        except (TypeError, ValueError):
            return None
    return out


def _resolve_syn_nums_from_simu_info(simu_info: dict, n_aff: int) -> list[int]:
    """
    Resolve aff-axis syn counts for one run.
    Prefer explicit aff_list in simulation_params for aff_mode=custom.
    """
    aff_mode = str(simu_info.get("aff_mode", "")).lower()
    aff_list = _safe_int_list(simu_info.get("aff_list"))
    if aff_mode == "custom" and aff_list:
        aff_list = list(dict.fromkeys([0] + aff_list))
        if len(aff_list) >= n_aff:
            return aff_list[:n_aff]
        return aff_list + list(range(len(aff_list), n_aff))

    n_syn_per_clus = int(simu_info.get("number of synapses per cluster", max(0, n_aff - 1)))
    if aff_mode == "full":
        if n_aff == 1:
            return [n_syn_per_clus]
        if n_aff == 2:
            return [0, n_syn_per_clus]
        return [0] + list(range(1, n_aff))

    if aff_mode == "linear":
        iter_step = int(simu_info.get("effective_iter_step", simu_info.get("iter_step", 1)))
        iter_step = max(1, iter_step)
        candidates = list(range(0, n_syn_per_clus + 1, iter_step))
        if len(candidates) >= n_aff:
            return candidates[:n_aff]

    return list(range(n_aff))


def extract_trial_metrics_by_syn(
    folder: str,
    *,
    window_ms: tuple[float, float] = (-20.0, 100.0),
    dt_seconds: float = 1 / 40000,
    stim_time_key: str = "time point of stimulation",
    anal_loc: str = "basal",
    metric: str = "peak",
) -> dict[int, np.ndarray] | None:
    """
    Returns:
      dict[syn_num] -> per-trial metric array (shape: [n_trials]).

    EPSP is computed against aff baseline index 0:
      delta[:, aff_idx, trial] = trace[:, aff_idx, trial] - trace[:, 0, trial]
    """
    trace_raw, simu_info, _ = _load_trace_folder(folder, anal_loc=anal_loc)
    if trace_raw is None:
        return None

    trace = _normalize_trace_shape(trace_raw)  # (T, A, Trials)
    if trace.shape[1] <= 0:
        return None

    if stim_time_key not in simu_info:
        raise KeyError(f"simulation_params.json missing key: '{stim_time_key}' in {folder}")
    t_ms = float(simu_info[stim_time_key])

    spm = _samples_per_ms_from_dt(dt_seconds)
    t_start = int(round((t_ms + window_ms[0]) * spm))
    t_end = int(round((t_ms + window_ms[1]) * spm))
    T = trace.shape[0]
    t_start = max(0, min(T, t_start))
    t_end = max(0, min(T, t_end))
    if t_end <= t_start:
        return None

    seg = trace[t_start:t_end, :, :]  # (time, aff, trial)
    delta = seg - seg[:, [0], :]  # baseline = aff index 0
    n_aff = delta.shape[1]
    syn_nums = _resolve_syn_nums_from_simu_info(simu_info, n_aff=n_aff)

    out: dict[int, np.ndarray] = {}
    if metric == "peak":
        values_by_aff = np.max(delta, axis=0)  # (aff, trials)
    elif metric == "area":
        n_samples = t_end - t_start
        x = np.arange(0, n_samples) * dt_seconds
        values_by_aff = np.trapz(np.clip(delta, 0, None), x, axis=0)  # (aff, trials)
    else:
        raise ValueError(f"metric must be 'peak' or 'area', got {metric!r}")

    for aff_idx in range(n_aff):
        syn_num = int(syn_nums[aff_idx]) if aff_idx < len(syn_nums) else aff_idx
        out[syn_num] = np.asarray(values_by_aff[aff_idx], dtype=float)
    return out


def build_peak_dataframe_from_base(
    *,
    base_clus_invivo_prefix: str,
    suffix: str,
    syn_nums: tuple[int, ...] | None = None,
    conditions: tuple[str, str] = CONDITIONS,
    window_ms: tuple[float, float] = (-20.0, 100.0),
    dt_seconds: float = 1 / 40000,
    anal_locs: tuple[str, str] = ANAL_LOCS,
    metrics: tuple[str, str] = METRICS,
) -> pd.DataFrame:
    rows: list[dict] = []
    syn_filter = set(syn_nums) if syn_nums is not None else None

    for anal_loc in anal_locs:
        for metric in metrics:
            for cond in conditions:
                exp_dir = _exp_dir_for_anal_loc(
                    base_clus_invivo_prefix,
                    anal_loc=anal_loc,
                    condition=cond,
                    suffix=suffix,
                )
                epoch_folders = _find_epoch_folders_two_level(exp_dir)
                if not epoch_folders:
                    continue

                for epoch_idx, folder in epoch_folders:
                    by_syn = extract_trial_metrics_by_syn(
                        folder,
                        window_ms=window_ms,
                        dt_seconds=dt_seconds,
                        anal_loc=anal_loc,
                        metric=metric,
                    )
                    if not by_syn:
                        continue

                    for syn_num, values in by_syn.items():
                        if syn_filter is not None and syn_num not in syn_filter:
                            continue
                        vals = np.asarray(values).ravel()
                        vals = vals[np.isfinite(vals)]
                        for p in vals:
                            rows.append(
                                {
                                    "epoch": int(epoch_idx),
                                    "suffix": suffix,
                                    "condition": cond,
                                    "peak": float(p),
                                    "folder": folder,
                                    "anal_loc": anal_loc,
                                    "metric": metric,
                                    "syn_num": int(syn_num),
                                }
                            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            f"No EPSP values extracted for suffix={suffix}. "
            "Check recordings exist and aff axis matches requested syn_nums."
        )
    return df


def _syn_num_offsets(n_syn: int) -> np.ndarray:
    if n_syn == 1:
        return np.array([0.0], dtype=float)
    return np.linspace(-0.28, 0.28, n_syn)


def _plot_single_syn_violin(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    conditions: tuple[str, str] = CONDITIONS,
) -> None:
    cond_centers = {"clus": 1.0, "distr": 2.0}
    color_map = {"clus": "tab:red", "distr": "tab:blue"}
    for cond in conditions:
        vals = df[df["condition"] == cond]["peak"].to_numpy()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        position = cond_centers[cond]
        parts = ax.violinplot(
            dataset=[vals],
            positions=[position],
            widths=0.55,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        body = parts["bodies"][0]
        body.set_facecolor(color_map[cond])
        body.set_edgecolor("none")
        body.set_alpha(0.35)

        bp = ax.boxplot(
            [vals],
            positions=[position],
            widths=0.12,
            patch_artist=True,
            showfliers=False,
            whis=1.5,
        )
        for box in bp["boxes"]:
            box.set_facecolor("none")
            box.set_edgecolor("black")
            box.set_linewidth(1.0)
        for med in bp["medians"]:
            med.set_color("black")
            med.set_linewidth(1.0)
        for whisk in bp["whiskers"]:
            whisk.set_color("black")
            whisk.set_linewidth(1.0)
        for cap in bp["caps"]:
            cap.set_color("black")
            cap.set_linewidth(1.0)

    ax.set_xticks([cond_centers[c] for c in conditions])
    ax.set_xticklabels(list(conditions))
    ax.set_xlabel("Condition")


def _plot_multi_syn_violin(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    conditions: tuple[str, str] = CONDITIONS,
) -> None:
    syn_nums = sorted(int(x) for x in df["syn_num"].dropna().unique())
    offsets = _syn_num_offsets(len(syn_nums))
    syn_to_offset = {syn_num: offsets[idx] for idx, syn_num in enumerate(syn_nums)}
    syn_to_alpha = {
        syn_num: 0.25 + 0.45 * (idx + 1) / len(syn_nums)
        for idx, syn_num in enumerate(syn_nums)
    }
    cond_centers = {"clus": 1.0, "distr": 2.0}
    color_map = {"clus": "tab:red", "distr": "tab:blue"}
    violin_width = 0.16 if len(syn_nums) >= 3 else 0.20

    for cond in conditions:
        for syn_num in syn_nums:
            vals = df[(df["condition"] == cond) & (df["syn_num"] == syn_num)]["peak"].to_numpy()
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            position = cond_centers[cond] + syn_to_offset[syn_num]
            parts = ax.violinplot(
                dataset=[vals],
                positions=[position],
                widths=violin_width,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            body = parts["bodies"][0]
            body.set_facecolor(color_map[cond])
            body.set_edgecolor("none")
            body.set_alpha(syn_to_alpha[syn_num])

            bp = ax.boxplot(
                [vals],
                positions=[position],
                widths=0.06,
                patch_artist=True,
                showfliers=False,
                whis=1.5,
            )
            for box in bp["boxes"]:
                box.set_facecolor("none")
                box.set_edgecolor("black")
                box.set_linewidth(1.0)
            for med in bp["medians"]:
                med.set_color("black")
                med.set_linewidth(1.0)
            for whisk in bp["whiskers"]:
                whisk.set_color("black")
                whisk.set_linewidth(1.0)
            for cap in bp["caps"]:
                cap.set_color("black")
                cap.set_linewidth(1.0)

    legend_handles = [
        Patch(
            facecolor="0.5",
            edgecolor="none",
            alpha=syn_to_alpha[syn_num],
            label=f"syn={syn_num}",
        )
        for syn_num in syn_nums
    ]
    ax.legend(handles=legend_handles, title="Syn num", frameon=False, loc="upper left")
    ax.set_xticks([cond_centers[c] for c in conditions])
    ax.set_xticklabels(list(conditions))
    ax.set_xlabel("Condition")


def _plot_multi_syn_line(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    conditions: tuple[str, str] = CONDITIONS,
) -> None:
    color_map = {"clus": "tab:red", "distr": "tab:blue"}
    syn_nums = sorted(int(x) for x in df["syn_num"].dropna().unique())
    for cond in conditions:
        medians: list[float] = []
        yerr_low: list[float] = []
        yerr_high: list[float] = []
        x_valid: list[int] = []
        for syn_num in syn_nums:
            vals = df[(df["condition"] == cond) & (df["syn_num"] == syn_num)]["peak"].to_numpy()
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            q25, q50, q75 = np.percentile(vals, [25, 50, 75])
            x_valid.append(syn_num)
            medians.append(float(q50))
            yerr_low.append(float(q50 - q25))
            yerr_high.append(float(q75 - q50))

        if not x_valid:
            continue
        ax.errorbar(
            x_valid,
            medians,
            yerr=np.vstack([yerr_low, yerr_high]),
            marker="o",
            markersize=4.0,
            linewidth=1.8,
            capsize=3.0,
            color=color_map[cond],
            label=cond,
            zorder=3,
        )
    ax.set_xlabel("Syn num")
    ax.set_xticks(syn_nums)
    ax.legend(frameon=False, loc="best")


def _plot_multi_syn_variance_line(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    conditions: tuple[str, str] = CONDITIONS,
) -> None:
    color_map = {"clus": "tab:red", "distr": "tab:blue"}
    syn_nums = sorted(int(x) for x in df["syn_num"].dropna().unique())
    for cond in conditions:
        variances: list[float] = []
        x_valid: list[int] = []
        for syn_num in syn_nums:
            sub = df[(df["condition"] == cond) & (df["syn_num"] == syn_num)]
            if sub.empty:
                continue
            epoch_values = (
                sub.groupby("epoch", sort=True)["peak"]
                .mean()
                .to_numpy(dtype=float)
            )
            epoch_values = epoch_values[np.isfinite(epoch_values)]
            if epoch_values.size == 0:
                continue
            ddof = 1 if epoch_values.size > 1 else 0
            x_valid.append(syn_num)
            variances.append(float(np.var(epoch_values, ddof=ddof)))

        if not x_valid:
            continue
        ax.plot(
            x_valid,
            variances,
            marker="o",
            markersize=4.0,
            linewidth=1.8,
            color=color_map[cond],
            label=cond,
            zorder=3,
        )
    ax.set_xlabel("Syn num")
    ax.set_xticks(syn_nums)
    ax.legend(frameon=False, loc="best")


def _style_axis(ax: plt.Axes, anal_loc: str, metric: str) -> None:
    ax.set_title(f"{anal_loc} | {metric}")
    ax.set_ylabel(_ylabel_for_anal_metric(anal_loc, metric))
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def build_meta_figure(
    df_suffix: pd.DataFrame,
    *,
    syn_nums: tuple[int, ...],
    plot_violin: bool,
    suffix: str,
) -> plt.Figure:
    include_variance = len(syn_nums) > 1 and not plot_violin
    if include_variance:
        fig, axes = plt.subplots(4, 2, figsize=(11.0, 13.0), sharex=False)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), sharex=False)

    for metric_idx, metric in enumerate(METRICS):
        for col_idx, anal_loc in enumerate(ANAL_LOCS):
            row_idx = metric_idx * 2 if include_variance else metric_idx
            ax = axes[row_idx, col_idx]
            sub = df_suffix[(df_suffix["metric"] == metric) & (df_suffix["anal_loc"] == anal_loc)]
            if sub.empty:
                ax.text(
                    0.5,
                    0.5,
                    "missing data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color="0.4",
                )
                _style_axis(ax, anal_loc, metric)
                continue

            if len(syn_nums) == 1:
                _plot_single_syn_violin(ax, sub)
            elif plot_violin:
                _plot_multi_syn_violin(ax, sub)
            else:
                _plot_multi_syn_line(ax, sub)
            _style_axis(ax, anal_loc, metric)

            if include_variance:
                var_ax = axes[row_idx + 1, col_idx]
                if sub.empty:
                    var_ax.text(
                        0.5,
                        0.5,
                        "missing data",
                        ha="center",
                        va="center",
                        transform=var_ax.transAxes,
                        color="0.4",
                    )
                else:
                    _plot_multi_syn_variance_line(var_ax, sub)
                _style_axis(var_ax, anal_loc, f"{metric} variance")
                var_ax.set_ylabel(f"Variance of {_ylabel_for_anal_metric(anal_loc, metric)}")

    plot_mode = "violin" if (len(syn_nums) == 1 or plot_violin) else "line+errorbar"
    if include_variance:
        plot_mode += "+variance"
    syn_text = ",".join(str(x) for x in syn_nums)
    fig.suptitle(
        f"{suffix} var meta | mode={plot_mode} | syn=[{syn_text}]",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def visualize_meta_figures_from_base(
    *,
    root_dir: str,
    range_idx: int,
    suffixes: tuple[str, ...],
    syn_nums: tuple[int, ...] | None,
    output_dir: str,
    fig_format: str,
    plot_violin: bool,
    window_ms: tuple[float, float],
    dt_seconds: float = 1 / 40000,
    show: bool = False,
) -> dict[str, pd.DataFrame]:
    os.makedirs(output_dir, exist_ok=True)
    results: dict[str, pd.DataFrame] = {}
    for suffix in suffixes:
        base_prefix = f"{root_dir.rstrip('/')}/basal_range{range_idx}_clus_invivo_"
        df_suffix = build_peak_dataframe_from_base(
            base_clus_invivo_prefix=base_prefix,
            suffix=suffix,
            syn_nums=syn_nums,
            conditions=CONDITIONS,
            window_ms=window_ms,
            dt_seconds=dt_seconds,
            anal_locs=ANAL_LOCS,
            metrics=METRICS,
        )
        if syn_nums is None:
            syn_nums_used = tuple(sorted(int(x) for x in df_suffix["syn_num"].dropna().unique()))
        else:
            syn_nums_used = tuple(sorted(dict.fromkeys(int(x) for x in syn_nums)))

        fig = build_meta_figure(
            df_suffix,
            syn_nums=syn_nums_used,
            plot_violin=plot_violin,
            suffix=suffix,
        )
        mode_tag = "violin" if (len(syn_nums_used) == 1 or plot_violin) else "line"
        syn_tag = "_".join(str(x) for x in syn_nums_used)
        out_name = f"meta_{suffix}_range{range_idx}_syn{syn_tag}_{mode_tag}_vivo.{fig_format}"
        save_path = os.path.join(output_dir, out_name)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved {save_path}  (n={len(df_suffix)} points)")
        if show:
            plt.show()
        else:
            plt.close(fig)
        results[suffix] = df_suffix
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Var meta EPSP figures from simulation folders. "
            "One figure per suffix; multi-syn line mode adds variance rows under the response rows."
        )
    )
    parser.add_argument(
        "--root_dir",
        default="/G/results/simulation_singclus_supple_May26",
        help="Directory containing <anal_loc>_range<N>_clus/distr_invivo_<suffix> trees.",
    )
    parser.add_argument(
        "--range_idx",
        type=int,
        default=1,
        help="Range index N in path names.",
    )
    parser.add_argument(
        "--suffixes",
        nargs="+",
        default=list(VAR_SUFFIXES),
        help="Experiment suffixes; each suffix produces one meta figure.",
    )
    parser.add_argument(
        "--syn_nums",
        type=int,
        nargs="+",
        default=None,
        help="One or more synapse counts, e.g. --syn_nums 72 or --syn_nums 0 12 24 36 48 60 72. "
             "If omitted, use all available syn numbers in data.",
    )
    parser.add_argument(
        "--plot_violin",
        action="store_true",
        help="When syn_nums has multiple values, use grouped violin instead of line+errorbar+variance.",
    )
    parser.add_argument(
        "--window_ms",
        nargs=2,
        type=float,
        default=[-20.0, 100.0],
        metavar=("START", "END"),
        help="Time window around stimulation (ms).",
    )
    parser.add_argument(
        "--output_dir",
        default="./results/violin_supple/var_meta",
        help="Directory for saved figures.",
    )
    parser.add_argument(
        "--fig_format",
        choices=["pdf", "png"],
        default="pdf",
        help="Figure file format.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Call plt.show() after saving.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.syn_nums is not None:
        syn_nums = tuple(sorted(dict.fromkeys(int(x) for x in args.syn_nums)))
    else:
        syn_nums = None

    window_ms = (float(args.window_ms[0]), float(args.window_ms[1]))
    visualize_meta_figures_from_base(
        root_dir=args.root_dir,
        range_idx=int(args.range_idx),
        suffixes=tuple(args.suffixes),
        syn_nums=syn_nums,
        output_dir=args.output_dir,
        fig_format=args.fig_format,
        plot_violin=bool(args.plot_violin),
        window_ms=window_ms,
        show=bool(args.show),
    )


if __name__ == "__main__":
    main()
