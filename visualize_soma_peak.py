import os
import re
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------
# IO helpers
# ---------------------------
def _load_soma_folder(folder: str):
    soma_path = os.path.join(folder, "soma_v_array.npy")
    info_path = os.path.join(folder, "simulation_params.json")
    if (not os.path.exists(soma_path)) or (not os.path.exists(info_path)):
        return None, None
    soma = np.load(soma_path)
    with open(info_path, "r") as f:
        simu_info = json.load(f)
    return soma, simu_info


def _normalize_soma_shape(soma: np.ndarray):
    """
    Normalize soma to (T, A, Trials).
    """
    if soma.ndim == 3:
        return soma
    if soma.ndim == 4:
        return np.mean(soma, axis=1)
    if soma.ndim == 5:
        return np.mean(soma, axis=1)
    raise ValueError(f"Unsupported soma ndim={soma.ndim}, shape={soma.shape}")


def _samples_per_ms_from_dt(dt_seconds: float) -> int:
    return int(round(1e-3 / dt_seconds))


# ---------------------------
# Path logic you asked for
# ---------------------------
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
    where base_root is extracted from '*_clus_invivo_'.
    """
    base_root = _infer_base_root_from_clus_prefix(base_clus_invivo_prefix)
    condition = condition.strip()
    suffix = suffix.strip().lstrip("_")
    return f"{base_root}_{condition}_invivo_{suffix}"


def _find_epoch_folders_two_level(exp_dir: str):
    """
    Your layout:
      exp_dir/<replicate>/<epoch>/soma_v_array.npy

    We scan exp_dir/*/* and take basename(epoch_dir) as epoch int if numeric.
    Return list of (epoch_int, epoch_dir_path).
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



def extract_soma_peak_trials(
    folder: str,
    window_ms: tuple[float, float] = (-20.0, 100.0),
    dt_seconds: float = 1 / 40000,
    stim_time_key: str = "time point of stimulation",
):
    """
    peak per trial:
      max_t( soma[t, -1, trial] - soma[t, 0, trial] ) within window
    """
    soma_raw, simu_info = _load_soma_folder(folder)
    if soma_raw is None:
        return None

    soma = _normalize_soma_shape(soma_raw)  # (T, A, Trials)

    if stim_time_key not in simu_info:
        raise KeyError(f"simulation_params.json missing key: '{stim_time_key}'")
    t_ms = float(simu_info[stim_time_key])

    spm = _samples_per_ms_from_dt(dt_seconds)
    t_start = int(round((t_ms + window_ms[0]) * spm))
    t_end = int(round((t_ms + window_ms[1]) * spm))

    T = soma.shape[0]
    t_start = max(0, min(T, t_start))
    t_end = max(0, min(T, t_end))
    if t_end <= t_start:
        return None

    delta = soma[t_start:t_end, -1, :] - soma[t_start:t_end, 0, :]
    peaks = np.max(delta, axis=0)  # (Trials,)
    return peaks


def build_peak_dataframe_from_base(
    base_clus_invivo_prefix: str,
    suffixes: list[str],
    conditions: tuple[str, str] = ("clus", "distr"),
    window_ms: tuple[float, float] = (-20.0, 100.0),
    dt_seconds: float = 1 / 40000,
):
    """
    For each suffix:
      exp_dir = {base_root}_{cond}_invivo_{suffix}
    Then scan exp_dir/*/* for epochs.

    DataFrame columns:
      epoch, suffix, condition, peak, folder
    """
    rows = []
    for suffix in suffixes:
        for cond in conditions:
            exp_dir = _exp_dir_from_base(base_clus_invivo_prefix, cond, suffix)
            epoch_folders = _find_epoch_folders_two_level(exp_dir)
            if not epoch_folders:
                # allow missing condition/suffix combos quietly
                continue

            for epoch_idx, folder in epoch_folders:
                peaks = extract_soma_peak_trials(
                    folder=folder,
                    window_ms=window_ms,
                    dt_seconds=dt_seconds,
                )
                if peaks is None:
                    continue

                for p in np.asarray(peaks).ravel():
                    rows.append(
                        dict(
                            epoch=epoch_idx,
                            suffix=suffix,
                            condition=cond,
                            peak=float(p),
                            folder=folder,
                        )
                    )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            "No peaks extracted. Check paths exist:\n"
            "  {base_root}_{clus/distr}_invivo_{suffix}/<rep>/<epoch>/soma_v_array.npy"
        )
    return df


def plot_peak_violins(
    df,
    conditions=("clus", "distr"),
    figsize_per_suffix=(5.2, 4.2),
    show_points=False,
):
    """
    x-axis: categorical condition only: {clus, distr}
    y-axis: peak
    each suffix: one subplot (expand to the right)
    violin color: clus=red, distr=blue
    inside violin: boxplot
    remove median line from violin + remove error bar
    """
    suffix_list = sorted(df["suffix"].unique().tolist())
    ncol = len(suffix_list)

    fig_w = max(6.0, figsize_per_suffix[0] * ncol)
    fig_h = figsize_per_suffix[1]
    fig, axes = plt.subplots(1, ncol, figsize=(fig_w, fig_h), sharey=True)
    if ncol == 1:
        axes = [axes]

    color_map = {
        "clus": "tab:red",
        "distr": "tab:blue",
    }

    for ax, suffix in zip(axes, suffix_list):
        sdf = df[df["suffix"] == suffix]

        data_list = []
        for cond in conditions:
            vals = sdf[sdf["condition"] == cond]["peak"].to_numpy()
            vals = vals[np.isfinite(vals)]
            data_list.append(vals)

        positions = np.arange(1, len(conditions) + 1, dtype=float)

        # --- Violin (no median line, no extrema) ---
        parts = ax.violinplot(
            dataset=data_list,
            positions=positions,
            widths=0.85,
            showmeans=False,
            showmedians=False,   # 删除你说的“蓝色横线”
            showextrema=False,
        )
        for body, cond in zip(parts["bodies"], conditions):
            body.set_facecolor(color_map.get(cond, "gray"))
            body.set_edgecolor("none")
            body.set_alpha(0.35)

        # --- Boxplot inside violin (reference style) ---
        bp = ax.boxplot(
            data_list,
            positions=positions,
            widths=0.18,
            patch_artist=True,
            showfliers=False,
            whis=1.5,
        )
        # box style: transparent fill, black edge
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

        # optional raw points
        if show_points:
            for x0, vals, cond in zip(positions, data_list, conditions):
                if vals.size == 0:
                    continue
                jitter = (np.random.rand(vals.size) - 0.5) * 0.12
                ax.scatter(
                    np.full(vals.size, x0) + jitter,
                    vals,
                    s=6,
                    alpha=0.25,
                    color=color_map.get(cond, "gray"),
                )

        ax.set_xticks(positions)
        ax.set_xticklabels(list(conditions))
        ax.set_xlabel("Condition")
        ax.set_title(f"suffix: {suffix}")
        ax.grid(True, axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Soma peak (mV)  [max_t(soma[-1]-soma[0])]")
    fig.tight_layout()
    return fig, axes


def visualize_soma_peak_from_base(
    base_clus_invivo_prefix: str,
    suffixes: list[str] = ("spktimevar",),
    window_ms: tuple[float, float] = (-20.0, 100.0),
    dt_seconds: float = 1 / 40000,
    conditions: tuple[str, str] = ("clus", "distr"),
    save_path: str | None = None,
    show: bool = True,
):
    df = build_peak_dataframe_from_base(
        base_clus_invivo_prefix=base_clus_invivo_prefix,
        suffixes=list(suffixes),
        conditions=conditions,
        window_ms=window_ms,
        dt_seconds=dt_seconds,
    )
    fig, axes = plot_peak_violins(df, conditions=conditions)

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return df, fig, axes

if __name__ == "__main__":

    # ------- independent naming variables -------
    root_dir = "/G/results/simulation_singclus_supple_Feb26"
    anal_loc = "basal"          # e.g., basal / tuft / trunk ...
    range_idx = 1               # e.g., 0-2
    suffixes = ["spktimevar"]   # can extend: ["spktimevar", "bgtimevar", ...]

    window_ms = (-20, 100)

    # ------- assemble base prefix (ends with underscore) -------
    # Note: visualize_soma_peak_from_base expects "*_clus_invivo_" prefix to infer base_root
    # so here we intentionally build the clus-prefix.
    base_clus_invivo_prefix = (
        f"{root_dir}/{anal_loc}_range{range_idx}_clus_invivo_"
    )

    # ------- run for each suffix (will expand to the right if you pass multiple suffixes at once) -------
    # Option A (recommended): pass all suffixes together -> one figure with multiple columns
    df, fig, axes = visualize_soma_peak_from_base(
        base_clus_invivo_prefix=base_clus_invivo_prefix,
        suffixes=suffixes,
        window_ms=window_ms,
        save_path=f"soma_peak_violin_{anal_loc}_range{range_idx}.png",
    )

    


