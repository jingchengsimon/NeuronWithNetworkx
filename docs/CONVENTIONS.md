# CONVENTIONS.md — 缩写表、魔法常量、Schema 定义

---

## 1  缩写表

| 缩写 | 全称 | 上下文 |
|------|------|--------|
| `L5PN` | Layer 5 Pyramidal Neuron | 细胞类型 |
| `exc` | excitatory | 突触/输入类型 |
| `inh` | inhibitory | 突触/输入类型 |
| `bg` | background | 背景活动（非 cluster stimulus） |
| `syn` | synapse | — |
| `sec` | section | NEURON section |
| `seg` | segment | NEURON segment（section 的子区间） |
| `clus` | clustered | 空间条件 |
| `distr` | distributed | 空间条件（散布控制） |
| `ctr` | center | cluster 中心突触 |
| `suc` | successor | DiG 中的子 section |
| `pre` | predecessor | DiG 中的父 section |
| `aff` | afferent | 传入纤维 / activated preunit |
| `perm` | permutation | preunit 激活排列 |
| `DiG` | Directed Graph | networkx.DiGraph 形态树 |
| `spk` / `spt` | spike / spike time | — |
| `rnd` | random (RNG instance) | — |
| `loc` | location [0,1] | section 上的归一化位置 |
| `df` | DataFrame | — |
| `ap` | action potential | `with_ap` 标志 |
| `globrec` | global recording | `with_global_rec` 标志 |
| `invivo` / `invitro` | in vivo / in vitro | 仿真条件 |

---

## 2  魔法常量

### 2.1 突触数量（文献值）

| 常量 | 值 | 来源 / 含义 |
|------|-----|------------|
| `NUM_SYN_BASAL_EXC` | 10042 | basal dendrite 上的兴奋性突触数 |
| `NUM_SYN_APIC_EXC` | 16070 | apical dendrite 上的兴奋性突触数 |
| `NUM_SYN_BASAL_INH` | 1023 | basal 抑制性突触（E/I ≈ 10:1） |
| `NUM_SYN_APIC_INH` | 1637 | apical 抑制性突触 |
| `NUM_SYN_SOMA_INH` | 150 | soma 抑制性突触 |

### 2.2 形态特异索引

| 常量 | 值 | 含义 |
|------|-----|------|
| `sections_apical[36]` | — | apical nexus / tuft root (硬编码) |
| section_id `0` | — | soma |
| section_id `121` | — | apical nexus 在 all_sections 中的索引 (= 85+36) |
| section_id `71` | — | basal tip recording branch |
| section_id `152` | — | tuft tip recording branch |
| `apic[121-85]` = `apic[36]` | — | apical nexus 录制点 |

> ⚠️ 更换形态文件时，以上所有索引必须重新确定。

### 2.3 距离分区阈值

| 类型 | zone 划分依据 | 每 zone 突触数 |
|------|--------------|---------------|
| basal | `distance_to_soma` | 3000 per zone |
| apical (tuft) | `distance_to_tuft` | 2500 per zone |

`distance_to_root` 取 0/1/2 选择 zone。

### 2.4 突触参数

| 参数 | 值 | 单位 | 说明 |
|------|-----|------|------|
| `syn_param_exc` | `[0, 0.3, 1.8]` | `[mV, ms, ms]` | `[e_syn, tau1, tau2]` (Exp2Syn 时) |
| `syn_param_inh` | `[-86, 1, 8, 0.00069]` | `[mV, ms, ms, µS]` | `[e_syn, tau1, tau2, weight]` |
| `initW` default | 0.0004 | µS | E[W] of log-normal |
| log-normal σ | 1 | — | 固定不变 |
| log-normal μ | `log(initW) - 0.5` | — | 使 E[W] = initW |
| `netcon.weight[0]` | 1 | — | 开关值，非实际权重 |
| `inh_delay` | 4.0 | ms | feedforward inhibition delay |

### 2.5 Simulation 常量

| 参数 | 默认值 | 单位 |
|------|--------|------|
| `h.celsius` | 37 | °C |
| `h.v_init` | `e_pas ≈ -90` | mV |
| `dt` (implicit) | 0.025 | ms |
| `T` (time points) | `1 + 40 × SIMU_DURATION` | — |
| `stim_time` | 500 | ms |
| `stim_time_var` | 5 | ms |
| `cluster_radius` | 5.0 | µm |
| `num_syn_per_clus` | 72 | — |
| `num_conn_per_preunit` | 3 | — |
| `dropout_p` (exc bg) | 0.5 | — |
| `dropout_p` (inh) | 0.5 | — |

### 2.6 Pink Noise IIR 滤波器系数

```python
B = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
A = [1, -2.494956002, 2.017265875, -0.522189400]
burn_in = 2000  # samples discarded
```

### 2.7 Branch Index 硬编码（cluster 集中度控制）

```python
basal_branch_idx_list = [40, 41, 41]
apic_branch_idx_list  = [138, 138, 138]
```

> 当前代码中定义但未在 single-cluster 模式下使用（用于 multi-cluster 集中模式）。

---

## 3  `simulation_params` JSON Schema

`simulation_params.json` 在每个 epoch 输出目录中保存，包含以下 key：

```
cell model                          : str     ("L5PN")
NUM_SYN_BASAL_EXC                   : int
NUM_SYN_APIC_EXC                    : int
NUM_SYN_BASAL_INH                   : int
NUM_SYN_APIC_INH                    : int
NUM_SYN_SOMA_INH                    : int
SIMU DURATION                       : int     (ms)
STIM DURATION                       : int     (ms)
simulation condition                : str     ("invivo" | "invitro")
synaptic spatial condition          : str     ("clus" | "distr")
basal channel type                  : str     ("AMPANMDA" | "AMPA")
channel_suffix                      : str
section type                        : str     ("basal" | "apical")
distance from clusters to root      : int     (0 | 1 | 2)
number of clusters                  : int
cluster radius                      : float   (µm)
background excitatory frequency     : float   (Hz)
background inhibitory frequency     : float   (Hz)
input ratio of basal to apical      : float
background excitatory channel type  : str     ("AMPANMDA" | "AMPA")
initial weight of AMPANMDA synapses : float   (µS)
use_fixedW                          : bool
fixedW                              : float   (µS)
number of functional groups         : int
delay of inhibitory inputs          : float   (ms)
number of stimuli                   : int
time point of stimulation           : int     (ms)
number of connection per preunit    : int
number of synapses per cluster      : int
number of trials                    : int
syn_pos_seed                        : int
bg_spike_gen_seed                   : int
clus_spike_gen_seed                 : int
with_ap                             : bool
with_global_rec                     : bool
use_replay_bg                       : bool
replay_bg_csv                       : str | null
segment_nmda_spike_rate_npz         : str | null
```

---

## 4  `section_synapse_df` 完整列定义

| # | Column | dtype | Nullable | 说明 |
|---|--------|-------|----------|------|
| 0 | `section_id_synapse` | int | N | all_sections 索引 |
| 1 | `section_synapse` | NEURON Section | N | 突触所在 section（CSV 中为 str） |
| 2 | `segment_synapse` | NEURON Segment | N | 突触所在 segment（CSV 中为 str） |
| 3 | `loc` | float [0,1] | N | section 上的归一化位置 |
| 4 | `type` | str ('A'/'B') | N | A=exc, B=inh |
| 5 | `distance_to_soma` | float (µm) | N | 到 soma 的电缆距离 |
| 6 | `distance_to_tuft` | float (µm) | N | 到 tuft root 的距离；非 tuft 区为 -1 |
| 7 | `cluster_flag` | int (-1/1) | N | -1=background, 1=cluster member |
| 8 | `cluster_center_flag` | int (-1/0/1) | N | 1=center, 0=surround, -1=non-cluster |
| 9 | `cluster_id` | int | N | cluster 编号；-1=non-cluster |
| 10 | `pre_unit_id` | int | N | 前突触单元 ID；-1=unassigned |
| 11 | `region` | str | N | 'basal' / 'apical' / 'soma' |
| 12 | `branch_idx` | int/object | N | section_df 中的 branch_idx |
| 13 | `syn_w` | float (nS) | Y | 突触权重（nS = 1000 × µS） |
| 14 | `synapse` | NEURON object | Y | AMPA/NMDA 或 Exp2Syn；CSV 中 None |
| 15 | `netstim` | NEURON object | Y | h.VecStim()；CSV 中 None |
| 16 | `netcon` | NEURON object | Y | h.NetCon()；CSV 中 None |
| 17 | `spike_train` | list[list] | Y | clustered stimulus spike times |
| 18 | `spike_train_bg` | list[list] | Y | background spike times |

---

## 5  Output 数组维度约定

| 数组类别 | Shape | 说明 |
|---------|-------|------|
| Standard (soma_v, apic_v, trunk_v, etc.) | `(T, S, A, R)` | T=time, S=num_stim, A=num_aff, R=num_trials |
| Dendritic (dend_v, dend_nmda_i, etc.) | `(C, T, S, A, R)` | C=num_clusters_sampled |
| Global seg (seg_v, seg_ina, seg_inmda) | `(N, T, S, A, R)` | N=num_segments_noaxon |

`T = 1 + 40 × SIMU_DURATION`，默认 1000 ms → T = 40001。

---

## 6  CLI 参数 → 内部变量映射

| CLI arg | 内部变量 | 传递给 |
|---------|---------|--------|
| `--num_syn_basal_exc` | `NUM_SYN_BASAL_EXC` | `add_synapses()` |
| `--bg_exc_freq` | `bg_exc_freq` → `self.FREQ_EXC` | `add_background_exc_inputs()` |
| `--initW` | `initW` → `self.initW` | `_exc_init_w_distr_array()` |
| `--syn_pos_seed` | `syn_pos_seed` (default: epoch) | `self.rnd`, `random.seed()`, `clus_loc_rnd` |
| `--bg_spike_gen_seed` | `bg_spike_gen_seed` (default: epoch) | `generate_synapse_seed()`, `make_noise()` |
| `--clus_spike_gen_seed` | `clus_spike_gen_seed` (default: epoch) | `clus_spk_rnd` for vecstim & perm |
| `--with_ap` | `with_ap` | biophys hoc file selection |
| `--with_global_rec` | `with_global_rec` | seg_v/ina/inmda recording + NMDA spike rate |
| `--use_replay_bg` | `replay_bg_csv` | replay pipeline activation |
| `--use_fixedW` | `use_fixedW` | weight mode switch |
| `--fixedW` | `fixedW` | fixed weight value (µS) |
