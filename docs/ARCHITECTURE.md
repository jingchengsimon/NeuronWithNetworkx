# ARCHITECTURE.md — 模块依赖、数据流、模型架构

---

## 1  模块依赖图

```
L5b_simulation.py (CLI + 并行调度: num_epochs / max_workers_epoch / max_workers_synapse)
  ├── analysis/nmda_spike_detection.py  ← segment NMDA spike rate
  │
  └── utils/l5pn_model.py            ← L5PNModel, h.run(), 输出保存
        ├── utils/morphology_graph.py      ← 形态 → DiG, branch order
        ├── utils/cable_distance.py        ← cable distance (递归)
        ├── utils/synaptic_inputs.py       ← 突触创建 + spike train 注入
        │     ├── utils/synapse_models.py    ← AMPANMDA wrapper
        │     └── utils/pink_noise.py        ← 1/f noise (纯 numpy)
        ├── utils/cluster_protocol.py      ← preunit 映射 + presynaptic stimulus schedule
        └── utils/random_streams.py        ← seed fallback + named RNG policy
```

### 依赖规则

- `pink_noise.py`、`cluster_protocol.py` 和 `random_streams.py` 是 **纯计算模块**，不 import NEURON。
- `l5pn_model.py` 持有 `L5PNModel`、调用 `h.run()`、负责输出保存，并协调各核心模块。
- `synaptic_inputs.py` 负责创建/连接输入相关 NEURON 对象和 DataFrame 写入，但不调用 `h.run()`。
- `morphology_graph.py` 和 `cable_distance.py` 只操作形态信息，不接触突触。

---

## 2  数据流图

```
                         ┌─────────────────────────┐
                         │  cell1.asc (morphology)  │
                         └───────────┬─────────────┘
                                     │
                         ┌───────────▼─────────────┐
                         │     L5PNModel.__init__    │
                         │  • all_sections/segments  │
                         │  • DiG (directed graph)   │
                         │  • section_df             │
                         └───────────┬─────────────┘
                                     │
                         ┌───────────▼─────────────┐
                         │ initialize_synapse_layout() │
                         │  • length-weighted random │
                         │  → section_synapse_df     │
                         └───────────┬─────────────┘
                                     │
            ┌────────────────────────▼──────────────────────────┐
            │              assign_synapse_clusters()             │
            │  • distance thresholds → zone selection           │
            │  • generate_indices() → preunit-cluster mapping   │
            │  • exponential dist → cluster member recruitment  │
            │  → cluster_flag, cluster_id, pre_unit_id updated  │
            └────────────────────────┬──────────────────────────┘
                                     │
            ┌────────────────────────▼──────────────────────────┐
            │             run_stimulation_protocol()             │
            │                                                    │
            │  ┌──────────────────────────────────────────┐     │
            │  │ 4a. add_background_exc_inputs()          │     │
            │  │  • make_noise() → pink noise array       │     │
            │  │  • per-synapse: rectify → scale → Poisson│     │
            │  │  • 50% dropout → spike_train_bg          │     │
            │  │  • create AMPANMDA/Exp2Syn + VecStim     │     │
            │  │  • log-normal weights → syn_w            │     │
            │  └──────────────┬───────────────────────────┘     │
            │                 │                                  │
            │  ┌──────────────▼───────────────────────────┐     │
            │  │ for each (num_activated, num_stim, trial):│     │
            │  │                                           │     │
            │  │  4b. add_clustered_inputs()               │     │
            │  │   • spt_unit_array[perm[:k]] → VecStim   │     │
            │  │   • merge bg + stim → unique spike times  │     │
            │  │                                           │     │
            │  │  4c. add_background_inh_inputs()          │     │
            │  │   • total_spikes = Σ(exc + clus spikes)   │     │
            │  │   • λ = FREQ_INH × total/mean(total)      │     │
            │  │   • Poisson → dropout → spike_train_inh   │     │
            │  │                                           │     │
            │  │  4d. _run_single_trial()                  │     │
            │  │   • h.run()                               │     │
            │  │   • record → arrays                       │     │
            │  └───────────────────────────────────────────┘     │
            └────────────────────────┬──────────────────────────┘
                                     │
            ┌────────────────────────▼──────────────────────────┐
            │                   Save Outputs                    │
            │  • *.npy (recording arrays)                       │
            │  • section_synapse_df.csv                          │
            │  • simulation_params.json                          │
            │  • preunit assignment.txt                          │
            │  • segment_nmda_spike_rate.npz (if globrec)        │
            └───────────────────────────────────────────────────┘
```

---

## 3  AMPANMDA 突触模型

### 3.1  模型文件

- **mod mechanism**: 自定义 AMPA + NMDA 双组分突触（mod 源码未纳入）
- **参数文件**: `model/AMPANMDA.json`
- **Python wrapper**: `utils/synapse_models.py` → `AMPANMDA(syn_params, loc, section, channel_type)`

### 3.2  关键属性

| 属性 | 说明 |
|------|------|
| `synapse.initW` | 突触权重 (µS)，创建时设置 |
| `synapse.i_AMPA` | AMPA 电流 (nA) |
| `synapse.i_NMDA` | NMDA 电流 (nA)；voltage-dependent Mg²⁺ block |
| `synapse.g_AMPA` | AMPA 电导 (µS) |
| `synapse.g_NMDA` | NMDA 电导 (µS) |
| `synapse.i` | 总电流 |

### 3.3  通道类型选择

| `channel_type` | 行为 |
|----------------|------|
| `'AMPANMDA'` | AMPA + NMDA 双组分，NMDA 有 Mg²⁺ block |
| `'AMPA'` | 仅 AMPA 组分（无 NMDA nonlinearity） |
| `'Exp2Syn'` | NEURON 内置双指数突触（实验性，较少使用） |

---

## 4  Biophysics 模型

### 4.1  Hoc 模板选择

| 文件 | `with_ap` | 内容 |
|------|-----------|------|
| `L5PCbiophys3.hoc` | False | passive + HCN 等，**无 Na/Ca 通道**（无 AP） |
| `L5PCbiophys3withNaCa.hoc` | True | 完整 active conductances（可产生 AP） |

### 4.2  形态

- `cell1.asc` / `cell1.swc`: Neurolucida 重建的 L5 锥体神经元
- `L5PCtemplate.hoc`: 定义 `L5PCtemplate` class，包含 `soma`, `basal`, `apical` SectionList

---

## 5  Cluster 分配算法详解

### 5.1  距离分区

```
sorted_distances = sort(all exc synapse distances in region)
thresholds = [sorted[3000-1], sorted[6000-1], max]  # basal
           = [sorted[2500-1], sorted[5000-1], max]  # tuft

zone[dis_to_root] = [thresholds[dis_to_root], thresholds[dis_to_root+1]]
```

### 5.2  Center 选取

1. 从 zone 内未分配突触中随机选 center（`clus_loc_rnd.choice`）。
2. 失败（`exceed_flag`）时重新选取。

### 5.3  Surround 招募

```
distances_from_center = |loc_center - loc_surround| × L  (同 section)
                       + cable distance              (跨 section)

marks = sort(Exponential(cluster_radius, max_size - 1))
selected = distance_synapse_mark_compare(distances, marks)
members = clus_loc_rnd.choice(selected, num_syn_per_clus - 1, replace=False)
```

### 5.4  跨 Section 扩展

- 沿 DiG successors / predecessors 迭代扩展
- 排除 soma (id=0) 和 apical nexus predecessor (id=121)
- `exceed_flag`: 当 successor 和 predecessor 都耗尽但突触数不足时触发重选

---

## 6  Preunit 激活与 Permutation

### 6.1  Preunit 数量

```python
num_preunit = num_syn_per_clus × ceil(num_clusters / 3)
```

### 6.2  Preunit → Cluster 映射 (`generate_indices`)

Round-robin 策略：每个 preunit 连接 `num_conn_per_preunit` 个 cluster，优先选择连接数最少的 cluster。

### 6.3  Permutation 特殊约束

```python
perm = clus_spk_rnd.permutation(num_preunit)
# 强制第一个 cluster 的 center synapse 对应的 pre_unit_id 排在 perm[0]
```

### 6.4  激活列表

```python
# 'expected' 模式 (iter_step=1): 每次只激活一个新 preunit
num_activated_preunit_list = [0, 1, 2, ..., num_preunit]

# 非 expected 模式 (iter_step=2): 步长为 2
num_activated_preunit_list = [0, 2, 4, ..., num_preunit]
```

---

## 7  Replay 系统架构

```
Reference run (normal mode)
  → section_synapse_df.csv  (含 spike_train_bg 列)
       │
       ▼
New run (--use_replay_bg)
  ├── populate_section_synapse_df_from_csv()  # 复原突触位置
  ├── replay_assign_cluster_metadata()        # 复原 cluster 分配
  ├── _add_background_exc_inputs_replay()     # 从 CSV 读取 exc bg spikes
  └── _add_background_inh_inputs_replay()     # 从 CSV 读取 inh bg spikes
```

- **匹配键**: `row_syn_key(section)` — 基于 section name + loc + type 的唯一标识
- **前提**: `syn_pos_seed` 必须与 reference run 一致
- **隔离**: cluster stimulus 不受 replay 影响（由 `clus_spike_gen_seed` 独立生成）

---

## 9  并行调度（`L5b_simulation.py`）

### 9.1  两层并发

```
run_combination(args)
  │
  ├─ for spat_cond in args.spat_cond:     ← 串行（clus → distr）
  │     combinations = flatten(
  │         epoch ∈ [start_epoch, start_epoch + num_epochs)
  │         × simu_cond × sec_type × dis_to_root
  │     )
  │     ProcessPoolExecutor(max_workers=max_workers_epoch)
  │       └─ build_cell(run_args)  × N
  │             └─ L5PNModel(..., max_workers_synapse=...)
  │                   ├─ initialize_synapse_layout() → ThreadPoolExecutor(max_workers_synapse)
  │                   └─ run_stimulation_protocol()  → synaptic_inputs (same thread limit)
  │                         └─ _run_single_trial()    ← 主线程 h.run()
  │
  └─ (无 epoch_mode / num_batches / 外部分批)
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--num_epochs` | 1 | 与 `--start_epoch` 共同定义 epoch 范围 |
| `--start_epoch` | 1 | 起始 epoch 编号 |
| `--max_workers_epoch` | 20 | 每个 `spat_cond` 内最大并行进程数 |
| `--max_workers_synapse` | 30 | 每个进程内突触/输入 prep 最大线程数 |

### 9.2  约束

- `clus` 与 `distr` **不可**在同一 ProcessPool 中交错——distr 读取 clus 输出的 `section_synapse_df.csv`。
- `list(executor.map(...))` 必须保留，确保子进程异常向上抛出。
- RAM 粗估：并发进程数 × ~8 GiB/进程（`with_global_rec` 时更高）。

---

## 10  录制点一览

### 10.1  Standard Recording

| 变量 | 位置 | NEURON ref |
|------|------|------------|
| `soma_v` | soma(0.5) | `_ref_v` |
| `apic_v` | apic[36](1) | `_ref_v` |
| `apic_ica` | apic[36](1) | `_ref_ica` |
| `trunk_v` | apic[3](0) | `_ref_v` |
| `basal_v` | apic[70](0.8) | `_ref_v` |
| `tuft_v` | apic[67](0.5) | `_ref_v` |
| `soma_i` | SEClamp @ soma(0.5) | `_ref_i` |

### 10.2  Per-cluster Recording

对每个 cluster (前 `num_clusters_sampled` 个)：
- `dend_v`: cluster center segment 膜电位
- `dend_i`: 该 cluster 所在 section(s) 上所有 exc synapse 的 `i` 之和
- `dend_nmda_i`, `dend_ampa_i`: NMDA/AMPA 分量之和
- `dend_nmda_g`, `dend_ampa_g`: NMDA/AMPA 电导之和

### 10.3  Background Current Recording

- section_id 71 上所有 exc synapses 的 `i_NMDA` / `i_AMPA` 平均值 → `basal_bg_i_*`
- section_id 152 上所有 exc synapses 的 `i_NMDA` / `i_AMPA` 平均值 → `tuft_bg_i_*`

### 10.4  Global Recording (`with_global_rec=True`)

- `seg_v_array`: 所有 non-axon segments 的膜电位
- `seg_ina_array`: 所有 non-axon segments 的 Na 电流
- `seg_inmda_array`: 每个 segment 上所有 exc synapses 的 NMDA 电流之和
