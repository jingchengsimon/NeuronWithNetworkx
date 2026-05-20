# AGENT.md — L5 Pyramidal Neuron Dendritic Clustering Simulation

> 本文件是 AI agent（Codex / Claude Code）操作本代码库的 **主规范**。  
> 详细参考：`docs/CONVENTIONS.md`（常量表 & schema）、`docs/ARCHITECTURE.md`（模块依赖 & 数据流）。  
> Cursor 规则：`.cursor/rules/naming.mdc`、`.cursor/rules/architecture.mdc`、`.cursor/rules/workflow.mdc`。

---

## 1  项目一句话描述

NEURON-based compartmental simulation of a biophysically detailed **Layer 5 pyramidal neuron (L5PN)**，研究 **dendritic synaptic clustering** 对 somatic / dendritic **nonlinear integration** 的影响。形态学来自重建数据 `cell1.asc`，biophysics 经 hoc 模板加载，excitatory synapses 使用自定义 AMPA+NMDA mod mechanism。

---

## 2  代码库结构

```
.
├── L5b_simulation.py          # 主入口：CLI + 并行调度
├── modelFile/                 # 形态 (cell1.asc/swc)、hoc 模板、AMPANMDA.json
├── mod/                       # NEURON mechanism（编译后 .so/.dll，源码未纳入）
├── utils/
│   ├── cell_with_networkx.py  # CellWithNetworkx 类 + h.run() + 输出保存
│   ├── add_inputs_utils.py    # 背景 / 聚类输入的突触创建与 spike train 注入
│   ├── generate_pink_noise.py # 1/f pink noise 生成器
│   ├── generate_stim_utils.py # preunit→cluster 映射 + VecStim 刺激生成
│   ├── generate_init_firing_utils.py  # (segment-level pink noise, 实验性)
│   ├── synapses_models.py     # AMPANMDA wrapper
│   ├── graph_utils.py         # 形态 → networkx DiG + branch order
│   ├── distance_utils.py      # cable distance 递归计算
│   ├── replay_background_spikes.py   # replay 模式: CSV→spike map
│   ├── replay_layout_from_csv.py     # replay 模式: CSV→突触布局
│   ├── nmda_detection_utils.py       # segment NMDA spike rate 检测
│   └── visualize_utils.py     # 可视化辅助
├── utils_anal/                # 分析脚本（不在仿真路径中）
├── utils_viz/                 # 可视化脚本
└── docs/                      # 约束文档（本目录）
```

---

## 3  核心工作流（调用顺序不可更改）

```
1. CellWithNetworkx.__init__()          # 加载形态 + graph
2.   .add_synapses()                    # 突触按 length-weighted 分布
3.   .assign_clustered_synapses()       # cluster 分配（distance range 内）
4.   .add_inputs()                      # 以下子步骤 ↓
   4a. add_background_exc_inputs()      # 兴奋性背景 (pink noise Poisson)
   4b. [loop] add_clustered_inputs()    # cluster stimulus (VecStim)
   4c. [loop] add_background_inh_inputs() # 抑制性背景 (tracking exc)
   4d. [loop] run_simulation()          # h.run() + 录制
5. 保存 npy / csv / json / npz
```

> **红线**：4a → 4b → 4c 的顺序 **不可调换**——inhibitory tracking 依赖 exc + clus 的 spike 统计。

---

## 4  随机种子三分体（最易出 bug 的部分）

| 种子 | 维度 | 控制对象 | RNG 类型 |
|------|------|----------|----------|
| `syn_pos_seed` | 空间 | 突触位置、cluster 中心、syn_w (log-normal) | `default_rng` + `random.seed` |
| `bg_spike_gen_seed` | 背景时间 | pink noise, Poisson spike, inh tracking | per-synapse `default_rng` via hash |
| `clus_spike_gen_seed` | 刺激时间 | `generate_vecstim` jitter, `perm` 排列 | `RandomState` |

**规则**：
- 三种种子 **严格隔离**，不可混用。
- `assign_clustered_synapses` 中 `clus_loc_rnd = RandomState(syn_pos_seed)`——必须是 `RandomState` 而非 `default_rng`。
- 多线程背景输入使用 `generate_synapse_seed(base_seed, i)` 保证确定性——不可回退到共享 RNG。
- 默认 fallback: 三种种子均回退到 `epoch`。

---

## 5  绝对禁止的操作 (Hard Stops)

| # | 禁止操作 |
|---|----------|
| H1 | 修改三种种子的用途或影响范围 |
| H2 | 改变 `section_synapse_df` 的列名 / 类型 / 语义 |
| H3 | 在 `h.run()` 之前删除 NEURON 对象的 Python 引用（VecStim, Vector, NetCon 等） |
| H4 | 将 `ThreadPoolExecutor` 用于包含 `h.run()` 的代码路径 |
| H5 | 修改 output 数组维度顺序 `(time, stim, aff, trial)` |
| H6 | 移除 inhibitory tracking 对 exc spike 的依赖 |
| H7 | 将 `syn_w` 字段单位从 nS 改为其他 |
| H8 | 修改 `folder_path` 命名模式（除非同步更新全部下游分析脚本） |
| H9 | 移除 `exceed_flag` 的 cluster 重选逻辑 |
| H10 | 在 replay 模式下生成新的 background spikes |

---

## 6  单位约定

| 量 | 代码变量 | 存储单位 | 换算 |
|----|----------|----------|------|
| 突触权重 | `initW`, `fixedW` (参数) | µS | — |
| 突触权重 | `section_synapse_df['syn_w']` | nS | × 1000 |
| 频率 | `FREQ_EXC`, `FREQ_INH` | Hz (/s) | 代码内 `/1000` 转 /ms |
| 距离 | `distance_to_soma`, `cluster_radius` | µm | — |
| 时间 | `SIMU_DURATION`, `stim_time` | ms | — |
| 电位 | 数组 / `h.v_init` | mV | — |
| 电流 | `i_NMDA`, `i_AMPA` | nA | NEURON 默认 |

---

## 7  推荐扩展模式

### 添加新录制变量
1. `__init__()` 声明 `self.xxx_array = None`
2. `add_inputs()` 分配空间（`common_shape` 或 `dend_shape`）
3. `run_simulation()` 创建 `h.Vector().record(...)` + 仿真后写入数组
4. `arrays_to_save` 字典注册

### 添加新 CLI 参数
1. `create_parser()` → `add_argument()`
2. `build_cell()` 提取 → 传递
3. `simulation_params` 字典注册

### 添加新突触类型
1. `section_synapse_df['type']` 新增标识
2. `add_single_synapse()` 添加分支
3. `add_inputs()` 添加处理逻辑

---

## 8  环境注意

- **不要 `pip install neuron`**：NEURON 需系统级安装 + mod 编译。
- **mod 文件未纳入版本控制**：agent 无法运行仿真，可做代码分析 / 重构 / mock 测试。
- Linux: `./mod/x86_64/.libs/libnrnmech.so`；Windows: `./mod/nrnmech.dll`。
- `h.celsius = 37`，`h.v_init = e_pas ≈ -90 mV`。
