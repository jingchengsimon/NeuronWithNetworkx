# 命令行使用说明

## 基本用法

### 使用默认设置（with AP，即使用 L5PCbiophys3withNaCa.hoc）

```bash
python simpleModelVer2.py
```

或者显式指定：

```bash
python simpleModelVer2.py --with_ap
```

### 不使用 AP（使用 L5PCbiophys3.hoc）

```bash
python simpleModelVer2.py --no_ap
```

## 完整示例

### 示例 1：使用默认的 with AP 模型，自定义其他参数

```bash
python simpleModelVer2.py \
    --epoch 1 \
    --num_clusters 1 \
    --sec_type basal \
    --distance_to_root 1 \
    --simu_duration 1000 \
    --with_ap
```

### 示例 2：不使用 AP 模型

```bash
python simpleModelVer2.py \
    --epoch 1 \
    --num_clusters 1 \
    --sec_type basal \
    --distance_to_root 1 \
    --simu_duration 1000 \
    --no_ap
```

### 示例 3：组合多个参数

```bash
python simpleModelVer2.py \
    --epoch 5 \
    --num_clusters 3 \
    --sec_type apical \
    --distance_to_root 2 \
    --simu_duration 2000 \
    --bg_exc_freq 1.5 \
    --bg_inh_freq 5.0 \
    --no_ap
```

## 参数说明

- `--with_ap`: 使用 `L5PCbiophys3withNaCa.hoc`（包含动作电位和钙动力学）。这是默认选项。
- `--no_ap`: 使用 `L5PCbiophys3.hoc`（不包含动作电位和钙动力学）。如果指定此参数，会覆盖 `--with_ap`。

## 注意事项

1. **默认行为**：如果不指定任何参数，默认使用 `--with_ap`（即使用 `L5PCbiophys3withNaCa.hoc`）
2. **参数优先级**：如果同时指定 `--with_ap` 和 `--no_ap`，`--no_ap` 会生效（因为它在参数列表中后定义）
3. **查看所有参数**：使用 `python simpleModelVer2.py --help` 查看所有可用参数

## 查看帮助信息

```bash
python simpleModelVer2.py --help
```

这将显示所有可用的命令行参数及其说明。

