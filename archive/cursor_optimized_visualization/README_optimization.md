# 可视化性能优化指南

## 概述

本优化方案针对神经元网络可视化过程中的性能瓶颈进行了全面优化，主要包括数据加载、并行计算和内存管理三个方面的改进。

## 主要优化策略

### 1. 数据加载优化

#### 原始问题
- 串行加载多个.npy文件
- 重复加载相同数据
- 缺乏错误处理机制

#### 优化方案
- **并行文件加载**: 使用`ThreadPoolExecutor`同时加载多个文件
- **数据缓存**: 实现内存缓存机制，避免重复加载
- **错误处理**: 改进异常处理，提高代码健壮性

```python
# 优化前：串行加载
v = np.load(os.path.join(root_folder_path, exp, 'dend_v_array.npy'))
i = np.load(os.path.join(root_folder_path, exp, 'dend_i_array.npy'))
# ... 更多文件

# 优化后：并行加载
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {name: executor.submit(load_single_file, path) 
              for name, path in file_paths.items()}
```

### 2. 并行计算优化

#### 原始问题
- 串行处理多个epoch
- 计算密集型操作未并行化

#### 优化方案
- **epoch并行处理**: 使用`ProcessPoolExecutor`并行处理不同epoch
- **向量化计算**: 优化numpy操作，减少循环

```python
# 优化前：串行处理
for epoch_idx in range(num_epochs):
    result = process_epoch_data(epoch_path, epoch_idx, ...)

# 优化后：并行处理
with ProcessPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(process_epoch_data, epoch_path, epoch_idx, ...) 
              for epoch_idx, epoch_path in enumerate(epoch_paths)]
```

### 3. 内存管理优化

#### 原始问题
- 大量数据同时加载到内存
- 缺乏内存清理机制

#### 优化方案
- **选择性加载**: 只加载必要的数据
- **缓存管理**: 提供缓存清理功能
- **内存监控**: 优化内存使用模式

## 使用方法

### 基本使用

```python
from optimized_visualization import full_nonlinearity_visualization_optimized, clear_cache

# 设置参数
exp_list = ['your_experiment_name']
idx_list = [1]
rec_loc_list = ['dend']  # 'dend', 'soma', 'nexus'
attr_list = ['peak']     # 'peak', 'area'
num_epochs = 10

# 运行优化版本（并行）
avg_EPSP_list, EPSP_list_list = full_nonlinearity_visualization_optimized(
    exp_list, idx_list, rec_loc_list, attr_list, 
    num_epochs=num_epochs, use_parallel=True
)

# 清理缓存
clear_cache()
```

### 性能对比

```python
from performance_comparison import performance_test, benchmark_different_epochs

# 运行性能测试
performance_results = performance_test()

# 测试不同epoch数量的性能
epoch_results = benchmark_different_epochs()
```

### 参数说明

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `exp_list` | List[str] | 实验名称列表 | - |
| `idx_list` | List[int] | 实验索引列表 | - |
| `rec_loc_list` | List[str] | 记录位置列表 | - |
| `attr_list` | List[str] | 属性类型列表 | - |
| `num_epochs` | int | epoch数量 | 10 |
| `use_parallel` | bool | 是否使用并行处理 | True |

### 记录位置选项

- `'dend'`: 树突记录
- `'soma'`: 胞体记录  
- `'nexus'`: 轴突丘记录

### 属性类型选项

- `'peak'`: 峰值分析
- `'area'`: 面积分析

## 性能提升效果

### 预期加速比

| 优化项目 | 加速比 | 说明 |
|----------|--------|------|
| 数据加载 | 2-4x | 并行文件加载 |
| 并行处理 | 2-8x | 取决于CPU核心数 |
| 总体加速 | 3-10x | 综合优化效果 |

### 实际测试结果

运行`performance_comparison.py`可以获得具体的性能测试结果：

```bash
python performance_comparison.py
```

## 高级配置

### 调整并行度

```python
# 在optimized_visualization.py中修改
max_workers = min(8, mp.cpu_count())  # 调整并行进程数
```

### 缓存配置

```python
# 启用/禁用缓存
load_data_optimized(exp, use_cache=True)  # 启用缓存
load_data_optimized(exp, use_cache=False) # 禁用缓存

# 手动清理缓存
clear_cache()
```

### 内存优化

```python
# 对于大数据集，可以分批处理
batch_size = 5
for i in range(0, num_epochs, batch_size):
    batch_epochs = min(batch_size, num_epochs - i)
    # 处理批次数据
    clear_cache()  # 清理缓存
```

## 故障排除

### 常见问题

1. **内存不足**
   ```python
   # 解决方案：减少并行度或分批处理
   max_workers = min(4, mp.cpu_count())  # 减少并行进程数
   ```

2. **文件路径错误**
   ```python
   # 检查数据路径
   root_folder_path = '/G/results/simulation/'
   # 确保路径存在且包含必要的数据文件
   ```

3. **并行处理失败**
   ```python
   # 回退到串行处理
   use_parallel = False
   ```

### 调试模式

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查数据加载过程
def debug_load_data(exp):
    print(f"Loading data for: {exp}")
    # ... 加载过程
```

## 最佳实践

### 1. 数据组织
- 确保数据文件结构一致
- 使用相对路径避免硬编码
- 定期清理临时文件

### 2. 性能监控
- 监控内存使用情况
- 记录执行时间
- 定期运行性能测试

### 3. 代码维护
- 定期更新依赖包
- 保持代码版本控制
- 文档化配置参数

## 扩展功能

### 自定义数据处理

```python
def custom_data_processor(data):
    """自定义数据处理函数"""
    # 添加你的数据处理逻辑
    return processed_data

# 在优化函数中使用
def process_epoch_data_custom(exp_path, epoch_idx, ...):
    # 使用自定义处理器
    data = custom_data_processor(raw_data)
    return data
```

### 批量处理

```python
def batch_process_experiments(exp_list, batch_size=5):
    """批量处理多个实验"""
    results = []
    for i in range(0, len(exp_list), batch_size):
        batch = exp_list[i:i+batch_size]
        batch_results = process_batch(batch)
        results.extend(batch_results)
        clear_cache()  # 清理缓存
    return results
```

## 联系支持

如果在使用过程中遇到问题，请：

1. 检查错误日志
2. 确认数据文件完整性
3. 验证参数设置
4. 运行性能测试脚本

---

**注意**: 本优化方案针对Linux系统进行了优化，在其他操作系统上可能需要调整并行处理参数。 