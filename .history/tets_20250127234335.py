from neuron import h
import time
from simpleModelVer2 import build_cell, generate_simu_params

params = generate_simu_params('basal', 'clus', 0)
params_with_epoch = params[0].copy()
params_with_epoch['epoch'] = 1

# 测试 CPU 运行时间
start_time = time.time()
build_cell(**params_with_epoch)


# 测试 GPU 运行时间（使用 CoreNEURON）
try:
    from neuron import coreneuron

    # 启用 CoreNEURON
    coreneuron.enable = True
    coreneuron.gpu = True  # 启用 GPU 加速
    build_cell(**params_with_epoch)

except ImportError:
    print("未安装 CoreNEURON，无法测试 GPU 运行时间。")