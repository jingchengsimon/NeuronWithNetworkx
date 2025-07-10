# 优化版本的可视化函数调用示例
# 基于原始代码的使用方式

from optimized_visualization import full_nonlinearity_visualization_optimized, clear_cache
import time

# 原始参数设置
anal_loc, range_idx, col_idx, num_epochs = 'basal', 0, 0, 10 # range_idx: 0-2, col_idx: 0-5
iter_start_idx, iter_end_idx = 1, 2
iter_step, iter_times = 1, iter_end_idx - iter_start_idx

print("开始优化版本的可视化分析...")
print(f"分析位置: {anal_loc}")
print(f"Epoch数量: {num_epochs}")
print(f"迭代范围: {iter_start_idx} - {iter_end_idx}")

# 记录总开始时间
total_start_time = time.time()

# 使用优化版本进行批量分析
for attr in ['peak', 'area']:
    for rec_loc in ['dend', 'soma']:
        for range_idx in range(3):
            print(f"\n处理: {anal_loc}_{attr}_{rec_loc}_{range_idx}")
            
            # 记录单个分析开始时间
            start_time = time.time()
            
            # 调用优化版本函数
            globals()[f'vitro_N+A_{anal_loc}_{attr}_{rec_loc}_{range_idx}_EPSP_list'], \
            globals()[f'vitro_N+A_{anal_loc}_{attr}_{rec_loc}_{range_idx}_EPSP_list_list'] = \
                full_nonlinearity_visualization_optimized(
                    [anal_loc+f'_range{range_idx}_clus_invitro_singclus'] * iter_times,
                    list(range(iter_start_idx, iter_end_idx, iter_step)),
                    [rec_loc] * iter_times, 
                    [attr] * iter_times,
                    num_epochs=num_epochs,
                    use_parallel=True  # 启用并行处理
                )
            
            # 记录单个分析完成时间
            single_time = time.time() - start_time
            print(f"完成时间: {single_time:.2f} 秒")
            
            # 清理缓存以释放内存
            clear_cache()

# 记录总完成时间
total_time = time.time() - total_start_time
print(f"\n总分析时间: {total_time:.2f} 秒")

# 打印结果摘要
print("\n=== 分析结果摘要 ===")
for attr in ['peak', 'area']:
    for rec_loc in ['dend', 'soma']:
        for range_idx in range(3):
            epsp_list = globals()[f'vitro_N+A_{anal_loc}_{attr}_{rec_loc}_{range_idx}_EPSP_list']
            epsp_list_list = globals()[f'vitro_N+A_{anal_loc}_{attr}_{rec_loc}_{range_idx}_EPSP_list_list']
            
            print(f"{anal_loc}_{attr}_{rec_loc}_{range_idx}:")
            print(f"  EPSP列表长度: {len(epsp_list)}")
            print(f"  Epoch数量: {len(epsp_list_list)}")
            if len(epsp_list) > 0:
                print(f"  最大EPSP值: {max(epsp_list):.3f}")
                print(f"  平均EPSP值: {sum(epsp_list)/len(epsp_list):.3f}")

print("\n分析完成！") 