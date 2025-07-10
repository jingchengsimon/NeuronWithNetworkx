# 批量优化分析脚本
# 提供更高效的批量处理功能

from optimized_visualization import full_nonlinearity_visualization_optimized, clear_cache
import time
import concurrent.futures
from typing import List, Dict, Tuple
import numpy as np

class BatchAnalyzer:
    """批量分析器类，提供高效的批量处理功能"""
    
    def __init__(self, use_parallel: bool = True, max_workers: int = 4):
        self.use_parallel = use_parallel
        self.max_workers = max_workers
        self.results = {}
        self.timing = {}
    
    def analyze_single_condition(self, anal_loc: str, attr: str, rec_loc: str, 
                               range_idx: int, iter_start_idx: int, iter_end_idx: int, 
                               num_epochs: int) -> Tuple[str, Dict]:
        """分析单个条件"""
        condition_key = f'{anal_loc}_{attr}_{rec_loc}_{range_idx}'
        
        print(f"开始分析: {condition_key}")
        start_time = time.time()
        
        try:
            iter_times = iter_end_idx - iter_start_idx
            epsp_list, epsp_list_list = full_nonlinearity_visualization_optimized(
                [anal_loc+f'_range{range_idx}_clus_invitro_singclus'] * iter_times,
                list(range(iter_start_idx, iter_end_idx, 1)),
                [rec_loc] * iter_times, 
                [attr] * iter_times,
                num_epochs=num_epochs,
                use_parallel=self.use_parallel
            )
            
            analysis_time = time.time() - start_time
            print(f"完成分析: {condition_key} (耗时: {analysis_time:.2f}秒)")
            
            return condition_key, {
                'epsp_list': epsp_list,
                'epsp_list_list': epsp_list_list,
                'analysis_time': analysis_time,
                'status': 'success'
            }
            
        except Exception as e:
            analysis_time = time.time() - start_time
            print(f"分析失败: {condition_key} (耗时: {analysis_time:.2f}秒) - 错误: {e}")
            
            return condition_key, {
                'epsp_list': [],
                'epsp_list_list': [],
                'analysis_time': analysis_time,
                'status': 'failed',
                'error': str(e)
            }
    
    def batch_analyze(self, anal_loc: str, attrs: List[str], rec_locs: List[str], 
                     range_indices: List[int], iter_start_idx: int, iter_end_idx: int, 
                     num_epochs: int) -> Dict:
        """批量分析所有条件"""
        print(f"开始批量分析...")
        print(f"分析位置: {anal_loc}")
        print(f"属性类型: {attrs}")
        print(f"记录位置: {rec_locs}")
        print(f"范围索引: {range_indices}")
        print(f"Epoch数量: {num_epochs}")
        
        total_start_time = time.time()
        
        # 生成所有分析条件
        analysis_conditions = []
        for attr in attrs:
            for rec_loc in rec_locs:
                for range_idx in range_indices:
                    analysis_conditions.append((anal_loc, attr, rec_loc, range_idx))
        
        print(f"总共需要分析 {len(analysis_conditions)} 个条件")
        
        if self.use_parallel and len(analysis_conditions) > 1:
            # 并行处理
            print("使用并行处理模式...")
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                for anal_loc, attr, rec_loc, range_idx in analysis_conditions:
                    future = executor.submit(
                        self.analyze_single_condition,
                        anal_loc, attr, rec_loc, range_idx,
                        iter_start_idx, iter_end_idx, num_epochs
                    )
                    futures.append(future)
                
                # 收集结果
                for future in concurrent.futures.as_completed(futures):
                    try:
                        condition_key, result = future.result()
                        self.results[condition_key] = result
                        self.timing[condition_key] = result['analysis_time']
                    except Exception as e:
                        print(f"并行处理出错: {e}")
        else:
            # 串行处理
            print("使用串行处理模式...")
            for anal_loc, attr, rec_loc, range_idx in analysis_conditions:
                condition_key, result = self.analyze_single_condition(
                    anal_loc, attr, rec_loc, range_idx,
                    iter_start_idx, iter_end_idx, num_epochs
                )
                self.results[condition_key] = result
                self.timing[condition_key] = result['analysis_time']
                
                # 清理缓存
                clear_cache()
        
        total_time = time.time() - total_start_time
        self.total_time = total_time
        
        print(f"\n批量分析完成！总耗时: {total_time:.2f}秒")
        
        return self.results
    
    def get_summary(self) -> Dict:
        """获取分析摘要"""
        summary = {
            'total_conditions': len(self.results),
            'successful_conditions': sum(1 for r in self.results.values() if r['status'] == 'success'),
            'failed_conditions': sum(1 for r in self.results.values() if r['status'] == 'failed'),
            'total_time': self.total_time,
            'average_time_per_condition': np.mean(list(self.timing.values())) if self.timing else 0,
            'conditions': {}
        }
        
        for condition_key, result in self.results.items():
            if result['status'] == 'success':
                epsp_list = result['epsp_list']
                epsp_list_list = result['epsp_list_list']
                
                summary['conditions'][condition_key] = {
                    'status': 'success',
                    'epsp_list_length': len(epsp_list),
                    'epoch_count': len(epsp_list_list),
                    'max_epsp': max(epsp_list) if epsp_list else 0,
                    'mean_epsp': np.mean(epsp_list) if epsp_list else 0,
                    'analysis_time': result['analysis_time']
                }
            else:
                summary['conditions'][condition_key] = {
                    'status': 'failed',
                    'error': result.get('error', 'Unknown error'),
                    'analysis_time': result['analysis_time']
                }
        
        return summary
    
    def print_summary(self):
        """打印分析摘要"""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("批量分析摘要")
        print("="*60)
        print(f"总条件数: {summary['total_conditions']}")
        print(f"成功分析: {summary['successful_conditions']}")
        print(f"失败分析: {summary['failed_conditions']}")
        print(f"总耗时: {summary['total_time']:.2f}秒")
        print(f"平均每条件耗时: {summary['average_time_per_condition']:.2f}秒")
        
        print("\n详细结果:")
        for condition_key, condition_summary in summary['conditions'].items():
            print(f"\n{condition_key}:")
            if condition_summary['status'] == 'success':
                print(f"  EPSP列表长度: {condition_summary['epsp_list_length']}")
                print(f"  Epoch数量: {condition_summary['epoch_count']}")
                print(f"  最大EPSP值: {condition_summary['max_epsp']:.3f}")
                print(f"  平均EPSP值: {condition_summary['mean_epsp']:.3f}")
                print(f"  分析耗时: {condition_summary['analysis_time']:.2f}秒")
            else:
                print(f"  状态: 失败")
                print(f"  错误: {condition_summary['error']}")
                print(f"  耗时: {condition_summary['analysis_time']:.2f}秒")
    
    def save_results_to_globals(self):
        """将结果保存到全局变量（兼容原始代码）"""
        for condition_key, result in self.results.items():
            if result['status'] == 'success':
                # 解析条件键
                parts = condition_key.split('_')
                if len(parts) >= 4:
                    anal_loc, attr, rec_loc, range_idx = parts[0], parts[1], parts[2], parts[3]
                    
                    # 创建全局变量名
                    epsp_list_name = f'vitro_N+A_{anal_loc}_{attr}_{rec_loc}_{range_idx}_EPSP_list'
                    epsp_list_list_name = f'vitro_N+A_{anal_loc}_{attr}_{rec_loc}_{range_idx}_EPSP_list_list'
                    
                    # 保存到全局变量
                    globals()[epsp_list_name] = result['epsp_list']
                    globals()[epsp_list_list_name] = result['epsp_list_list']
        
        print("结果已保存到全局变量")

# 使用示例
def run_optimized_batch_analysis():
    """运行优化批量分析"""
    
    # 参数设置（与原始代码相同）
    anal_loc, range_idx, col_idx, num_epochs = 'basal', 0, 0, 10
    iter_start_idx, iter_end_idx = 1, 2
    
    # 创建批量分析器
    analyzer = BatchAnalyzer(use_parallel=True, max_workers=4)
    
    # 运行批量分析
    results = analyzer.batch_analyze(
        anal_loc=anal_loc,
        attrs=['peak', 'area'],
        rec_locs=['dend', 'soma'],
        range_indices=[0, 1, 2],
        iter_start_idx=iter_start_idx,
        iter_end_idx=iter_end_idx,
        num_epochs=num_epochs
    )
    
    # 打印摘要
    analyzer.print_summary()
    
    # 保存到全局变量（兼容原始代码）
    analyzer.save_results_to_globals()
    
    return analyzer

if __name__ == "__main__":
    # 运行优化批量分析
    analyzer = run_optimized_batch_analysis()
    
    print("\n分析完成！所有结果已保存到全局变量中。") 