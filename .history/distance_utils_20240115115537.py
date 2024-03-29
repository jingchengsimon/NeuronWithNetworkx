def _distance_synapse_mark_compare(self, dis_syn_from_ctr, dis_mark_from_ctr):
        # 创建一个包含原始索引的列表
        original_indices = list(range(len(dis_syn_from_ctr)))
        index = []

        for value in dis_mark_from_ctr:
            # 计算与value差值最小的元素的索引
            min_index = min(original_indices, key=lambda i: abs(dis_syn_from_ctr[i] - value)) 
            # 将该索引加入结果列表，并从original_indices中移除
            index.append(min_index)
            original_indices.remove(min_index)
        
        return index