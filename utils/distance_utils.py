# import numpy as np

def distance_synapse_mark_compare(dis_syn_from_ctr, dis_mark_from_ctr):
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


# def calculate_distance_matrix(self, distance_limit=2000):
#     loc_array, section_id_synapse_list = self.loc_array, self.section_id_synapse_list
#     parentID_list, length_list = self.section_df['parent_id'].values, self.section_df['length'].values
    
#     length_list = np.array(length_list)
#     distance_matrix = np.zeros((self.num_syn, self.num_syn))
#     for i in range(self.num_syn):
#         for j in range(i + 1, self.num_syn):
#                 m = section_id_synapse_list[i]
#                 n = section_id_synapse_list[j]

#                 path = self.sp[m][n]

#                 if len(path) > 1:
#                     loc_i = loc_array[i] * (parentID_list[m] == path[1]) + (1-loc_array[i]) * (parentID_list[m] != path[1])
#                     loc_j = loc_array[j] * (parentID_list[n] == path[-2]) + (1-loc_array[j]) * (parentID_list[n] != path[-2])
                    
#                     mask_i = np.array(path) == m
#                     mask_j = np.array(path) == n

#                     distance = np.sum(length_list[path] * (mask_i * loc_i + mask_j * loc_j))
#                     # for k in path:
#                     #     if k == m:
#                     #         distance = length_list[k]*loc_i
#                     #     if k == n:
#                     #         distance = length_list[k]*loc_j
#                     #     distance = distance + length_list[k]
#                 else:
#                     distance = length_list[m] * abs(loc_array[i] - loc_array[j])

#                 distance_matrix[i, j] = distance_matrix[j, i] = distance

#     self.distance_matrix = distance_matrix

    # return distance_matrix