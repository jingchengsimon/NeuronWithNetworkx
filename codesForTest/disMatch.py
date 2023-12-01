list_A = [1, 3, 5, 7]
list_B = [2, 8, 4, 10, 7]

# 创建一个包含原始索引的列表
original_indices = list(range(len(list_B)))

indices = []

for value in list_A:
    # 计算与value差值最小的元素的索引
    min_index = min(original_indices, key=lambda i: abs(list_B[i] - value))
    
    # 将该索引加入结果列表，并从original_indices中移除
    indices.append(min_index)
    original_indices.remove(min_index)

print("Original Indices in B with minimum differences for each element in A:", indices)
