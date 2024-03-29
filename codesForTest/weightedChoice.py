import random

list_A = ['A', 'B', 'C', 'D']
list_length = [0, 1, 2, 0]

# 使用random.choices进行有权重的随机抽样
sample_size = 9  # 你想要抽取的样本数量
weighted_sample = random.choices(list_A, weights=list_length, k=sample_size)

print("Weighted Random Sample:", weighted_sample)
