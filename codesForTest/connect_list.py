import random

n = 5  # 重复操作的次数
k = 5  # 范围

results = []  # 用于存储生成的列表
indices = []
for _ in range(n):
    sampled = random.sample(range(k), 3)  # 从范围中选择三个不同的整数
    results.append(sampled)

# 查找包含从0到k-1的列表的索引
for i in range(k):
    index_list = [j for j, lst in enumerate(results) if i in lst]
    indices.append(index_list)

print("生成的列表:")
for lst in results:
    print(lst)

print("包含从0到k-1的列表的索引:")
for lst in indices:
    print(lst)

