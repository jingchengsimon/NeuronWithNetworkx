import pandas as pd
import numpy as np

rnd = np.random.RandomState(10)
# 创建一个空的DataFrame，每一列对应list1, list2, list3
df = pd.DataFrame({'list1': [], 'list2': [], 'list3': []})

# 假设有多个循环来分别对list1, list2, list3进行append操作
for i in range(3):  # 假设循环3次
    # 在每次循环中，假设有不同的数据要添加到list1, list2, list3中
    data_to_append_list1 = rnd.choice(['A', 'B', 'C'])
    data_to_append_list2 = rnd.choice([1, 2, 3])
    data_to_append_list3 = rnd.choice([True, False, True])
    
    # 创建一个字典，键是DataFrame的列名，值是要添加的数据
    data_to_append = {'list1': data_to_append_list1, 'list2': data_to_append_list2, 'list3': data_to_append_list3}
    
    # 将数据添加到DataFrame中
    df = df.append(data_to_append, ignore_index=True)

# 打印结果
print(df)
