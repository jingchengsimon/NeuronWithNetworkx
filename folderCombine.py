import os
import shutil

# 定义文件夹路径
folder1_path = '/G/results/simulation/basal_range0_clus_invitro'
folder2_path = '/G/results/simulation/basal_range0_clus_invitro1'

# 遍历 folder1 中的 1-42 主文件夹
for index in range(1, 43):
    folder1_subfolder = os.path.join(folder1_path, str(index))
    folder2_subfolder = os.path.join(folder2_path, str(index))
    
    if not os.path.exists(folder1_subfolder):
        print(f"Skipping {folder1_subfolder} as it does not exist.")
        continue
    if not os.path.exists(folder2_subfolder):
        print(f"Skipping {folder2_subfolder} as it does not exist.")
        continue

    used_folder2_indices = set()  # 记录已经使用过的 folder2 子文件夹编号

    # 遍历 folder1 的子文件夹
    for sub_index in range(1, 11):  # 包括可能需要的 8-10 号子文件夹
        folder1_child = os.path.join(folder1_subfolder, str(sub_index))
        soma_file_path = os.path.join(folder1_child, "soma_v_array.npy")

        # 如果 folder1 的子文件夹不存在，先创建
        if not os.path.exists(folder1_child):
            os.makedirs(folder1_child)
            print(f"Created directory: {folder1_child}")

        # 如果子文件夹中没有 soma_v_array.npy，则从 folder2 中复制内容
        if not os.path.exists(soma_file_path):
            for folder2_sub_index in range(1, 9):  # 遍历 folder2 的子文件夹
                if folder2_sub_index in used_folder2_indices:  # 跳过已使用的文件夹
                    continue

                folder2_child = os.path.join(folder2_subfolder, str(folder2_sub_index))

                # 如果 folder2 的子文件夹存在且有内容，进行复制
                if os.path.exists(folder2_child) and os.listdir(folder2_child):
                    # 清空 folder1 的当前子文件夹内容
                    for item in os.listdir(folder1_child):
                        item_path = os.path.join(folder1_child, item)
                        if os.path.isfile(item_path) or os.path.islink(item_path):
                            os.unlink(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)

                    # 将 folder2 的子文件夹内容复制到 folder1 的子文件夹
                    for item in os.listdir(folder2_child):
                        src_path = os.path.join(folder2_child, item)
                        dest_path = os.path.join(folder1_child, item)
                        if os.path.isfile(src_path):
                            shutil.copy2(src_path, dest_path)
                        elif os.path.isdir(src_path):
                            shutil.copytree(src_path, dest_path)
                    
                    # 记录已使用的 folder2 子文件夹编号
                    used_folder2_indices.add(folder2_sub_index)
                    print(f"Copied contents from {folder2_child} to {folder1_child}.")
                    break  # 复制完成后退出内层循环
            else:
                print(f"No more unused content in {folder2_subfolder} to copy for {folder1_child}.")
        else:
            print(f"Skipped {folder1_child} as soma_v_array.npy already exists.")
