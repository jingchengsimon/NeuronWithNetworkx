import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

def search_folders_by_json_values(root_folder, target_values):
    matching_folders = []

    # 遍历文件夹
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)

        # 检查文件夹下是否有json文件
        json_file_path = os.path.join(folder_path, 'simulation_params.json').replace('\\', '/')
        if os.path.exists(json_file_path) and os.path.isfile(json_file_path):
            # 读取JSON文件
            with open(json_file_path, 'r') as json_file:
                try:
                    data = json.load(json_file)

                    # 检查JSON中是否包含所有目标键值对
                    if all(key in data and data[key] == value for key, value in target_values.items()):
                        matching_folders.append(folder_name)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in {json_file_path}: {e}")

    return matching_folders

def plot_figure_in_folder(root_folder_path, folder_path, figure_name):
    figure_path = os.path.join(root_folder_path, folder_path, figure_name).replace('\\', '/')
    if os.path.exists(figure_path) and os.path.isfile(figure_path):
        # cv2 is the only way to add a title to the window (compared to plt and PIL)
        img = cv2.imread(figure_path) 
        cv2.imshow(folder_path, img) 
    else:
        print(f"Figure '{figure_name}' not found in folder '{folder_path}'")


# 示例用法
root_folder_path = './results/simulation/pseudo/'

target_values_1 = {
    'basal channel type': 'AMPANMDA',
    'distance from basal clusters to soma': 4,
    'number of clusters': 10,
    'number of synapses in each cluster': 10,
    'cluster radius': 5,
    'background synapse frequency': 1,
    'number of stimuli': 5}

target_values_2 = {
    'basal channel type': 'AMPANMDA',
    'distance from basal clusters to soma': 3,
    'number of clusters': 10,
    'number of synapses in each cluster': 20,
    'cluster radius': 5,
    'background synapse frequency': 1,
    'number of stimuli': 5}

target_values_3 = {
    'basal channel type': 'AMPANMDA',
    'distance from basal clusters to soma': 3,
    'number of clusters': 20,
    'number of synapses in each cluster': 10,
    'cluster radius': 5,
    'background synapse frequency': 1,
    'number of stimuli': 5}

# 多组键值对
target_values_list = [
    target_values_1,
    target_values_2,
    target_values_3,
]

for target_values in target_values_list:
    result_folders = search_folders_by_json_values(root_folder_path, target_values)

    print("Matching folders:")
    for folder_name in result_folders:
        print(folder_name)

        
    if result_folders:
        for folder_path in result_folders:
            # 指定要打开的图像文件名
            dend_v_npy_path = os.path.join(root_folder_path, folder_path, 'dend_v_array.npy').replace('\\', '/')
            dend_v_array = np.load(dend_v_npy_path)
            soma_v_npy_path = os.path.join(root_folder_path, folder_path, 'soma_v_array.npy').replace('\\', '/')
            soma_v_array = np.load(soma_v_npy_path)

            # dend_v_averaged = np.mean(dend_v_array, axis=3)[0,:,:]
            # soma_v_averaged = np.mean(soma_v_array, axis=2)
            # plt.figure()
            # for num_syn_per_cluster in range(0,dend_v_averaged.shape[1],2):
            #     # plt.plot(dend_v_averaged[:,num_syn_per_cluster], label=f"Number of synapses per cluster: {num_syn_per_cluster + 1}")
            #     plt.plot(soma_v_averaged[:,num_syn_per_cluster], linestyle='dashed', label=f"Soma, Number of synapses per cluster: {num_syn_per_cluster + 1}")
            # plt.xlabel("Time")
            # plt.ylabel("Voltage")
            # plt.legend()
            # plt.show()

            print("Shape of the array:", dend_v_array.shape)
            print("Shape of the array:", soma_v_array.shape)

            # dend_v_peak = np.mean(np.max(dend_v_array[:,20000:24000,:,:],axis=1),axis=2)
            # soma_v_peak = np.mean(np.max(soma_v_array[20000:24000,:,:],axis=0),axis=1)

            dend_v_peak = np.max(np.mean(dend_v_array, axis=3)[:,20000:24000,:], axis=1)
            soma_v_peak = np.max(np.mean(soma_v_array, axis=2)[20000:24000,:], axis=0)
            plt.figure()
            for cluster_index in range(dend_v_array.shape[0]):
                plt.plot(dend_v_peak[cluster_index, :], label=f"Cluster {cluster_index + 1}")
            plt.plot(soma_v_peak, linestyle='dashed', label="Soma")
            
            plt.xlabel("Number of synapses")
            plt.ylabel("Voltage")
            plt.legend()
            # plt.show()
            plt.savefig(os.path.join(root_folder_path, folder_path, 'figure_volatge_numOfSyn.png').replace('\\', '/'))

            # 对 num_trials 维度取平均
            # average_data = np.mean(dend_v_array, axis=3)

            # 取 num_syn_per_cluster 的第一个切片
            # first_syn_slice = average_data[:, :, 0]

            # # 绘制电压-时间曲线
            # plt.figure()

            # for cluster_index in range(dend_v_array.shape[0]):
            #     plt.plot(first_syn_slice[cluster_index], label=f"Cluster {cluster_index + 1}")

            # plt.xlabel("Time")
            # plt.ylabel("Voltage")
            # plt.title("Voltage-Time Curves for Different Clusters")
            # plt.legend()
            # plt.grid(True)
            # plt.show()

            # figure_name_to_open = 'figure_volatge_numOfSyn.png'  # 替换为您要打开的图像文件名
            # plot_figure_in_folder(root_folder_path, folder_path, figure_name_to_open)
    else:
        print("No matching folders found.")

cv2.waitKey(0)

