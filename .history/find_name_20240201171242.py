import os
import json

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

return matching_folders

def plot_figure_in_folder(folder_path, figure_name):
    figure_path = os.path.join(folder_path, figure_name)
    if os.path.exists(figure_path) and os.path.isfile(figure_path):
        plt.figure()
        plt.imshow(plt.imread(figure_path))
        plt.show()
    else:
        print(f"Figure '{figure_name}' not found in folder '{folder_path}'")

# 示例用法
root_folder_path = '/path/to/your/folder'
target_values_to_match = {
    'key1': 'value1',
    'key2': 'value2',
    # 添加更多的键值对...
}
result_folders = search_folders_by_json_values(root_folder_path, target_values_to_match)

if result_folders:
    for folder_path in result_folders:
        # 指定要打开的图像文件名
        figure_name_to_open = 'example_figure.png'  # 替换为您要打开的图像文件名
        plot_figure_in_folder(folder_path, figure_name_to_open)
else:
    print("No matching folders found.")




