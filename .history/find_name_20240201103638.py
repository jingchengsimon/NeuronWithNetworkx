import os
import json

def search_folders_by_json_values(root_folder, target_values):
    matching_folders = []

    # 遍历文件夹
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)

        # 检查文件夹下是否有json文件
        json_file_path = os.path.join(folder_path, 'data.json')
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

# 示例用法
root_folder_path = '/path/to/your/folder'
target_values_to_match = {
    'key1': 'value1',
    'key2': 'value2',
    # 添加更多的键值对...
}
result = search_folders_by_json_values(root_folder_path, target_values_to_match)

print("Matching folders:")
for folder_name in result:
    print(folder_name)

root_folder_path = 'C:\Users\Windows\Desktop\MIMOlab\Codes\NeuronWithNetworkx\results\simulation\pseudo'
target_key = 
target_value = 'desired_value'
result = search_folders_by_json_value(root_folder_path, target_key, target_value)

print("Matching folders:")
for folder_name in result:
    print(folder_name)
