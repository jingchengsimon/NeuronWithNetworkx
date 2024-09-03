import os
import json

def search_folders_by_json_value(root_folder, target_value):
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

                    # 检查JSON中特定项的值是否匹配
                    if 'target_key' in data and data['target_key'] == target_value:
                        matching_folders.append(folder_name)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in {json_file_path}: {e}")

    return matching_folders

# 示例用法
root_folder_path = '/path/to/your/folder'
target_value_to_match = 'desired_value'
result = search_folders_by_json_value(root_folder_path, target_value_to_match)

print("Matching folders:")
for folder_name in result:
    print(folder_name)
