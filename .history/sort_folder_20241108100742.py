# I have 60 folders in the path and each folder include a simulation_params.json file, 
# sort the folders based on the information in the simulation_params.json file.
# I have 'simulation condition', 'synaptic spatial condition', 'section type', 'distance from clusters to root' and 
# 'number of clusters' in the simulation_params.json file,  
# I want basal distance from clusters to root 1 2 3 and apical distance from clusters to root 1 2 3 in the same folder. 
# And sort the folders based on the simulation condition, synaptic spatial condition and number of clusters.

import os
import json
from collections import defaultdict

# Define the path to the folders
base_path = '/G/results/simulation/20241107_2207/'

# Initialize a dictionary to store folder information
folders_info = []

# Read the simulation_params.json files from each folder
for folder_name in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder_name)
    if os.path.isdir(folder_path):
        json_file_path = os.path.join(folder_path, 'simulation_params.json')
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as json_file:
                params = json.load(json_file)
                folders_info.append({
                    'folder_name': folder_name,
                    'simulation_condition': params.get('simulation condition'),
                    'synaptic_spatial_condition': params.get('synaptic spatial condition'),
                    'section_type': params.get('section type'),
                    'distance_from_clusters_to_root': params.get('distance from clusters to root'),
                    'number_of_clusters': params.get('number of clusters')
                })

# Group folders based on section type and distance from clusters to root
grouped_folders = defaultdict(list)
for info in folders_info:
    key = (info['section_type'], info['distance_from_clusters_to_root'])
    grouped_folders[key].append(info)

# Filter groups to include only those with basal and apical distances 1, 2, 3
# filtered_groups = {}
# for key, group in grouped_folders.items():
#     section_type, distance = key
#     if section_type == 'basal' and distance in [0, 1, 2]:
#         filtered_groups[key] = group
#     elif section_type == 'apical' and distance in [0, 1, 2]:
#         filtered_groups[key] = group

# Sort the folders based on simulation condition, synaptic spatial condition, and number of clusters
sorted_folders = sorted(folders_info, key=lambda x: (
    x['simulation_condition'],
    x['synaptic_spatial_condition'],
    x['number_of_clusters']
))

# Print the sorted folder names
# for folder in sorted_folders:
    # print(folder['folder_name'])

# Create new folders and move the original folders into them
for folder in sorted_folders:
    new_folder_name = f"{folder['simulation_condition']}_{folder['synaptic_spatial_condition']}_{folder['number_of_clusters']}"
    new_folder_path = os.path.join(output_base_path, new_folder_name)
    os.makedirs(new_folder_path, exist_ok=True)
    shutil.move(folder['folder_path'], new_folder_path)

print("Folders have been sorted and moved successfully.")