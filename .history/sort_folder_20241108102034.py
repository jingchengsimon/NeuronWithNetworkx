import os
import json
import shutil

def sort_folders(base_path, output_base_path):

    """
    I have 60 folders in the path and each folder include a simulation_params.json file, 
    sort the folders based on the information in the simulation_params.json file.
    I have 'simulation condition', 'synaptic spatial condition', 'section type', 'distance from clusters to root' and 
    'number of clusters' in the simulation_params.json file,  
    I want basal distance from clusters to root 1 2 3 and apical distance from clusters to root 1 2 3 in the same folder. 
    And sort the folders based on the simulation condition, synaptic spatial condition and number of clusters.
    """

    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)

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
                        'folder_path': folder_path,
                        'simulation_condition': params.get('simulation condition'),
                        'synaptic_spatial_condition': params.get('synaptic spatial condition'),
                        'section_type': params.get('section type'),
                        'distance_from_clusters_to_root': params.get('distance from clusters to root'),
                        'number_of_clusters': params.get('number of clusters')
                    })

    # Sort the folders based on simulation condition, synaptic spatial condition, and number of clusters
    sorted_folders = sorted(folders_info, key=lambda x: (
        x['simulation_condition'],
        x['synaptic_spatial_condition'],
        x['number_of_clusters']
    ))

    # Create new folders and move the original folders into them
    for folder in sorted_folders:
        new_folder_name = f"{folder['simulation_condition']}_{folder['synaptic_spatial_condition']}_{folder['number_of_clusters']*80}syns"
        new_folder_path = os.path.join(output_base_path, new_folder_name)
        os.makedirs(new_folder_path, exist_ok=True)
        shutil.move(folder['folder_path'], new_folder_path)

    print("Folders have been sorted and moved successfully.")


def rename_sorted_folders(output_base_path):
"""

First, rename each folder in the output_base_path, change the number after the last underscore to the number of synapses in the folder.
Which is the number of clusters * 80. So the previous name like 'invitro_clus_1' now should be 'invitro_clus_80syns'.

Second, rename the subfolders in each folder, that is, in each folder, the original index are from the previous folder, 
now it should be based on the order in the new folders. If a folder includes subfolder 1 6 11 16 21 and 26, 
then the new name for each folder should be 1 2 3 4 5 and 6. 
"""
    # First, rename each folder in the output_base_path
    for folder_name in os.listdir(output_base_path):
        folder_path = os.path.join(output_base_path, folder_name)
        if os.path.isdir(folder_path):
            # Extract the number of clusters from the folder name
            parts = folder_name.split('_')
            num_clusters = int(parts[-1])
            num_synapses = num_clusters * 80
            new_folder_name = f"{'_'.join(parts[:-1])}_{num_synapses}syns"
            new_folder_path = os.path.join(output_base_path, new_folder_name)
            os.rename(folder_path, new_folder_path)
    
    # Second, rename the subfolders in each folder
    for folder_name in os.listdir(output_base_path):
        folder_path = os.path.join(output_base_path, folder_name)
        if os.path.isdir(folder_path):
            subfolders = sorted(os.listdir(folder_path), key=lambda x: int(x))
            for i, subfolder_name in enumerate(subfolders, start=1):
                subfolder_path = os.path.join(folder_path, subfolder_name)
                if os.path.isdir(subfolder_path):
                    new_subfolder_name = str(i)
                    new_subfolder_path = os.path.join(folder_path, new_subfolder_name)
                    os.rename(subfolder_path, new_subfolder_path)

    print("Folders and subfolders have been renamed successfully.")

# Define the path to the folders
base_path = '/G/results/simulation/20241107_2207/'
output_base_path = '/G/results/simulation/invitro_simulation/'

sort_folders(base_path, output_base_path)