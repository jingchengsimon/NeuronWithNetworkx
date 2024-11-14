import os
import json
import shutil
import glob

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

    # Rename the subfolders in each folder
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

def delete_npy_files(path):
    for root, dirs, files in os.walk(path):
        # Check for .npy files in each subfolder
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
    
# Specify the path


# Call the function to delete .npy files

# delete_npy_files(path)

# # Define the path to the folders
# base_path = '/G/results/simulation/20241107_2255/'
# output_base_path = '/G/results/simulation/invivo_simulation/'

# sort_folders(base_path, output_base_path)