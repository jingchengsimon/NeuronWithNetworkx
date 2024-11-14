# I have 60 folders in the path and each folder include a simulation_params.json file, 
# sort the folders based on the information in the simulation_params.json file.
# I have 'simulation condition', 'synaptic spatial condition', 'section type', 'distance from clusters to root' and 
# 'number of clusters' in the simulation_params.json file,  
# I want basal distance from clusters to root 1 2 3 and apical distance from clusters to root 1 2 3 in the same folder. 
# And sort the folders based on the simulation condition, synaptic spatial condition and number of clusters.

import os

root_folder_path = '/G/results/simulation/'

# Get all the folders in the root folder
folders = os.listdir(root_folder_path)
