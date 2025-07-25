import os

parent_dir = '/G/results/simulation/'

for name in os.listdir(parent_dir):
    if name.endswith('singclus_2'):
        old_path = os.path.join(parent_dir, name)
        new_name = name.replace('singclus_AMPA_2', 'singclus_AMPA')
        new_path = os.path.join(parent_dir, new_name)
        os.rename(old_path, new_path)
        print(f"{old_path} -> {new_path}")