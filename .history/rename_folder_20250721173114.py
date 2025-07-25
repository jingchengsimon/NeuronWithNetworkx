import os

for name in os.listdir('/G/results/simulation/'):
    if name.endswith('singclus_2'):
        new_name = name.replace('singclus_2', 'singclus')
        os.rename(name, new_name)
        print(f"{name} -> {new_name}")