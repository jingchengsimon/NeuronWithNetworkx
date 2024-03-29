import os

root_folder_path = './results/simulation/pseudo/AMPANMDA_20240209_1653_1/'
file_path = root_folder_path + 'dend_v_array.npy'  # 替换为你的 .npy 文件路径

try:
    arr = np.load(file_path)
    print("Shape of the array:", arr.shape)
except FileNotFoundError:
    print(f"{file_path} does not exist.")
except Exception as e:
    print("An error occurred:", e)

if os.path.exists(file_path):
    # file_size = os.path.getsize(file_path)
    print(f"File size of {file_path}: {file_size} bytes")
else:
    print(f"{file_path} does not exist.")
