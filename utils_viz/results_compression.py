import os
import shutil
import zipfile

# Define the root folder and output zip file
root_folder = "/G/results/simulation/basal_range0_clus_invitro_variedW_concentest_multi/28"
output_zip = "morpho_syn_files_clus_on_multi_branches.zip"

# Create a temporary folder to store matching files
temp_dir = "temp_html_files"
os.makedirs(temp_dir, exist_ok=True)

# Iterate through folders 1 to 20
for i in range(1, 21):
    subfolder = os.path.join(root_folder, str(i))  # Path to each subfolder
    if os.path.isdir(subfolder):  # Ensure it's a valid folder
        for file in os.listdir(subfolder):
            if "morpho_syn" in file and file.endswith(".html"):  # Match required files
                src_path = os.path.join(subfolder, file)
                dst_path = os.path.join(temp_dir, file)
                shutil.copy2(src_path, dst_path)  # Copy file

# Create a zip file containing all copied HTML files
with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
    for file in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, file)
        zipf.write(file_path, arcname=file)  # Add to zip with only filename

# Clean up temporary folder
shutil.rmtree(temp_dir)

print(f"Compression complete: {output_zip}")