import os
import json

def find_folders_with_3d_acquisition(base_dir):
    folders_with_3d = []

    # Walk through the directory structure
    for root, dirs, files in os.walk(base_dir):
        print(f"Checking directory: {root}")  # Debugging line
        # Check each file in the directory
         # Check if the file is a JSON file
         # and if it contains "MRAcquisitionType" with value "3D"
        for file in files:
            if file.endswith(".json"):  # Look for JSON files
                json_path = os.path.join(root, file)
                try:
                    # Open and parse the JSON file
                    with open(json_path, "r") as f:
                        data = json.load(f)
                    
                    # Check if "MRAcquisitionType" is "3D"
                    if data.get("MRAcquisitionType") == "3D":
                        folders_with_3d.append(root)
                        break  # No need to check other files in this folder
                except Exception as e:
                    print(f"Error reading {json_path}: {e}")

    return folders_with_3d

# Base directory to search
base_directory = "/media/backup_16TB/sean/Monai/Dataset/SOOP"

# Find folders and print them
folders = find_folders_with_3d_acquisition(base_directory)
print("Folders with MRAcquisitionType = '3D':")
for folder in folders:
    print(folder)