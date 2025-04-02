#%%
import preprocessing_utils as preprocess
import csv

preprocess.change_cwd()
videolist = preprocess.retrieve_nii_gz_paths("/media/backup_16TB/sean/VJEPA/jepa/data/IXI2/**/*.mp4")
print(len(videolist))
print(videolist[:10])

#%%

# Output CSV file path
output_csv = "/media/backup_16TB/sean/Monai/videolist.csv"

# Write to CSV
with open(output_csv, mode="w", newline="") as file:
    writer = csv.writer(file, delimiter=" ")  # Use space as the delimiter
    for video in videolist:
        writer.writerow([video, 1])  # Write video path and integer 1

print(f"CSV file created at: {output_csv}")