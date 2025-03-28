#%%
import cv2
import os
import glob
import preprocessing_utils as preprocess
from natsort import natsorted  # Natural sorting

nii_path = "../PreProcessedData/IXI2/IXI002-Guys-0828-T1.nii.gz"
main_dataset = preprocess.dataset_extractor("../PreProcessedData/IXI2/IXI002-Guys-0828-T1.nii.gz")
listdir = preprocess.makedirs(nii_path,main_dataset,makedir=False)
# Define the input and output paths
Ordered_list = ["","Axial", "Coronal", "Sagittal"]
for listdir_curr_index in range(len(listdir)):
    ## it runs in alphabetical order, Axial, Coronal, Sagittal
    if listdir_curr_index == 0:
        continue
    curr_path = listdir[listdir_curr_index]
    output_video = f"{listdir[0]}/{Ordered_list[listdir_curr_index]}.mp4"

    # Get all images (Assumes they are PNG/JPG and sorted in order)
    image_files = natsorted(glob.glob(os.path.join(curr_path, "*.png")))  # Change to "*.jpg" if needed

    # Read first image to get dimensions
    first_img = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)  # Read in grayscale
    height, width = first_img.shape

    # Define video writer (MP4, 30 FPS, same resolution as images)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
    video_writer = cv2.VideoWriter(output_video, fourcc, 30, (width, height), isColor=False)

    # Loop through images, repeating each 4 times
    for img_path in image_files:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read grayscale
        for _ in range(4):  # Repeat each frame 4 times
            video_writer.write(img)

    # Release the video writer
    video_writer.release()
    print(f"Video saved as {output_video}")

# %%
