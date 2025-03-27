#%%
import cv2
import os
import glob
import preprocessing_utils as preprocess
from natsort import natsorted  # Natural sorting

nii_path = "../PreProcessedData/IXI2/IXI002-Guys-0828-T1.nii.gz"
main_dataset = preprocess.dataset_extractor(nii_path)
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

    # Ensure there are exactly 48 slices
    if len(image_files) != 48:
        print(f"Error: Expected 48 slices, but found {len(image_files)} in {curr_path}. Skipping...")
        continue

    # Read first image to get dimensions
    first_img = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)  # Read in grayscale
    height, width = first_img.shape

    # Define video writer (MP4, 30 FPS, same resolution as images)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
    video_writer = cv2.VideoWriter(output_video, fourcc, 30, (width, height), isColor=True)

    # Split slices into R, G, B groups
    red_slices = image_files[:16]  # First 16 slices
    green_slices = image_files[16:32]  # Next 16 slices
    blue_slices = image_files[32:]  # Last 16 slices

    # Loop through slices and create RGB frames
    for i in range(16):  # Loop through the first 16 slices (one frame per iteration)
        # Read slices for each channel
        red_img = cv2.imread(red_slices[i], cv2.IMREAD_GRAYSCALE)  # Red channel
        green_img = cv2.imread(green_slices[i], cv2.IMREAD_GRAYSCALE)  # Green channel
        blue_img = cv2.imread(blue_slices[i], cv2.IMREAD_GRAYSCALE)  # Blue channel

        # Stack into an RGB image
        rgb_img = cv2.merge((blue_img, green_img, red_img))  # OpenCV uses BGR order

        # Repeat each frame 4 times
        for _ in range(4):  # Repeat each frame 4 times
            video_writer.write(rgb_img)

    # Release the video writer
    video_writer.release()
    print(f"Video saved as {output_video}")

# %%
