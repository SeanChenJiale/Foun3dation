import SimpleITK as sitk
import os
import glob
import numpy as np

file = os.path.dirname(os.path.abspath(__file__))
os.chdir(file)
# print(os.getcwd())

image_paths = glob.glob("../PreProcessedData/SALD/**/*.nii.gz",recursive=True)
print(len(image_paths))

for image_path in image_paths:
    # for image_path in image_paths:

    curr_image = sitk.ReadImage(image_path)  
    # print(curr_image)

    # Convert image to an array
    image_array = sitk.GetArrayFromImage(curr_image).astype(float)

    # Compute 1st and 99th percentiles
    p1 = np.percentile(image_array, 1)
    p99 = np.percentile(image_array, 99)
    # print(p1,p99,image_array.max())

    # Normalize using percentile range
    normalized_array = (image_array - p1) / (p99 - p1)

    # Clip values to range [0,1]
    normalized_array = np.clip(normalized_array, 0, 1)
    
    # print(normalized_array.max(),normalized_array.min())
    # Convert back to a SimpleITK image
    normalized_image = sitk.GetImageFromArray(normalized_array)

    # Copy the original image's metadata (spacing, origin, direction)
    normalized_image.CopyInformation(curr_image)

    # Save or use the normalized image 
    sitk.WriteImage(normalized_image, image_path)

    print(f"write complete, {image_path}, standardized to [0,1]")
# 
        