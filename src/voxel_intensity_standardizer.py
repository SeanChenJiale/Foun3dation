import SimpleITK as sitk
import os
import glob

file = os.path.dirname(os.path.abspath(__file__))
os.chdir(file)
# print(os.getcwd())

image_paths = glob.glob("../PreProcessedData/**/*.nii.gz",recursive=True)

for image_path in image_paths:
    # for image_path in image_paths:

    curr_image = sitk.ReadImage(image_path)  # MNI template

    # print(curr_image)

    # Convert image to an array
    image_array = sitk.GetArrayFromImage(curr_image).astype(float)

    # Normalize to range [0,1]
    min_val = image_array.min()
    max_val = image_array.max()
    normalized_array = (image_array - min_val) / (max_val - min_val)

    # Convert back to a SimpleITK image
    normalized_image = sitk.GetImageFromArray(normalized_array)

    # Copy the original image's metadata (spacing, origin, direction)
    normalized_image.CopyInformation(curr_image)

    # Save or use the normalized image
    sitk.WriteImage(normalized_image, image_path)

    print(f"write complete, {image_path}, standardized to [0,1]")

        