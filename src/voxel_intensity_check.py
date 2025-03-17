import SimpleITK as sitk
import os
import glob

file = os.path.dirname(os.path.abspath(__file__))
os.chdir(file)
# print(os.getcwd())

image_paths = glob.glob("../PreProcessedData/**/*.nii.gz",recursive=True)
print(image_paths[1],len(image_paths))
# for image_path in image_paths:

# IXI_image = sitk.ReadImage("../PreProcessedData/IXI/IXI012-HH-1211-T1.nii.gz")  # MNI template
# print(IXI_image)

# # Convert image to an array
# image_array = sitk.GetArrayFromImage(IXI_image).astype(float)

# # Normalize to range [0,1]
# min_val = image_array.min()
# max_val = image_array.max()
# normalized_array = (image_array - min_val) / (max_val - min_val)

# # Convert back to a SimpleITK image
# normalized_image = sitk.GetImageFromArray(normalized_array)

# # Copy the original image's metadata (spacing, origin, direction)
# normalized_image.CopyInformation(IXI_image)

# # Save or use the normalized image
# sitk.WriteImage(normalized_image, "IXI012.nii.gz")

# print("write complete, standardized to [0,1]")