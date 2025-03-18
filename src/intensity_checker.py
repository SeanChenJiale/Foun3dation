import SimpleITK as sitk
import os
import glob

file = os.path.dirname(os.path.abspath(__file__))
os.chdir(file)
# print(os.getcwd())


image_path = "../PreProcessedData/IXI/IXI634-HH-2690-T1.nii.gz"
# for image_path in image_paths:

curr_image = sitk.ReadImage(image_path)  # MNI template

# Convert to a NumPy array
image_array = sitk.GetArrayFromImage(curr_image)

# Extract min and max intensity values
min_intensity = image_array.min()
max_intensity = image_array.max()

print(f"Minimum Intensity: {min_intensity}")
print(f"Maximum Intensity: {max_intensity}")