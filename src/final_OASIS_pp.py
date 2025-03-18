#%%

import SimpleITK as sitk
import glob
import pandas as pd
import os 
import numpy as np

file = os.path.dirname(os.path.abspath(__file__))
os.chdir(file)
print(os.getcwd())

OASIS_hdrpathlist = glob.glob('../Dataset/OASIS/**/*.hdr', recursive=True)
OASIS_imgpathlist = glob.glob('../Dataset/OASIS/**/*.img', recursive=True)
OASIS_hdrpathlist = [string for string in OASIS_hdrpathlist if "mpr-1" in string]
OASIS_imgpathlist = [string for string in OASIS_imgpathlist if "mpr-1" in string]
# Create a dictionary to store corresponding paths
hdr_img_pairs = {}

# Iterate over all the .hdr files
for hdr_path in OASIS_hdrpathlist:
    # Get the base filename without extension
    base_filename = os.path.splitext(os.path.basename(hdr_path))[0]
    
    # Find the corresponding .img file by matching the base filename
    corresponding_img = next((img for img in OASIS_imgpathlist if os.path.splitext(os.path.basename(img))[0] == base_filename), None)
    
    # If the corresponding .img file exists, add to the dictionary
    if corresponding_img:
        hdr_img_pairs[hdr_path] = corresponding_img

# Now `hdr_img_pairs` contains the mapping of .hdr to .img
# print("Mapped .hdr and .img paths:")
# for hdr, img in hdr_img_pairs.items():
#     print(f"Header: {hdr} -> Image: {img}")

# print(len(OASIS_hdrpathlist),len(OASIS_imgpathlist))

fixed_image = sitk.ReadImage("../Dataset/MNI/MNI152_T1_1mm.nii.gz")  # MNI template
#%%
### Reorientation just for OASIS This is RAI 
# Define the affine transformation matrix
affine_transform = np.array([
    [ -1,  0,  0],  # Flip X-axis (Left to Right)
    [ 0,  -1,  0],  # Flip Y-axis (Posterior to Anterior)
    [ 0,  0, 1]    # keep inferior and superior
])

# Define the matrix for swapping XY and YZ planes
swap_xy_yz = np.array([
    [0, 0, 1],  # X goes to Z
    [1, 0, 0],  # Y goes to X
    [0, 1, 0]   # Z goes to Y
])

# Combined transformation: affine + swap
combined_transform = affine_transform @ swap_xy_yz


for hdr, img in hdr_img_pairs.items():
    print("Starting preprocessing")
    # Load the NIfTI image
    input_image = sitk.ReadImage(hdr)
    # Convert to a float type for better correctionko
    input_image = sitk.Cast(input_image, sitk.sitkFloat32)

    direction = np.array(input_image.GetDirection()).reshape(3, 3)  # Convert to 3x3 matrix

    # Apply new direction (modify existing direction matrix)
    new_direction = tuple((combined_transform @ direction).flatten())  # Apply affine to direction
    input_image.SetDirection(new_direction)

    # Apply N4 Bias Field Correction
    bias_corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_image = bias_corrector.Execute(input_image)
        
    sitk.WriteImage(corrected_image,  "../PreProcessedData/OASIS_input/" + os.path.splitext(img.split('/')[-1])[0] + ".nii.gz")

    print("N4 bias field correction applied successfully!")
    
#     #  resampling test
#     import SimpleITK as sitk

#     # Load the original NIfTI image
#     input_image = sitk.ReadImage("./Dataset/trial_OASIS002-Guys-0828-T1_N4.nii.gz")

    # Get original spacing and size
    original_spacing = corrected_image.GetSpacing() # input_image.GetSpacing()  # (dx, dy, dz)
    original_size = input_image.GetSize()        # (nx, ny, nz)

    # Define new isotropic spacing (1mm x 1mm x 1mm)
    new_spacing = (1.0, 1.0, 1.0)

    # Compute new size while preserving physical dimensions
    new_size = [
        int(round(original_size[i] * (original_spacing[i] / new_spacing[i])))
        for i in range(3)
    ]

    # Define resampling interpolator (use linear for intensity images, nearest for labels)
    interpolator = sitk.sitkLinear

    # Perform resampling
    resampled_image = sitk.Resample(
        input_image,
        new_size,
        sitk.Transform(),
        interpolator,
        input_image.GetOrigin(),
        new_spacing,
        input_image.GetDirection(),
        0,  # Default pixel value for areas outside original image
        input_image.GetPixelID()
    )

#     # Save the resampled image
#     sitk.WriteImage(resampled_image, "./Dataset/trial_resample_OASIS002-Guys-0828-T1_N4.nii.gz")

    print("Resampling to 1x1x1 mm isotropic voxels complete!")

#     # Rigid registration https://discourse.itk.org/t/3d-mri-image-registration/3144
#     import SimpleITK as sitk
#     import os

#     # Load images
#     fixed_image = sitk.ReadImage("./Dataset/MNI/MNI152_T1_1mm.nii.gz")  # MNI template
    moving_image = resampled_image # sitk.ReadImage("./Dataset/trial_resample_OASIS002-Guys-0828-T1_N4.nii.gz")  # Your T1 MRI

    # Convert images to float32
    fixed_image_cast = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)
    
    initial_transform=sitk.CenteredTransformInitializer(fixed_image_cast,moving_image,sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    moving_resampled = sitk.Resample(moving_image, fixed_image_cast, initial_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsCorrelation()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(sitk.sitkNearestNeighbor) ##update on this. It performs better in terms of registration.
    registration_method.SetOptimizerAsGradientDescent(learningRate=0.1, numberOfIterations=500)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    final_transform = registration_method.Execute(sitk.Cast(fixed_image_cast, sitk.sitkFloat32), sitk.Cast(moving_image, sitk.sitkFloat32))
    moving_resampled = sitk.Resample(moving_image, fixed_image_cast, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

    # print("Final Metric Value:", registration_method.GetMetricValue())
    # print(
    #     f"Optimizer's stopping condition, {registration_method.GetOptimizerStopConditionDescription()}"
    # )
    # Save the registered image
    sitk.WriteImage(moving_resampled,  "../PreProcessedData/OASIS_input_KNN/" + os.path.splitext(img.split('/')[-1])[0] + ".nii.gz")
    sitk.WriteTransform(final_transform, "../PreProcessedData/OASIS_mask_KNN/" + os.path.splitext(img.split('/')[-1])[0] + ".tfm")
    print("Rigid registration to MNI152 completed! Saved as ", img.split('/')[-1] )

# # HD-Bet

# # import os
# # os.system("cd C:/Sean/PhD/Dataset")
# # os.system("hd-bet -i ./trial_input -o ./trial_output_hdbet -device cpu --disable_tta")
#hd-bet -i ./sean/Monai/PreProcessedData/OASIS_input -o ./sean/Monai/PreProcessedData/OASIS