#%%

# bias field test

import SimpleITK as sitk
import glob
import pandas as pd
import os 

file = os.path.dirname(os.path.abspath(__file__))
os.chdir(file)
print(os.getcwd())

IXI = glob.glob('../Dataset/IXI/**/*.nii.gz', recursive=True)

fixed_image = sitk.ReadImage("../Dataset/MNI/MNI152_T1_1mm.nii.gz")  # MNI template
#%%

for name in IXI:
    print("Starting preprocessing")
    # Load the NIfTI image
    input_image = sitk.ReadImage(name)

    # Convert to a float type for better correctionko
    input_image = sitk.Cast(input_image, sitk.sitkFloat32)
    # Apply N4 Bias Field Correction
    bias_corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_image = bias_corrector.Execute(input_image)
        
    # sitk.WriteImage(corrected_image, "./PreProcessedData/IXI/" + name.split('/')[-1])

    print("N4 bias field correction applied successfully!")
#     #  resampling test
#     import SimpleITK as sitk

#     # Load the original NIfTI image
#     input_image = sitk.ReadImage("./Dataset/trial_IXI002-Guys-0828-T1_N4.nii.gz")

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
#     sitk.WriteImage(resampled_image, "./Dataset/trial_resample_IXI002-Guys-0828-T1_N4.nii.gz")

    print("Resampling to 1x1x1 mm isotropic voxels complete!")

#     # Rigid registration https://discourse.itk.org/t/3d-mri-image-registration/3144
#     import SimpleITK as sitk
#     import os

#     # Load images
#     fixed_image = sitk.ReadImage("./Dataset/MNI/MNI152_T1_1mm.nii.gz")  # MNI template
    moving_image = resampled_image # sitk.ReadImage("./Dataset/trial_resample_IXI002-Guys-0828-T1_N4.nii.gz")  # Your T1 MRI

    #  Convert images to float32
    fixed_image_cast = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)
                            
    initial_transform=sitk.CenteredTransformInitializer(fixed_image_cast,moving_image,sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    moving_resampled = sitk.Resample(moving_image, fixed_image_cast, initial_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsCorrelation()
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

    # Save the registered image
    sitk.WriteImage(moving_resampled,  "../PreProcessedData/IXI_input_KNN/" + name.split('/')[-1])
    sitk.WriteTransform(final_transform, "../PreProcessedData/IXI_mask_KNN/" + os.path.splitext(name.split('/')[-1])[0] + ".tfm")
    print("Rigid registration to MNI152 completed!")

# # HD-Bet

# # import os
# # os.system("cd C:/Sean/PhD/Dataset")
# # os.system("hd-bet -i ../trial_input -o ../trial_output_hdbet -device cpu --disable_tta")
