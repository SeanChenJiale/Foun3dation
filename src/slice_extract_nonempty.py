#%%
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import preprocessing_utils as preprocess
from PIL import Image
from pathlib import Path
import os

nii_path = "../PreProcessedData/IXI/IXI002-Guys-0828-T1.nii.gz"
main_dataset = preprocess.dataset_extractor(nii_path)
listofcreateddirs = preprocess.makedirs(nii_path,main_dataset)
img_data,affine = preprocess.load_nii(nii_path)

# Target shape
target_size = max([max(img_data.shape),224])

# Calculate padding
pad_0 = ((target_size - img_data.shape[0]) // 2, (target_size - img_data.shape[0] + 1) // 2)
pad_2 = ((target_size - img_data.shape[2]) // 2, (target_size - img_data.shape[2] + 1) // 2)

# Apply padding
img_data = np.pad(img_data, (pad_0, (0, 0), pad_2), mode='constant', constant_values=0)

# Identify the nonzero voxel coordinates
nonzero_coords = np.array(np.nonzero(img_data))
print(nonzero_coords)
# Function to get non-empty slices
def get_non_black_slices(image_array, axis):
    """
    Extract indices of slices that are not fully black along a given axis.
    """
    non_black_slices = []
    num_slices = image_array.shape[axis]

    for i in range(num_slices):
        if axis == 0:  # Axial/Horizontal Plane
            slice_ = image_array[i, :, :]
        elif axis == 1:  # Coronal Plane
            slice_ = image_array[:, i, :]
        elif axis == 2:  # Sagittal Plane
            slice_ = image_array[:, :, i]
        
        if np.any(slice_ > 0):  # Check if there are nonzero pixels
            non_black_slices.append(i)

    return non_black_slices

# Extract slices for each plane
axial_slices = get_non_black_slices(img_data, axis=0)
coronal_slices = get_non_black_slices(img_data, axis=1)
sagittal_slices = get_non_black_slices(img_data, axis=2)



# Total number of rows
num_slices = 16

# Compute slice indices spaced across the volume
x_slices = np.linspace(axial_slices[0], axial_slices[-1], num_slices, dtype=int)
y_slices = np.linspace(coronal_slices[0], coronal_slices[-1] - 1, num_slices, dtype=int)
z_slices = np.linspace(sagittal_slices[0], sagittal_slices[-1] - 1, num_slices, dtype=int)

# Set up a 16x3 plot grid
fig, axes = plt.subplots(num_slices, 3, figsize=(12, num_slices * 1.5))

for i in range(num_slices):
    curr_sagittal_slice = img_data[x_slices[i], :, :]
    curr_coronal_slice = img_data[:, y_slices[i], :]
    curr_axial_slice = img_data[:, :, z_slices[i]]
    # Sagittal (x-axis)
    axes[i, 0].imshow(np.rot90(curr_sagittal_slice), cmap='gray')
    axes[i, 0].set_title(f'x={x_slices[i]}', fontsize=8)
    axes[i, 0].axis('off')

    # Coronal (y-axis)
    axes[i, 1].imshow(np.rot90(img_data[:, y_slices[i], :]), cmap='gray')
    axes[i, 1].set_title(f'y={y_slices[i]}', fontsize=8)
    axes[i, 1].axis('off')

    # Axial (z-axis)
    axes[i, 2].imshow(np.rot90(img_data[:, :, z_slices[i]]), cmap='gray')
    axes[i, 2].set_title(f'z={z_slices[i]}', fontsize=8)
    axes[i, 2].axis('off')

        # Normalize to 0-255
    curr_axial_slice = ((curr_axial_slice - np.min(curr_axial_slice)) / (np.max(curr_axial_slice) - np.min(curr_axial_slice)) * 255).astype(np.uint8)
    curr_coronal_slice = ((curr_coronal_slice - np.min(curr_coronal_slice)) / (np.max(curr_coronal_slice) - np.min(curr_coronal_slice)) * 255).astype(np.uint8)
    curr_sagittal_slice = ((curr_sagittal_slice - np.min(curr_sagittal_slice)) / (np.max(curr_sagittal_slice) - np.min(curr_sagittal_slice)) * 255).astype(np.uint8)
    # Save as PNG (lossless format, recommended for medical images)
    Image.fromarray(curr_axial_slice).save(f"{listofcreateddirs[1]}/{i}.png")
    Image.fromarray(curr_coronal_slice).save(f"{listofcreateddirs[2]}/{i}.png")    
    Image.fromarray(curr_sagittal_slice).save(f"{listofcreateddirs[3]}/{i}.png")
# Set column titles
axes[0, 0].set_ylabel('Sagittal', fontsize=10)
axes[0, 1].set_ylabel('Coronal', fontsize=10)
axes[0, 2].set_ylabel('Axial', fontsize=10)

plt.tight_layout()
plt.show()
# %%

# Normalize to 0-255
img_slice = ((curr_axial_slice - np.min(curr_axial_slice)) / (np.max(curr_axial_slice) - np.min(curr_axial_slice)) * 255).astype(np.uint8)

# Save as PNG (lossless format, recommended for medical images)
Image.fromarray(img_slice).save(f"{i}axial.png")


# %%
