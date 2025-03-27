#%%
import concurrent.futures
import numpy as np
import os
from preprocessing_utils import load_MNI_template, retrieve_nii_gz_paths ,bias_field_correction ,\
    rescale_intensity, equalize_hist , \
    load_nii , save_nii ,change_cwd , linear_interpolator, load_itk,\
    rigid_registration, filename_extract_from_path_with_ext, setup_logger, main_preprocessing
import SimpleITK as sitk

change_cwd()
dataset_to_preprocess = "SALD"
MNI_template, pathlist, finished_pathlist, finished_pp, undone_paths, logger = main_preprocessing(dataset_to_preprocess,
                                                                                                  search_string="*T1w.nii.gz")

# %%


def process_file(filepath):
    try:
        filename_to_save = filename_extract_from_path_with_ext(filepath)
        temp_file = f"../temp/{filename_to_save}_temp.nii.gz"
        img, affine = load_nii(filepath)

        img = bias_field_correction(img)
        save_nii(data=img, affine=affine, path=temp_file)

        # Linear interpolation
        img = load_itk(temp_file)
        img = linear_interpolator(img)
        sitk.WriteImage(img, temp_file)

        # Intensity normalization
        img, affine = load_nii(temp_file)
        volume = rescale_intensity(img, percentils=[0.5, 99.5], bins_num=256)
        volume = equalize_hist(volume, bins_num=256)
        save_nii(data=volume, affine=affine, path=temp_file)  # Save processed volume

        # Rigid registration
        img = load_itk(temp_file)
        img = rigid_registration(MNI_template, img)
        output_path = f"../temp/{dataset_to_preprocess}/{filename_to_save}"
        sitk.WriteImage(img, output_path)

        logger.info(f"{output_path} was saved.")
        # Remove the temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)
            logger.info(f"Temporary file {temp_file} was removed.")        
    except Exception as e:
        logger.error(f"Error processing {filepath}: {e}")

# Run processing in parallel
# when coding this, ensure that each thread's variables are independant of another thread.
def process_all_files_parallel(filepaths, num_workers=4):
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        executor.map(process_file, filepaths)

# Example Usage
if __name__ == "__main__":
    num_workers = os.cpu_count()  # Use all available CPU cores
    process_all_files_parallel(undone_paths, int(num_workers/2) )

    # hd-bet -i IXI2 -o ../PreProcessedData/IXI2