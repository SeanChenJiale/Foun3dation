#%%
from preprocessing_utils import load_MNI_template, retrieve_nii_gz_paths ,bias_field_correction ,\
    rescale_intensity, equalize_hist , \
    load_nii , save_nii ,change_cwd , linear_interpolator, load_itk,\
    rigid_registration, filename_extract_from_path_with_ext, setup_logger
import SimpleITK as sitk


change_cwd() 

main_datasetname = "ICBM"

MNI_template = load_MNI_template()

temp_file = "../temp/temp.nii.gz"

pathlist = retrieve_nii_gz_paths(f'../Dataset/{main_datasetname}/**/*.nii.gz')

finished_pathlist = retrieve_nii_gz_paths(f'../temp/{main_datasetname}/*.nii.gz')

finished_pp= []

for path in finished_pathlist:
    finished_pp.append(filename_extract_from_path_with_ext(path))

undone_paths = [path for path in pathlist if not any(sub in path for sub in finished_pp)]

logger = setup_logger(f"../logs/{main_datasetname}preprocess.log")


#%%

for filepath in undone_paths:
    
    try:
        filename_to_save = filename_extract_from_path_with_ext(filepath)

        img, affine = load_nii(filepath)

        img = bias_field_correction(img)

        save_nii(data = img, affine = affine, path=temp_file)

        ## start linear)interpolation

        img = load_itk(temp_file)

        img = linear_interpolator(img)

        sitk.WriteImage(img,temp_file)

        ## intensity normalization

        img, affine = load_nii(temp_file)

        volume = rescale_intensity(img, percentils=[0.5, 99.5],  bins_num=256)

        volume = equalize_hist(volume, bins_num=256)

        save_nii(data = img, affine = affine, path=temp_file)

        ## rigid_registration

        img = load_itk(temp_file)

        img = rigid_registration(MNI_template,img)

        sitk.WriteImage(img,f"../temp/{main_datasetname}/{filename_to_save}")
        logger.info(f"../temp/{main_datasetname}/{filename_to_save} was saved.")

    except:

        logger.error(f"../temp/{main_datasetname}/{filename_to_save} had an error.")

# %%
