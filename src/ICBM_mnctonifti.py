#%%

import preprocessing_utils

# if __name__ == "__main__":
#     if len(sys.argv) < 2 or len(sys.argv) > 3:
#         print("Usage:")
#         print("  ./mnc2nii.py input.mnc                # Default output: input.nii.gz")
#         print("  ./mnc2nii.py input.mnc output.nii.gz  # Custom output path")
#         sys.exit(1)
#     input_mnc_file = sys.argv[1]
#     output_nii_file = sys.argv[2] if len(sys.argv) == 3 else None
#     convert_mnc_to_nifti(input_mnc_file, output_nii_file)
preprocessing_utils.change_cwd()
mnclist = preprocessing_utils.retrieve_nii_gz_paths("../Dataset/ICBM/**/*.mnc")


for inputmnc in mnclist:
    basefilename = preprocessing_utils.filename_extract_from_path_with_ext(inputmnc).split(".")[0]
    outputfilename = f"../Dataset/ICBM/T1_w_nii/{basefilename}.nii.gz"
    print("converting")
    preprocessing_utils.convert_mnc_to_nifti(inputmnc,outputfilename)



# %%

