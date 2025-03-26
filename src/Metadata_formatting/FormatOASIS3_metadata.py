#%%
import glob
import pandas as pd
import os
import sys

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add it to sys.path
sys.path.append(parent_dir)

import preprocessing_utils
### instructions place into ./Metadata/sub_information_SALD.xlsx
### preprocessed images should be in ./PreProcessedData/SALD/<all SALD_nii.gz files here>
print('OASIS3 csv formatter')

preprocessing_utils.change_cwd()
file = os.path.dirname(os.path.abspath(__file__))
os.chdir(file)
# print(os.getcwd())

AOMIC = glob.glob('../../PreProcessedData/OASIS_3/*.nii.gz',recursive=True) ## change to the file of the pre-processed images
for name in AOMIC:
    print(name)

# Replace backslashes with forward slashes
AOMIC = [path.replace("\\", "/") for path in AOMIC]

df = pd.read_csv("../../Metadata/Oasis3_fullMRI.csv") ### metadatafolder here

# Split into three columns
df[['MR_ID1', 'MR_ID2', 'MR_ID3']] = df['MR_ID'].str.split('_', expand=True)

# ## sanity check
# print(df.columns)

# print(df.head)

# Function to check if any element in the list contains both Col1 and Col3 substrings
def match_substrings(row):
    return [item for item in AOMIC if row['MR_ID1'] in item and row['MR_ID3'] in item]

# Apply function to create a new column with matches
df['filepath'] = df.apply(match_substrings, axis=1)

df = df[['Age_flo', 'filepath']].dropna()

df = df.rename(columns={'Age_flo':"age"})

df.to_csv("../../Metadata/Oasis3_cleaned.csv",index=False)
# %%
