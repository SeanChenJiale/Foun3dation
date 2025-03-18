import glob
import pandas as pd
import os
### instructions place into ./Metadata/sub_information_SALD.xlsx
### preprocessed images should be in ./PreProcessedData/SALD/<all SALD_nii.gz files here>
print('AOMIC csv formatter')

file = os.path.dirname(os.path.abspath(__file__))
os.chdir(file)
# print(os.getcwd())

AOMIC = glob.glob('../PreProcessedData/AOMIC/*.nii.gz') ## change to the file of the pre-processed images
for name in AOMIC:
    print(name)

# Replace backslashes with forward slashes
AOMIC = [path.replace("\\", "/") for path in AOMIC]

df = pd.read_csv("../Metadata/participants_Aomic.csv") ### metadatafolder here

# ## sanity check
# print(df.columns)

# print(df.head)

# Find matching paths for each IXI_ID
df['filepath'] = df['participant_id'].apply(
    lambda x: next((path for path in AOMIC if x in path), None)
)

df = df[['age', 'filepath']].dropna()

df = df.rename(columns={'age':"age"})

df.to_csv("../Metadata/AOMIC_cleaned.csv",index=False)

