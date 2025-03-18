import glob
import pandas as pd
import os
### instructions place into ./Metadata/sub_information_SALD.xlsx
### preprocessed images should be in ./PreProcessedData/SALD/<all SALD_nii.gz files here>
print('OASIS csv formatter')

file = os.path.dirname(os.path.abspath(__file__))
os.chdir(file)
# print(os.getcwd())

OASIS = glob.glob('../PreProcessedData/OASIS/*.nii.gz') ## change to the file of the pre-processed images
for name in OASIS:
    print(name)

# Replace backslashes with forward slashes
OASIS = [path.replace("\\", "/") for path in OASIS]

df = pd.read_excel("../Metadata/oasis1_metadata.xlsx") ### metadatafolder here

# ## sanity check
# print(df.columns)

# print(df.head)

# Find matching paths for each IXI_ID
df['filepath'] = df['ID'].apply(
    lambda x: next((path for path in OASIS if x in path), None)
)

df = df[['Age', 'filepath']].dropna()

df = df.rename(columns={'Age':"age"})

df.to_csv("../Metadata/OASIS_cleaned.csv",index=False)