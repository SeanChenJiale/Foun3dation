#%%
import glob
import pandas as pd
import os 

file = os.path.dirname(os.path.abspath(__file__))
os.chdir(file)
print(os.getcwd())

IXI = glob.glob('../../PreProcessedData/ICBM/**/*.nii.gz',recursive=True) ## change to the file of the pre-processed images
for name in IXI:
    print(name)
    
# Replace backslashes with forward slashes
IXI = [path.replace("\\", "/") for path in IXI]
   
df = pd.read_csv("../../Metadata/idaSearch_3_11_2025.csv")

# # Convert to formatted IXI IDs
# df['Formatted_IXI_ID'] = df['Subject_ID'].apply(lambda x: f'IXI{x:03}')

# Find matching paths for each IXI_ID
df['filepath'] = df['Subject_ID'].apply(
    lambda x: next((path for path in IXI if x in path), None)
)
df = df[['Age', 'filepath']].dropna()

df = df.rename(columns={'Age':"age"})

df.to_csv("../../Metadata/ICBM_cleaned.csv",index=False)

# print('\nNamed with wildcard *:')
# IXI = glob.glob('C:/Sean/PhD/Dataset/ixi_tiny/ixi_tiny/**/*.nii.gz')
# for name in IXI:
#     print(name)    

# %%
