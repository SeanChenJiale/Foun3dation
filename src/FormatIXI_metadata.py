
import glob
import pandas as pd
import os 

file = os.path.dirname(os.path.abspath(__file__))
os.chdir(file)
print(os.getcwd())

IXI = glob.glob('../PreProcessedData/IXI/**/*.nii.gz',recursive=True) ## change to the file of the pre-processed images
for name in IXI:
    print(name)
    
# Replace backslashes with forward slashes
IXI = [path.replace("\\", "/") for path in IXI]
   
df = pd.read_csv("../Metadata/IXI_metadata.csv")

# Convert to formatted IXI IDs
df['Formatted_IXI_ID'] = df['IXI_ID'].apply(lambda x: f'IXI{x:03}')

# Find matching paths for each IXI_ID
df['filepath'] = df['Formatted_IXI_ID'].apply(
    lambda x: next((path for path in IXI if x in path), None)
)
df = df[['AGE', 'filepath']].dropna()

df = df.rename(columns={'AGE':"age"})

df.to_csv("../Metadata/IXI_cleaned.csv",index=False)

# print('\nNamed with wildcard *:')
# IXI = glob.glob('C:/Sean/PhD/Dataset/ixi_tiny/ixi_tiny/**/*.nii.gz')
# for name in IXI:
#     print(name)    
