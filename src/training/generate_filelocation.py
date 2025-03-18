
# generate a csv file with file paths and corresponding labels.

import pandas as pd 
import os

file = os.path.dirname(os.path.abspath(__file__))
os.chdir(file)

print(os.path.abspath("../../Metadata/AOMIC_cleaned.csv"))  # Check absolute path
print(os.path.exists("../../Metadata/AOMIC_cleaned.csv"))   # Check if the file exists
aomic_df = pd.read_csv("../../Metadata/AOMIC_cleaned.csv")
ixi_df = pd.read_csv("../../Metadata/IXI_cleaned.csv")
oasis_df = pd.read_csv("../../Metadata/OASIS_cleaned.csv")
sald_df = pd.read_csv("../../Metadata/SALD_cleaned.csv")

print(aomic_df.head())