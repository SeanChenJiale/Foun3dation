
# generate a csv file with file paths and corresponding labels.

import pandas as pd 
import os

file = os.path.dirname(os.path.abspath(__file__))
os.chdir(file)

aomic_df = pd.read_csv("../../Metadata/AOMIC_cleaned.csv")
ixi_df = pd.read_csv("../../Metadata/IXI_cleaned.csv")
oasis_df = pd.read_csv("../../Metadata/OASIS_cleaned.csv")
sald_df = pd.read_csv("../../Metadata/SALD_cleaned.csv")

all_df = pd.concat([aomic_df,ixi_df,oasis_df,sald_df])
# all_df = pd.concat([ixi_df,oasis_df,sald_df])
all_df.to_csv("../../Metadata/ao_ix_oa_sa.csv",index = False)