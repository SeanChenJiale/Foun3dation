import SimpleITK as sitk
import glob
import pandas as pd
import os 

AOMIC_meta = pd.read_csv("AOMIC_cleaned.csv")
SALD_meta = pd.read_csv("SALD_cleaned.csv")
IXI_meta = pd.read_csv("IXI_cleaned.csv")