#%%
import os
import sys
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import monai
from monai.data import DataLoader, ImageDataset
from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    RandRotate90,
    Resize,
    ScaleIntensity,
)
from monai.networks.nets import Regressor
from monai.networks.nets import ResNet
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

file = os.path.dirname(os.path.abspath(__file__))
os.chdir(file)
if not os.getcwd() == "/media/backup_16TB/sean/Monai/src/training":
    os.chdir("./sean/Monai/src/training/")
    
pin_memory = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv("../../Metadata/ao_ix_oa_sa.csv")
print(len(df))
df = df[(df["age"] < 40) & (df["age"] > 0)]
df = df.sample(frac=1,random_state=42).reset_index(drop=True) ## random reshuffle of datarows
print(len(df))
print(df.head())
# IXI dataset as a demo, downloadable from https://brain-development.org/ixi-dataset/
images = df['filepath']

# ages of subjects
ages = np.array(
df['age']
)
test_index= int(np.floor(len(df)*0.2))
train_index= int(np.floor(len(df)*0.8))
training_images_list= list(images[:train_index])
test_images_list= list(images[-test_index:])
training_ages_list= list(ages[:train_index])
test_ages_list= list(ages[-test_index:])

# Define bins
bins = [(15, 20), (20, 25), (25, 30), (30, 35), (35, 40)]

# Group data into bins
binned_dfs = {b: df[(df["age"] >= b[0]) & (df["age"] < b[1])] for b in bins}

# Find the minimum count across bins
min_count = min(len(binned_df) for binned_df in binned_dfs.values())

# Resample each bin to match min_count
resampled_dfs = []
for b, binned_df in binned_dfs.items():
    if len(binned_df) > min_count:
        resampled_dfs.append(binned_df.sample(min_count, random_state=42))  # Downsample
    else:
        resampled_dfs.append(binned_df.sample(min_count, replace=True, random_state=42))  # Upsample

# Combine resampled data
train_df = pd.concat(resampled_dfs).reset_index(drop=True)

# Plot original vs. resampled distribution
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Assuming df and train_df are already defined
val_df = df[~df['filepath'].isin(train_df['filepath'])].drop_duplicates(subset=['filepath'])



sns.histplot(df["age"], bins=30, kde=True, color="blue", ax=ax[0])
ax[0].set_title(f"Original Age Distribution  {len(df)} ")

sns.histplot(train_df["age"], bins=30, kde=True, color="green", ax=ax[1])
ax[1].set_title(f"Uniform Age Distribution train, {len(train_df)} ")

# sns.histplot(val_df["age"], bins=30, kde=True, color="purple", ax=ax[2])
# ax[2].set_title(f"Uniform Age Distribution val, {len(val_df)} ")

plt.show()

# train_df.to_csv("../../Metadata/train_uniform.csv",index=False)
# val_df.to_csv("../../Metadata/val.csv",index=False)


# %%
