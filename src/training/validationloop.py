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
from torch import nn
from monai.networks.nets import Regressor
from monai.networks.nets import ResNet
import pandas as pd
from monai.networks.nets.resnet import resnet50

def init_mednet() -> monai.networks.nets.ResNet:
    ## pull resnet50
    model = resnet50(pretrained=False,n_input_channels = 1, feed_forward = False , shortcut_type = "B", bias_downsample = False )

    # print(medicalnet.parameters)
    # Freeze all layers except the FC layer
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(2048, 1)

    return model

file = os.path.dirname(os.path.abspath(__file__))
os.chdir(file)

if not os.getcwd() == "/media/backup_16TB/sean/Monai/src/training":
    os.chdir("./sean/Monai/src/training")
print(os.getcwd())
pin_memory = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
df = pd.read_csv("../../Metadata/ICBM_cleaned.csv")
print(len(df))
# df = df[(df["age"] >40 )]
df = df.sample(frac=1, random_state=42).reset_index(drop=True) ## random reshuffle of datarows
print(len(df))

# IXI dataset as a demo, downloadable from https://brain-development.org/ixi-dataset/
images = df['filepath']

# ages of subjects
ages = np.array(
df['age']
)
test_index= len(df) #int(np.floor(len(df)*0.2)) #or use 80/20 otherwise
train_index= len(df) #int(np.floor(len(df)*0.8))
training_images_list= list(images[:train_index])
test_images_list= list(images[-test_index:])
training_ages_list= list(ages[:train_index])
test_ages_list= list(ages[-test_index:])
# Define transforms
train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((128,128,128)), RandRotate90()])

val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((128,128,128))])


print("validation data loader")
# create a validation data loader
val_ds = ImageDataset(image_files=test_images_list , labels=test_ages_list, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=16, num_workers=2, pin_memory=pin_memory)

def init_scratch() -> monai.networks.nets.ResNet:
        
    # Define parameters
    block_type = "bottleneck"  # or "basic"
    layers = [3, 4, 6, 3]  # Example for ResNet-50
    block_inplanes = [64, 128, 256, 512]  # Standard ResNet feature sizes
    spatial_dims = 3  # 3D images
    n_input_channels = 1  # MRI is grayscale (single-channel)
    conv1_t_size = 7
    conv1_t_stride = 2
    num_classes = 1  # Age Regressor 

    # Instantiate ResNet for MRI images
    model = ResNet(
        block=block_type,
        layers=layers,
        block_inplanes=block_inplanes,
        spatial_dims=spatial_dims,
        n_input_channels=n_input_channels,  # Set to 1 for MRI
        conv1_t_size=conv1_t_size,
        conv1_t_stride=conv1_t_stride,
        num_classes=num_classes
    )

    return model

# model = Regressor(in_shape=[1, 128,128,128], out_shape=1, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2))

model = init_scratch()

model.load_state_dict(torch.load("./Models/scratch.pth",weights_only=True))
model.to(device)
print("successfully loaded model")
model.eval()
all_labels = []
all_val_outputs = []
for val_data in val_loader:
    val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
    all_labels.extend(val_labels.cpu().detach().numpy())
    with torch.no_grad():
        val_outputs = model(val_images)
        flattened_val_outputs = [val for sublist in val_outputs.cpu().detach().numpy() for val in sublist]
        all_val_outputs.extend(flattened_val_outputs)

mse = np.absolute(np.subtract(all_labels, all_val_outputs)).mean()
print(mse)
print(all_labels,all_val_outputs)
#%%
for index in range(10):
    print(f"actual age : {all_labels[index]},pred age : {all_val_outputs[index]}")


# %%
# Remove NaNs while keeping corresponding elements
# Convert lists to NumPy arrays
all_labels = np.array(all_labels, dtype=np.float64)
all_val_output = np.array(all_val_outputs, dtype=np.float64)

# Remove NaNs while keeping corresponding elements
mask = ~np.isnan(all_labels) & ~np.isnan(all_val_output)
filtered_labels = all_labels[mask]
filtered_outputs = all_val_output[mask]

import matplotlib.pyplot as plt
# Scatter plot
plt.figure(figsize=(6, 6))
plt.scatter(all_labels, all_val_outputs, color='blue', alpha=0.5, label='Predictions')

# Reference line (y = x) for perfect predictions
plt.plot(all_labels, all_labels, color='red', linestyle='--', label='Ideal')

# Labels and title
plt.xlabel("True Values (all_labels)")
plt.ylabel("Predicted Values (all_val_output)")
plt.title("Scatter Plot of Predictions vs. Ground Truth")
plt.legend()
plt.grid(True)

# Show plot
plt.show()
# %%
len(filtered_labels) 
len(filtered_outputs)
# Compute MAE
mae = np.mean(np.abs(filtered_labels - filtered_outputs))

print(f"Mean Absolute Error (MAE): {mae:.4f}")
# %%
