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
from monai.networks.nets.resnet import resnet50
from torch import nn
import pandas as pd

file = os.path.dirname(os.path.abspath(__file__))
os.chdir(file)

pin_memory = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv("../../Metadata/train_uniform.csv")
print(len(df))
# df = df[(df["age"] < 40) & (df["age"] > 0)]
df = df.sample(frac=1, random_state=42).reset_index(drop=True) ## random reshuffle of datarows
print(len(df))

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
# Define transforms
train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((128,128,128)), RandRotate90()])

val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((128,128,128))])


print("validation data loader")
# create a validation data loader
val_ds = ImageDataset(image_files=test_images_list , labels=test_ages_list, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=16, num_workers=2, pin_memory=pin_memory)

print("creating data loader")
# create a training data loader
train_ds = ImageDataset(image_files=training_images_list , labels=training_ages_list , transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2, pin_memory=pin_memory)


## pull resnet50
model = resnet50(pretrained=True, n_input_channels = 1, feed_forward = False , shortcut_type = "B", bias_downsample = False )

# print(medicalnet.parameters)
# Freeze all layers except the FC layer
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(2048, 1)

# # Print the requires_grad status of the parameters
# for name, param in medicalnet.named_parameters():
#     print(name, param.requires_grad)


if torch.cuda.is_available():
    model.cuda()
# It is important that we use nn.MSELoss for regression.

# # Load the saved model weights if needed to continue.
# checkpoint = torch.load("medicalnet_finetuneallages.pth")
# model.load_state_dict(checkpoint)

loss_function = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), 1e-4)

# start a typical PyTorch training
val_interval = 2
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
val_loss_values = []
writer = SummaryWriter()
max_epochs = 50

lowest_mse= sys.float_info.max
for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0

    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels.float())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_len = len(train_ds) // train_loader.batch_size
        print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
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
        val_loss_values.append(mse)

        if mse < lowest_mse:
            lowest_mse = mse
            lowest_mse_epoch = epoch + 1
            torch.save(model.state_dict(), "medicalnet_finetuneallages.pth")
            print("saved new best metric model")

        print(f"Current epoch: {epoch+1} current MAE: {mse:.4f} ")
        print(f"Best MAE: {lowest_mse:.4f} at epoch {lowest_mse_epoch}")
        writer.add_scalar("val_mae", mse, epoch + 1)

print(f"Training completed, lowest_mae: {lowest_mse:.4f} at epoch: {lowest_mse_epoch}")
writer.close()

print(epoch_loss_values)

# Convert list to DataFrame
epoch_loss_df = pd.DataFrame({'epoch_loss': epoch_loss_values})

# Save to CSV
epoch_loss_df.to_csv('epoch_loss_values_mednet_ctn.csv', index=False)

# Convert list to DataFrame
val_loss_df = pd.DataFrame({'epoch_loss': val_loss_values})

# Save to CSV
val_loss_df.to_csv('val_loss_values_mednet_ctn.csv', index=False)