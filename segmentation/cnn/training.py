from math import floor
from torch.nn.functional import interpolate
from torch.nn.modules.loss import BCELoss
from torch.utils.data.dataset import random_split
from dataset import SegmentationDataset
from cnn import  CNNSegmentation

from torch.utils.data import DataLoader
from torchsummary import summary

import ray
from ray import tune

import torch
import torch.optim as optim

import os

# General params
crop_size = 100

def train(config, checkpoint_dir=None):
    # CNN params
    kernel_size = config["kernel_size"]
    input_channels = 3
    intermidiate_channels = config["intermidiate_channels"]
    num_layers = config["num_layers"]
    thin =  config["thin"]
    positional_encoding = config["positional_encoding"]
    padding = config["padding"]
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]

    # check device
    device = "cuda:0" if torch.cuda.is_available else "cpu"

    # CNN
    net = CNNSegmentation(kernel_size, input_channels, intermidiate_channels, num_layers, thin, positional_encoding, padding)
    net = net.to(device)

    criterion = BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # Train 
    train_set = SegmentationDataset("/home/martin/Videos/ondrej_et_al/bf/segmentation/cnn/train_input", "/home/martin/Videos/ondrej_et_al/bf/segmentation/cnn/train_output", crop_size)

    split_abs = int(len(train_set) * 0.8)
    train_subset, valid_subset = random_split(train_set,  [split_abs, len(train_set) - split_abs])

    train_loader = DataLoader(train_subset, batch_size, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_subset, batch_size, shuffle=True, num_workers=8)

    if checkpoint_dir:
        checkpoint = os.path_join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint)
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    for epoch in range(10):
        running_loss = 0
        net.train()
        for i, data in enumerate(train_loader):
            inputs, labels= data

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)

            # interpolate if we don't have same size
            outputs = interpolate(outputs, (crop_size, crop_size), mode="bilinear", align_corners=False) if not padding else outputs
            outputs = torch.squeeze(outputs)

            if list(outputs.shape) != list(labels.shape):
                ray.util.pdb.set_trace()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                break

        val_losses = 0
        val_accuracy = 0
        step = 0
        net.eval()
        for inputs, labels in valid_loader:
            with torch.no_grad():
                if step >= 2000:
                    break
                inputs = inputs.to(device)
                labels = labels.to(device)

                output = net(inputs)
                output = interpolate(output, (crop_size, crop_size), mode="bilinear", align_corners=False) if not padding else output
                output = torch.squeeze(output)

                # compute validation loss
                val_loss = criterion(outputs, labels)
                val_losses += val_loss.item()

                # compute validation accuracy
                accuracy = torch.mean((labels == (output > 0.5).type(torch.uint8)).type(torch.float))
                val_accuracy += accuracy.item()
                step += 1

        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            ck_path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), optimizer.state_dict()), ck_path)

        mean_val_loss = val_losses / 2000
        mean_val_acc = val_accuracy / 2000
        tune.report(loss=mean_val_loss, accuracy=mean_val_acc)

def test_best_model(best_trial):
    config = best_trial.config

    # setup params
    kernel_size = config["kernel_size"]
    input_channels = 3
    intermidiate_channels = config["intermidiate_channels"]
    num_layers = config["num_layers"]
    thin =  config["thin"]
    positional_encoding = config["positional_encoding"]
    padding = config["padding"]
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]

    device = "cuda:0" if torch.cuda.is_available else "cpu"

    # create net from params
    net = CNNSegmentation(kernel_size, input_channels, intermidiate_channels, num_layers, thin, positional_encoding, padding)
    net.to(device)

    # load model
    ck_path = os.path.join(best_trial.checkpoint.value, "checkpoint")
    model_state, optim_state = torch.load(ck_path)
    net.load_state_dict(model_state)

    # Get DAtaloader
    valid_set = SegmentationDataset("/home/martin/Videos/ondrej_et_al/bf/segmentation/cnn/valid_input", "/home/martin/Videos/ondrej_et_al/bf/segmentation/cnn/valid_output", crop_size)
    valid_loader = DataLoader(valid_set, batch_size=32, shuffle=True, num_workers=2)

    # Compute accuracy
    mean_accuracy = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(valid_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)
            outputs = interpolate(outputs, (crop_size, crop_size), mode="bilinear", align_corners=False) if not padding else outputs
 
            accuracy = torch.mean((labels == (outputs > 0.5).type(torch.uint8)).type(torch.float))
            mean_accuracy += accuracy.item()

            if i >= 22862:
                break

    mean_accuracy /= 22862 
    best_trial(f"Best trial on validation set: {mean_accuracy}")