from math import floor
from torch.nn.functional import interpolate
from torch.nn.modules.loss import BCELoss
from torch.utils.data.dataset import random_split

from segmentation.data.dataset import SegmentationDataset
from segmentation.helper import create_cnn

from torch.utils.data import DataLoader

import ray
from ray import tune

import torch
import torch.optim as optim

import os

# General params
def create_train(create_model, crop_size, cvt_flag, add_encoding):
    def train(config, checkpoint_dir=None):
        # Other params
        learning_rate = config["learning_rate"]
        batch_size = config["batch_size"]
        padding = config["padding"]

        # check device
        device = "cuda:0" if torch.cuda.is_available else "cpu"

        # Model
        net = create_model(config)

        net = net.to(device)

        criterion = BCELoss()
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)

        # Train 
        train_set = SegmentationDataset(
            "/home/martin/Videos/ondrej_et_al/bf/segmentation/cnn/train_input",
            "/home/martin/Videos/ondrej_et_al/bf/segmentation/cnn/train_output",
            crop_size,
            cvt_flag,
            add_encoding
        )

        split_abs = int(len(train_set) * 0.7)
        train_subset, valid_subset = random_split(train_set,  [split_abs, len(train_set) - split_abs])

        train_loader = DataLoader(train_subset, batch_size, shuffle=True, num_workers=8)
        valid_loader = DataLoader(valid_subset, batch_size, shuffle=True, num_workers=8)

        if checkpoint_dir:
            checkpoint = os.path_join(checkpoint_dir, "checkpoint")
            model_state, optimizer_state = torch.load(checkpoint)
            net.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
        
        max_steps = len(train_loader) / 100

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

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i > max_steps:
                    break

            val_losses = 0
            val_accuracy = 0
            net.eval()
            for inputs, labels in valid_loader:
                with torch.no_grad():
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

            with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                ck_path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((net.state_dict(), optimizer.state_dict()), ck_path)

            mean_val_loss = val_losses / len(valid_loader)
            mean_val_acc = val_accuracy / len(valid_loader)
            tune.report(loss=mean_val_loss, accuracy=mean_val_acc, training_loss=running_loss/max_steps)
    
    return train

def create_test_best(create_model, crop_size, cvt_flag, add_encoding):
    def test_best_model(best_trial):
        config = best_trial.config

        device = "cuda:0" if torch.cuda.is_available else "cpu"

        # create net from params
        net = create_model(config)
        net.to(device)

        # load model
        ck_path = os.path.join(best_trial.checkpoint.value, "checkpoint")
        model_state, _ = torch.load(ck_path)
        net.load_state_dict(model_state)

        # Get DAtaloader
        valid_set = SegmentationDataset(
            "/home/martin/Videos/ondrej_et_al/bf/segmentation/cnn/valid_input",
            "/home/martin/Videos/ondrej_et_al/bf/segmentation/cnn/valid_output",
            crop_size,
            cvt_flag,
            add_encoding
        )
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

        mean_accuracy = mean_accuracy / len(valid_loader)
        best_trial(f"Best trial on validation set: {mean_accuracy}")

    return test_best_model