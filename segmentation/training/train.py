from torch.nn.functional import interpolate
from torch.nn.modules.loss import BCELoss

from segmentation.data.dataset import TestDataset, TrainDataset 

from torch.utils.data import DataLoader

from ray import tune

import torch
import torch.optim as optim

import os

# General params
def create_train(create_model, crop_size, cvt_flag, add_encoding, use_tune=True, add_padding=None):
    def train(config, checkpoint_dir=None):
        # Other params
        learning_rate = config["learning_rate"]
        batch_size = config["batch_size"]
        padding = config["padding"] if "padding" in config else add_padding

        # check device
        device = "cuda:0" if torch.cuda.is_available else "cpu"

        # Model
        net = create_model(config)
        net = net.to(device)

        criterion = BCELoss()
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)

        # Train 
        train_set = TrainDataset(
            "/home/martin/Videos/ondrej_et_al/bf/segmentation/cnn/train_input",
            "/home/martin/Videos/ondrej_et_al/bf/segmentation/cnn/train_output",
            crop_size,
            cvt_flag,
            add_encoding
        )

        valid_set = TestDataset(
            "/home/martin/Videos/ondrej_et_al/bf/segmentation/cnn/valid_input",
            "/home/martin/Videos/ondrej_et_al/bf/segmentation/cnn/valid_output",
            5,
            cvt_flag,
            add_encoding
        )

        train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=8)
        valid_loader = DataLoader(valid_set, 1)

        if checkpoint_dir:
            checkpoint = os.path.join(checkpoint_dir, "checkpoint")
            model_state, optimizer_state = torch.load(checkpoint)
            net.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
        
        # Train on equal num of expamples no matter the batchsize
        # len(train_loader) = len(train_set)/ batch_size
        max_steps_train = len(train_loader) // (1000 / batch_size)

        for epoch in range(1000):
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
                if i > max_steps_train:
                    break

            val_losses = 0
            val_accuracy = 0
            net.eval()
            for i, (inputs, labels) in enumerate(valid_loader):
                with torch.no_grad():
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    output = net(inputs)

                    output = interpolate(output, (inputs.shape[2], inputs.shape[3]), mode="bilinear", align_corners=False) if not padding else output
                    output = torch.squeeze(output, 1)

                    # compute validation loss
                    val_loss = criterion(output, labels)
                    val_losses += val_loss.item()

                    # compute validation accuracy
                    accuracy = torch.mean((labels == (output > 0.5).type(torch.uint8)).type(torch.float))
                    val_accuracy += accuracy.item()

            mean_train_loss = running_loss / max_steps_train
            mean_val_loss = val_losses / len(valid_loader)
            mean_val_acc = val_accuracy / len(valid_loader)

            if tune and use_tune:
                with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                    ck_path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save((net.state_dict(), optimizer.state_dict()), ck_path)

                tune.report(val_loss=mean_val_loss, val_accuracy=mean_val_acc, train_loss=mean_train_loss)
            else:
                print(f"Mean validation_loss:     {mean_val_loss}")    
                print(f"Mean Training loss:       {mean_train_loss}")
                print(f"Mean validation accuracy: {mean_val_acc}")
    
    return train

def create_test_best(create_model, split_size, cvt_flag, add_encoding, add_padding=None):
    def test_best_model(best_trial):
        config = best_trial.config
        padding = config["padding"] if "padding" in config else add_padding

        device = "cuda:0" if torch.cuda.is_available else "cpu"

        # create net from params
        net = create_model(config)
        net.to(device)

        # load model
        ck_path = os.path.join(best_trial.checkpoint.value, "checkpoint")
        model_state, _ = torch.load(ck_path)
        net.load_state_dict(model_state)

        # Get DAtaloader
        valid_set = TestDataset(
            "/home/martin/Videos/ondrej_et_al/bf/segmentation/cnn/test_input",
            "/home/martin/Videos/ondrej_et_al/bf/segmentation/cnn/test_output",
            split_size,
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
                outputs = interpolate(outputs, (split_size, split_size), mode="bilinear", align_corners=False) if not padding else outputs
    
                accuracy = torch.mean((labels == (outputs > 0.5).type(torch.uint8)).type(torch.float))
                mean_accuracy += accuracy.item()

        mean_accuracy = mean_accuracy / len(valid_loader)
        best_trial(f"Best trial on validation set: {mean_accuracy}")

    return test_best_model