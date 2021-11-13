from torch._C import Value
from torch.nn.functional import interpolate
from torch.nn.modules.loss import BCELoss

from segmentation.data.dataset import TestDataset, TrainDataset 

from torch.utils.data import DataLoader

from ray import tune

import torch
import torch.optim as optim

import os

def crop_or_scale(outputs, targets, transform):
    """
        Helper function that checks whether the output of a model has the same dimension as the target
        If they don't the output gets either scaled, or cropped
    """
    if outputs.shape[1] != targets.shape[1] or outputs.shape[2] != outputs.shape[2]:
        if transform == "scale":
            outputs = interpolate(outputs, (targets.shape[1], targets.shape[1]), mode="bilinear", align_corners=False)
        elif transform == "crop":
            diff_y = (targets.shape[1] - outputs.shape[1]) // 2
            diff_x = (targets.shape[2] - outputs.shape[2]) // 2
            targets = targets[:, diff_y: diff_y + outputs.shape[1], diff_x: diff_x + outputs.shape[2]]
        else:
            raise ValueError(f"Outputs of the Model has not the same shape as target.\nOutput shape{outputs.shape} Target shape: {targets.shape}\nPlease provide the right transform argument in the training function")
    return outputs,targets 

def jaccard_index(pred_labels, target_labels, device="cpu"):
    """
        Computes the Jaccard index
    """
    # Cast to same type
    pred_labels = pred_labels.int()
    target_labels = target_labels.int()

    # Compute union and intersection
    union = torch.bitwise_or(pred_labels, target_labels).sum()
    intersection = torch.bitwise_and(pred_labels, target_labels).sum()

    # Exception when union is 0
    if union == 0:
        return torch.tensor(1.0, device=device)
    else:
        return intersection / union




# General params
def create_train(
    create_model,  # The function that creates a model (unet, cnn, etc) from a given config
    crop_size,     # Dataset parameter that says how big the images should get cropped
    cvt_flag,      # Conversion flag for the Dataset. Can be used to create a hsv or gray-scale image
    add_encoding,  # If the dataset should add sin/linear positional encoding
    use_tune=True, # Disable when running without tune hyper-paramtersearch
    transform=None # Either "crop" or "scale". Is used with to decide whether to crop/scale the outputs to the right size
):
    def train(config, checkpoint_dir=None):
        # Other params
        learning_rate = config["learning_rate"]
        batch_size = config["batch_size"]

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
            crop_size,
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

            # Training loop
            for i, data in enumerate(train_loader):
                inputs, labels= data

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = net(inputs)

                outputs = torch.squeeze(outputs)

                # either interpolate outputs or crop target if it does not have same size
                outputs, labels = crop_or_scale(outputs, labels, transform)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i > max_steps_train:
                    break

            val_losses = 0
            val_accuracy = 0
            val_j_index = 0
            net.eval()
            # Validation loop
            for i, (inputs, labels) in enumerate(valid_loader):
                with torch.no_grad():
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = net(inputs)
                    outputs = torch.squeeze(outputs, 1)
                    
                    # Either crop or scale the labels, oututs
                    outputs, labels = crop_or_scale(outputs, labels, transform)

                    # compute validation loss
                    val_loss = criterion(outputs, labels)
                    val_losses += val_loss.item()

                    # compute jaccard index
                    output_labels = (outputs > 0.5).type(torch.uint8)
                    val_j_index += jaccard_index(output_labels, labels, device=device)

                    # compute validation accuracy
                    accuracy = torch.mean((labels == output_labels).type(torch.float))
                    val_accuracy += accuracy.item()

            mean_train_loss = running_loss / max_steps_train
            mean_val_loss = val_losses / len(valid_loader)
            mean_val_acc = val_accuracy / len(valid_loader)
            mean_val_j = val_j_index / len(valid_loader)

            if tune and use_tune:
                with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                    ck_path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save((net.state_dict(), optimizer.state_dict()), ck_path)

                tune.report(val_loss=mean_val_loss, val_accuracy=mean_val_acc, val_j_index=mean_val_j, train_loss=mean_train_loss)
            else:
                print(f"Mean validation_loss:     {mean_val_loss}")    
                print(f"Mean validation accuracy: {mean_val_acc}")
                print(f"Mean validation j_index:  {mean_val_j}")
                print(f"Mean Training loss:       {mean_train_loss}")
    
    return train

def create_test_best(create_model, crop_size, cvt_flag, add_encoding, transform=None):
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

        # Get Dataloader
        valid_set = TestDataset(
            "/home/martin/Videos/ondrej_et_al/bf/segmentation/cnn/test_input",
            "/home/martin/Videos/ondrej_et_al/bf/segmentation/cnn/test_output",
            crop_size,
            cvt_flag,
            add_encoding
        )

        valid_loader = DataLoader(valid_set, batch_size=1)

        # Compute accuracy
        mean_accuracy = 0
        mean_j_index = 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(valid_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = net(inputs)
                outputs.squeeze(1)

                outputs, labels = crop_or_scale(outputs, labels, transform)

                output_labels = (outputs > 0.5).type(torch.uint8)
    
                j_index = jaccard_index(output_labels, labels, device=device)
                accuracy = torch.mean((labels == output_labels).type(torch.float))

                mean_j_index += j_index
                mean_accuracy += accuracy.item()

        mean_accuracy = mean_accuracy / len(valid_loader)
        mean_j_index = mean_j_index / len(valid_loader)

        print(f"Best trial on validation set - Accuracy {mean_accuracy} J-Index {mean_j_index}")

    return test_best_model