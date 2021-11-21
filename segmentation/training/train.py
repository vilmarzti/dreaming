import os
import cv2

import torch
import torch.optim as optim

from ray import tune

from torch.utils.data import DataLoader
from torch.nn.modules.loss import BCELoss
from torch.nn.functional import interpolate 

from segmentation.helper import preprocessing
from segmentation.data.dataset import TestDataset, TrainDataset 
from segmentation.helper.metrics import jaccard_index


def crop_or_scale(predictions, targets, transform="scale"):
    """
        Helper function that that shapes the predictions and targets into the same format.

    Args:
        predictions (torch.tensor): The predictions of size (B, C, H, W) 
            Where B is the number of Batches, C the number of Channels, H is height and W is width
        targets (torch.tensor):  The targets of size (B, C, H, W)
            Where B is the batch-size, C is the number of channels, H the height and W is the width
        transform (str, optional): Should be one of ["scale", "crop"]. If None is provided raises an error. Defaults to "scale".

    Raises:
        ValueError: If transform is None and the target and prediction shape doesn't match

    Returns:
        [(torch.Tensor, torch.Tensor)]: A dict of the transformed predictions and targets
    """
    if predictions.shape[2] != targets.shape[2] or predictions.shape[3] != targets.shape[3]:
        if transform == "scale":
            predictions = interpolate(predictions, (targets.shape[2], targets.shape[3]), mode="bilinear", align_corners=False)
        elif transform == "crop":
            diff_y = (targets.shape[2] - predictions.shape[2]) // 2
            diff_x = (targets.shape[3] - predictions.shape[3]) // 2
            targets = targets[:, :, diff_y: diff_y + predictions.shape[1], diff_x: diff_x + predictions.shape[2]]
        else:
            raise ValueError(f"predictions of the Model has not the same shape as target.\nOutput shape{predictions.shape} Target shape: {targets.shape}\nPlease provide the right transform argument in the training function")
    return predictions, targets 

def create_train(create_model, train_set, test_set, use_tune=True, transform=None):
    """ Creates a train function that can be called from ray.tune or started with a costum config.

    Args:
        create_model (function): A function that takes in a config-dict and generates a  corresponding model.
            For example create_cnn creates a cnn from a given config. Check out segmentation.helper.create_models for available methods.
        train_set (TrainDataset): A dataset with the train data
        test_set (TestDataset): A dataset with the test data.
        use_tune (bool, optional): Whether to use tune during execution of the returned train function. 
            Set this to False when executing training without tune. Defaults to True.
        transform (str, optional): How to transform the output of the model or the labels if their sizes mismatch. 
            Can be either "crop" for cropping the targets or "scale" for interpolating the predictions.
            Compare with method crop_or_scale. Defaults to None.

    Returns:
        A training function that runs the trainig for a specific model. The arguments to the function are a config file for creating
        the model and a checkpoint_dir if tune wants to resume from a checkpoint.
    """

    def train(config, checkpoint_dir=None):
        """Trains a network on a created model given a config.

        Args:
            config (dict): Contains the parameters of how to create the model. 
                Compare to the create_model parameter of the parent function.
            checkpoint_dir (str, optional): A model checkpoint-dir from which to load a model.
                The config of the model should correspond to the model in the checkpoint. Defaults to None.
        """
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

        train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=8)
        test_loader = DataLoader(test_set, 1)

        if checkpoint_dir:
            checkpoint = os.path.join(checkpoint_dir, "checkpoint")
            model_state, optimizer_state = torch.load(checkpoint)
            net.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
        
        # Train on equal num of expamples no matter the batchsize
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

                outputs = torch.squeeze(outputs, 1)
                labels = torch.squeeze(labels, 1)

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
            for i, (inputs, labels) in enumerate(test_loader):
                with torch.no_grad():
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # Compute output of net
                    outputs = net(inputs)
 
                    # Remove dimension
                    outputs = torch.squeeze(outputs, 1)
                    labels = torch.squeeze(labels, 1)                   

                    # Either crop or scale the labels, oututs
                    outputs, labels = crop_or_scale(outputs, labels, transform)

                    # compute validation loss
                    val_loss = criterion(outputs, labels)
                    val_losses += val_loss.item()

                    # compute jaccard index
                    output_labels = (outputs > 0.5).type(torch.uint8)
                    val_j_index += jaccard_index(output_labels, labels, device=device).detach().cpu().item()

                    # compute validation accuracy
                    accuracy = torch.mean((labels == output_labels).type(torch.float))
                    val_accuracy += accuracy.item()

            mean_train_loss = running_loss / max_steps_train
            mean_val_loss = val_losses / len(test_loader)
            mean_val_acc = val_accuracy / len(test_loader)
            mean_val_j = val_j_index / len(test_loader)

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

def create_test_best(create_model, crop_size, add_encoding, transform=None):
    """Generates a function for testing a model

    Args:
        create_model (function): A function that creates a model from a given config.
            See segmentation.helper.create_model for available functions
        crop_size (int or tuple of ints): Tells the test-dataset how much to crop the samples.
        add_encoding (bool): Add positional encoding to the datasets if true. If not no positional encoding is provided
        transform (str, optional): How to transform the output of the model or the labels if their sizes mismatch. 
            Can be either "crop" for cropping the targets or "scale" for interpolating the predictions.
            Compare with method crop_or_scale. Defaults to None.
    
    Returns:
        A function that test a model given a ray.tune trial

    """
    def test_best_model(best_trial):
        """ Tests a trial on a given test-set and print results.

        Args:
            best_trial (ray.tune.trial): The trial we want to test. The trial model should correspond to
                the create_model method supplied in the create_test_best function.
        """
        config = best_trial.config

        device = "cuda:0" if torch.cuda.is_available else "cpu"

        # create net from params
        net = create_model(config)
        net.to(device)

        # load model
        ck_path = os.path.join(best_trial.checkpoint.value, "checkpoint")
        model_state, _ = torch.load(ck_path)
        net.load_state_dict(model_state)

        # Create Dataset
        input_preprocess = [preprocessing.add_encoding] if add_encoding else []
        input_preprocess.append(preprocessing.subtract_mean)

        valid_set = TestDataset(
            [
                "/home/martin/Videos/ondrej_et_al/bf/segmentation/cnn/test_input",
                "/home/martin/Videos/ondrej_et_al/bf/segmentation/cnn/test_output",
            ],
            crop_size,
            read_flags=[
                cv2.IMREAD_COLOR,
                cv2.IMREAD_GRAYSCALE
            ],
            preprocess=[
                preprocessing.compose(input_preprocess),
                preprocessing.threshold
            ]
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
                
                outputs = torch.squeeze(outputs, 1)
                labels = torch.squeeze(labels, 1)

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