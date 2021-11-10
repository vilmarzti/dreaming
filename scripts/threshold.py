from torch.nn.functional import interpolate
from torch.utils.data.dataloader import DataLoader

from segmentation.data import TrainDataset, TestDataset 
from segmentation.models import RangeImage

from torch.nn.modules.loss import BCELoss

import torch
import torch.optim as optim

from cv2 import COLOR_BGR2HSV, COLOR_BGR2GRAY

import numpy as np

def train(gray=True):
    crop_size = 300
    device = "cuda:0" if torch.cuda.is_available else "cpu"
    flag = COLOR_BGR2GRAY if gray else COLOR_BGR2HSV
    lr = 0.0001
    batch_size = 10
    max_steps = 100

    # CNN
    net = RangeImage(1, 4) if gray else RangeImage(3, 4)
    net = net.to(device)

    criterion = BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    train_set = TrainDataset("/home/martin/Videos/ondrej_et_al/bf/segmentation/cnn/train_input", "/home/martin/Videos/ondrej_et_al/bf/segmentation/cnn/train_output", crop_size, flag, False)
    valid_set = TestDataset("/home/martin/Videos/ondrej_et_al/bf/segmentation/cnn/valid_input", "/home/martin/Videos/ondrej_et_al/bf/segmentation/cnn/valid_output", 5, flag, False)

    train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_set, 1, shuffle=True, num_workers=8)

    for epoch in range(5):
        train_loss = 0
        train_accuracy = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)

            outputs = interpolate(outputs, (crop_size, crop_size), mode="bilinear", align_corners=False)
            outputs = torch.squeeze(outputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            accuracy = torch.mean((labels == (outputs > 0.5).type(torch.uint8)).type(torch.float))
            train_accuracy += accuracy.item()
            train_loss += loss.item()

            if i % (max_steps / 2) == (max_steps / 2) - 1:
                train_loss /= (max_steps / 2)
                train_accuracy /= (max_steps / 2)
                print(f"Train loss: {train_loss} Train accuracy: {train_accuracy}")
            
            if i >= max_steps:
                break
        
        with torch.no_grad():
            valid_accuracy = 0
            valid_loss = 0
            for i, (inputs, labels) in enumerate(valid_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = net(inputs)
                outputs = interpolate(outputs, (inputs.shape[2], inputs.shape[3]), mode="bilinear", align_corners=False)
                outputs = torch.squeeze(outputs)

                loss = criterion(outputs, labels)

                accuracy = torch.mean((labels == (outputs > 0.5).type(torch.uint8)).type(torch.float))

                valid_accuracy += accuracy.item()
                valid_loss += loss.item()

            valid_accuracy /= len(valid_loader)
            valid_loss /= len(valid_loader) 
            print(f"Validation loss: {valid_loss} Validation accuracy: {valid_accuracy}")

    
    if gray:
        torch.save(net.state_dict(), "data/models/threshold_gray.pth")
    else:
        torch.save(net.state_dict(), "data/models/threshold_hsv.pth")