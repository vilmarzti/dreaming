from torch.utils.data.dataloader import DataLoader
from segmentation.data.dataset import SegmentationDataset
from segmentation.threshold.threshold import RangeImage

from torch.nn.modules.loss import BCELoss

import torch
import torch.optim as optim

import cv2
from cv2 import COLOR_BGR2HSV, COLOR_BGR2GRAY

import numpy as np

def train(gray=True):
    crop_size = 300
    device = "cuda:0" if torch.cuda.is_available else "cpu"
    flag = COLOR_BGR2GRAY if gray else COLOR_BGR2HSV
    lr = 0.01
    batch_size = 32
    max_steps = 4000

    # CNN
    net = RangeImage(1) if gray else RangeImage(3)
    net = net.to(device)

    criterion = BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    train_set = SegmentationDataset("/home/martin/Videos/ondrej_et_al/bf/segmentation/cnn/train_input", "/home/martin/Videos/ondrej_et_al/bf/segmentation/cnn/train_output", crop_size, flag, False)
    valid_set = SegmentationDataset("/home/martin/Videos/ondrej_et_al/bf/segmentation/cnn/valid_input", "/home/martin/Videos/ondrej_et_al/bf/segmentation/cnn/valid_output", crop_size, flag, False)

    train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_set, batch_size, shuffle=True, num_workers=8)

    for epoch in range(2):
        train_loss = 0
        train_accuracy = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            outputs = torch.squeeze(outputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            accuracy = torch.mean((labels == (outputs > 0.5).type(torch.uint8)).type(torch.float))
            train_accuracy += accuracy.item()
            train_loss += loss.item()

            if i % (max_steps / 10) == (max_steps / 10) - 1:
                train_loss /= (max_steps / 10)
                train_accuracy /= (max_steps / 10)
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
                outputs = torch.squeeze(outputs)

                loss = criterion(outputs, labels)

                accuracy = torch.mean((labels == (outputs > 0.5).type(torch.uint8)).type(torch.float))

                valid_accuracy += accuracy.item()
                valid_loss += loss.item()

                if i >= max_steps:
                    valid_accuracy /= max_steps
                    valid_loss /= max_steps
                    print(f"Validation loss: {valid_loss} Validation accuracy: {valid_accuracy}")
                    break

    
    if gray:
        torch.save(net.state_dict(), "data/models/threshold_gray.pth")
    else:
        torch.save(net.state_dict(), "data/models/threshold_hsv.pth")