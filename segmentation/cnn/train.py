import cv2
import numpy as np
from torch.nn.modules.loss import BCELoss, MSELoss
from dataset import SegmentationDataset
from model import  CNNSegmentation

from torch.utils.data import DataLoader
from torchsummary import summary

import torch
import torch.nn as nn
import torch.optim as optim


batch_size = 32
device = "cuda:0"

train_set = SegmentationDataset("/home/martin/Videos/ondrej_et_al/bf/segmentation/cnn/train_input", "/home/martin/Videos/ondrej_et_al/bf/segmentation/cnn/train_output", 100)
valid_set = SegmentationDataset("/home/martin/Videos/ondrej_et_al/bf/segmentation/cnn/valid_input", "/home/martin/Videos/ondrej_et_al/bf/segmentation/cnn/valid_output", 100)

train_loader = DataLoader(train_set, batch_size, shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size, shuffle=True, pin_memory=True)

net = CNNSegmentation(5, 3, 3, 3, False, "sin", padding="zeros")
net = net.to(device)

summary(net, (9, 300, 300))

criterion = BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)


def train():
    for epoch in range(100):
        running_loss = 0
        net.train()
        for i, data in enumerate(train_loader):
            inputs, labels= data

            optimizer.zero_grad()

            inputs = inputs.to(device)

            labels = labels.to(device)

            outputs = net(inputs)
            outputs = torch.squeeze(outputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:
                print(f"{epoch} {running_loss/200}")
                running_loss = 0.0

            if i % 2000 == 1999:
                break

        with torch.no_grad():
            val_losses = 0
            val_accuracy = 0
            step = 0
            net.eval()
            for inputs, labels in valid_loader:
                if step >= 2000:
                    break
                inputs = inputs.to(device)
                labels = labels.to(device)

                output = net(inputs)
                output = torch.squeeze(output)

                # compute validation loss
                val_loss = criterion(outputs, labels)
                val_losses += val_loss.item()

                # compute validation accuracy
                accuracy = torch.mean((labels == (output > 0.5).type(torch.uint8)).type(torch.float))
                val_accuracy += accuracy.item()
                step += 1


            mean_val_loss = val_losses / 2000
            mean_val_acc = val_accuracy / 2000
            print(f"The mean validiation loss is {mean_val_loss}")
            print(f"The mean accuracy loss is {mean_val_acc}")

if __name__ == "__main__":
    train()