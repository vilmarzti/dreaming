from cv2 import circle
from numpy.core.fromnumeric import size
from segmentation.helper.create_model import create_unet
from segmentation.models import UNet
from torchsummary import summary

from time import time
import torch
from torch.nn.modules.loss import BCELoss

from segmentation.training.train import create_train

net = UNet(3, 2, 9, False, "sin")

net = net.to("cuda")

"""
difference = {}
min_difference = 300
min_size = 0
for size in range(250, 400):
    input = torch.zeros((16, 9, size, size), device="cuda")
    output = net(input)
    diff = size - output.shape[2]
    if diff < min_difference:
        print(output.shape)
        min_difference = diff
        min_size = size

print(min_size, min_difference)
"""


config = {
    "deepness": 2,
    "starting_multiplier": 3,
    "use_thin": False,
    "positional_encoding": None,
    "learning_rate": 0.01,
    "batch_size": 6
}


train_func = create_train(
    create_unet,
    252,
    None,
    True,
    False,
    "crop"
)

train_func(config)