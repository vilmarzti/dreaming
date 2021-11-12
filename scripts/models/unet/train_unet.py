from cv2 import circle
from numpy.core.fromnumeric import size
from segmentation.models import UNet
from torchsummary import summary

from time import time
import torch
from torch.nn.modules.loss import BCELoss

net = UNet(3, 4, 9, True, "sin")

net = net.to("cuda")

for x in range(110, 400, 10):
    try:
        summary(net, (9, x, x))
        break
    except RuntimeError: 
        print(f"{x} does not work")