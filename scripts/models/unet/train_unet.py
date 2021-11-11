from segmentation.models import UNet
from torchsummary import summary

from time import time

net = UNet(3, 4, 10, False, "sin")
net = net.to("cuda")

start = time()
summary(net, (9, 300, 300))
duration = time() - start
print(f"The pass took {duration} seconds")