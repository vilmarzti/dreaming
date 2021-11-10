from segmentation.models import UNet
from torchsummary import summary

net = UNet(3, 9)

net = net.to("cuda")

summary(net, (9, 100, 100))