import torch
from torch import nn
from torch.nn import functional as F 

class SPP(nn.Module):
    def __init__(self, input_channel):
        super().__init__()
        self.branch1 = nn.Sequential(nn.AvgPool2d((64,64), stride=(64,64)),
                                     nn.Conv2d(in_channels=input_channel, out_channels=32, kernel_size=3, padding=1, stride=1),nn.BatchNorm2d(32))
        self.branch2 = nn.Sequential(nn.AvgPool2d((32,32), stride=(32,32)),
                                     nn.Conv2d(in_channels=input_channel, out_channels=32, kernel_size=3, padding=1, stride=1),nn.BatchNorm2d(32))
        self.branch3 = nn.Sequential(nn.AvgPool2d((16,16), stride=(16,16)),
                                     nn.Conv2d(in_channels=input_channel, out_channels=32, kernel_size=3, padding=1, stride=1),nn.BatchNorm2d(32))
        self.branch4 = nn.Sequential(nn.AvgPool2d((8,8), stride=(8,8)),
                                     nn.Conv2d(in_channels=input_channel, out_channels=32, kernel_size=3, padding=1, stride=1),nn.BatchNorm2d(32))
        self.last_conv1 = nn.Sequential(nn.Conv2d(in_channels=input_channel+64+128, out_channels=128, kernel_size=3, padding=1, stride=1),nn.BatchNorm2d(128))
        self.last_conv2 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1, padding=0, stride=1)

    def forward(self, x_low, x):
        branch1 = F.relu(self.branch1(x))
        branch1 = F.interpolate(branch1, x.shape[2:4], mode = "bilinear")

        branch2 = F.relu(self.branch2(x))
        branch2 = F.interpolate(branch2, x.shape[2:4], mode = "bilinear")

        branch3 = F.relu(self.branch3(x))
        branch3 = F.interpolate(branch3, x.shape[2:4], mode = "bilinear")

        branch4 = F.relu(self.branch4(x))
        branch4 = F.interpolate(branch4, x.shape[2:4], mode = "bilinear")

        out = torch.cat((x_low, x, branch1, branch2, branch3, branch4), 1)
        out = F.relu(self.last_conv1(out))
        # out = F.relu(self.last_conv1(out))
        out = self.last_conv2(out)
        # out = F.relu(self.last_conv2(out))

        return out