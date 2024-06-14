import torch
from torch import nn
from torch.nn import functional as F 

class downsampleblock(nn.Module):
    def __init__(self, input_channel, output_channel, stride=2) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=input_channel, out_channels=output_channel, kernel_size=3, 
                               padding=1, stride=stride)        
        self.conv2 = nn.Conv3d(in_channels=output_channel, out_channels=output_channel, kernel_size=3, 
                               padding=1, stride=1)

        self.bn1 = nn.BatchNorm3d(output_channel)
        self.bn2 = nn.BatchNorm3d(output_channel)
    
    def forward(self, cost):
        Y = F.leaky_relu(self.bn1(self.conv1(cost)))
        Y = self.bn2(self.conv2(Y))
        return Y

class hourglass(nn.Module):
    def __init__(self, input_channel, stride=2) -> None:
        super().__init__()
        self.downsample1_1 = downsampleblock(input_channel,input_channel*2,2)
        self.downsample1_2 = downsampleblock(input_channel*2,input_channel*2,2)

        self.upsample1_1 = nn.Sequential(nn.ConvTranspose3d(input_channel*2, input_channel*2, kernel_size=3, padding=1, output_padding=1, stride=2),nn.BatchNorm3d(input_channel*2))
        self.onebyone1_1 = nn.Conv3d(in_channels=input_channel*2, out_channels=input_channel*2, kernel_size=1, padding=0, stride=1)

        self.upsample1_2 = nn.Sequential(nn.ConvTranspose3d(input_channel*2, input_channel, kernel_size=3, padding=1, output_padding=1, stride=2),nn.BatchNorm3d(input_channel))
        self.onebyone1_2 = nn.Conv3d(in_channels=input_channel, out_channels=input_channel, kernel_size=1, padding=0, stride=1)

    def forward(self, cost):

        down1 = F.leaky_relu(self.downsample1_1(cost), negative_slope=0.1, inplace=True)
        down2 = F.leaky_relu(self.downsample1_2(down1), negative_slope=0.1, inplace=True)

        up1 = F.leaky_relu(self.upsample1_1(down2) + self.onebyone1_1(down1), negative_slope=0.1, inplace=True)
        up2 = F.leaky_relu(self.upsample1_2(up1) + self.onebyone1_2(cost), negative_slope=0.1, inplace=True)

        return up2
    
    
class StackHourglass(nn.Module):
    def __init__(self, input_channels) -> None:
        super().__init__()
        self.conv_in1 = nn.Sequential(nn.Conv3d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1, stride=1),nn.BatchNorm3d(32),nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                      nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1),nn.BatchNorm3d(32),nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.conv_in2 = nn.Sequential(nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1),nn.BatchNorm3d(32),nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                      nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1),nn.BatchNorm3d(32))

        # stage 1
        self.hourglass1 = hourglass(32)
        self.hourglass2 = hourglass(32)
        self.hourglass3 = hourglass(32)
        
        # output
        self.out_conv1 = nn.Sequential(nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1),
                                       nn.BatchNorm3d(32),nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                      nn.Conv3d(in_channels=32, out_channels=1, kernel_size=3, padding=1, stride=1))
        self.out_conv2 = nn.Sequential(nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1),
                                       nn.BatchNorm3d(32),nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                      nn.Conv3d(in_channels=32, out_channels=1, kernel_size=3, padding=1, stride=1))
        self.out_conv3 = nn.Sequential(nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1),
                                       nn.BatchNorm3d(32),nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                      nn.Conv3d(in_channels=32, out_channels=1, kernel_size=3, padding=1, stride=1))

    def forward(self, cost):
        cost = self.conv_in1(cost)
        cost = self.conv_in2(cost) + cost

        cost1 = self.hourglass1(cost)
        cost2 = self.hourglass2(cost1)
        cost3 = self.hourglass3(cost2)

        cost1 = self.out_conv1(cost1)
        cost2 = self.out_conv2(cost2)
        cost3 = self.out_conv3(cost3)

        return cost1, cost2, cost3
    