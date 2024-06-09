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
        self.conv3 = nn.Conv3d(in_channels=output_channel, out_channels=output_channel, kernel_size=3, 
                               padding=1, stride=1)

        self.bn1 = nn.BatchNorm3d(output_channel)
        self.bn2 = nn.BatchNorm3d(output_channel)
        self.bn3 = nn.BatchNorm3d(output_channel)
    
    def forward(self, cost):
        Y = F.relu(self.bn1(self.conv1(cost)))
        Y = F.relu(self.bn2(self.conv2(Y)))
        Y = F.relu(self.bn3(self.conv3(Y)))
        return Y
    
class Hourglass(nn.Module):
    def __init__(self, input_channel) -> None:
        super().__init__()
        self.conv_in1 = nn.Conv3d(in_channels=input_channel, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.conv_in2 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.bn_in1 = nn.BatchNorm3d(32)
        self.bn_in2 = nn.BatchNorm3d(32)

        # downsample layer
        self.downsample1 = nn.Sequential(downsampleblock(32,64,2))
        self.downsample2 = nn.Sequential(downsampleblock(64,64,2))
        self.downsample3 = nn.Sequential(downsampleblock(64,64,2))
        self.downsample4 = nn.Sequential(downsampleblock(64,128,2))

        # upsample layer
        self.upsample1 = nn.ConvTranspose3d(128, 64, kernel_size=3, padding=1, output_padding=1, stride=2)
        self.debn1 = nn.BatchNorm3d(64)
        self.upsample2 = nn.ConvTranspose3d(64, 64, kernel_size=3, padding=1, output_padding=1, stride=2)
        self.debn2 = nn.BatchNorm3d(64)
        self.upsample3 = nn.ConvTranspose3d(64, 64, kernel_size=3, padding=1, output_padding=1, stride=2)
        self.debn3 = nn.BatchNorm3d(64)
        self.upsample4 = nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2)
        self.debn4 = nn.BatchNorm3d(32)
        self.upsample5 = nn.ConvTranspose3d(32, 1, kernel_size=3, padding=1, output_padding=1, stride=2)

    def forward(self, cost):
        cost_in1 = F.relu(self.bn_in1(self.conv_in1(cost)))
        cost_in1 = F.relu(self.bn_in2(self.conv_in2(cost_in1)))

        #downsample
        cost_down1 = self.downsample1(cost_in1)
        cost_down2 = self.downsample2(cost_down1)   
        cost_down3 = self.downsample3(cost_down2)
        cost_down4 = self.downsample4(cost_down3)

        #upsample
        cost_up1 = self.debn1(self.upsample1(cost_down4))
        cost_up1 = F.relu( cost_up1 + cost_down3 , inplace=True)
        cost_up2 = self.debn2(self.upsample2(cost_up1))
        cost_up2 = F.relu( cost_up2 + cost_down2 , inplace=True)
        cost_up3 = self.debn3(self.upsample3(cost_up2))
        cost_up3 = F.relu( cost_up3 + cost_down1 , inplace=True)
        cost_up4 = self.debn4(self.upsample4(cost_up3))
        cost_up4 = F.relu( cost_up4 + cost_in1 , inplace=True)
        cost_out = self.upsample5(cost_up4)
        return cost_out