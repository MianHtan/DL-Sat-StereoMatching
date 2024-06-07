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
        Y = F.relu(self.bn1(self.conv1(cost)))
        Y = self.bn2(self.conv2(Y))
        return Y
    
class StackHourglass(nn.Module):
    def __init__(self, input_channels) -> None:
        super().__init__()
        self.conv_in1 = nn.Sequential(nn.Conv3d(in_channels=input_channels, out_channels=input_channels, kernel_size=3, padding=1, stride=1),nn.BatchNorm3d(input_channels),nn.ReLU(inplace=True),
                                      nn.Conv3d(in_channels=input_channels, out_channels=input_channels, kernel_size=3, padding=1, stride=1),nn.BatchNorm3d(input_channels),nn.ReLU(inplace=True))
        self.conv_in2 = nn.Sequential(nn.Conv3d(in_channels=input_channels, out_channels=input_channels, kernel_size=3, padding=1, stride=1),nn.BatchNorm3d(input_channels),nn.ReLU(inplace=True),
                                      nn.Conv3d(in_channels=input_channels, out_channels=input_channels, kernel_size=3, padding=1, stride=1),nn.BatchNorm3d(input_channels))

        # stage 1
        self.downsample1_1 = downsampleblock(input_channels,input_channels*2,2)
        self.downsample1_2 = downsampleblock(input_channels*2,input_channels*2,2)
        self.upsample1_1 = nn.Sequential(nn.ConvTranspose3d(input_channels*2, input_channels*2, kernel_size=3, padding=1, output_padding=1, stride=2),nn.BatchNorm3d(input_channels*2))
        self.upsample1_2 = nn.Sequential(nn.ConvTranspose3d(input_channels*2, input_channels, kernel_size=3, padding=1, output_padding=1, stride=2),nn.BatchNorm3d(input_channels))
        # stage 2
        self.downsample2_1 = downsampleblock(input_channels,input_channels*2,2)
        self.downsample2_2 = downsampleblock(input_channels*2,input_channels*2,2)
        self.upsample2_1 = nn.Sequential(nn.ConvTranspose3d(input_channels*2, input_channels*2, kernel_size=3, padding=1, output_padding=1, stride=2),nn.BatchNorm3d(input_channels*2))
        self.upsample2_2 = nn.Sequential(nn.ConvTranspose3d(input_channels*2, input_channels, kernel_size=3, padding=1, output_padding=1, stride=2),nn.BatchNorm3d(input_channels))
        # stage 3
        self.downsample3_1 = downsampleblock(input_channels,input_channels*2,2)
        self.downsample3_2 = downsampleblock(input_channels*2,input_channels*2,2)
        self.upsample3_1 = nn.Sequential(nn.ConvTranspose3d(input_channels*2, input_channels*2, kernel_size=3, padding=1, output_padding=1, stride=2),nn.BatchNorm3d(input_channels*2))
        self.upsample3_2 = nn.Sequential(nn.ConvTranspose3d(input_channels*2, input_channels, kernel_size=3, padding=1, output_padding=1, stride=2),nn.BatchNorm3d(input_channels))
        
        # output
        self.out_conv1 = nn.Sequential(nn.Conv3d(in_channels=input_channels, out_channels=input_channels, kernel_size=3, padding=1, stride=1),
                                       nn.BatchNorm3d(input_channels),nn.ReLU(inplace=True),
                                      nn.Conv3d(in_channels=input_channels, out_channels=1, kernel_size=3, padding=1, stride=1))
        self.out_conv2 = nn.Sequential(nn.Conv3d(in_channels=input_channels, out_channels=input_channels, kernel_size=3, padding=1, stride=1),
                                       nn.BatchNorm3d(input_channels),nn.ReLU(inplace=True),
                                      nn.Conv3d(in_channels=input_channels, out_channels=1, kernel_size=3, padding=1, stride=1))
        self.out_conv3 = nn.Sequential(nn.Conv3d(in_channels=input_channels, out_channels=input_channels, kernel_size=3, padding=1, stride=1),
                                       nn.BatchNorm3d(input_channels),nn.ReLU(inplace=True),
                                      nn.Conv3d(in_channels=input_channels, out_channels=1, kernel_size=3, padding=1, stride=1))

    def forward(self, cost):
        cost = self.conv_in1(cost)
        cost = self.conv_in2(cost) + cost

        #stage 1
        cost_down1_1 = F.relu(self.downsample1_1(cost))
        cost_down1_2 = F.relu(self.downsample1_2(cost_down1_1))
        cost_up1_1 = F.relu( self.upsample1_1(cost_down1_2) + cost_down1_1 )
        cost_up1_2 = self.upsample1_2(cost_up1_1) + cost

        #stage 2
        cost_down2_1 = F.relu(self.downsample2_1(cost_up1_2) + cost_up1_1)
        cost_down2_2 = F.relu(self.downsample2_2(cost_down2_1))
        cost_up2_1 = F.relu( self.upsample2_1(cost_down2_2) + cost_down1_1 )
        cost_up2_2 = self.upsample2_2(cost_up2_1) + cost

        #stage 3
        cost_down3_1 = F.relu(self.downsample3_1(cost_up1_2) + cost_up2_1)
        cost_down3_2 = F.relu(self.downsample3_2(cost_down3_1))
        cost_up3_1 = F.relu( self.upsample3_1(cost_down3_2) + cost_down1_1 )
        cost_up3_2 = self.upsample3_2(cost_up3_1) + cost

        out1 = self.out_conv1(cost_up1_2)
        out2 = self.out_conv2(cost_up2_2) + out1
        out3 = self.out_conv3(cost_up3_2) + out2

        return out1, out2, out3
    