import torch
from torch import nn
from torch.nn import functional as F 

def convbn(in_channel, out_channel, kernel_size, stride, padding, dilation=1):
    return nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, 
                                   padding=((kernel_size//2)*dilation if dilation > 1 else padding),dilation=dilation), 
                         nn.BatchNorm2d(out_channel))


class BasciBlock(nn.Module):
    def __init__(self, input_channel, output_channel, dilation=1, use_1x1conv=False,
                 stride=1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=3, 
                               padding=dilation if dilation>1 else 1, dilation=dilation, stride=stride)
        self.conv2 = nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=3, 
                               padding=dilation if dilation>1 else 1, dilation=dilation, stride=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=1, 
                               padding=0, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.bn2 = nn.BatchNorm2d(output_channel)
    
    def forward(self, X):
        Y = F.leaky_relu(self.bn1(self.conv1(X)), negative_slope=0.2, inplace=True)
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y = Y + X
        return F.leaky_relu(Y, negative_slope=0.2, inplace=True)
    
class Stereonet_Extractor(nn.Module):
    def __init__(self, input_channel, output_channel, k) -> None:
        super().__init__()
        self.downsample = self._make_downsample_layer(in_channel=input_channel, out_channel=output_channel, num_layer=k, stride=2)
        
        self.resblock = self._make_res_layer(channel=output_channel, num_block=6)
        self.last_conv = nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=3, 
                               padding=1, stride=1)
    
    def _make_downsample_layer(self, in_channel, out_channel, num_layer, stride):
        downsample_list = []
        for i in range(num_layer):
            if i == 0:
                downsample_list.append(convbn(in_channel, out_channel, 5, stride, 2))
                downsample_list.append(nn.ReLU(inplace=True))
            else:
                downsample_list.append(convbn(out_channel, out_channel, 5, stride, 2))
                downsample_list.append(nn.ReLU(inplace=True))
        return nn.Sequential(*downsample_list)
    def _make_res_layer(self, channel, num_block):
        res_list = []
        for i in range(num_block):
                res_list.append(BasciBlock(channel, channel))
        return nn.Sequential(*res_list)    
    
    def forward(self, x):
        y1 = self.downsample(x)
        y2 = self.resblock(y1)
        y2 = self.last_conv(y2)
        return y2

class EdgeRefinement(nn.Module):
    def __init__(self,image_channel=3):
        super().__init__()
        self.conv1 = convbn(in_channel=image_channel+1, out_channel=32, kernel_size=3, stride=1, padding=1)
        self.resblock = nn.Sequential(BasciBlock(input_channel=32, output_channel=32, dilation=1, use_1x1conv=False, stride=1),
                                      BasciBlock(input_channel=32, output_channel=32, dilation=2, use_1x1conv=False, stride=1),
                                      BasciBlock(input_channel=32, output_channel=32, dilation=4, use_1x1conv=False, stride=1),
                                      BasciBlock(input_channel=32, output_channel=32, dilation=8, use_1x1conv=False, stride=1),
                                      BasciBlock(input_channel=32, output_channel=32, dilation=1, use_1x1conv=False, stride=1),
                                      BasciBlock(input_channel=32, output_channel=32, dilation=1, use_1x1conv=False, stride=1))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, stride=1)

    def forward(self, imgL, disp):
        input = torch.cat([imgL, disp], dim=1)
        output = self.conv1(input)
        output = F.leaky_relu(output, negative_slope=0.2, inplace=True)
        output = self.resblock(output)
        output = self.conv2(output)
        return disp+output
    
 