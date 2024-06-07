import torch
from torch import nn
from torch.nn import functional as F 

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
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y = Y + X
        return F.relu(Y)
    
class PSM_Extractor(nn.Module):
    def __init__(self, input_channel, output_channel) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=32, kernel_size=3, padding=1, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(32)
        
        self.resblock1 = self._make_layer(in_channel=32, out_channel=32, num_resblock=3, stride=1, dilation = 1)
        self.resblock2 = self._make_layer(in_channel=32, out_channel=64, num_resblock=16, stride=2, dilation = 1)
        self.resblock3 = self._make_layer(in_channel=64, out_channel=output_channel, num_resblock=3, stride=1, dilation = 2)
        self.resblock4 = self._make_layer(in_channel=output_channel, out_channel=output_channel, num_resblock=3, stride=1, dilation = 4)
    
    def _make_layer(self, in_channel, out_channel, num_resblock, stride, dilation):
        resblk = []
        for i in range(num_resblock):
            if (i == 0) and ((in_channel != out_channel) or (stride != 1)):
                resblk.append(BasciBlock(in_channel, out_channel, stride=stride, dilation = dilation, use_1x1conv=True))
            else:
                resblk.append(BasciBlock(out_channel, out_channel, dilation = dilation))
        return nn.Sequential(*resblk)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
    
        x = self.resblock1(x)
        y1 = self.resblock2(x)
        y2 = self.resblock3(y1)
        y2 = self.resblock4(y2)
        return y1, y2
    
def convbn(in_channel, out_channel, kernel_size, stride, padding, dilation=1):
    return nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, 
                                   padding=((kernel_size//2)*dilation if dilation > 1 else padding),dilation=dilation), 
                         nn.BatchNorm2d(out_channel))

class EdgeRefinement(nn.Module):
    def __init__(self,image_channel=3):
        super().__init__()
        self.conv1 = convbn(in_channel=image_channel+1, out_channel=32, kernel_size=3, stride=1, padding=1)
        self.resblock = nn.Sequential(BasciBlock(input_channel=32, output_channel=32, dilation=1, use_1x1conv=False, stride=1),
                                      BasciBlock(input_channel=32, output_channel=32, dilation=2, use_1x1conv=False, stride=1),
                                      BasciBlock(input_channel=32, output_channel=32, dilation=4, use_1x1conv=False, stride=1),
                                      BasciBlock(input_channel=32, output_channel=32, dilation=1, use_1x1conv=False, stride=1))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, stride=1)

    def forward(self, imgL, disp):
        input = torch.cat([imgL, disp], dim=1)
        output = self.conv1(input)
        output = F.leaky_relu(output, negative_slope=0.2, inplace=True)
        output = self.resblock(output)
        output = self.conv2(output)
        return disp+output