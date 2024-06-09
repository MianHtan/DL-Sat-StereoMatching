from torch import nn
from torch.nn import functional as F 

class BasciBlock(nn.Module):
    def __init__(self, input_channel, output_channel, use_1x1conv=False,
                 stride=1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=3, 
                               padding=1, stride=stride)
        self.conv2 = nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=3, 
                               padding=1, stride=stride)
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
    
class GC_Extractor(nn.Module):
    def __init__(self, input_channel, output_channel, num_resblock) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=5, padding=2, stride=2)
        self.bn_in = nn.BatchNorm2d(output_channel)
        self.resblock = self._make_layer(num_channel=output_channel, num_resblock=num_resblock)
        self.conv_last = nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=3, padding=1, stride=1)
    
    def _make_layer(self, num_channel, num_resblock):
        resblk = []
        for i in range(num_resblock):
            resblk.append(BasciBlock(num_channel, num_channel))
        return nn.Sequential(*resblk)    
    
    def forward(self, x):
        x = self.conv_in(x)
        x = self.bn_in(x)
        x = F.relu(x)
        x = self.resblock(x)
        Y = self.conv_last(x)
        return Y