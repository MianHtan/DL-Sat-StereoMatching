import torch
from torch import nn
from torch.nn import functional as F 
import math

from .Extractor import GC_Extractor
from .EncoderDecoder import Hourglass
from utils.cost_volume import concat_volume, SoftArgMax

class GCNet(nn.Module):
    def __init__(self, image_channel = 3):
        super().__init__()
        self.fea1 = GC_Extractor(image_channel, 32, 8)
        self.hourglass = Hourglass(input_channel=64)

    def forward(self, imgL, imgR, min_disp, max_disp):
        regression = SoftArgMax(min_disp, max_disp)
        regression = regression.to(imgL.device)
        #extract feature map
        featureL = self.fea1(imgL) 
        featureR = self.fea1(imgR) # shape -> C * H/2 * W/2

        # construct cost volume
        cost_vol = concat_volume(featureL, featureR, min_disp//2, max_disp//2) # shape -> B * 2C * (maxdisp-mindisp)/2 * H/2 * W/2

        # cost filtering
        cost_vol = self.hourglass(cost_vol) # shape -> B * 1 * (maxdisp-mindisp) * H * W
        # disparity regression
        disp = {}
        disp['final_disp'] = regression(cost_vol) # shape -> B * H * W
        return disp
    
    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()