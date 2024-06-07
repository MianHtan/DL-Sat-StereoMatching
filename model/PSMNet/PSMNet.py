import torch
from torch import nn
from torch.nn import functional as F 
from torch.autograd import Variable
import math

from .Extractor import PSM_Extractor
from .SPP import SPP
from .EncoderDecoder import StackHourglass
from .cost_volume import concat_volume

class SoftArgMax(nn.Module):
    def __init__(self, min_disp, max_disp):
        super().__init__()
        self.min_disp = min_disp
        self.max_disp = max_disp
        disp_num = torch.arange(min_disp, max_disp).type(torch.cuda.FloatTensor)
        disp_num = disp_num.unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(4)

        self.disp_regression = nn.Conv3d(1, 1, (max_disp-min_disp, 1, 1), 1, 0, bias=False)

        self.disp_regression.weight.data = disp_num
        self.disp_regression.weight.requires_grad = False

    def forward(self, cost_volume):
        # shape -> B * 1 * (maxdisp-mindisp) * H * W
        cost_softmax = F.softmax(cost_volume, dim=2)
        
        disp = self.disp_regression(cost_softmax) # shape -> B * 1 * 1 * H * W
        disp = disp.squeeze(1).squeeze(1) # shape -> B * H * W
        return disp

class PSMNet(nn.Module):
    def __init__(self, min_disp, max_disp, image_channel=3):
        super().__init__()
        self.min_disp = min_disp
        self.max_disp = max_disp
        self.range = max_disp - min_disp

        self.fea1 = PSM_Extractor(image_channel, 128)
        self.spp = SPP(128)
        self.hourglass = StackHourglass()
        self.regression = SoftArgMax(min_disp, max_disp)

    def forward(self, imgL, imgR):
        #extract feature map
        featureL1, featureL2 = self.fea1(imgL) 
        featureR1, featureR2 = self.fea1(imgR) # shape -> 32 * H/4 * W/4

        featureL = self.spp(featureL1, featureL2)
        featureR = self.spp(featureR1, featureR2)
        # construct cost volume
        cost_vol = concat_volume(featureL, featureR, self.min_disp//4, self.max_disp//4) # shape -> B * 64 * (maxdisp-mindisp)/4 * H/4 * W/4

        # cost filtering
        cost_vol1, cost_vol2, cost_vol3 = self.hourglass(cost_vol) # shape -> B * 1 * (maxdisp-mindisp)/4 * H/4 * W/4

        disp_pred = {}
        if self.training:
            # shape -> B * 1 * (maxdisp-mindisp) * H * W
            cost_vol1 = F.interpolate(cost_vol1, [self.range, imgL.size()[2], imgL.size()[3]], mode='trilinear')
            cost_vol2 = F.interpolate(cost_vol2, [self.range, imgL.size()[2], imgL.size()[3]], mode='trilinear')
            # disparity regression
            # disp_pred['disp1'] = self.softargmax(cost_vol1, min_disp, max_disp) # shape -> B * H * W
            # disp_pred['disp2'] = self.softargmax(cost_vol2, min_disp, max_disp) # shape -> B * H * W
            disp_pred['disp1'] = self.regression(cost_vol1)
            disp_pred['disp2'] = self.regression(cost_vol2)

        cost_vol3 = F.interpolate(cost_vol3, [self.range, imgL.size()[2], imgL.size()[3]], mode='trilinear')  
        # # disparity regression
        # disp_pred['final_disp'] = self.softargmax(cost_vol3, min_disp, max_disp)  # shape -> B * H * W
        disp_pred['final_disp'] = self.regression(cost_vol3)
            
        return disp_pred
    
    def softargmax(self, cost, min_disp, max_disp):
        cost_softmax = F.softmax(cost, dim = 2)
        vec = torch.arange(min_disp, max_disp).to(cost.device)
        vec = vec.unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(4)
        vec = vec.expand_as(cost_softmax).type_as(cost_softmax)
        disp = torch.sum(vec*cost_softmax, dim=2)
        disp = disp.squeeze(1)
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
                

