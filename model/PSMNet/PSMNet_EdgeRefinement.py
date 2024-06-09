import torch
from torch import nn
from torch.nn import functional as F 
from torch.autograd import Variable
import math

from .Extractor import PSM_Extractor,EdgeRefinement
from .SPP import SPP
from .EncoderDecoder import StackHourglass
from utils.cost_volume import concat_volume, SoftArgMax


class PSMNet_Edge(nn.Module):
    def __init__(self, image_channel=3):
        super().__init__()
        self.fea1 = PSM_Extractor(image_channel, 128)
        self.spp = SPP(128)
        self.hourglass = StackHourglass()
        # self.edgeRefine = nn.ModuleList()
        self.edgeRefine = EdgeRefinement(image_channel=image_channel)

    def forward(self, imgL, imgR, min_disp, max_disp):
        regression = SoftArgMax(min_disp, max_disp)
        #extract feature map
        featureL1, featureL2 = self.fea1(imgL) 
        featureR1, featureR2 = self.fea1(imgR) # shape -> 32 * H/4 * W/4

        featureL = self.spp(featureL1, featureL2)
        featureR = self.spp(featureR1, featureR2)
        # construct cost volume
        cost_vol = concat_volume(featureL, featureR, min_disp, max_disp) # shape -> B * 64 * (maxdisp-mindisp)/4 * H/4 * W/4

        # cost filtering
        cost_vol1, cost_vol2, cost_vol3 = self.hourglass(cost_vol) # shape -> B * 1 * (maxdisp-mindisp)/4 * H/4 * W/4

        disp_pred = {}        
        if self.training:
            # shape -> B * 1 * (maxdisp-mindisp) * H * W
            cost_vol1 = F.interpolate(cost_vol1, [max_disp-min_disp, imgL.size()[2], imgL.size()[3]], mode='trilinear')
            cost_vol2 = F.interpolate(cost_vol2, [max_disp-min_disp, imgL.size()[2], imgL.size()[3]], mode='trilinear')

            disp1 = regression(cost_vol1) # shape -> B * H * W
            disp2 = regression(cost_vol2) # shape -> B * H * W
            disp_pred['disp1'] = disp1
            disp_pred['disp2'] = disp2

        cost_vol3 = F.interpolate(cost_vol3, [max_disp-min_disp, imgL.size()[2], imgL.size()[3]], mode='trilinear')   
        # # disparity regression
        disp3 = regression(cost_vol3) # shape -> B * H * W
        disp_pred['disp3'] = disp3
        
        disp4 = disp3.unsqueeze(1)
        disp4 = self.edgeRefine(imgL, disp4)
        disp4 = disp4.squeeze(1)
        disp_pred['final_disp'] = disp4        
            
        return disp_pred
    
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

