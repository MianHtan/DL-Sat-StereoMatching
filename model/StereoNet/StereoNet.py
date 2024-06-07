import torch
from torch import nn
from torch.nn import functional as F 
from torch.autograd import Variable
import math

from .Extractor import Stereonet_Extractor, EdgeRefinement
from .cost_volume import concat_volume

def convbn3d(in_channel, out_channel, kernel_size, stride, padding):
    return nn.Sequential(nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding), 
                         nn.BatchNorm3d(out_channel))


class StereoNet(nn.Module):
    ''' 
    @para: k: downsample times
           Warning: refinement_time should larger than k 
    '''
    def __init__(self, image_channel=3, k=3, refinement_time=4):
        super().__init__()
        self.k = k
        self.refinement_time = refinement_time
        self.fea1 = Stereonet_Extractor(input_channel=image_channel, output_channel=32, k=k)
        self.cost_filter = nn.Sequential(convbn3d(64, 64, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True), 
                                         convbn3d(64, 64, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True), 
                                         convbn3d(64, 64, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True), 
                                         convbn3d(64, 64, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True), 
                                         nn.Conv3d(64, 1, 3, 1, 1))
        self.edgeRefine = nn.ModuleList()
        for _ in range(refinement_time):
            self.edgeRefine.append(EdgeRefinement(image_channel=image_channel))
        
    def forward(self, imgL, imgR, min_disp, max_disp):
        #extract feature map
        featureL = self.fea1(imgL) 
        featureR = self.fea1(imgR) # shape -> 32 * H/(2**k) * W/(2**k)

        # construct cost volume
        cost_vol = concat_volume(featureL, featureR, min_disp, max_disp) # shape -> B * 64 * (maxdisp-mindisp)/(2**k) * H/(2**k) * W/(2**k)
        # cost filtering
        cost_vol = self.cost_filter(cost_vol)
        # disparity regression
        disp1 = self.softargmax(cost_vol, min_disp=min_disp// (2**self.k), max_disp=max_disp// (2**self.k))

        # upsample disparity and record
        disp = F.interpolate(disp1, size=(imgL.shape[2:4]), mode='bilinear', align_corners=False)
        disp = disp * (2**self.k)
        disp_pred = {}
        disp_pred['disp0'] = disp
        
        # iterative refinement
        i = 0
        for edgerefine_layer in self.edgeRefine:
            if i < self.k:
                # upsample and refine
                disp1 = F.interpolate(disp1, scale_factor=2, mode='bilinear', align_corners=False)
                disp1 = disp1 * 2
                imgL1 = F.interpolate(imgL, size=(disp1.shape[2:4]), mode='bilinear', align_corners=False)
                disp1 = edgerefine_layer(imgL1, disp1)
                # record
                disp = F.interpolate(disp1, size=(imgL.shape[2:4]), mode='bilinear', align_corners=False)
                disp = disp * (2**(self.k-i-1))
                disp_pred[f'disp{i+1}'] = disp
                i = i+1
            else:
                disp1 = edgerefine_layer(imgL, disp1)
                if i == self.refinement_time-1:
                    disp_pred['final_disp'] = disp1
                else:
                    disp_pred[f'disp{i+1}'] = disp1
                i = i+1

        return disp_pred
    
    
    def softargmax(self, cost, min_disp, max_disp):
        cost_softmax = F.softmax(cost, dim = 2)
        vec = torch.arange(min_disp, max_disp).to(cost.device)
        vec = vec.unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(4)
        vec = vec.expand_as(cost_softmax).type_as(cost_softmax)
        disp = torch.sum(vec*cost_softmax, dim=2)
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

