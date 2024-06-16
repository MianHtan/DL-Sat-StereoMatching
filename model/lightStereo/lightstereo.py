import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from utils.cost_volume import concat_volume, Gwc_volume
from utils.cost_volume import SoftArgMax

import math

from .backbone import Backbone, FPNLayer, BasicConv2d, BasicDeconv2d

from .aggregation import Aggregation

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, padding, dilation):
        super(BasicBlock, self).__init__()
        self.conv1 = BasicConv2d(in_channels=in_planes, out_channels=out_planes,
                                 norm_layer=nn.BatchNorm2d,
                                 act_layer=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True),
                                 kernel_size=3, stride=stride, padding=padding, dilation=dilation)

        self.conv2 = BasicConv2d(in_channels=out_planes, out_channels=out_planes,
                                 norm_layer=nn.BatchNorm2d, act_layer=None,
                                 kernel_size=3, stride=1, padding=padding, dilation=dilation)

        self.final_act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += x
        out = self.final_act(out)
        return out

def context_upsample(disp_low, up_weights, scale_factor=4):
    # disp_low [b,1,h,w]
    # up_weights [b,9,4*h,4*w]

    b, c, h, w = disp_low.shape
    disp_unfold = F.unfold(disp_low, kernel_size=3, dilation=1, padding=1)  # [bz, 3x3, hxw]
    disp_unfold = disp_unfold.reshape(b, -1, h, w)  # [bz, 3x3, h, w]
    disp_unfold = F.interpolate(disp_unfold, (h * scale_factor, w * scale_factor), mode='nearest')  # [bz, 3x3, 4h, 4w]
    disp = (disp_unfold * up_weights).sum(1)  # # [bz, 4h, 4w]

    return disp

class Refinement(nn.Module):
    def __init__(self):
        super(Refinement, self).__init__()

        # Original StereoNet: left, disp
        self.conv = BasicConv2d(4, 32, kernel_size=3, stride=1, padding=1,
                                norm_layer=nn.BatchNorm2d,
                                act_layer=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True))

        self.dilation_list = [2]
        self.dilated_blocks = nn.ModuleList()

        for dilation in self.dilation_list:
            self.dilated_blocks.append(BasicBlock(32, 32, stride=1,
                                                  padding=dilation, dilation=dilation))

        self.dilated_blocks = nn.Sequential(*self.dilated_blocks)

        self.final_conv = nn.Conv2d(32, 1, 3, 1, 1)

    def forward(self, low_disp, left_img):
        """Upsample low resolution disparity prediction to
        corresponding resolution as image size
        Args:
            low_disp: [B, 1, H, W]
            left_img: [B, 3, H, W]
        """
        scale_factor = left_img.size(-1) / low_disp.size(-1)
        if scale_factor == 1.0:
            disp = low_disp
        else:
            disp = F.interpolate(low_disp, size=left_img.size()[-2:], mode='bilinear', align_corners=False)
            disp = disp * scale_factor  # scale correspondingly

        concat = torch.cat((disp, left_img), dim=1)  # [B, 4, H, W]
        out = self.conv(concat)
        out = self.dilated_blocks(out)
        residual_disp = self.final_conv(out)

        disp = F.relu(disp + residual_disp, inplace=True)  # [B, 1, H, W]

        return disp


class LightStereo(nn.Module):
    def __init__(self):
        super().__init__()
        self.left_att = True

        # backbobe
        self.backbone = Backbone()
        
        # aggregation
        self.cost_agg = Aggregation(in_channels=48,
                                    left_att=self.left_att,
                                    blocks=[8, 16, 32],
                                    expanse_ratio=8)

        # disp refine
        self.refine_1 = nn.Sequential(
            BasicConv2d(48, 24, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.InstanceNorm2d, act_layer=nn.LeakyReLU),
            BasicConv2d(24, 24, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.InstanceNorm2d, act_layer=nn.ReLU))

        self.stem_2 = nn.Sequential(
            BasicConv2d(3, 16, kernel_size=3, stride=2, padding=1,
                        norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU),
            BasicConv2d(16, 16, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU))
        self.refine_2 = FPNLayer(24, 16)

        self.refine_3 = BasicDeconv2d(16, 9, kernel_size=4, stride=2, padding=1)

    def forward(self, image1, image2, min_disp, max_disp):
        assert (max_disp-min_disp == 192), 'LightStereo only support disp range within 192'
        
        regression = SoftArgMax(min_disp//4, max_disp//4)
        regression.to(image1.device)

        features_left = self.backbone(image1)
        features_right = self.backbone(image2)
        gwc_volume = Gwc_volume(features_left[0], features_right[0], min_disp//4, max_disp // 4, 1)
        gwc_volume = gwc_volume.squeeze(1)
        encoding_volume = self.cost_agg(gwc_volume, features_left)  # [bz, 1, max_disp/4, H/4, W/4]

        init_disp = regression(encoding_volume)  # [bz, 1, H/4, W/4]
        init_disp = init_disp.unsqueeze(1)

        xspx = self.refine_1(features_left[0])
        xspx = self.refine_2(xspx, self.stem_2(image1))
        xspx = self.refine_3(xspx)
        spx_pred = F.softmax(xspx, 1)  # [bz, 9, H, W]
        disp_pred = context_upsample(init_disp * 4., spx_pred.float()).unsqueeze(1)  # # [bz, 1, H, W]

        result = {'final_disp': disp_pred.squeeze(1)}

        if self.training:
            disp_4 = F.interpolate(init_disp, image1.shape[2:], mode='bilinear', align_corners=False)
            disp_4 *= 4
            result['disp_4'] = disp_4.squeeze(1)

        return result

    def get_loss(self, model_pred, disp_gt, valid):

        disp_pred = model_pred['final_disp']
        loss = 1.0 * F.smooth_l1_loss(disp_pred[valid], disp_gt[valid], reduction='mean')

        disp_4 = model_pred['disp_4']
        loss += 0.3 * F.smooth_l1_loss(disp_4[valid], disp_gt[valid], reduction='mean')

        return loss
    
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

    def _disable_batchnorm_tracking(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
