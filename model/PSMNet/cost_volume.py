import torch

def concat_volume(feaL:torch.tensor, feaR:torch.tensor, min_disp, max_disp) -> torch.tensor:
    B, C, H, W = feaL.shape
    device = feaL.device

    # feature map has been downsample, so disparity range should be devided by 2
    max_disp = max_disp
    min_disp = min_disp
    cost = torch.zeros(B, C*2, max_disp-min_disp, H, W).to(device)
    # cost[:, 0:C, :, :, :] = feaL.unsqueeze(2).repeat(1,1,max_disp-min_disp,1,1)

    for i in range(min_disp, max_disp):
        if i < 0:
            cost[:, 0:C, i-min_disp, :, 0:W+i] = feaL[:, :, :, 0:W+i]
            cost[:, C:, i-min_disp, :, 0:W+i] = feaR[:, :, :, -i:]
        if i >= 0:
            cost[:, 0:C, i-min_disp, :, i:] = feaL[:, :, :, i:]
            cost[:, C:, i-min_disp, :, i:] = feaR[:, :, :, :W-i]
    cost = cost.contiguous()
    return cost

def Gwc_volume(feaL:torch.tensor, feaR:torch.tensor, min_disp, max_disp, groups) -> torch.tensor:
    B, C, H, W = feaL.shape
    device = feaL.device
    max_disp = max_disp 
    min_disp = min_disp 
    cost = torch.zeros(B, groups, max_disp-min_disp, H, W).to(device)
    for i in range(min_disp, max_disp):
        if i >= 0:
            cost[:, :, i-min_disp, :, i:] = groupwise_correlation(feaL[:, :, :, i:], feaR[:, :, :, :W-i], groups)
        elif i < 0 :
            cost[:, :, i-min_disp, :, 0:W+i] = groupwise_correlation(feaL[:, :, :, 0:W+i], feaR[:, :, :, -i:], groups)
    cost = cost.contiguous()
    return cost

def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost
