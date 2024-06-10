import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from pathlib import Path

from utils.stereo_datasets import fetch_dataset
from validation import evaluation

class Tripleloss(nn.Module):
    def __init__(self):
        super(Tripleloss, self).__init__()
    def forward(self, gt, disp_pred, valid):
        loss = 0.5*F.smooth_l1_loss(disp_pred['disp1'][valid], gt[valid], reduction='mean') \
            + 0.7*F.smooth_l1_loss(disp_pred['disp2'][valid], gt[valid], reduction='mean') \
            + 0.9 * F.smooth_l1_loss(disp_pred['final_disp'][valid], gt[valid], reduction='mean')
        return loss

class Tripleloss_edge(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, gt, disp_pred, valid):
        loss = 0.5*F.smooth_l1_loss(disp_pred['disp1'][valid], gt[valid], reduction='mean') \
        + 0.7*F.smooth_l1_loss(disp_pred['disp2'][valid], gt[valid], reduction='mean') \
        + 0.9 * F.smooth_l1_loss(disp_pred['disp3'][valid], gt[valid], reduction='mean') \
        + F.smooth_l1_loss(disp_pred['final_disp'][valid], gt[valid], reduction='mean')
        return loss

class HierarchicalLoss(nn.Module):
    def __init__(self):
        super(HierarchicalLoss, self).__init__()
    def forward(self, gt, disp_pred, vaild):
        loss = 0
        for k in disp_pred:
            loss += torch.sum(torch.sqrt(torch.pow(gt[vaild] - disp_pred[k][vaild], 2) + 4) /2 - 1) 
        return loss/vaild.sum()

def export_config(path, args):
    # export the config
    config = vars(args)
    path = path + '/config.txt'
    with open(path, 'w+') as f:
        for k in config:
            str = k + ':' + f"{getattr(args, k)}"
            f.write(str + '\n')
    print("Config has been exported to {}".format(path))

def train_psm(model_name, net, dataset_name, batch_size, root, min_disp, max_disp, iters, init_lr, resize, device, log_dir='logs', save_frequency=None, require_validation=False, pretrain = None):
    print("Train on:", device)

    # tensorboard log file
    writer = SummaryWriter(log_dir=log_dir)

    # define model
    net.to(device)
    if pretrain is not None:
        net.load_state_dict(torch.load(pretrain), strict=True)
        print("Finish loading pretrain model!")
    else:
        net._init_params()
        print("Model parameters has been random initialize!")
    net.train()

    # fetch traning data
    train_loader = fetch_dataset(dataset_name = dataset_name, root = root,
                                batch_size = batch_size, resize = resize, 
                                min_disp = min_disp, max_disp = max_disp, mode = 'training')
    
    steps_per_iter = train_loader.__len__()
    num_steps = steps_per_iter * iters    
    
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in net.parameters()])))
    # initialize the optimizer and lr scheduler
    optimizer = torch.optim.AdamW(net.parameters(), lr=init_lr)
    # optimizer = torch.optim.RMSprop(net.parameters(), lr=init_lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, init_lr, num_steps + 100,
                                              pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')
    
    if model_name == 'PSMNet_Edge':
        criterion = Tripleloss_edge().to(device)
    elif model_name == 'StereoNet':
        criterion = HierarchicalLoss().to(device)
    elif model_name == 'GCNet':
        criterion = nn.SmoothL1Loss().to(device)
    else:
        criterion = Tripleloss().to(device)

    # start traning
    should_keep_training = True
    total_steps = 0
    while should_keep_training:
        for i_batch, data_blob in enumerate(tqdm(train_loader, ncols=80, desc=f'epoch:{int(total_steps/steps_per_iter) + 1}/{iters}')):
            optimizer.zero_grad()
            image1, image2, disp_gt, valid = [x.to(device) for x in data_blob]
            valid = valid.detach_()

            net.training
            disp_pred = net(image1, image2, min_disp, max_disp)
            assert net.training
            if model_name == 'GCNet':
                loss = criterion(disp_gt[valid], disp_pred['final_disp'][valid])
            else:
                loss = criterion(disp_gt, disp_pred, valid)
            loss.backward()
            optimizer.step()
            scheduler.step()
            del disp_pred

            # code of validation
            if total_steps % save_frequency == (save_frequency - 1):
                # load validation data 
                if require_validation:
                    metric = evaluation(net, dataset_name, root, device, min_disp, max_disp, batch_size, resize)

                    writer.add_scalars(main_tag="metric/epe", tag_scalar_dict = {'train()': metric['epe'][0]}, global_step=total_steps+1)
                    writer.add_scalars(main_tag="metric/epe", tag_scalar_dict = {'eval()':metric['epe'][1]}, global_step=total_steps+1)
                    writer.add_scalars(main_tag="metric/1px", tag_scalar_dict = {'train()': metric['1px'][0]*100}, global_step=total_steps+1)
                    writer.add_scalars(main_tag="metric/1px", tag_scalar_dict = {'eval()':metric['1px'][1]*100}, global_step=total_steps+1)
                    writer.add_scalars(main_tag="metric/2px", tag_scalar_dict = {'train()': metric['2px'][0]*100}, global_step=total_steps+1)
                    writer.add_scalars(main_tag="metric/2px", tag_scalar_dict = {'eval()':metric['2px'][1]*100}, global_step=total_steps+1)
                    writer.add_scalars(main_tag="metric/3px", tag_scalar_dict = {'train()':metric['3px'][0]*100}, global_step=total_steps+1)
                    writer.add_scalars(main_tag="metric/3px", tag_scalar_dict = {'eval()':metric['3px'][1]*100}, global_step=total_steps+1)
                    
                    # torch.save(net.state_dict(), f'{log_dir}/{model_name}_{dataset_name}_{total_steps+1}.pth')
                net.train()
            
            # write loss and lr to log
            writer.add_scalar(tag="loss/training loss", scalar_value=loss, global_step=total_steps+1)
            writer.add_scalar(tag="lr/lr", scalar_value=scheduler.get_last_lr()[0], global_step=total_steps+1)
            total_steps += 1

            if total_steps > num_steps:
                should_keep_training = False
                break

        if len(train_loader) >= 1000:
            cur_iter = int(total_steps/steps_per_iter)
            save_path = Path(f'{log_dir}/%d_epoch_{model_name}_%s.pth' % (cur_iter, dataset_name))
            torch.save(net.state_dict(), save_path)

    print("FINISHED TRAINING")

    final_outpath = f'{log_dir}/{model_name}_{dataset_name}.pth'
    torch.save(net.state_dict(), final_outpath)
    print("model has been save to path: ", final_outpath)