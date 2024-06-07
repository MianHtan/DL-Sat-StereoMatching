import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from pathlib import Path

from utils.stereo_datasets import fetch_dataset
from StereoNet.StereoNet import StereoNet

class HierarchicalLoss(nn.Module):
    def __init__(self):
        super(HierarchicalLoss, self).__init__()
    def forward(self, gt, pred_list, refinement_times, vaild):
        loss = 0
        count = len(torch.nonzero(vaild))
        for i in range(refinement_times+1):
            loss += torch.sum(torch.sqrt(torch.pow(gt[vaild] - pred_list[i][vaild], 2) + 4) /2 - 1) 
        return loss/count
    
class HierarchicalLoss_SmoothL1(nn.Module):
    def __init__(self):
        super(HierarchicalLoss_SmoothL1, self).__init__()
    def forward(self, gt, pred_list, refinement_times, vaild):
        loss = 0
        for i in range(refinement_times+1):
            loss += F.smooth_l1_loss(pred_list[i][vaild], gt[vaild], reduction='mean')
        return loss

def train(net, dataset_name, batch_size, root, min_disp, max_disp, refinement_times, iters, init_lr, resize, 
          device, save_frequency=None, require_validation=False, log_dir='logs', pretrain = None):
    print("Train on:", device)
    Path("training_checkpoints").mkdir(exist_ok=True, parents=True)

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
    optimizer = torch.optim.AdamW(net.parameters(), lr=init_lr, weight_decay=0.0005)
    # optimizer = torch.optim.RMSprop(net.parameters(), lr=init_lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, init_lr, num_steps + 100,
                                              pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')
    
    criterion = HierarchicalLoss().to(device)
    # criterion = HierarchicalLoss_SmoothL1().to(device)

    # start traning
    should_keep_training = True
    total_steps = 0
    while should_keep_training:
        # print('--- start new epoch ---')
        for i_batch, data_blob in enumerate(tqdm(train_loader, ncols=80, desc=f"Epoch:{int(total_steps/steps_per_iter)+1}/{iters}")):
            optimizer.zero_grad()
            image1, image2, disp_gt, valid = [x.to(device) for x in data_blob]
            valid = valid.detach_()

            net.training
            pred_list = net(image1, image2, min_disp, max_disp)
            assert net.training

            loss = criterion(disp_gt, pred_list, refinement_times, valid)
            loss.backward()
            optimizer.step()
            scheduler.step()
            # code of validation
            if total_steps % save_frequency == (save_frequency - 1):
                # save checkpoints
                # save_path = Path('training_checkpoints/%dsteps_StereoNet%s_%s.pth' % (total_steps + 1, refinement_times, dataset_name))
                # torch.save(net.state_dict(), save_path)

                # load validation data 
                if require_validation:
                    # print("--- start validation ---")
                    test_loader = fetch_dataset(dataset_name = dataset_name, root = root,
                                    batch_size = batch_size, resize = resize, 
                                    min_disp = min_disp, max_disp = max_disp, mode = 'testing')
                    
                    val_loss_train = 0
                    val_loss_eval = 0
                    with torch.no_grad():
                        for i_batch, data_blob in enumerate(tqdm(test_loader, ncols=80, desc="Validation")):
                            image1, image2, disp_gt, valid = [x.to(device) for x in data_blob]

                            net.eval()
                            pred_list = net(image1, image2, min_disp, max_disp)
                            val_loss_eval += criterion(disp_gt, pred_list, refinement_times, valid)

                            net.train()
                            pred_list = net(image1, image2, min_disp, max_disp)
                            val_loss_train += criterion(disp_gt, pred_list, refinement_times, valid)
                        val_loss_eval = val_loss_eval / test_loader.__len__()
                        val_loss_train = val_loss_train / test_loader.__len__()
                    writer.add_scalars(main_tag="loss/vaildation loss", tag_scalar_dict = {'train()': val_loss_train}, global_step=total_steps+1)
                    writer.add_scalars(main_tag="loss/vaildation loss", tag_scalar_dict = {'eval()':val_loss_eval}, global_step=total_steps+1)

                net.train()
            
            # write loss and lr to log
            writer.add_scalar(tag="loss/training loss", scalar_value=loss, global_step=total_steps+1)
            writer.add_scalar(tag="lr/lr", scalar_value=scheduler.get_last_lr()[0], global_step=total_steps+1)
            total_steps += 1

            if total_steps > num_steps:
                should_keep_training = False
                break

        cur_iter = int(total_steps/steps_per_iter)
        save_path = Path('training_checkpoints/%d_epoch_StereoNet%s_%s.pth' % (cur_iter, refinement_times, dataset_name))
        torch.save(net.state_dict(), save_path)

    print("FINISHED TRAINING")

    final_outpath = f'training_checkpoints/StereoNet{refinement_times}_{dataset_name}.pth'
    torch.save(net.state_dict(), final_outpath)
    print("model has been save to path: ", final_outpath)

if __name__ == '__main__':
    

    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

    net = StereoNet(image_channel=3, k=3, refinement_time=4)

    ''' 
    training set keywords: 
    1.DFC2019, 
    2.WHUStereo, 
    3.all 
    '''
    # '/home/lab1/datasets/DFC2019_track2_grayscale_8bit'
    # '/home/lab1/datasets/whu_stereo_8bit/with_ground_truth'
    # train(net=net, dataset_name='DFC2019', root = '/home/lab1/datasets/DFC2019_track2_grayscale_8bit', 
    #       batch_size=2, min_disp=-96, max_disp=96, refinement_times=4, iters=20, init_lr=0.0002,
    #       resize = [1024,1024], save_frequency = 2000, require_validation=True, 
    #       device=device, log_dir='logs/dfc_ft_15+20epoch_96',
    #       pretrain='training_checkpoints/96/15_epoch_StereoNet4_DFC2019.pth')
    train(net=net, dataset_name='WHUStereo', root = '/home/lab1/datasets/whu_stereo_8bit/with_ground_truth', 
        batch_size=2, min_disp=-96, max_disp=96, refinement_times=4, iters=20, init_lr=0.001,
        resize = [1024,1024], save_frequency = 610, require_validation=True, 
        device=device, log_dir='logs/dfc_whu_3_96_35+20epoch',
        pretrain='training_checkpoints/96/35_epoch_StereoNet4_DFC2019.pth')