import torch
import argparse
import sys
sys.path.append('.')

from model.model_builder import build_model
from utils.read_data import *

from utils.stereo_datasets import fetch_dataset
from tqdm import tqdm
from utils.metric import epe_tensor, px_error_tensor


def evaluation(net, dataset_name, root, device, min_disp, max_disp, batch_size=1, resize=[1024, 1024]):
    metric = {'epe':[0,0], '1px':[0,0], '2px':[0,0], '3px':[0,0]}
    net  = net.to(device)
    test_loader = fetch_dataset(dataset_name = dataset_name, root = root,
                        batch_size = batch_size, resize = resize, 
                        min_disp = min_disp, max_disp = max_disp, mode = 'testing', shuffle=False)

    with torch.no_grad():
        for i_batch, data_blob in enumerate(tqdm(test_loader, ncols=80, desc="Validation  eval()")):
            image1, image2, disp_gt, valid = [x.to(device) for x in data_blob]
            
            net.eval()
            # net._disable_batchnorm_tracking()
            with torch.no_grad():
                disp_pred = net(image1, image2, min_disp, max_disp)
                metric['epe'][1] += epe_tensor(disp_pred['final_disp'], disp_gt, valid)
                metric['1px'][1] += px_error_tensor(disp_pred['final_disp'], disp_gt, valid, 1)
                metric['2px'][1] += px_error_tensor(disp_pred['final_disp'], disp_gt, valid, 2)
                metric['3px'][1] += px_error_tensor(disp_pred['final_disp'], disp_gt, valid, 3)
        for k in metric:
            metric[k][1] /= test_loader.__len__()

    with torch.no_grad():
        for i_batch, data_blob in enumerate(tqdm(test_loader, ncols=80, desc="Validation train()")):
            image1, image2, disp_gt, valid = [x.to(device) for x in data_blob]          
            net.train()
            with torch.no_grad():
                disp_pred = net(image1, image2, min_disp, max_disp)
                metric['epe'][0] += epe_tensor(disp_pred['final_disp'], disp_gt, valid).item()
                metric['1px'][0] += px_error_tensor(disp_pred['final_disp'], disp_gt, valid, 1)
                metric['2px'][0] += px_error_tensor(disp_pred['final_disp'], disp_gt, valid, 2)
                metric['3px'][0] += px_error_tensor(disp_pred['final_disp'], disp_gt, valid, 3)
        for k in metric:
            metric[k][0] /= test_loader.__len__()

    return metric

def print_metric(metric):
    print('--------------------------------------------')
    print('Result:')
    for k in metric:
        if k == 'epe':
            print(f'{k} -- (train): {metric[k][0] :.2f} -- (eval): {metric[k][1] :.2f}')
        else:
            print(f'{k} -- (train): {metric[k][0]*100 :.2f}% -- (eval): {metric[k][1]*100 :.2f}%')
    print('--------------------------------------------')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training Config')

    # model config
    parser.add_argument('--model', type=str, default='GwcNet', help='model name: PSMNet, PSMNet_Edge, GwcNet')
    parser.add_argument('--image_channels', default=3, type=int, help='image channel: 3 for RGB, 1 for grayscale')
    # GwcNet
    parser.add_argument('--groups', default=32, type=int, help='number of groups for group convolution')

    # parser.add_argument('--dataset_name', type=str, default='DFC2019', help='testing set keywords: "DFC2019", "WHUStereo", "all"')
    # parser.add_argument('--root', type=str, default='/media/win_d/honghao/training_data/DFC2019/track2_grayscale', help='root path of testing set')
    parser.add_argument('--dataset_name', type=str, default='WHUStereo', help='training set keywords: "DFC2019", "WHUStereo", "all"')
    parser.add_argument('--root', type=str, default='/media/win_d/honghao/training_data/WHUStereo/WHUStereo_8UC3//with_ground_truth', help='root path of training set')

    parser.add_argument('--min_disp', type=int, default=-96, help='minimum disparity')
    parser.add_argument('--max_disp', type=int, default=96, help='maximum disparity')

    parser.add_argument('--ckpt', type=str, help='checkpoint path')

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
   
    print('Model:', args.model)
    print('Disp range:', args.min_disp, '~', args.max_disp)
    print('Testing Dataset:', args.dataset_name)
    print('Checkpoint:', args.ckpt)

    net = build_model(args)
    net.load_state_dict(torch.load(args.ckpt), strict=True)

    # net.eval()
    net = net.to(device)   

    metric= evaluation(net, args.dataset_name, args.root, device, args.min_disp, args.max_disp)

    print_metric(metric)
