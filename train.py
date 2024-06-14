import torch

import argparse
import os
from datetime import datetime

from model.model_builder import build_model

from tools.train_psm import train_psm

def export_config(path, args):
    # export the config
    config = vars(args)
    path = path + '/config.txt'
    with open(path, 'w+') as f:
        for k in config:
            str = k + ':' + f"{getattr(args, k)}"
            f.write(str + '\n')
    print("Config has been exported to {}".format(path))

if __name__ == '__main__':
    

    parser = argparse.ArgumentParser(description='Training Config')

    '''model config'''
    parser.add_argument('--model', type=str, default='PSMNet', help='model name: PSMNet, PSMNet_Edge, GwcNet, StereoNet')
    parser.add_argument('--image_channels', default=3,   type=int, help='image channel: 3 for RGB, 1 for grayscale')
    parser.add_argument('--min_disp',       default=-96, type=int, help='minimum disparity')
    parser.add_argument('--max_disp',        default=96, type=int, help='maximum disparity')
    # GwcNet
    parser.add_argument('--groups', default=32, type=int, help='number of groups for group convolution')
    # StereoNet
    parser.add_argument('--k', default=3, type=int, help='downsample scale of StereoNet')
    parser.add_argument('--refinement_time', default=4, type=int, help='number of times to refine disparity')

    '''training config'''
    parser.add_argument('--dataset_name',   type=str, default='WHUStereo', 
                        help='training set keywords: "DFC2019", "WHUStereo", "all"')
    # parser.add_argument('--root',           type=str, default='/media/win_d/honghao/training_data/DFC2019/track2_grayscale', 
    #                     help='root path of training set')
    parser.add_argument('--root', type=str, default='/media/win_d/honghao/training_data/WHUStereo/WHUStereo_8UC3//with_ground_truth', 
                        help='root path of training set')
    parser.add_argument('--batch_size',     type=int,   default=1, help='batch size')

    parser.add_argument('--epoch',          type=int,   default=5,      help='number of epoch')
    parser.add_argument('--init_lr',        type=float, default=0.001,  help='initial learning rate')
    parser.add_argument('--resize',         type=list,  default=[1024,1024], help='resize image to [height, width]')
    parser.add_argument('--save_frequency', type=int,   default=1000,   help='save model every save_frequency iterations')
    parser.add_argument('--require_validation', type=bool, default=True, help='require validation during training')
    parser.add_argument('--pretrain',       type=str,  default=None,   help='pretrained model path')
    args = parser.parse_args()

    log_dir = f'logs_{args.model}/' + datetime.now().strftime('%Y-%m-%d_%H:%M') + f'_{args.dataset_name}'
    os.makedirs(log_dir, exist_ok=True)
    export_config(log_dir, args)

    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

    net = build_model(args)

    train_psm(model_name=args.model, net = net, dataset_name=args.dataset_name, root = args.root,
        batch_size=args.batch_size, min_disp=args.min_disp, max_disp=args.max_disp, iters=args.epoch, init_lr=args.init_lr,
        resize = args.resize, save_frequency = args.save_frequency, require_validation=args.require_validation,
        device=device, log_dir=log_dir, pretrain=args.pretrain)