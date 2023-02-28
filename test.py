import utils
import time
import torch
import matplotlib
import numpy as np
import os
import random
import argparse
import torch.nn as nn
import mtutils as mt

from PIL import Image
from utils import ext_transforms as et
from matplotlib import pyplot as plt
from metrics import StreamSegMetrics
from torchvision import transforms as T
from torch.utils.data import DataLoader
from utils.visualizer import Visualizer
from modeling import UNet, LBDNet
from tqdm import tqdm
from data import Crop_line

def get_argparser():
    parser = argparse.ArgumentParser()

    #Dataset Options
    parser.add_argument('--data_root', type=str, default='/media/rjg/SSD/crop_line',
                        help='path to Dataset')
    parser.add_argument('--dataset', type=str, default='crop_line',
                        help='Name of Dataset')
    parser.add_argument('--input_channel', type=int, default=3,
                        help='the channel of the input image')                  
    parser.add_argument('--num_classes', type=int, default=2,
                        help='num classes in seg_task')
    parser.add_argument('--num_classes_aux', type=int, default=6,
                        help='num classes in aux_task')

    #Model Options
    parser.add_argument('--model', type=str, default='UNet',
                        choices=['UNet', 'LBDNet'],
                        help='model name')
    parser.add_argument('--need_scnn', action='store_true',default=False,
                        help='add SCNN-Blocks')
    parser.add_argument('--need_aux', action='store_true',default=False,
                        help='add aux-training branch')
    parser.add_argument('--num_stages', type=int, default=6,
                        help='total number of stages in DBLNet')

    #Train Options
    parser.add_argument("--ckpt", default='/home/rjg/project/lines_detector/checkpoints_1/last_UNet.pth', type=str,
                        help="restore from checkpoint")
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of CPU workers, cat /proc/cpuinfo 查看cpu核心数')
    parser.add_argument('--expnum',  type=int, default=1,
                        help='the number of my train')
    parser.add_argument('--epoch_num', type=int, default=300,
                        help='epoch number')
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'],
                        help='the type of optimizer')
    parser.add_argument('--init_lr', type=float, default=1e-3,
                        help='init learning rate')
    parser.add_argument('--lr_policy', type=str, default='step', choices=['step', 'poly'],
                        help='learning rate scheduler policy')
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--step_size', type=int, default=2,
                        help='when to change LR')
    parser.add_argument("--batch_size", type=int, default=2,
                        help='batch size (default: 2)')
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=5,
                        help="print interval of loss (default: 5)")
    parser.add_argument("--val_interval", type=int, default=1,
                        help="iter interval for eval (default: 1)")
    parser.add_argument("--loss_type", type=str, default='BCE', choices=['BCE', 'cross_entropy', 'focal_loss'], 
                        help="loss type (depend on which model chosen)")
    
    # Visdom options
    parser.add_argument("--enable_vis", action='store_false', default=True,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='8097',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')

    return parser
    
def get_dataset(opts):
    """ Dataset And Augmentation
    """
    train_transform = et.ExtCompose([
            #et.ExtRandomHorizontalFlip(),
            et.ExtRandomRotation((-2.0, 2.0)),
            et.ExtResize(size=opts.crop_size),
            #et.ExtRandomScale((0.5, 2.0)),
            #et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtCenterCrop(opts.crop_size),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5]),
            ])

    val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                et.ExtCenterCrop(opts.crop_size),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5]),
                ])
    
    train_dst = Crop_line(root=opts.data_root, 
                          image_set='train',
                          transform=train_transform)

    val_dst = Crop_line(root=opts.data_root, 
                          image_set='val',
                          transform=val_transform)    

    return train_dst, val_dst 

def convert_label(lbls):
    mask_1, mask_2 = (lbls == 1), (lbls == 2)#[2 256 256], [2 256 256]
    lbl1 = torch.masked_fill(lbls, mask_2, 1).unsqueeze(1) #[2 1 256 256]
    lbl2 = torch.masked_fill(lbls, mask_1, 0)
    mask = (lbl2 == 2)
    lbl2 = torch.masked_fill(lbl2, mask, 1).unsqueeze(1) #[2 1 256 256]
    return lbl1, lbl2

def main():
    opts = get_argparser().parse_args()

    train_dst, val_dst = get_dataset(opts)
    val_loader = DataLoader(
        val_dst, batch_size=opts.batch_size, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    model_map = {
                'UNet':UNet,
                'LBDNet':LBDNet,
                }
    
    model = model_map[opts.model](opts)
    if opts.ckpt is not None:
        model.load_state_dict(torch.load(opts.ckpt)['model_state'])
        print('Pretrain Model successfully! ! !')
    else:
        print('Pretrain fail! !')
    model.to(device)

    metrics_1 = StreamSegMetrics(opts.num_classes)
    metrics_2 = StreamSegMetrics(opts.num_classes)
    denorm = utils.Denormalize(mean=[0.5, 0.5, 0.5], 
                                std=[0.5, 0.5, 0.5])
    metrics_1.reset()
    metrics_2.reset()

    ################################
    #      IOU指标评估
    ###############################
    for inputs, lbls in tqdm(val_loader):
        inputs = inputs.to(device, dtype=torch.float32) #(2 3 256 256)
        preds = model(inputs) #(2,1,256,256)  (2,1,256,256)
        lbl1, lbl2 = convert_label(lbls)#(2 1 256 256) (2 1 256 256)
        out1, out2 = preds[0].detach(), preds[1].detach()
        mask1, mask2 = out1.squeeze().sigmoid(), out2.squeeze().sigmoid() #(2 256 256), (2 256 256)
        roi = torch.ones_like(mask1, dtype=torch.uint8)
        ng = torch.zeros_like(mask1, dtype=torch.uint8)
        pred1 = torch.where(mask1>0.5, roi, ng).cpu().numpy()
        pred2 = torch.where(mask2>0.5, roi, ng).cpu().numpy()
        tgt1, tgt2 = lbl1.detach().squeeze().cpu().numpy(), lbl2.detach().squeeze().cpu().numpy()#(2 256 256), (2 256 256)
        metrics_1.update(tgt1, pred1)
        metrics_2.update(tgt2, pred2)
        score1 = metrics_1.get_results()
        score2 = metrics_2.get_results()
    print('score1: ',score1)
    print('score2: ',score2)
    filename = '../record/checkpoints_%d/metrics.txt'%opts.expnum
    with open(filename,'w') as f:
        f.write(f'score1: {score1}\n\nscore2: {score2}')
    print(f'Store {filename} successfully')


    #############################
    #        测试集preds可视化
    #############################
    # for inputs, lbls in val_loader:
    #     inputs = inputs.to(device, dtype=torch.float32) #(2 3 256 256)
    #     preds = model(inputs) #(2,1,256,256)  (2,1,256,256)
    #     out1, out2 = preds[0].detach(), preds[1].detach()
    #     mask1, mask2 = out1[0].squeeze().sigmoid(), out2[0].squeeze().sigmoid()
    #     roi = torch.ones_like(mask1, dtype=torch.uint8)
    #     ng = torch.zeros_like(mask1, dtype=torch.uint8)
    #     mask1 = torch.where(mask1>0.5, roi, ng).cpu().numpy()
    #     mask2 = torch.where(mask2>0.5, roi*2, ng).cpu().numpy()

    #     input = inputs.detach().cpu().numpy()[0]
    #     image = (denorm(input) * 255).transpose(1, 2, 0).astype(np.uint8)
    #     rgb1 = val_loader.dataset.decode_target(mask1).astype(np.uint8)
    #     rgb2 = val_loader.dataset.decode_target(mask2).astype(np.uint8)

    #     fig = plt.figure()
    #     plt.imshow(image)
    #     plt.axis('off')
    #     #plt.imshow(rgb1, alpha=0.7)
    #     plt.imshow(rgb2, alpha=0.5)
    #     plt.show()


    ########################################
    #      src_images和labels的可视化
    ###################################

        # for inputs, lbls in train_loader:
    #     print('inputs.shape: ',inputs.shape)
    #     print('lbls.shape: ',lbls.shape)
    #     outputs = model(inputs)
    #     print(len(outputs))
    #     output1, output2 = outputs
    #     print('outputs.shape: ',output1.shape) #(2,1,256,256)

    #     input = np.array(inputs)[0]
    #     image = (denorm(input) * 255).transpose(1, 2, 0).astype(np.uint8)

    #     mask_1, mask_2 = (lbls == 1), (lbls == 2)
    #     lbls1 = torch.masked_fill(lbls, mask_2, 1)
    #     lbls2 = torch.masked_fill(lbls, mask_1, 0)
    #     print(lbls1.unique())
    #     print(lbls2.unique())
    #     lbl = np.array(lbls)[0]
    #     lbl1 = np.array(lbls1)[0]
    #     lbl2 = np.array(lbls2)[0]
    #     rgb = train_loader.dataset.decode_target(lbl).astype(np.uint8)
    #     rgb1 = train_loader.dataset.decode_target(lbl1).astype(np.uint8)
    #     rgb2 = train_loader.dataset.decode_target(lbl2).astype(np.uint8)

    #     print(rgb.shape, rgb1.shape, rgb2.shape)
    #     mt.PIS(rgb,rgb1,rgb2)
        # fig = plt.figure()
        # plt.imshow(image)
        # plt.axis('off')
        # plt.imshow(rgb, alpha=0.7)
        # plt.show()
if __name__ == '__main__':
    main()