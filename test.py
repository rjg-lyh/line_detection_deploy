import utils
import time
import torch
import matplotlib
import numpy as np
import os
import sys
import random
import argparse
import torch.nn as nn
import cv2

from PIL import Image
from utils import ext_transforms as et
from matplotlib import pyplot as plt
from metrics import StreamSegMetrics
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torchstat import stat
from utils.visualizer import Visualizer
from modeling import UNet, DGLNet, AttU_Net, R2AttU_Net, deeplab_resnet50, deeplab_mobilenetv2, Scnn_AttU_Net
from tqdm import tqdm
from data import Crop_line


def get_argparser():
    parser = argparse.ArgumentParser()

    #Test configure
    parser.add_argument('--note',  type=str, default='Test Code: all_rows, left_main, right_main, navigation_line Detector',
                        help='the note of the train experiment')
    
    parser.add_argument('--expnum',  type=int, default=1,
                        help='the number of my train')
    parser.add_argument("--ckpt", default='/root/project/record/checkpoints_1/best_AttU_Net.pth', type=str,
                        help="restore from checkpoint")
    parser.add_argument("--save_val_results", action='store_false', default=True,
                        help="save segmentation results to \"./results\"")
    
    
    
    #below no matter...
    parser.add_argument("--batch_size", type=int, default=3,
                        help='batch size (default: 6)')
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--init_lr', type=float, default=0.02,
                        help='init learning rate')
    parser.add_argument('--epoch_num', type=int, default=8,
                        help='epoch number')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")

    #Dataset Options
    # parser.add_argument('--data_root', type=str, default='/root/autodl-tmp/dataset_SunAndShadow',
    #                     help='path to Dataset')
    parser.add_argument('--data_root', type=str, default='/root/autodl-tmp/test_dataset',
                        help='path to Dataset')
    parser.add_argument('--dataset', type=str, default='crop_line',
                        help='Name of Dataset')

    #Model Options
    parser.add_argument('--model', type=str, default='AttU_Net',
                        choices=['UNet', 'DGLNet', 'AttU_Net', 'Scnn_AttU_Net', 'R2AttU_Net', 'deeplab_resnet50', 'deeplab_mobilenetv2'],
                        help='model name')
    parser.add_argument('--input_channel', type=int, default=3,
                        help='the channel of the input image')                  
    parser.add_argument('--num_classes', type=int, default=4,
                        help='num classes in seg_task')

    #Train Options
    parser.add_argument('--num_workers', type=int, default=15,
                        help='number of CPU workers, cat /proc/cpuinfo 查看cpu核心数')
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'],
                        help='the type of optimizer')
    
    parser.add_argument('--lr_policy', type=str, default='step', choices=['step', 'poly'],
                        help='learning rate scheduler policy')
    parser.add_argument('--step_size', type=int, default=2,
                        help='when to change LR')
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--print_interval", type=int, default=1,
                        help="tmp print interval of loss (default: 1)")
    parser.add_argument("--val_interval", type=int, default=1,
                        help="iter interval for eval (default: 1)")
    parser.add_argument("--loss_type", type=str, default='bce', choices=['bce', 'ce'], 
                        help="loss type (depend on which model chosen)")
    parser.add_argument("--need_focal", action='store_true', default=False,
                        help="need focal loss")
    
    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='8097',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')

    return parser
    
def get_test_dataset(opts):
    """ Dataset And Augmentation
    """

    test_transform = et.ExtCompose([
                et.ExtResize(size=[opts.crop_size, opts.crop_size]),
                #et.ExtCenterCrop(opts.crop_size),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5]),
                ])
    test_dst = Crop_line(root=opts.data_root, 
                          image_set='test',
                          transform=test_transform)    

    return test_dst                 

def convert_label(lbls):
    false_mask = torch.zeros_like(lbls)
    true_mask = torch.ones_like(lbls)
    lbl1 = torch.where(lbls== 1, true_mask, false_mask).unsqueeze(1)#[2 1 256 256]  #left_main
    lbl2 = torch.where(lbls== 2, true_mask, false_mask).unsqueeze(1)                #right_main
    lbl3_1 = torch.where(lbls== 3, true_mask, false_mask).unsqueeze(1)              #all_rows
    lbl4 = torch.where(lbls== 4, true_mask, false_mask).unsqueeze(1)                #navigation_line
    lbl3 = lbl3_1 + lbl1 + lbl2
    lbl5 = lbl1 + lbl2                                                              #main_rows(merge, not split)
    return lbl1, lbl2, lbl3, lbl4, lbl5

def table_cmp(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    table = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        table[i] = np.array([r, g, b])

    table = table/255 if normalized else table
    return table

def line_fitness(dots, image, type = cv2.DIST_L1, color=(255, 0, 0)):
    image_ = image.copy()
    h, w = image_.shape[0]-1, image_.shape[1]-1
    [vx, vy, x0, y0] = cv2.fitLine(dots, type, 0, 0.01, 0.01)
    k = vy/vx
    list1 = []
    #左竖交点 (0, ?)
    x1 = 0
    y1 = int(k*(x1-x0)+y0)
    if(0<= y1 <= h): list1.append((x1, y1))
    #右竖交点 (w, ?)
    x2 = w
    y2 = int(k*(w-x0)+y0)
    if(0 <= y2 <= h): list1.append((x2, y2))
    
    #上横交点 (?, 0)
    if(len(list1) != 2):
        y3 = 0
        x3 = int((y3-y0)/k+x0)
        if(0<= x3 <= w): list1.append((x3, y3))
    #下横交点 (?, h)
    if(len(list1) != 2):
        y4 = h
        x4 = int((y4-y0)/k+x0)
        if(0<= x4 <= w): list1.append((x4, y4))
    
    cv2.line(image_, list1[0], list1[1], color, 2)
    return image_, list1[0], list1[1]

def main():
    opts = get_argparser().parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)
    
    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    #Setup dataloader
    test_dst = get_test_dataset(opts)
    test_loader = DataLoader(
        test_dst, batch_size=opts.batch_size, shuffle=False)
    print("Dataset: %s, test set: %d" %
          (opts.dataset, len(test_dst)))
    
    # Setup metrics
    metrics_1 = StreamSegMetrics(opts.num_classes - 2) #all_rows
    metrics_2 = StreamSegMetrics(opts.num_classes - 2) #main_rows(not split)
    metrics_3 = StreamSegMetrics(opts.num_classes - 2) #left_main
    metrics_4 = StreamSegMetrics(opts.num_classes - 2) #right_main
    metrics_5 = StreamSegMetrics(opts.num_classes - 2) #navigation_line
    
    # Setup model
    model_map = {
                'UNet':UNet,
                'DGLNet':DGLNet,
                'AttU_Net':AttU_Net, 
                'R2AttU_Net':R2AttU_Net,
                'Scnn_AttU_Net':Scnn_AttU_Net,
                'deeplab_resnet50':deeplab_resnet50,
                'deeplab_mobilenetv2':deeplab_mobilenetv2,
                }
    model = model_map[opts.model](opts)
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print('restored fail! !')
        sys.exit()
    
    #Set up logs.txt
    if os.path.exists('../test_log/checkpoints_%d'%opts.expnum):
        print("document repeat!!")
        sys.exit()
    else:
        os.mkdir('../test_log/checkpoints_%d'%opts.expnum)
    
    if opts.save_val_results:
        save_path = '../test_log/checkpoints_%d/results'%(opts.expnum)
        utils.mkdir(save_path)
        img_id = 1
    logs_filename = '../test_log/checkpoints_%d/logs.txt'%opts.expnum

    denorm = utils.Denormalize(mean=[0.5, 0.5, 0.5], 
                                std=[0.5, 0.5, 0.5])
    metrics_1.reset()
    metrics_3.reset()
    metrics_4.reset()
    metrics_5.reset()
    
    ################################
    #      test: IOU指标评估
    ###############################
    model.eval()
    with torch.no_grad():

        for images, labels in tqdm(test_loader):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            preds = model(images)#[2 2 256 256]
            lbl1, lbl2, lbl3, lbl4, lbl5 = convert_label(labels)#(2 1 256 256) ,(2 1 256 256), ...
                
            out_sigm1 = preds[:,0].detach().sigmoid()#(2 256 256)
            out_sigm2 = preds[:,1].detach().sigmoid()#(2 256 256)
            out_sigm3 = preds[:,2].detach().sigmoid()#(2 256 256)
            out_sigm4 = preds[:,3].detach().sigmoid()#(2 256 256)
            roi = torch.ones_like(out_sigm1, dtype=torch.uint8)
            ng = torch.zeros_like(out_sigm1, dtype=torch.uint8)
            mask1 = torch.where(out_sigm1>0.5, roi, ng).cpu().numpy()
            mask2 = torch.where(out_sigm2>0.5, roi, ng).cpu().numpy()
            mask3 = torch.where(out_sigm3>0.5, roi, ng).cpu().numpy()
            mask4 = torch.where(out_sigm4>0.5, roi, ng).cpu().numpy()
            tgt1, tgt2, tgt3, tgt4 = lbl3.detach().squeeze(1).cpu().numpy(), lbl1.detach().squeeze(1).cpu().numpy(), lbl2.detach().squeeze(1).cpu().numpy(), lbl4.detach().squeeze(1).cpu().numpy()#(2 256 256), (2 256 256)

            metrics_1.update(tgt1, mask1)
            metrics_3.update(tgt2, mask2)
            metrics_4.update(tgt3, mask3)
            metrics_5.update(tgt4, mask4)
            #绘制导航线，并保存图片
            if opts.save_val_results:
                table = table_cmp()
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    pred1 = mask1[i]
                    pred2 = mask2[i]
                    pred3 = mask3[i]
                    pred4 = mask4[i]
                    mask_1 =table[pred1].astype(np.uint8)
                    mask_2 = table[pred2*2].astype(np.uint8)
                    mask_3 =table[pred3*3].astype(np.uint8)
                    mask_4 = table[pred4*4].astype(np.uint8)
                    
                    dot2s = np.nonzero(mask_2)
                    dot2s = np.concatenate([dot2s[1].reshape(1,-1).T, dot2s[0].reshape(1,-1).T], axis=1)
                    dot3s = np.nonzero(mask_3)
                    dot3s = np.concatenate([dot3s[1].reshape(1,-1).T, dot3s[0].reshape(1,-1).T], axis=1)
                    dot4s = np.nonzero(mask_4)
                    dot4s = np.concatenate([dot4s[1].reshape(1,-1).T, dot4s[0].reshape(1,-1).T], axis=1)     
                    image_, dot21, dot22 = line_fitness(dot2s, image, type = cv2.DIST_L1, color=(0, 255, 0))
                    image_, dot31, dot32 = line_fitness(dot3s, image_, type = cv2.DIST_L1, color=(255, 255, 0))
                    image_, dot41, dot42 = line_fitness(dot4s, image_, type = cv2.DIST_L1, color=(255, 0, 0))
                    
                    pic_path1 = save_path + '/' + str(img_id) + '_mask.png'
                    pic_path2 = save_path + '/' + str(img_id) + '_line.png'
                    fig = plt.figure()
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(mask_2, alpha=0.7)
                    plt.imshow(mask_3, alpha=0.5)
                    plt.imshow(mask_4, alpha=0.5)
                    plt.savefig(pic_path1, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    
                    fig = plt.figure()
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.imshow(image_)
                    plt.axis('off')
                    plt.savefig(pic_path2, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    
                    img_id += 1                   
            
    score1, score2, score3, score4 = metrics_1.get_results(), metrics_3.get_results(), metrics_4.get_results(), metrics_5.get_results()
    pp1 = metrics_1.to_str(score1)
    pp2 = metrics_2.to_str(score2)
    pp3 = metrics_3.to_str(score3)
    pp4 = metrics_4.to_str(score4)
    
    print('作物行IOU: %f'%score1['Class IoU'][1])
    print('主左作物行IOU: %f'%score2['Class IoU'][1])
    print('主右作物行IOU: %f'%score3['Class IoU'][1])
    print('导航线IOU: %f'%score4['Class IoU'][1])
    
    device1 = torch.device('cpu')
    Total_Params = stat(model.to(device1), (opts.input_channel, opts.crop_size, opts.crop_size))
    
    #记录opts、params、valid_result
    with open(logs_filename, 'w') as f:
        f.writelines(f'{vars(opts)}\n\n\n')
    with open(logs_filename,'a') as f:
        f.write(f'\n\n{Total_Params}\n\n作物行:\n{pp1}\n\n主左作物行:\n{pp2}\n\n主右作物行:\n{pp3}\n\n导航线:\n{pp4}')

    print(f'test complated.')




if __name__ == '__main__':
    main()