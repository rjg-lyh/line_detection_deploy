import cv2
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
from modeling import UNet, DGLNet, AttU_Net, R2AttU_Net, deeplab_resnet50, deeplab_mobilenetv2, Scnn_AttU_Net
from tqdm import tqdm
from data import Crop_line

import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--demo_img',  type=str, default='/home/rjg/dataset2/images/val/IMG_20221006_125547_aug0.jpg',
                        help='the path of the demo image')
    parser.add_argument('--ckpt_path',  type=str, default='/home/rjg/下载/best_AttU_Net.pth',
                        help='the path of the pretrained .pth')
    parser.add_argument('--model', type=str, default='AttU_Net',
                        choices=['UNet', 'LBDNet', 'AttU_Net', 'Scnn_AttU_Net', 'R2AttU_Net', 'deeplab_resnet50', 'deeplab_mobilenetv2'],
                        help='model name')
    parser.add_argument('--input_channel', type=int, default=3,
                        help='the channel of the input image')                  
    parser.add_argument('--num_classes', type=int, default=4,
                        help='num classes in seg_task')
    return parser

def preprocess(demo_img):
    img = F.rotate(demo_img, 12)
    img = F.resize(demo_img, 256, InterpolationMode.BILINEAR)
    img = F.center_crop(img, 256)
    img = F.to_tensor(img)
    input = F.normalize(img, mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5])
    return input

def postprocess(output):
    pass

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

def watch(image, pred):
    image = image.squeeze().numpy()#[3 256 256]
    denorm = utils.Denormalize(mean=[0.5, 0.5, 0.5], 
                                std=[0.5, 0.5, 0.5])
    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)

    pred = pred.squeeze()#(4 256 256)
    out_sigm1 = pred[0].detach().sigmoid()#(256 256)
    out_sigm2 = pred[1].detach().sigmoid()
    out_sigm3 = pred[2].detach().sigmoid()
    out_sigm4 = pred[3].detach().sigmoid()
    roi = torch.ones_like(out_sigm1, dtype=torch.uint8)
    ng = torch.zeros_like(out_sigm1, dtype=torch.uint8)
    mask1 = torch.where(out_sigm1>0.5, roi, ng).cpu().numpy()
    mask2 = torch.where(out_sigm2>0.5, roi, ng).cpu().numpy()
    mask3 = torch.where(out_sigm3>0.5, roi, ng).cpu().numpy()
    mask4 = torch.where(out_sigm4>0.5, roi, ng).cpu().numpy()

    table = table_cmp()
    mask_1 =table[mask1].astype(np.uint8)
    mask_2 = table[mask2*2].astype(np.uint8)
    mask_3 =table[mask3*3].astype(np.uint8)
    mask_4 = table[mask4*4].astype(np.uint8)
    fig = plt.figure()
    plt.imshow(image)
    plt.axis('off')
    plt.imshow(mask_2, alpha=0.3)
    plt.imshow(mask_3, alpha=0.3)
    plt.imshow(mask_4, alpha=0.3)
    ax = plt.gca()
    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
    plt.savefig('./overlay1.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    return image, mask2, mask3, mask4
    
def line_fitness(dots, image, type = cv2.DIST_L1, color=(255, 0, 0)):
    image_ = image.copy()
    w = image_.shape[0]
    [vx, vy, x, y] = cv2.fitLine(dots, type, 0, 0.01, 0.01)
    y1 = int((-x * vy / vx) + y)
    y2 = int(((w - x) * vy / vx) + y)
    cv2.line(image_, (w - 1, y2), (0, y1), color, 2)
    return image_, (w - 1, y2),(0, y1)

if __name__ == '__main__':
    opts = get_argparser().parse_args()

    model_map = {
            'UNet':UNet,
            'LBDNet':DGLNet,
            'AttU_Net':AttU_Net, 
            'R2AttU_Net':R2AttU_Net,
            'Scnn_AttU_Net':Scnn_AttU_Net,
            'deeplab_resnet50':deeplab_resnet50,
            'deeplab_mobilenetv2':deeplab_mobilenetv2,
            }

    demo_img = Image.open(opts.demo_img).convert('RGB')
    model = model_map[opts.model](opts)
    pretrained = torch.load(opts.ckpt_path, map_location=torch.device('cpu'))
    model.load_state_dict(pretrained["model_state"])

    input = preprocess(demo_img)#[3 256 256]
    input = input.unsqueeze(0)#[1 3 256 256]

    pred = model(input)#[1 4 256 256]

    image, mask2, mask3, mask4 = watch(input, pred) #二值化图 numpy 0、1
    start_time = time.time()
    # mask2_ = cv2.distanceTransform(mask2, distanceType=cv2.DIST_L1, maskSize=3) #计算出来的结果是float32，用opencv显示必须convertScaleAbs，pyplot无所谓
    # mask2_dist = cv2.convertScaleAbs(mask2_)
    # mask3_ = cv2.distanceTransform(mask3, distanceType=cv2.DIST_L1, maskSize=3) #计算出来的结果是float32，用opencv显示必须convertScaleAbs，pyplot无所谓
    # mask3_dist = cv2.convertScaleAbs(mask3_)
    # mask4_ = cv2.distanceTransform(mask4, distanceType=cv2.DIST_L1, maskSize=3) #计算出来的结果是float32，用opencv显示必须convertScaleAbs，pyplot无所谓
    # mask4_dist = cv2.convertScaleAbs(mask4_)

    # mask2_thin = np.where(mask2_dist>8, np.ones_like(mask2_dist, dtype=np.uint8)*255, np.zeros_like(mask2_dist, dtype=np.uint8))
    # mask3_thin = np.where(mask3_dist>8, np.ones_like(mask3_dist, dtype=np.uint8)*255, np.zeros_like(mask3_dist, dtype=np.uint8))
    # mask4_thin = np.where(mask4_dist>8, np.ones_like(mask4_dist, dtype=np.uint8)*255, np.zeros_like(mask4_dist, dtype=np.uint8))

    dot2s = np.nonzero(mask2)
    dot2s = np.concatenate([dot2s[1].reshape(1,-1).T, dot2s[0].reshape(1,-1).T], axis=1)
    dot3s = np.nonzero(mask3)
    dot3s = np.concatenate([dot3s[1].reshape(1,-1).T, dot3s[0].reshape(1,-1).T], axis=1)
    dot4s = np.nonzero(mask4)
    dot4s = np.concatenate([dot4s[1].reshape(1,-1).T, dot4s[0].reshape(1,-1).T], axis=1)
    dot5s = np.concatenate([dot2s, dot3s], axis=0)
    print(dot2s.shape, dot3s.shape, dot5s.shape)
    image_, _, _ = line_fitness(dot2s, image, type = cv2.DIST_L1, color=(0, 0, 255))
    image_, _, _ = line_fitness(dot3s, image_, type = cv2.DIST_L1, color=(0, 255, 0))
    image_, _, _ = line_fitness(dot4s, image_, type = cv2.DIST_L1, color=(255, 0, 0))

    image_2, _, _ = line_fitness(dot5s, image_, type = cv2.DIST_L1, color=(128, 0, 128))
    end_time = time.time()
    print('time cost: ', end_time-start_time)
    image_overlay = np.array(Image.open('./overlay1.png'))

    mt.PIS(image, image_overlay, image_, image_2)

    
    
    
    
