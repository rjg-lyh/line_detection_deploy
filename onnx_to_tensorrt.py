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
import tensorrt as trt
import onnx
import onnxruntime
def get_argparser():
    parser = argparse.ArgumentParser()  #IMG_20221006_125547_aug0  IMG_20221013_124427  IMG_20221013_125206_aug2（典型 rotate=15 IoU_threshold=0.8）
                                        #IMG_20221013_124515_aug1 IMG_20221006_132459(杂草) IMG_20221013_124448（train 正常 偏移 rotate=-15 threshold=0.5）
                                        #IMG_20221006_131948(杂草)
    parser.add_argument('--demo_img',  type=str, default='/home/nvidia/project/ceshi.jpg',
                        help='the path of the demo image')
    parser.add_argument('--ckpt_path',  type=str, default='/home/nvidia/project/best_AttU_Net.pth',
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
    img = F.rotate(demo_img, 13)
    img = F.resize(img, [256, 256], InterpolationMode.BILINEAR) #[256 312]
    #img = F.center_crop(img, 256)
    img = F.to_tensor(img)
    input = F.normalize(img, mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5])
    return input

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
    # print(mask_2.shape)
    # print(mask_3.shape)
    # print(mask_4.shape)
    fig = plt.figure()
    plt.imshow(image)
    plt.axis('off')
    plt.imshow(mask_2, alpha=0.5)
    plt.imshow(mask_3, alpha=0.5)
    plt.imshow(mask_4, alpha=0.5)
    ax = plt.gca()
    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
    plt.savefig('./overlay1.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    return image, mask2, mask3, mask4



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
    #demo_img1 = np.array(demo_img)
    
    model = model_map[opts.model](opts)
    pretrained = torch.load(opts.ckpt_path, map_location=torch.device('cpu'))
    model.load_state_dict(pretrained["model_state"])
    model.eval()

    input = preprocess(demo_img)#[3 256 256]
    input = input.unsqueeze(0)#[1 3 256 256]

    #生成onnx中间表示模型:
    # x = torch.randn(1, 3, 256, 256) 
    # with torch.no_grad(): 
    #     torch.onnx.export( 
    #         model, 
    #         x, 
    #         "attn_unet.onnx", 
    #         opset_version=11, 
    #         input_names=['input'], 
    #         output_names=['output'])

    input_arr = np.array(input)

    #ONNX转TensorRT推理模型:
    device = torch.device('cuda:0')
    onnx_model = onnx.load('attn_unet.onnx') 

    try: 
        onnx.checker.check_model(onnx_model) 
    except Exception: 
        print("Model incorrect") 
    else: 
        print("Model correct")
    
    ort_session = onnxruntime.InferenceSession("attn_unet.onnx") 
    ort_inputs = {'input': input_arr} 
    y_trt = ort_session.run(['output'], ort_inputs)[0] 
    print(type(y_trt))
    print(y_trt.shape)
    
    y = model(input).detach().sigmoid()

    y_trt = torch.tensor(y_trt).detach().sigmoid()
    print(torch.max(torch.abs(y - y_trt)))

    #image, mask2, mask3, mask4 = watch(input, y_trt)




    #'''
    # create builder and network 
    logger = trt.Logger(trt.Logger.ERROR) 
    builder = trt.Builder(logger) 
    EXPLICIT_BATCH = 1 << (int)( 
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) 
    network = builder.create_network(EXPLICIT_BATCH) 
    
    # parse onnx 
    parser = trt.OnnxParser(network, logger) 
    
    if not parser.parse(onnx_model.SerializeToString()): 
        error_msgs = '' 
        for error in range(parser.num_errors): 
            error_msgs += f'{parser.get_error(error)}\n' 
        raise RuntimeError(f'Failed to parse onnx, {error_msgs}') 
    
    config = builder.create_builder_config() 
    config.max_workspace_size = 1<<20 
    profile = builder.create_optimization_profile() 
    
    profile.set_shape('input', [1, 3, 256, 256], [1, 3, 256, 256], [1, 3, 256, 256]) 
    config.add_optimization_profile(profile) 
    # create engine 
    with torch.cuda.device(device): 
        engine = builder.build_engine(network, config) 

    with open('model.engine', mode='wb') as f: 
        f.write(bytearray(engine.serialize())) 
        print("generating file done!") 
    #'''
    