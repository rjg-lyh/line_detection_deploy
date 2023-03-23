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
    img = F.rotate(demo_img, 15)
    img = F.resize(img, [256, 256], InterpolationMode.BILINEAR) #[256 312]
    #img = F.center_crop(img, 256)
    img = F.to_tensor(img)
    input = F.normalize(img, mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5])
    return input





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
    demo_img1 = np.array(demo_img)
    
    model = model_map[opts.model](opts)
    pretrained = torch.load(opts.ckpt_path, map_location=torch.device('cpu'))
    model.load_state_dict(pretrained["model_state"])

    input = preprocess(demo_img)#[3 256 256]
    input = input.unsqueeze(0)#[1 3 256 256]

    #生成onnx中间表示模型:
    x = torch.randn(1, 3, 256, 256) 
    with torch.no_grad(): 
        torch.onnx.export( 
            model, 
            x, 
            "attn_unet.onnx", 
            opset_version=11, 
            input_names=['input'], 
            output_names=['output'])

    input_arr = np.array(input)
    # ort_session = onnxruntime.InferenceSession("attn_unet.onnx") 
    # ort_inputs = {'input': input_arr} 
    # ort_output = ort_session.run(['output'], ort_inputs)[0] 
    # print(type(ort_output))
    # print(ort_output.shape)


    #ONNX转TensorRT推理模型:
    device = torch.device('cuda:0')
    onnx_model = onnx.load('attn_unet.onnx') 
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