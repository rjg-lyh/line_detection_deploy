from typing import Union, Optional, Sequence,Dict,Any 
import argparse
import time
import numpy as np
import torch 
import tensorrt as trt 
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from modeling import UNet, DGLNet, AttU_Net, R2AttU_Net, deeplab_resnet50, deeplab_mobilenetv2, Scnn_AttU_Net
import utils
import matplotlib
from matplotlib import pyplot as plt

def get_argparser():
    parser = argparse.ArgumentParser()  
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

def watch(image, pred, num):
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
    plt.savefig(f'./overlay{num}.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    return image, mask2, mask3, mask4

class TRTWrapper(torch.nn.Module): 
    def __init__(self,engine: Union[str, trt.ICudaEngine], 
                 output_names: Optional[Sequence[str]] = None) -> None: 
        super().__init__() 
        self.engine = engine 
        if isinstance(self.engine, str): 
            with trt.Logger() as logger, trt.Runtime(logger) as runtime: 
                with open(self.engine, mode='rb') as f: 
                    engine_bytes = f.read() 
                self.engine = runtime.deserialize_cuda_engine(engine_bytes) 
        self.context = self.engine.create_execution_context() 
        names = [_ for _ in self.engine] 
        input_names = list(filter(self.engine.binding_is_input, names))
        self._input_names = input_names 
        self._output_names = output_names 
 
        if self._output_names is None: 
            output_names = list(set(names) - set(input_names)) 
            self._output_names = output_names 
 
    def forward(self, inputs: Dict[str, torch.Tensor]): 
        assert self._input_names is not None 
        assert self._output_names is not None 
        bindings = [None] * (len(self._input_names) + len(self._output_names)) 
        profile_id = 0 
        for input_name, input_tensor in inputs.items(): 
            # check if input shape is valid 
            profile = self.engine.get_profile_shape(profile_id, input_name) 
            assert input_tensor.dim() == len( 
                profile[0]), 'Input dim is different from engine profile.' 
            for s_min, s_input, s_max in zip(profile[0], input_tensor.shape, 
                                             profile[2]): 
                assert s_min <= s_input <= s_max, 'Input shape should be between ' + f'{profile[0]} and {profile[2]}' + f' but get {tuple(input_tensor.shape)}.' 
            idx = self.engine.get_binding_index(input_name) 
            # All input tensors must be gpu variables 
            assert 'cuda' in input_tensor.device.type 
            input_tensor = input_tensor.contiguous() 
            if input_tensor.dtype == torch.long: 
                input_tensor = input_tensor.int() 
            self.context.set_binding_shape(idx, tuple(input_tensor.shape)) 
            bindings[idx] = input_tensor.contiguous().data_ptr() 
 
        # create output tensors 
        outputs = {} 
        for output_name in self._output_names: 
            idx = self.engine.get_binding_index(output_name) 
            dtype = torch.float32
            shape = tuple(self.context.get_binding_shape(idx)) 
 
            device = torch.device('cuda:0')
            print(device) 
            output = torch.empty(size=shape, dtype=dtype, device=device) 
            outputs[output_name] = output 
            bindings[idx] = output.data_ptr() 
        self.context.execute_async_v2(bindings, 
                                      torch.cuda.current_stream().cuda_stream) 
        return outputs 

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
    input = preprocess(demo_img)
    input = input.unsqueeze(0)
    #pytorch_model
    model_pytorch = model_map[opts.model](opts)
    model_pytorch.load_state_dict(torch.load(opts.ckpt_path, map_location=torch.device('cpu'))["model_state"])
    #model_pytorch.load_state_dict(torch.load(opts.ckpt_path)["model_state"])
    device = torch.device('cuda:0')
    model_pytorch.to(device)
    model_pytorch.eval()
    #print('aaaa: ', next(model_pytorch.parameters()).device)

    #pytorch_model
    start1 = time.time()
    out_pytorch = model_pytorch(input.to(device)).detach().sigmoid()
    end1 = time.time()
    #trt
    model_trt = TRTWrapper('../model.engine', ['output']) 
    start2 = time.time()
    out_trt = model_trt(dict(input = input.to(device)))['output'].detach().sigmoid()
    end2 = time.time()

    # print(out_pytorch.shape)    
    # print(out_trt.shape)
    # print(torch.max(torch.abs(out_pytorch - out_trt)))

    print("no_deploy inference: ", end1-start1)
    print("TrT_deploy inference: ",end2-start2)

    #print(torch.isclose(out_pytorch, out_trt))
    # print(out_pytorch[0,3].unique())
    # print(out_trt[0,3].unique())

    # watch(input, out_pytorch, 1)
    # watch(input, out_trt, 2)