from typing import Union, Optional, Sequence,Dict,Any 
import argparse
import numpy as np
import torch 
import tensorrt as trt 
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from modeling import UNet, DGLNet, AttU_Net, R2AttU_Net, deeplab_resnet50, deeplab_mobilenetv2, Scnn_AttU_Net

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
    img = F.rotate(demo_img, 15)
    img = F.resize(img, [256, 256], InterpolationMode.BILINEAR) #[256 312]
    #img = F.center_crop(img, 256)
    img = F.to_tensor(img)
    input = F.normalize(img, mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5])
    return input

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
 
            device = torch.device('cuda')
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
    model_pytorch.to('cuda')
    out_pytorch = model_pytorch(input.cuda())

    #trt
    model_trt = TRTWrapper('model.engine', ['output']) 
    out_trt = model_trt(dict(input = input.cuda()))['output']


    #compart
    print(out_pytorch[0,:])
    print(out_trt[0,:])

    #print(out_pytorch.shape)    
    #print(out_trt.shape)
    #print(torch.max(torch.abs(out_pytorch - out_trt)))
    #print(torch.isclose(out_pytorch, out_trt))
