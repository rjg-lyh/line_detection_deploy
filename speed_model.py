from torchstat import stat
from modeling import UNet, DGLNet, AttU_Net, R2AttU_Net, deeplab_resnet50, deeplab_mobilenetv2, Scnn_AttU_Net, U_Net_sim, U_Net_Plus_sim
import argparse
import torch
import os

def get_argparser():
    parser = argparse.ArgumentParser()

    #Experiment Note
    parser.add_argument('--note',  type=str, default='speed estimation of the model',
                        help='')
    
    #Experiment number
    parser.add_argument('--expnum',  type=int, default=6,
                        help='the number of my train')

    #Dataset Options
    parser.add_argument('--data_root', type=str, default='/home/rjg/dataset2',
                        help='path to Dataset')
    parser.add_argument('--dataset', type=str, default='crop_line',
                        help='Name of Dataset')

    #Model Options
    parser.add_argument('--model', type=str, default='DGLNet',
                        choices=['UNet', 'U_Net_sim', 'U_Net_Plus_sim', 'DGLNet', 'AttU_Net', 'Scnn_AttU_Net', 'R2AttU_Net', 'deeplab_resnet50', 'deeplab_mobilenetv2'],
                        help='model name')
    parser.add_argument('--input_channel', type=int, default=3,
                        help='the channel of the input image')                  
    parser.add_argument('--num_classes', type=int, default=3,
                        help='num classes in seg_task')

    #Train Options
    parser.add_argument("--ckpt", default='/home/rjg/project/best_U_Net_sim.pth', type=str,
                        help="restore from checkpoint")
    parser.add_argument("--save_val_results", action='store_false', default=True,
                        help="save segmentation results to \"./results\"")
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of CPU workers, cat /proc/cpuinfo 查看cpu核心数')
    parser.add_argument('--epoch_num', type=int, default=6,
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
    parser.add_argument("--batch_size", type=int, default=4,
                        help='batch size (default: 6)')
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=1,
                        help="tmp print interval of loss (default: 1)")
    parser.add_argument("--val_interval", type=int, default=1,
                        help="iter interval for eval (default: 1)")
    parser.add_argument("--loss_type", type=str, default='bce', choices=['bce', 'ce'], 
                        help="loss type (depend on which model chosen)")
    parser.add_argument("--need_focal", action='store_true', default=False,
                        help="need focal loss")
    
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



model_map = {
                'UNet':UNet,
                'U_Net_sim':U_Net_sim,
                'U_Net_Plus_sim':U_Net_Plus_sim,
                'DGLNet':DGLNet,
                'AttU_Net':AttU_Net, 
                'R2AttU_Net':R2AttU_Net,
                'Scnn_AttU_Net':Scnn_AttU_Net,
                'deeplab_resnet50':deeplab_resnet50,
                'deeplab_mobilenetv2':deeplab_mobilenetv2,
                }

opts = get_argparser().parse_args()   
model = model_map[opts.model](opts)
device = torch.device('cpu')
# if opts.ckpt is not None and os.path.isfile(opts.ckpt):
#         checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
#         load_dict = checkpoint["model_state"]
        #model.load_state_dict
        #model.to(device)
# for name, value in load_dict.items():
#     print(name, value.shape)

# result = filter(lambda p: p.requires_grad, model.parameters())
# print(type(result))
# for name, val in model.named_parameters():
#     if name.startswith('Ext'):
#         val.requires_grad = False

#print(type(list(model.parameters())[0]))

Total_Params = stat(model.to(device), (3, 256, 256))
print(Total_Params)

# model_dict = model.state_dict()
# count = 0
# for name, value in model_dict.items():
#     if not name.startswith('Ext'):
#         continue
#     tmp_name = '.'.join(name.split('.')[1:])
#     if tmp_name in load_dict:
#         count += 1
#         model_dict[name] = load_dict[tmp_name]
#         #print(f'{name} is matched successfully !')
# print('count: ', count)
# print(len(load_dict.keys()))

# model.load_state_dict(model_dict)
# print('load checkpoint successfully ! !')
# model.to(device)

# for name, val in model.named_parameters():
#     if name.startswith('Ext'):
#         val.requires_grad = False
# print('set false_grads successfully ! !')

# optimizer = torch.optim.SGD(params=filter(lambda p: p.requires_grad, model.parameters()), 
#                             lr=opts.init_lr, momentum=0.9, weight_decay=opts.weight_decay)
# print('Optimizer loads successfully ! !')
