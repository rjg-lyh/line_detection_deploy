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

from PIL import Image
from utils import ext_transforms as et
from matplotlib import pyplot as plt
from metrics import StreamSegMetrics
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torchstat import stat
from utils.visualizer import Visualizer
from modeling import UNet, DGLNet, AttU_Net, R2AttU_Net, deeplab_resnet50, deeplab_mobilenetv2, Scnn_AttU_Net, U_Net_sim
from tqdm import tqdm
from data import Crop_line



def get_argparser():
    parser = argparse.ArgumentParser()

    #Experiment Note
    parser.add_argument('--note',  type=str, default='all_rows Detector',
                        help='the note of the train experiment')
    
    #Experiment number
    parser.add_argument('--expnum',  type=int, default=3,
                        help='the number of my train')

    #Dataset Options
    parser.add_argument('--data_root', type=str, default='/root/autodl-tmp/dataset2',
                        help='path to Dataset')
    parser.add_argument('--dataset', type=str, default='crop_line',
                        help='Name of Dataset')

    #Model Options
    parser.add_argument('--model', type=str, default='U_Net_sim',
                        choices=['UNet', 'U_Net_sim', 'DGLNet', 'AttU_Net', 'Scnn_AttU_Net', 'R2AttU_Net', 'deeplab_resnet50', 'deeplab_mobilenetv2'],
                        help='model name')
    parser.add_argument('--input_channel', type=int, default=3,
                        help='the channel of the input image')                  
    parser.add_argument('--num_classes', type=int, default=1,
                        help='num classes in seg_task')

    #Train Options
    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--save_val_results", action='store_false', default=True,
                        help="save segmentation results to \"./results\"")
    parser.add_argument('--num_workers', type=int, default=12,
                        help='number of CPU workers, cat /proc/cpuinfo 查看cpu核心数')
    parser.add_argument('--epoch_num', type=int, default=8,
                        help='epoch number')
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'],
                        help='the type of optimizer')
    parser.add_argument('--init_lr', type=float, default=6e-3,
                        help='init learning rate')
    parser.add_argument('--lr_policy', type=str, default='step', choices=['step', 'poly'],
                        help='learning rate scheduler policy')
    parser.add_argument("--weight_decay", type=float, default=6e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--step_size', type=int, default=2,
                        help='when to change LR')
    parser.add_argument("--batch_size", type=int, default=5,
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
    parser.add_argument("--enable_vis", action='store_true', default=False,
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
            et.ExtRandomRotation((-8.0, 8.0)),
            et.ExtResize(size=[opts.crop_size, opts.crop_size]),
            #et.ExtRandomScale((0.5, 2.0)),
            #et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            #et.ExtCenterCrop(opts.crop_size),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5]),
            ])

    val_transform = et.ExtCompose([
                et.ExtResize(size=[opts.crop_size, opts.crop_size]),
                #et.ExtCenterCrop(opts.crop_size),
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
    false_mask = torch.zeros_like(lbls)
    true_mask = torch.ones_like(lbls)
    lbl1 = torch.where(lbls== 1, true_mask, false_mask).unsqueeze(1)#[2 1 256 256]  #left_main
    lbl2 = torch.where(lbls== 2, true_mask, false_mask).unsqueeze(1)                #right_main
    lbl3_1 = torch.where(lbls== 3, true_mask, false_mask).unsqueeze(1)              #all_rows
    lbl4 = torch.where(lbls== 4, true_mask, false_mask).unsqueeze(1)                #navigation_line
    lbl3 = lbl3_1 + lbl1 + lbl2
    lbl5 = lbl1 + lbl2                                                              #main_rows(merge, not split)
    return lbl1, lbl2, lbl3, lbl4, lbl5

def save_best_img(opts, model, loader, device):
    utils.mkdir('/root/autodl-tmp/record/checkpoints_%d/results'%(opts.expnum))
    denorm = utils.Denormalize(mean=[0.5, 0.5, 0.5], 
                                std=[0.5, 0.5, 0.5])
    img_id = 0
    with torch.no_grad():
        for j, (images, labels) in tqdm(enumerate(loader)):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            preds = model(images)

            lbl1, lbl2, lbl3, lbl4, lbl5 = convert_label(labels)#(2 1 256 256) (2 1 256 256)
            out_sigm = preds[:, 0].detach().sigmoid()#(2 256 256)
            roi = torch.ones_like(out_sigm, dtype=torch.uint8)
            ng = torch.zeros_like(out_sigm, dtype=torch.uint8)
            mask = torch.where(out_sigm>0.5, roi, ng).cpu().numpy()
            tgt = lbl3.detach().squeeze(1).cpu().numpy()

            for i in range(len(images)):
                image = images[i].detach().cpu().numpy()
                target = tgt[i]
                pred = mask[i]


                image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                target = loader.dataset.decode_target(target).astype(np.uint8)#将掩码转化为了RGB图像
                pred= loader.dataset.decode_target(pred).astype(np.uint8)
            
                Image.fromarray(image).save('/root/autodl-tmp/record/checkpoints_%d/results/%d_image.png' % (opts.expnum, img_id))#保存为.png格式的图像
                Image.fromarray(target).save('/root/autodl-tmp/record/checkpoints_%d/results/%d_target.png' % (opts.expnum, img_id))
                Image.fromarray(pred).save('/root/autodl-tmp/record/checkpoints_%d/results/%d_pred.png' % (opts.expnum, img_id))


                fig = plt.figure()
                plt.imshow(image)
                plt.axis('off')
                plt.imshow(pred, alpha=0.6)
                ax = plt.gca()
                ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                plt.savefig('/root/autodl-tmp/record/checkpoints_%d/results/%d_overlay.png' % (opts.expnum, img_id), bbox_inches='tight', pad_inches=0)
                plt.close()
                img_id += 1
    print('best val_pictures are saved successfully! !')

def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    ret_samples = []

    model.eval()
    with torch.no_grad():
        
        for i, (images, labels) in tqdm(enumerate(loader)):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            preds = model(images)#[2 2 256 256]
            lbl1, lbl2, lbl3, lbl4, lbl5 = convert_label(labels)#(2 1 256 256) ,(2 1 256 256), ...
                
            out_sigm = preds[:,0].detach().sigmoid()#(2 256 256)
            roi = torch.ones_like(out_sigm, dtype=torch.uint8)
            ng = torch.zeros_like(out_sigm, dtype=torch.uint8)
            mask = torch.where(out_sigm>0.5, roi, ng).cpu().numpy()
            tgt = lbl3.detach().squeeze(1).cpu().numpy()

            metrics.update(tgt, mask)

            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), tgt[0], mask[0]))

        score = metrics.get_results()
    return score, ret_samples

def main():

    #Configs
    opts = get_argparser().parse_args()

    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))

    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id #"2, 3"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)
    
    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    #Setup dataloader
    train_dst, val_dst = get_dataset(opts)
    train_loader = DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers)
    val_loader = DataLoader(
        val_dst, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))
    
    # Setup model
    model_map = {
                'UNet':UNet,
                'U_Net_sim':U_Net_sim,
                'DGLNet':DGLNet,
                'AttU_Net':AttU_Net, 
                'R2AttU_Net':R2AttU_Net,
                'Scnn_AttU_Net':Scnn_AttU_Net,
                'deeplab_resnet50':deeplab_resnet50,
                'deeplab_mobilenetv2':deeplab_mobilenetv2,
                }
    
    model = model_map[opts.model](opts)

    # Setup metrics
    metrics_1 = StreamSegMetrics(opts.num_classes + 1) #all_rows
    metrics_2 = StreamSegMetrics(opts.num_classes + 1) #main_rows(not split)
    metrics_3 = StreamSegMetrics(opts.num_classes + 1) #left_main
    metrics_4 = StreamSegMetrics(opts.num_classes + 1) #right_main
    metrics_5 = StreamSegMetrics(opts.num_classes + 1) #navigation_line


    # Set up optimizer
    optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.init_lr, momentum=0.9, weight_decay=opts.weight_decay)

    # Setup scheduler
    if opts.lr_policy=='poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy=='step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    criterion = utils.criterion(type = opts.loss_type, need_focal = opts.need_focal)

    #Set up logs.txt
    logs_filename = '/root/autodl-tmp/record/checkpoints_%d/logs.txt'%opts.expnum

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            #"model_state": model.module.state_dict(),#用于多卡训练
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)
    if os.path.exists('/root/autodl-tmp/record/checkpoints_%d'%opts.expnum):
        print("document repeat!!")
        sys.exit()
    else:
        os.mkdir('/root/autodl-tmp/record/checkpoints_%d'%opts.expnum)
    # Restore
    best_score = 0.0
    best_info = None
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        #model = nn.DataParallel(model)
        model.to(device)
        # if opts.continue_training:
        #     optimizer.load_state_dict(checkpoint["optimizer_state"])
        #     scheduler.load_state_dict(checkpoint["scheduler_state"])
        #     cur_itrs = checkpoint["cur_itrs"]
        #     best_score = checkpoint['best_score']
        #     CONTINUE_Flag = True
        #     print("Continue training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        #model = nn.DataParallel(model)
        model.to(device)

    #================ Training =====================================
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples if opts.vis_num_samples < len(val_loader) else len(val_loader),
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # denormalization for ori images
    
    interval_loss = []
    len_train = len(train_loader)
    len_val = len(val_loader)
    
    per_epoch_print_num = 5
    opts.print_interval = len_train//per_epoch_print_num #每一轮epoch打印 5 次loss

    with open(logs_filename, 'w') as f:
        f.writelines(f'{vars(opts)}\n\n\n')

    for cur_epoch in range(opts.epoch_num):
        cur_itrs = 0
        with open(logs_filename, 'a') as f:
            division_line = f'Epoch_{cur_epoch}'.center(100, '-')
            f.writelines(f'{division_line}\nTrain:\n\n')
            
        model.train()
        for (images, labels) in train_loader:
            start_time = time.time()
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images) #contain one or two heatmaps
            lbl1, lbl2, lbl3, lbl4, lbl5 = convert_label(labels)
            #lbls = torch.concat([lbl3, lbl1, lbl2, lbl4], 1)
            loss = criterion(outputs, lbl3)

            loss.backward()
            optimizer.step()
            np_loss = loss.detach().cpu().numpy()
            interval_loss.append(np_loss)
            if vis is not None:
                vis.vis_scalar('Loss', ( cur_itrs+cur_epoch*len_train ), np_loss)
            end_time = time.time()
            if (cur_itrs) % opts.print_interval == 0:
                leave_itrs = opts.epoch_num*len_train - cur_itrs - cur_epoch*len_train #1405
                d, h, m, s =  utils.compute_eta(start_time, end_time, leave_itrs)
                interval_loss = sum(interval_loss)/len(interval_loss)
                print("Epoch [%d] Itrs [%d/%d], eta:%.0f days, %.0f:%.0f:%.0f, Loss=%.2f" %
                      (cur_epoch, cur_itrs, len_train, d, h, m, s, interval_loss))
                with open(logs_filename, 'a') as f:
                    f.writelines("\tEpoch [%d] Itrs [%d/%d], eta:%.0f days, %.0f:%.0f:%.0f, Loss=%.2f \n" %
                      (cur_epoch, cur_itrs, len_train, d, h, m, s, interval_loss))
                interval_loss = []

        if (cur_epoch) % opts.val_interval == 0:
            # save_ckpt('../record/checkpoints_%d/latest_%s.pth' %
            #     (opts.expnum, opts.model))
            print("validation...")
            model.eval()
            val_score, ret_samples = validate(
                    opts=opts, model=model, loader=val_loader, device=device, metrics=metrics_1, ret_samples_ids=vis_sample_id)
            print('作物行IOU:\n', metrics_1.to_str(val_score))
            if val_score['Mean IoU'] > best_score:  # when all_lines best
                best_score = val_score['Mean IoU']
                save_ckpt('/root/autodl-tmp/record/checkpoints_%d/best_%s.pth' %
                            (opts.expnum, opts.model))
                best_info = val_score     
                if vis is not None:  # visualize validation score and samples
                    # vis.vis_scalar("[Val] Overall Acc", cur_epoch, val_score['Overall Acc'])
                    # vis.vis_scalar("[Val] Mean IoU", cur_epoch, val_score['Mean IoU'])
                    # vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

                    for k, (img, tgt, mask) in enumerate(ret_samples):
                        img = (denorm(img) * 255).astype(np.uint8)
                        tgt = train_dst.decode_target(tgt).transpose(2, 0, 1).astype(np.uint8)
                        mask = train_dst.decode_target(mask).transpose(2, 0, 1).astype(np.uint8)
                        concat_img = np.concatenate((img, tgt, mask), axis=2)  # concat along width
                        vis.vis_image('Sample %d' % k, concat_img)
            with open(logs_filename, 'a') as f:
                miou = val_score['Mean IoU']
                class_iou = val_score['Class IoU']
                f.writelines('\nValid:\n\n')
                f.writelines(f'\t作物行   Mean IoU: {miou} Class IoU: {class_iou}\n')
        scheduler.step()

    best_info_pp1 = metrics_1.to_str(best_info)
    
    #Save val_results
    if opts.save_val_results:
        best_ckpt = '/root/autodl-tmp/record/checkpoints_%d/best_%s.pth' % (opts.expnum, opts.model)
        model.load_state_dict(torch.load(best_ckpt)["model_state"])
        save_best_img(opts=opts, model=model, loader=val_loader, device=device)
    
    device1 = torch.device('cpu')
    Total_Params = stat(model.to(device1), (opts.input_channel, opts.crop_size, opts.crop_size))
    #Save 'config_info' and 'best_info'
    with open(logs_filename,'a') as f:
        f.write(f'\n\n{Total_Params}\n\n作物行:\n{best_info_pp1}')
    #Print total params
    print(Total_Params)
    print('finish all epochs, best_metrics is shown below:')
    print('作物行_best_iou:\n', best_info_pp1)



if __name__=='__main__':
    main()