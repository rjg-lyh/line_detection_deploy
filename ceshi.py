import cv2
import torch
import numpy as np
import math
from visdom import Visdom
from PIL import Image

# gaussian1d = cv2.getGaussianKernel(5, 1.15)
# print(gaussian1d.flatten())

def put_heatmap(heatmap, plane_idx, center, sigma):
    """
    Parameters
    -heatmap: 热图（heatmap）
    - plane_idx：关键点列表中第几个关键点（决定了在热图中通道）
    - center： 关键点的位置
    - sigma: 生成高斯分布概率时的一个参数
    Returns
    - heatmap: 热图
    """
 
    center_x, center_y = center  #mou发
    _, height, width = heatmap.shape[:3]
 
    th = 4.6052
    delta = math.sqrt(th * 2)
 
    x0 = int(max(0, center_x - delta * sigma + 0.5))
    y0 = int(max(0, center_y - delta * sigma + 0.5))
 
    x1 = int(min(width - 1, center_x + delta * sigma + 0.5))
    y1 = int(min(height - 1, center_y + delta * sigma + 0.5))
 
    exp_factor = 1 / 2.0 / sigma / sigma
 
    ## fast - vectorize
    arr_heatmap = heatmap[plane_idx, y0:y1 + 1, x0:x1 + 1]
    y_vec = (np.arange(y0, y1 + 1) - center_y)**2  # y1 included
    x_vec = (np.arange(x0, x1 + 1) - center_x)**2
    xv, yv = np.meshgrid(x_vec, y_vec)
    arr_sum = exp_factor * (xv + yv)
    arr_exp = np.exp(-arr_sum)
    arr_exp[arr_sum > th] = 0
    heatmap[plane_idx, y0:y1 + 1, x0:x1 + 1] = np.maximum(arr_heatmap, arr_exp)
    return heatmap


def convert_label(lbls):
    mask_1, mask_2 = (lbls == 1), (lbls == 2)#[2 256 256], [2 256 256]
    lbl1 = torch.masked_fill(lbls, mask_2, 1).unsqueeze(1) #[2 1 256 256]
    lbl2 = torch.masked_fill(lbls, mask_1, 0)
    mask = (lbl2 == 2)
    lbl2 = torch.masked_fill(lbl2, mask, 1).unsqueeze(1) #[2 1 256 256]
    return lbl1, lbl2

def loss_compute(outputs, lbls):
    out1, out2 = outputs #[2 1 256 256], [2 1 256 256]
    lbl1, lbl2 = convert_label(lbls)
    lbl1 = lbl1.to(torch.float32)
    lbl2 = lbl2.to(torch.float32)
    bce = torch.nn.BCEWithLogitsLoss()
    loss = bce(out1, lbl1) + bce(out2, lbl2)
    return loss

if __name__ == '__main__':
    # wind = Visdom()

    # heatmap = np.zeros((1,10,10))
    # heatmap = put_heatmap(heatmap, 0, [2,2], 6)
    # wind.heatmap(heatmap[0], # X的第一个点的坐标
	# 	  win = 'train_loss', # 窗口的名称
	# 	  opts = dict(title = 'train_loss') # 图像的标例
    # )
    # src = Image.open('label.png')
    # img = np.array(src)
    # print(np.unique(img, return_counts=True))
    import torch.nn.functional as F
    import torch
    from PIL import Image

    # palette_data = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128,
    #             64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128,
    #             0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0]  # 调色板

    # src = Image.open('./1.png')
    # src.show()
    # lbls = torch.tensor(np.array(src))
    # print(lbls.dtype)
    # false_mask = torch.zeros_like(lbls)
    # true_mask = torch.ones_like(lbls)
    # print(false_mask.dtype)
    # lbl1_1 = torch.where(lbls== 1, true_mask, false_mask)
    # lbl1_2 = torch.where(lbls== 2, true_mask, false_mask)
    # lbl1_4 = torch.where(lbls== 4, true_mask, false_mask)
    # lbl1 = lbl1_1 + lbl1_2 + lbl1_4
    # print(lbl1)
    # tag = Image.fromarray(lbl1.numpy(), "P")
    # tag.putpalette(palette_data)
    # tag.show()

    # x = torch.randn(2,2,256,256)
    # x = torch.where(x>0, torch.ones_like(x), torch.zeros_like(x))
    # pred = torch.randn(2,2,256,256)
    # bce = torch.nn.BCEWithLogitsLoss(reduction='none')
    # print(bce(x, pred).shape)

    with open('./log.txt', 'w') as f:
        f.writelines('sdsd\n')
        f.writelines('\t55555\n')
        f.writelines('success\n\t')
    with open('./log.txt', 'a') as f:
        f.writelines('777')
    with open('./log.txt', 'a') as f:
        f.writelines('777')

