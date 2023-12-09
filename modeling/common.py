import torch
import torch.nn as nn
from torch.nn import functional as F
from .deeplab_inter import IntermediateLayerGetter
from ._deeplab import DeepLabHead, DeepLabHeadV3Plus, DeepLabV3
from .backbone import resnet
from .backbone import mobilenetv2
import math

class SpatialConv(nn.Module):
    # SCNN

    def __init__(self, num_channels=128):
        super().__init__()
        self.conv_d = nn.Conv2d(num_channels, num_channels, (1, 9), padding=(0, 4))
        self.conv_u = nn.Conv2d(num_channels, num_channels, (1, 9), padding=(0, 4))
        self.conv_r = nn.Conv2d(num_channels, num_channels, (9, 1), padding=(4, 0))
        self.conv_l = nn.Conv2d(num_channels, num_channels, (9, 1), padding=(4, 0))
        self._adjust_initializations(num_channels=num_channels)

    def _adjust_initializations(self, num_channels=128):
        # https://github.com/XingangPan/SCNN/issues/82
        bound = math.sqrt(2.0 / (num_channels * 9 * 5))
        nn.init.uniform_(self.conv_d.weight, -bound, bound)
        nn.init.uniform_(self.conv_u.weight, -bound, bound)
        nn.init.uniform_(self.conv_r.weight, -bound, bound)
        nn.init.uniform_(self.conv_l.weight, -bound, bound)

    def forward(self, input):
        x = input.clone()
        if False:
            # PyTorch index+add_ will be ignored in traced graph
            # Down
            for i in range(1, output.shape[2]):
                output[:, :, i:i + 1, :] = output[:, :, i:i + 1, :].add(F.relu(self.conv_d(output[:, :, i - 1:i, :])))
            # Up
            for i in range(output.shape[2] - 2, 0, -1):
                output[:, :, i:i + 1, :] = output[:, :, i:i + 1, :].add(
                    F.relu(self.conv_u(output[:, :, i + 1:i + 2, :])))
            # Right
            for i in range(1, output.shape[3]):
                output[:, :, :, i:i + 1] = output[:, :, :, i:i + 1].add(F.relu(self.conv_r(output[:, :, :, i - 1:i])))
            # Left
            for i in range(output.shape[3] - 2, 0, -1):
                output[:, :, :, i:i + 1] = output[:, :, :, i:i + 1].add(
                    F.relu(self.conv_l(output[:, :, :, i + 1:i + 2])))
        else:
            # First one remains unchanged (according to the original paper), why not add a relu afterwards?
            # Update and send to next
            # Down
            # for i in range(1, output.shape[2]):
            #     output[:, :, i:i + 1, :].add_(F.relu(self.conv_d(output[:, :, i - 1:i, :])))
            # # Up
            # for i in range(output.shape[2] - 2, 0, -1):
            #     output[:, :, i:i + 1, :].add_(F.relu(self.conv_u(output[:, :, i + 1:i + 2, :])))
            # # Right
            # for i in range(1, output.shape[3]):
            #     output[:, :, :, i:i + 1].add_(F.relu(self.conv_r(output[:, :, :, i - 1:i])))
            # # Left
            # for i in range(output.shape[3] - 2, 0, -1):
            #     output[:, :, :, i:i + 1].add_(F.relu(self.conv_l(output[:, :, :, i + 1:i + 2])))
            for i in range(1, x.shape[2]):
                tmp1 = x[:, :, i:i + 1, :].clone() + F.relu(self.conv_d(x[:, :, i - 1:i, :].clone()))
                x[:, :, i:i + 1, :] = tmp1
            # Up
            for i in range(x.shape[2] - 2, 0, -1):
                tmp2 = x[:, :, i:i + 1, :].clone() + F.relu(self.conv_u(x[:, :, i + 1:i + 2, :].clone()))
                x[:, :, i:i + 1, :] = tmp2
            # Right
            for i in range(1, x.shape[3]):
                tmp3 = x[:, :, :, i:i + 1].clone() + F.relu(self.conv_r(x[:, :, :, i - 1:i].clone()))
                x[:, :, :, i:i + 1] = tmp3
            # Left
            for i in range(x.shape[3] - 2, 0, -1):
                tmp4 = x[:, :, :, i:i + 1].clone() + F.relu(self.conv_l(x[:, :, :, i + 1:i + 2].clone()))
                x[:, :, :, i:i + 1] = tmp4
        return x

class W_Conv(nn.Module):
    def __init__(self, num_channels=128):
        super().__init__()
        self.conv_r = nn.Conv2d(num_channels, num_channels, (9, 1), padding=(4, 0))
        self.conv_l = nn.Conv2d(num_channels, num_channels, (9, 1), padding=(4, 0))
        self._adjust_initializations(num_channels=num_channels)

    def _adjust_initializations(self, num_channels=128):
        # https://github.com/XingangPan/SCNN/issues/82
        bound = math.sqrt(2.0 / (num_channels * 9 * 5))
        nn.init.uniform_(self.conv_r.weight, -bound, bound)
        nn.init.uniform_(self.conv_l.weight, -bound, bound)

    def forward(self, input):
        x = input.clone()
        # Right
        for i in range(1, x.shape[3]):
            tmp3 = x[:, :, :, i:i + 1].clone() + F.relu(self.conv_r(x[:, :, :, i - 1:i].clone()))
            x[:, :, :, i:i + 1] = tmp3
        #Left
        for i in range(x.shape[3] - 2, 0, -1):
            tmp4 = x[:, :, :, i:i + 1].clone() + F.relu(self.conv_l(x[:, :, :, i + 1:i + 2].clone()))
            x[:, :, :, i:i + 1] = tmp4
        return x

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class U_netplus(nn.Module):
    def __init__(self, opts):
        super(U_netplus, self).__init__()
        n1 = 64
        self.n_channels = opts.input_channel
        # self.n_classes = opts.num_classes
        self.n_classes = 1
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(self.n_channels, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], self.n_classes, kernel_size=1, stride=1, padding=0)

        #self.active = torch.nn.Sigmoid()


    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

      #  out = self.active(out)

        return out, d3

class Recurrent_block(nn.Module):
    """
    Recurrent Block for R2Unet_CNN
    """
    def __init__(self, out_ch, t=2):
        super(Recurrent_block, self).__init__()

        self.t = t
        self.out_ch = out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x = self.conv(x)
            out = self.conv(x + x)
        return out

class RRCNN_block(nn.Module):
    """
    Recurrent Residual Convolutional Neural Network Block
    """
    def __init__(self, in_ch, out_ch, t=2):
        super(RRCNN_block, self).__init__()

        self.RCNN = nn.Sequential(
            Recurrent_block(out_ch, t=t),
            Recurrent_block(out_ch, t=t)
        )
        self.Conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.Conv(x)
        x2 = self.RCNN(x1)
        out = x1 + x2
        return out

class SELayer_2d(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer_2d, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )


    def forward(self, X_input):
        b, c, _, _ = X_input.size()  	# shape = [32, 64, 2000, 80]
        
        y = self.avg_pool(X_input)		# shape = [32, 64, 1, 1]
        y = y.view(b, c)				# shape = [32,64]
        
        # 第1个线性层（含激活函数），即公式中的W1，其维度是[channel, channer/16], 其中16是默认的
        y = self.linear1(y)				# shape = [32, 64] * [64, 4] = [32, 4]
        
        # 第2个线性层（含激活函数），即公式中的W2，其维度是[channel/16, channer], 其中16是默认的
        y = self.linear2(y) 			# shape = [32, 4] * [4, 64] = [32, 64]
        y = y.view(b, c, 1, 1)			# shape = [32, 64, 1, 1]， 这个就表示上面公式的s, 即每个通道的权重

        return X_input*y.expand_as(X_input)

# 一堆操作
class DGLNet_Head(nn.Module):
    def __init__(self, in_channels):
        super(DGLNet_Head, self).__init__()
        
        # self.SE = SELayer_2d()
        
        self.aux1 = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)
        self.aux2 = nn.Conv2d(in_channels//2, 1, kernel_size=1, stride=1, padding=0)
        self.aux3 = nn.Conv2d(in_channels//2, 1, kernel_size=1, stride=1, padding=0)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=True)
        )
       
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1,stride=1,padding=0)
        )

    def forward(self, x):
        out1 = self.aux1(x)
        
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        out2 = self.aux2(x1)
        out3 = self.aux3(x2)
        
        x3 = torch.cat((x1, x2), dim=1)
        out4 = self.conv3(x3)
        
        out =  torch.cat((out1, out2, out3, out4),dim=1)
        return out
    
#四解耦头
class DGLNet_Head2(nn.Module):
    def __init__(self, in_channels):
        super(DGLNet_Head2, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)
        )
                
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out4 = self.conv4(x)
        
        out =  torch.cat((out1, out2, out3, out4),dim=1)
        return out
   
#三解耦头
class DGLNet_Head3(nn.Module):
    def __init__(self, in_channels):
        super(DGLNet_Head3, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 3, kernel_size=1, stride=1, padding=0)
        
        self.conv2 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        
        out =  torch.cat((out1, out2),dim=1)
        return out

#三解耦头v2
class DGLNet_Head4(nn.Module):
    def __init__(self, in_channels):
        super(DGLNet_Head4, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 3, kernel_size=1, stride=1, padding=0)
        
        self.conv2 = nn.Conv2d(in_channels+3, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out1 = self.conv1(x)
        x = torch.cat((x, out1), dim=1)
        out2 = self.conv2(x)
        
        out =  torch.cat((out1, out2),dim=1)
        return out
      
class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)
    #forward(d5, e4)
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out

def _segm_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone):

    if output_stride==8:
        replace_stride_with_dilation=[False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, True]
        aspp_dilate = [6, 12, 18]

    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)
    
    inplanes = 2048
    low_level_planes = 256

    if name=='deeplabv3plus':
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}#
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'layer4': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    #提取网络的第几层输出结果并给一个别名
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model

def _segm_mobilenet(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride==8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    backbone = mobilenetv2.mobilenet_v2(pretrained=pretrained_backbone, output_stride=output_stride)
    
    # rename layers
    backbone.low_level_features = backbone.features[0:4]
    backbone.high_level_features = backbone.features[4:-1]
    backbone.features = None
    backbone.classifier = None

    inplanes = 320
    low_level_planes = 24
    
    if name=='deeplabv3plus':
        return_layers = {'high_level_features': 'out', 'low_level_features': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'high_level_features': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model

if __name__ == '__main__':
    x = torch.randn(2,64, 256, 256)
    scnn = SpatialConv(x.shape[1])
    out = scnn(x)
    print(x.shape)
    print(out.shape)