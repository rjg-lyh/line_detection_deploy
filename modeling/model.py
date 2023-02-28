import torch
import torch.nn as nn
from .unet_parts import *
from .common import conv_block, up_conv, Attention_block, RRCNN_block, _segm_mobilenet, _segm_resnet, SpatialConv, U_netplus, W_Conv

def deeplab_resnet50(opts):
    model = _segm_resnet('deeplabv3', 'resnet50', num_classes=opts.num_classes, 
                        output_stride=8, pretrained_backbone=True)
    return model 

def deeplab_mobilenetv2(opts):
    model = _segm_mobilenet('deeplab', 'mobilenetv2', num_classes=opts.num_classes, 
                        output_stride=8, pretrained_backbone=True)
    return model 

class UNet(nn.Module):
    def __init__(self, opts, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = opts.input_channel
        self.n_classes = opts.num_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.head1 = DoubleConv(64, 64) # center_area detector
       # self.head2 = DoubleConv(64, 64) # main_lines detector
        self.out1 = OutConv(64, self.n_classes)
        #self.out2 = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        feature1 = self.head1(x)
        #feature2 = self.head2(x)
        out1 = self.out1(feature1)# center_area
       #out2 = self.out2(feature2)# main_lines
        return out1 #[2 2 256 256]

class U_Net_sim(nn.Module):
    def __init__(self, opts, bilinear=False):
        super(U_Net_sim, self).__init__()
        self.n_channels = opts.input_channel
        self.n_classes = 1
        self.bilinear = bilinear

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = up_conv(1024, 512)
        self.up2 = up_conv(512, 256)
        self.up3 = up_conv(256, 128)
        self.up4 = up_conv(128, 64)
        self.head1 = DoubleConv(64, 64) # center_area detector
        self.out1 = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)     
        d4 = self.up1(x5)
        d3 = self.up2(d4)
        d2 = self.up3(d3)
        d1 = self.up4(d2)
        feature1 = self.head1(d1)
        out1 = self.out1(feature1)
        return out1, x2, d2

class U_Net_Plus_sim(nn.Module):
    def __init__(self, opts, bilinear=False):
        super(U_Net_Plus_sim, self).__init__()
        self.n_channels = opts.input_channel
        self.n_classes = 1
        self.bilinear = bilinear

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = up_conv(1024, 512)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.up2 = up_conv(512, 256)
        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.up3 = up_conv(256, 128)
        self.up4 = up_conv(128, 64)
        self.head1 = DoubleConv(64, 64) # center_area detector
        self.out1 = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)  

        d4 = self.up1(x5)
        d4 = torch.cat((x4, d4), dim=1) 
        d4 = self.conv1(d4)

        d3 = self.up2(d4)
        d3 = torch.cat((x3, d3), dim=1) 
        d3 = self.conv2(d3)

        d2 = self.up3(d3)
        d1 = self.up4(d2)
        feature1 = self.head1(d1)
        out1 = self.out1(feature1)
        return out1, x2, d2



class AttU_Net(nn.Module):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """
    def __init__(self, opts):
        super(AttU_Net, self).__init__()

        n1 = 64
        self.n_channels = opts.input_channel
        self.n_classes = opts.num_classes
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
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
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

        #print(x5.shape)
        d5 = self.Up5(e5)
        #print(d5.shape)
        x4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

      #  out = self.active(out)

        return out

class R2AttU_Net(nn.Module):
    """
    Residual Recuurent Block with attention Unet
    Implementation : https://github.com/LeeJunHyun/Image_Segmentation
    """
    def __init__(self, opts):
        super(R2AttU_Net, self).__init__()

        t = 2
        n1 = 64
        self.n_channels = opts.input_channel
        self.n_classes = opts.num_classes
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.RRCNN1 = RRCNN_block(self.n_channels, filters[0], t=t)
        self.RRCNN2 = RRCNN_block(filters[0], filters[1], t=t)
        self.RRCNN3 = RRCNN_block(filters[1], filters[2], t=t)
        self.RRCNN4 = RRCNN_block(filters[2], filters[3], t=t)
        self.RRCNN5 = RRCNN_block(filters[3], filters[4], t=t)

        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_RRCNN5 = RRCNN_block(filters[4], filters[3], t=t)

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_RRCNN4 = RRCNN_block(filters[3], filters[2], t=t)

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_RRCNN3 = RRCNN_block(filters[2], filters[1], t=t)

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_RRCNN2 = RRCNN_block(filters[1], filters[0], t=t)

        self.Conv = nn.Conv2d(filters[0], self.n_classes, kernel_size=1, stride=1, padding=0)

       # self.active = torch.nn.Sigmoid()


    def forward(self, x):

        e1 = self.RRCNN1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.RRCNN2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.RRCNN3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.RRCNN4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.RRCNN5(e5)

        d5 = self.Up5(e5)
        e4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        e3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        e2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        e1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        out = self.Conv(d2)

      #  out = self.active(out)

        return out

class Scnn_AttU_Net(nn.Module):
    '''
    Just add SCNN module in AttU_Net
    '''
    def __init__(self, opts):
        super(Scnn_AttU_Net, self).__init__()

        n1 = 64
        self.n_channels = opts.input_channel
        self.n_classes = opts.num_classes
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
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.scnn = SpatialConv(filters[0])


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

        #print(x5.shape)
        d5 = self.Up5(e5)
        #print(d5.shape)
        x4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d2 = self.scnn(d2)

        out = self.Conv(d2)

      #  out = self.active(out)

        return out

class DGLNet1(nn.Module):
    def __init__(self, opts):
        super(DGLNet1, self).__init__()
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.Extractor = U_netplus(opts)
        self.Scnn = SpatialConv(filters[1])
        self.F1 = nn.Sequential(
            conv_block(filters[1], filters[1]),
            conv_block(filters[1], filters[1]),
        )
        self.F2 = nn.Sequential(
            conv_block(filters[1], filters[1]),
            conv_block(filters[1], filters[1]),
        )
        self.Attn1 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Attn2 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up = up_conv(filters[2], filters[1])
        self.head1 = nn.Sequential(
            W_Conv(filters[1]),
            conv_block(filters[1], filters[1]),
            conv_block(filters[1], filters[1]),
            nn.Conv2d(filters[1], 1, kernel_size=1, stride=1, padding=0)
        )
        self.head2 = nn.Sequential(
            W_Conv(filters[1]),
            conv_block(filters[1], filters[1]),
            conv_block(filters[1], filters[1]),
            nn.Conv2d(filters[1], 1, kernel_size=1, stride=1, padding=0)
        )
    
    def forward(self, x):
        out1, d3 = self.Extractor(x)

        x1 = self.F1(d3)
        x1 = self.Attn1(d3, x1)#128
        x2 = self.F2(d3)
        x2 = self.Attn2(d3, x2)

        x3 = torch.cat((x1, x2), dim=1)#256
        x3 = self.Up(x3)#128

        out2 = self.head1(x3)
        out3 = self.head2(x3)

        out = torch.cat((out1, out2, out3), dim = 1)
        return out

class DGLNet2(nn.Module):
    def __init__(self, opts):
        super(DGLNet2, self).__init__()
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.n_classes = 1
        self.Extractor = U_Net_sim(opts)
        self.conv1 = nn.Sequential(
            nn.Conv2d(n1 * 2, n1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(n1),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(n1 * 2, n1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(n1),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(n1, n1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n1)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(n1, n1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n1)
        )
        self.conve1 = nn.Sequential(
            nn.Conv2d(n1 * 2, n1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(n1),
            nn.ReLU(inplace=True),
        )
        self.conve2 = nn.Sequential(
            nn.Conv2d(n1, n1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n1)
        )
        self.filter1 = nn.Sequential(
            nn.Conv2d(n1, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.filter2 = nn.Sequential(
            nn.Conv2d(n1, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.up1 = up_conv(n1 * 2, n1)
        self.up2 = up_conv(n1 * 2, n1)
        self.out1 = OutConv(n1, self.n_classes)
        self.out2 = OutConv(n1, self.n_classes)

        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out1, e2, d2 = self.Extractor(x)#41.32
        e22 = self.conve1(e2)#降维128->64
        e2 = self.conve2(e22)
        x2_11 = self.conv1(d2)
        x2_22 = self.conv2(d2)
        x2_1 = self.conv3(x2_11)
        x2_2 = self.conv4(x2_22)       
        x2_1 = self.relu(x2_1 + e2)
        x2_2 = self.relu(x2_2 + e2)#41.34
        psi_1 = self.filter1(x2_1)
        psi_2 = self.filter1(x2_2)#41.34
        e2_1 = e22 * psi_1
        e2_2 = e22 * psi_2
        d2_1 = torch.cat((e2_1, x2_11), dim = 1)
        d2_2 = torch.cat((e2_2, x2_22), dim = 1)#41.3
        out2 = self.out1(self.up1(d2_1))#60.71
        out3 = self.out2(self.up2(d2_2))

        out = torch.cat((out1, out2, out3), dim = 1)
        return out

class DGLNet(nn.Module):
    def __init__(self, opts):
        super(DGLNet, self).__init__()
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.n_classes = 1
        self.Extractor = U_Net_Plus_sim(opts)
        self.conv1 = nn.Sequential(
            nn.Conv2d(n1 * 2, n1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(n1),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(n1 * 2, n1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(n1),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(n1, n1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n1)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(n1, n1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n1)
        )
        self.conve1 = nn.Sequential(
            nn.Conv2d(n1 * 2, n1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(n1),
            nn.ReLU(inplace=True),
        )
        self.conve2 = nn.Sequential(
            nn.Conv2d(n1, n1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n1)
        )
        self.filter1 = nn.Sequential(
            nn.Conv2d(n1, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.filter2 = nn.Sequential(
            nn.Conv2d(n1, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.up1 = up_conv(n1 * 2, n1)
        self.up2 = up_conv(n1 * 2, n1)
        self.out1 = OutConv(n1, self.n_classes)
        self.out2 = OutConv(n1, self.n_classes)

        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out1, e2, d2 = self.Extractor(x)#41.32
        e22 = self.conve1(e2)#降维128->64
        e2 = self.conve2(e22)
        x2_11 = self.conv1(d2)
        x2_22 = self.conv2(d2)
        x2_1 = self.conv3(x2_11)
        x2_2 = self.conv4(x2_22)       
        x2_1 = self.relu(x2_1 + e2)
        x2_2 = self.relu(x2_2 + e2)#41.34
        psi_1 = self.filter1(x2_1)
        psi_2 = self.filter1(x2_2)#41.34
        e2_1 = e22 * psi_1
        e2_2 = e22 * psi_2
        d2_1 = torch.cat((e2_1, x2_11), dim = 1)
        d2_2 = torch.cat((e2_2, x2_22), dim = 1)#41.3
        out2 = self.out1(self.up1(d2_1))#60.71
        out3 = self.out2(self.up2(d2_2))

        out = torch.cat((out1, out2, out3), dim = 1)
        return out