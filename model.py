import torch
from torch import nn
from torch.nn import functional as F


class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(out_ch),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(out_ch)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class up1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up1, self).__init__()

        self.conv = double_conv(out_ch, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.size()[2:], mode='bilinear', align_corners=True)
        x1 = self.conv1(x1)
        x = x1 + x2
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, need_feature_maps=False):
        super(UNet, self).__init__()
        # U-net1
        self.need_feature_maps = need_feature_maps
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        # print('x1:', x1.shape)
        x2 = self.down1(x1)
        # print('x2:', x2.shape)
        x3 = self.down2(x2)
        # print('x3:', x3.shape)
        x4 = self.down3(x3)
        # print('x4:', x4.shape)
        x5 = self.down4(x4)
        # print('x5:', x5.shape)
        x = self.up1(x5, x4)
        # print('up1:', x.shape)
        x = self.up2(x, x3)
        # print('up2:', x.shape)
        x = self.up3(x, x2)
        # print('up3:', x.shape)
        x = self.up4(x, x1)
        # print('up4:', x.shape)
        y = self.outc(x)
        # print('y:', y.shape)
        if self.need_feature_maps:
            return y, x
        return y


class Doc_UNet(nn.Module):
    def __init__(self, input_channels, n_classes):
        super(Doc_UNet, self).__init__()
        # U-net1
        self.U_net1 = UNet(input_channels, n_classes, need_feature_maps=True)
        self.U_net2 = UNet(64 + n_classes, n_classes, need_feature_maps=False)

    def forward(self, x):
        y1, feature_maps = self.U_net1(x)
        x = torch.cat((feature_maps, y1), dim=1)
        # print("x:",x.shape)
        # print("y1:",y1.shape)
        y2 = self.U_net2(x)
        # print("y2:",y2.shape)
        return y2
    