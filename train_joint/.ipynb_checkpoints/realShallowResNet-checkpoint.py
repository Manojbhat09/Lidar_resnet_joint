# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class conv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True):
        super(conv2DBatchNorm, self).__init__()

        self.cb_unit = nn.Sequential(nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                               padding=padding, stride=stride, bias=bias),
                                     nn.BatchNorm2d(int(n_filters)),)

    def forward(self, inputs):
        outputs = self.cb_unit(inputs)
        return outputs


class deconv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True):
        super(deconv2DBatchNorm, self).__init__()

        self.dcb_unit = nn.Sequential(nn.ConvTranspose2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                                         padding=padding, stride=stride, bias=bias),
                                      nn.BatchNorm2d(int(n_filters)),)

    def forward(self, inputs):
        outputs = self.dcb_unit(inputs)
        return outputs


class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True, dilation=1):
        super(conv2DBatchNormRelu, self).__init__()

        self.cbr_unit = nn.Sequential(nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                                padding=padding, stride=stride, bias=bias, dilation=dilation),
                                      nn.BatchNorm2d(int(n_filters)),
                                      nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs

class deconv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True):
        super(deconv2DBatchNormRelu, self).__init__()

        self.dcbr_unit = nn.Sequential(nn.ConvTranspose2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                                          padding=padding, stride=stride, bias=bias),
                                       nn.BatchNorm2d(int(n_filters)),
                                       nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.dcbr_unit(inputs)
        return outputs

    
class ResnetShallow(nn.Module):

    def __init__(self):  # Output Size: 30 * 30
        super(ResnetShallow, self).__init__()

        self.trunk = torchvision.models.resnet18(pretrained=True)

        self.upscale3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), )

        self.upscale4 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 7, stride=4, padding=3, output_padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), )

        self.shrink = conv2DBatchNormRelu(in_channels=384, n_filters=32,
                                          k_size=1, stride=1, padding=0)

    def forward(self, image):
        x = self.trunk.conv1(image)
        x = self.trunk.bn1(x)
        x = self.trunk.relu(x)
        x = self.trunk.maxpool(x)

        x = self.trunk.layer1(x)
        x2 = self.trunk.layer2(x)  # /8 the size
        x3 = self.trunk.layer3(x2)  # 16
        x4 = self.trunk.layer4(x3)  # 32

        x3u = self.upscale3(x3.detach())
        x4u = self.upscale4(x4.detach())

        xall = torch.cat((x2.detach(), x3u, x4u), dim=1)
        xall = F.interpolate(xall, size=(30, 30))
        output = self.shrink(xall)

        return output