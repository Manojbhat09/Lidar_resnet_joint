
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F




class Fire(nn.Module):
  def __init__(self, inplanes, squeeze_planes,
               expand1x1_planes, expand3x3_planes, bn_d=0.1):
    super(Fire, self).__init__()
    self.inplanes = inplanes
    self.bn_d = bn_d
    self.activation = nn.ReLU(inplace=True)
    self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
    self.squeeze_bn = nn.BatchNorm2d(squeeze_planes, momentum=self.bn_d)
    self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                               kernel_size=1)
    self.expand1x1_bn = nn.BatchNorm2d(expand1x1_planes, momentum=self.bn_d)
    self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                               kernel_size=3, padding=1)
    self.expand3x3_bn = nn.BatchNorm2d(expand3x3_planes, momentum=self.bn_d)

  def forward(self, x):
    x = self.activation(self.squeeze_bn(self.squeeze(x)))
    return torch.cat([
        self.activation(self.expand1x1_bn(self.expand1x1(x))),
        self.activation(self.expand3x3_bn(self.expand3x3(x)))
    ], 1)


class CAM(nn.Module):

  def __init__(self, inplanes, bn_d=0.1):
    super(CAM, self).__init__()
    self.inplanes = inplanes
    self.bn_d = bn_d
    self.pool = nn.MaxPool2d(7, 1, 3)
    self.squeeze = nn.Conv2d(inplanes, inplanes // 16,
                             kernel_size=1, stride=1)
    self.squeeze_bn = nn.BatchNorm2d(inplanes // 16, momentum=self.bn_d)
    self.relu = nn.ReLU(inplace=True)
    self.unsqueeze = nn.Conv2d(inplanes // 16, inplanes,
                               kernel_size=1, stride=1)
    self.unsqueeze_bn = nn.BatchNorm2d(inplanes, momentum=self.bn_d)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    # 7x7 pooling
    y = self.pool(x)
    # squeezing and relu
    y = self.relu(self.squeeze_bn(self.squeeze(y)))
    # unsqueezing
    y = self.sigmoid(self.unsqueeze_bn(self.unsqueeze(y)))
    # attention
    return y * x

# ******************************************************************************


class Encoder(nn.Module):
  """
     Class for Squeezeseg. Subclasses PyTorch's own "nn" module
  """

  def __init__(self, params):
    # Call the super constructor
    super(Encoder, self).__init__()
    print("Using SqueezeNet Backbone")
    self.use_range = params["input_depth"]["range"]
    self.use_xyz = params["input_depth"]["xyz"]
    self.use_remission = params["input_depth"]["remission"]
    self.extra_dim = params["input_depth"]["extra_dim"]
    self.extra_dim_num = params["input_depth"]["extra_dim_num"]
    self.bn_d = params["bn_d"]
    self.drop_prob = params["dropout"]
    self.OS = params["OS"]

    # input depth calc
    self.input_depth = 0
    self.input_idxs = []
    if self.use_range:
      self.input_depth += 1
      self.input_idxs.append(0)
    if self.use_xyz:
      self.input_depth += 3
      self.input_idxs.extend([1, 2, 3])
    if self.use_remission:
      self.input_depth += 1
      self.input_idxs.append(4)
    if self.extra_dim:
      self.input_depth += self.extra_dim_num
      self.input_idxs.extend([each for each in range(5, self.extra_dim_num+5)])
    print("Depth of backbone input = ", self.input_depth)

    # stride play
    self.strides = [2, 2, 2, 2]
    # check current stride
    current_os = 1
    for s in self.strides:
      current_os *= s
    print("Original OS: ", current_os)

    # make the new stride
    if self.OS > current_os:
      print("Can't do OS, ", self.OS,
            " because it is bigger than original ", current_os)
    else:
      # redo strides according to needed stride
      for i, stride in enumerate(reversed(self.strides), 0):
        if int(current_os) != self.OS:
          if stride == 2:
            current_os /= 2
            self.strides[-1 - i] = 1
          if int(current_os) == self.OS:
            break
      print("New OS: ", int(current_os))
      print("Strides: ", self.strides)
    print("input depth is ", self.input_depth)
    # encoder
    self.conv1a = nn.Sequential(nn.Conv2d(self.input_depth, 64, kernel_size=3,
                                          stride=[1, self.strides[0]],
                                          padding=1),
                                nn.BatchNorm2d(64, momentum=self.bn_d),
                                nn.ReLU(inplace=True),
                                CAM(64, bn_d=self.bn_d))
    self.conv1b = nn.Sequential(nn.Conv2d(self.input_depth, 64, kernel_size=1,
                                          stride=1, padding=0),
                                nn.BatchNorm2d(64, momentum=self.bn_d))
    self.fire23 = nn.Sequential(nn.MaxPool2d(kernel_size=3,
                                             stride=[1, self.strides[1]],
                                             padding=1),
                                Fire(64, 16, 64, 64, bn_d=self.bn_d),
                                CAM(128, bn_d=self.bn_d),
                                Fire(128, 16, 64, 64, bn_d=self.bn_d),
                                CAM(128, bn_d=self.bn_d))
    self.fire45 = nn.Sequential(nn.MaxPool2d(kernel_size=3,
                                             stride=[1, self.strides[2]],
                                             padding=1),
                                Fire(128, 32, 128, 128, bn_d=self.bn_d),
                                Fire(256, 32, 128, 128, bn_d=self.bn_d))
    self.fire6789 = nn.Sequential(nn.MaxPool2d(kernel_size=3,
                                               stride=[1, self.strides[3]],
                                               padding=1),
                                  Fire(256, 48, 192, 192, bn_d=self.bn_d),
                                  Fire(384, 48, 192, 192, bn_d=self.bn_d),
                                  Fire(384, 64, 256, 256, bn_d=self.bn_d),
                                  Fire(512, 64, 256, 256, bn_d=self.bn_d))

    self._conv1 = nn.Sequential(
			nn.Conv2d(
				in_channels=512, out_channels=256, kernel_size=2, stride=2
			)
		)

    # output
    self.dropout = nn.Dropout2d(self.drop_prob)

    # last channels
    self.last_channels = 512

  def run_layer(self, x, layer, skips, os):
    y = layer(x)
    if y.shape[2] < x.shape[2] or y.shape[3] < x.shape[3]:
      skips[os] = x.detach()
      os *= 2
    x = y
    return x, skips, os

  def forward(self, x):
    # filter input
#     print("before filtering ", x.shape)
    x = x[:, self.input_idxs]

    # run cnn
    # store for skip connections
    skips = {}
    os = 1
#     print("/"*20, "starting encoder")
#     print(x.shape)
    x = x.float()
    # encoder
    skip_in = self.conv1b(x)
    x = self.conv1a(x)
    # first skip done manually
    skips[1] = skip_in.detach()
    os *= 2

    x, skips, os = self.run_layer(x, self.fire23, skips, os)
#     print("fire23 shape ", x.size())
    x, skips, os = self.run_layer(x, self.dropout, skips, os)
#     print("dropout shape ", x.size())
    x, skips, os = self.run_layer(x, self.fire45, skips, os)
#     print("fire45 shape ", x.size())
    x, skips, os = self.run_layer(x, self.dropout, skips, os)
#     print("dropout shape ", x.size())
    x, skips, os = self.run_layer(x, self.fire6789, skips, os)
#     print("fire6789 shape ", x.size())
#     x, skips, os = self.run_layer(x, self.dropout, skips, os)
#     print("dropout shape ", x.size())

    return x, skips

  def get_last_depth(self):
    return self.last_channels

  def get_input_depth(self):
    return self.input_depth
    
    

class FireUp(nn.Module):

  def __init__(self, inplanes, squeeze_planes,
               expand1x1_planes, expand3x3_planes, bn_d, stride):
    super(FireUp, self).__init__()
    self.inplanes = inplanes
    self.bn_d = bn_d
    self.stride = stride
    self.activation = nn.ReLU(inplace=True)
    self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
    self.squeeze_bn = nn.BatchNorm2d(squeeze_planes, momentum=self.bn_d)
    if self.stride == 2:
      self.upconv = nn.ConvTranspose2d(squeeze_planes, squeeze_planes,
                                       kernel_size=[1, 4], stride=[1, 2],
                                       padding=[0, 1])
    self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                               kernel_size=1)
    self.expand1x1_bn = nn.BatchNorm2d(expand1x1_planes, momentum=self.bn_d)
    self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                               kernel_size=3, padding=1)
    self.expand3x3_bn = nn.BatchNorm2d(expand3x3_planes, momentum=self.bn_d)

  def forward(self, x):
    x = self.activation(self.squeeze_bn(self.squeeze(x)))
    if self.stride == 2:
      x = self.activation(self.upconv(x))
    return torch.cat([
        self.activation(self.expand1x1_bn(self.expand1x1(x))),
        self.activation(self.expand3x3_bn(self.expand3x3(x)))
    ], 1)


# ******************************************************************************

class Decoder(nn.Module):
  """
     Class for DarknetSeg. Subclasses PyTorch's own "nn" module
  """

  def __init__(self, params, OS=32, feature_depth=512):
    super(Decoder, self).__init__()
    self.backbone_OS = OS
    self.backbone_feature_depth = feature_depth
    self.drop_prob = params["dropout"]
    self.bn_d = params["bn_d"]

    # stride play
    self.strides = [2, 2, 2, 2]
    # check current stride
    current_os = 1
    for s in self.strides:
      current_os *= s
    print("Decoder original OS: ", int(current_os))
    # redo strides according to needed stride
    for i, stride in enumerate(self.strides):
      if int(current_os) != self.backbone_OS:
        if stride == 2:
          current_os /= 2
          self.strides[i] = 1
        if int(current_os) == self.backbone_OS:
          break
    print("Decoder new OS: ", int(current_os))
    print("Decoder strides: ", self.strides)

    # decoder
    # decoder
    self.firedec10 = FireUp(self.backbone_feature_depth,
                            64, 128, 128, bn_d=self.bn_d,
                            stride=self.strides[0])
    self.firedec11 = FireUp(256, 32, 64, 64, bn_d=self.bn_d,
                            stride=self.strides[1])
    self.firedec12 = FireUp(128, 16, 32, 32, bn_d=self.bn_d,
                            stride=self.strides[2])
    self.firedec13 = FireUp(64, 16, 32, 32, bn_d=self.bn_d,
                            stride=self.strides[3])

    # for a bit of fun
    self.dropout = nn.Dropout2d(self.drop_prob)

    # last channels
    self.last_channels = 64
    
#     self._conv1 = nn.Sequential(
#         nn.Conv2d(
#             in_channels=512, out_channels=256, kernel_size=2, stride=2
#         ))
    
#     self._conv2 = nn.Sequential(
#         nn.Conv2d(
#             in_channels=256, out_channels=128, kernel_size=2, stride=2
#         ))
#     self._conv3 = nn.Sequential(
#         nn.Conv2d(
#             in_channels=128, out_channels=96, kernel_size=2, stride=2
#         ))
    
    self._conv1 = nn.Sequential(
        nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=6, stride=2
        ))
    


  def run_layer(self, x, layer, skips, os):
    feats = layer(x)  # up
    if feats.shape[-1] > x.shape[-1]:
      os //= 2  # match skip
      feats = feats + skips[os].detach()  # add skip
    x = feats
    return x, skips, os

  def forward(self, x, skips):
#     print("/"*30, " starting decoder ","/"*30)
    os = self.backbone_OS
#     print("input shape ", x.size())
    # run layers
    x, skips, os = self.run_layer(x, self.firedec10, skips, os)
#     print("firedec10 shape ", x.size())
    x, skips, os = self.run_layer(x, self.firedec11, skips, os)
#     print("firedec11 shape ", x.size())
    x, skips, os = self.run_layer(x, self.firedec12, skips, os)
#     print("firedec12 shape ", x.size())
    x, skips, os = self.run_layer(x, self.firedec13, skips, os)
#     print("firedec13 shape ", x.size())
    
#     out = self._conv1(x) # reduce dim 64 -> 32, channels 64 -> 32
#     out = self._conv2(out) # k=2, s=1. reduce dim 32 -> 30, channels 32 -> 32
    x = self._conv1(x) # k=6, s=2, reduce dim 64 -> 30, channels 64 -> 32
    x = self.dropout(x)
#     print("output shape ", x.size())
    return x

  def get_last_depth(self):
    return self.last_channels