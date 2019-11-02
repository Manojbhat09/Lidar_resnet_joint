import numpy as np
import torch
import torchvision

import torch.nn as nn

class ResNet18Loss(nn.Module):

	def __init__(self, shape=(30, 30)):
		
# 		super(self, ResNet18Loss).__init__()
		super(ResNet18Loss, self).__init__()
		_resnet = torchvision.models.resnet18(
			pretrained=True, progress=True
		)

		for layer in _resnet.parameters():
			layer.requires_grad = False

		self._conv1 = nn.Sequential(
			_resnet.conv1, 
			_resnet.bn1, 
			_resnet.relu, 
			_resnet.maxpool,
			_resnet.layer1, 
			_resnet.layer2
		)

		self._conv2 = nn.Sequential(
			_resnet.layer3,
			nn.ConvTranspose2d(
				256, 256,
				kernel_size=3, stride=2,
				padding=1, output_padding=1
			)
		)

		self._conv3 = nn.Sequential(
			_resnet.layer4,
			nn.ConvTranspose2d(
				512, 256,
				kernel_size=7, stride=4,
				padding=3, output_padding=3
			)
		)
        
		self._conv4 = nn.Sequential(
			nn.Conv2d(
				in_channels=256, out_channels=128, kernel_size=2, stride=2
			)
		)
		self._conv5 = nn.Sequential(
			nn.Conv2d(
				in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=2
			)
		)

		self._interp = nn.UpsamplingBilinear2d(
			size=shape
		)

		self._1x1conv = nn.Conv2d(
			in_channels=128, out_channels=32, kernel_size=1
			)


	def forward(self, tensor):
		
		channels = []
# 		print("input shape ", tensor.size())
		channels.append(self._conv1(tensor))
# 		print("cov1 ", channels[-1].size())
		channels.append(self._conv2(channels[-1]))
# 		print("cov2 ", channels[-1].size())
		channels.append(self._conv3(channels[-1]))
# 		print("cov3 ", channels[-1].size())
# 		print("final ", [each.size() for each in channels])
        

		concat = torch.cat(channels[:2], dim=1)
# 		print("sec last s ", concat.size())
		out_dash = self._conv4(channels[2])
		concat = torch.cat([concat, out_dash], dim=1)
# 		print("last s ", concat.size())
# 		interp = self._interp(output)
		out_dash_2 = self._conv5(concat)
# 		print("last s ", out_dash_2.size())
		map_rep = self._1x1conv(out_dash_2)

		return map_rep

