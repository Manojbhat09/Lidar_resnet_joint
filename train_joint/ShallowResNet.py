import numpy as np
import torch
import torchvision

import torch.nn as nn

class ResNet18Loss(nn.module):

	def __init__(self, shape=(30, 30)):
		
        super(self, ResNet18Loss).__init__()

		_resnet = torch.vision.models.resnet18(
			pretrained=True, progress=True
		)

		for layer in _resnet.parameters():
			layer.requires_grad = False

		self._conv1 = nn.Sequential(
			_resnet.conv1, _resnet.bn1, _resnet.relu, _resnet.maxpool,
			_resnet.layer1, _resnet.layer2
		)

		self._conv2 = nn.Sequential(
			_resnet.layer3,
			nn.ConvTranspose2d(
				256, 128,
				kernel_size=3, strides=2,
				padding=1, output_padding=1
			)
		)

		self._conv3 = nn.Sequential(
			_resnet.layer4,
			nn.ConvTranspose2d(
				512, 128,
				kernel_size=7, strides=4,
				padding=3, output_padding=3
			)
		)

		self._interp = nn.UpsamplingBilinear2d(
			size=shape
		)

		self._1x1conv = nn.Conv2d(
			inchannels=384, out_channels=32, kernel_size=1
			)


	def forward(self, tensor):
		
		channels = []

		channels.append(self._conv1(tensor))
		channels.append(self._conv2(channels[-1]))
		channels.append(self._conv3(channels[-1]))

		concat = torch.cat(channels, dim=1)

		interp = self._interp(output)

		map_rep = self._1x1conv(output)

		return output

