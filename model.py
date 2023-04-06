import torch
from torch import nn 
from torch.nn import functional

#256 1024
#128 512
#64 256
#32 128
#16 64
#8 32

ENCODER = 'encoder'
DECODER = 'decoder'

class ConvStack(nn.Module):
	def __init__(self, channels, layer_count) -> None:
		super().__init__()
		self.layers = [(nn.BatchNorm2d(channels), nn.Conv2d(channels, channels, kernel_size=3, padding='same'))]
	def forward(self, x):
		for bn, conv in self.layers:
			x = bn(functional.relu(conv(x)))
		return x

class DownMod(nn.Module):
	def __init__(self, channels_in, channels_out) -> None:
		super().__init__()
		self.stack = ConvStack(channels_in, 3)
		self.down = nn.Conv2d(channels_in, channels_out, kernel_size=2, stride=2)
		self.bn_down = nn.BatchNorm2d(channels_in)
	def forward(self, x):
		x = self.stack(x)
		x = self.bn_down(functional.relu(self.down(x))) 
		return x	
	
class UpMod(nn.Module):
	def __init__(self, channels_in, channels_out) -> None:
		super().__init__()
		self.up = nn.ConvTranspose2d(channels_in, channels_out, kernel_size=2, stride=2)
		self.bn_up = nn.BatchNorm2d(channels_in)
		self.stack = ConvStack(channels_in, 2)
	def forward(self, x):
		x = self.bn_up(functional.relu(self.up(x))) 
		x = self.stack(x)
		return x	
	
class Model1(nn.Module):
	def __init__(self, channels) -> None:
		super().__init__()
		self.layers = {
			ENCODER: [
				DownMod(channels, 32),
				DownMod(32, 64),
				DownMod(64, 128),
				DownMod(128, 192),
				DownMod(192, 256),
			],
			DECODER: [
				UpMod(256, 192),
				UpMod(192, 128),
				UpMod(128, 64),
				UpMod(64, 32),
				UpMod(32, channels),
			]
		}