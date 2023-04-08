import torch
from torch import nn 
from torch.nn import functional


ENCODER = 'encoder'
DECODER = 'decoder'

class BnConv(nn.Module):
	def __init__(self, channels_in, channels_out, kernel_size = 3, stride = 1, padding = 'same') -> None:
		super().__init__()
		self.bn = nn.BatchNorm2d(channels_out)
		self.conv = nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, stride=stride, padding=padding)
	def forward(self, x):
		return functional.relu(self.bn(self.conv(x)))
class ConvStack(nn.Module):
	def __init__(self, channels, layer_count) -> None:
		super().__init__()
		self.layers = nn.ModuleList([BnConv(channels, channels) for _ in range(layer_count)])
	def forward(self, x):
		for layer  in self.layers:
			x = layer(x)
		return x

class DownMod(nn.Module):
	def __init__(self, channels_in, channels_out) -> None:
		super().__init__()
		self.stack = ConvStack(channels_in, 3)
		self.down = BnConv(channels_in, channels_out, kernel_size=2, stride=2)
	def forward(self, x):
		x = self.stack(x)
		x = self.down(x)
		return x	
	
class UpMod(nn.Module):
	def __init__(self, channels_in, channels_out) -> None:
		super().__init__()
		self.up = nn.ConvTranspose2d(channels_in, channels_out, kernel_size=2, stride=2)
		self.bn_up = nn.BatchNorm2d(channels_out)
		self.stack = ConvStack(channels_out, 2)
	def forward(self, x):
		x = functional.relu(self.bn_up(self.up(x)))
		x = self.stack(x)
		return x	
	
#256 1024
#128 512
#64 256
#32 128
#16 64
#8 32
class Model1(nn.Module):
	def __init__(self, channels) -> None:
		super().__init__()
		self.sections = [ENCODER, DECODER]
		self.layers = {
			ENCODER: [
				BnConv(channels, 32),
				DownMod(32, 32),
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
				UpMod(32, 32),
				BnConv(32, channels),
			]
		}
		self.sub_modules = nn.ModuleDict({key: nn.Sequential(*layers) for key, layers in self.layers.items()}) 
	def forward(self, x):
		x = self.sub_modules[ENCODER](x)	
		x = self.sub_modules[DECODER](x)	
		return x