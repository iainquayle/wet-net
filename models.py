import torch
from torch import nn 
from torch.nn import functional


ENCODER = 'encoder'
DECODER = 'decoder'

#TODO: consider removing a layer
class BnConv(nn.Module):
	def __init__(self, channels_in, channels_out, kernel_size = 3, stride = 1, padding = 'same', res_connect=False, half_dilated = False, activation = torch.relu) -> None:
		super().__init__()
		padding = 'valid' if padding == 'same' and stride > 1 else padding 
		self.bn = nn.BatchNorm2d(channels_out)
		self.conv = nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, stride=stride, padding=padding)
		self.activation = activation
		self.res_connect = res_connect
	def forward(self, x):
		if self.res_connect:
			return self.activation(self.bn(self.conv(x))) + x
		else:
			return self.activation(self.bn(self.conv(x)))

class ConvStack(nn.Module):
	def __init__(self, channels, layers, res_connect=False, half_dilated = False) -> None:
		super().__init__()
		self.layers = nn.ModuleList([BnConv(channels, channels, res_connect=res_connect) for _ in range(layers)])
	def forward(self, x):
		for layer  in self.layers:
			x = layer(x)
		return x

class DownMod(nn.Module):
	def __init__(self, channels_in, channels_out, res_connect=False,  half_dilated = False) -> None:
		super().__init__()
		self.stack = ConvStack(channels_in, layers=2, res_connect=res_connect)
		self.down = BnConv(channels_in, channels_out, kernel_size=2, stride=2)
	def forward(self, x):
		x = self.stack(x)
		x = self.down(x)
		return x	
	
class UpMod(nn.Module):
	def __init__(self, channels_in, channels_out, res_connect=False, half_dilated = False) -> None:
		super().__init__()
		self.up = nn.ConvTranspose2d(channels_in, channels_out, kernel_size=2, stride=2)
		self.bn_up = nn.BatchNorm2d(channels_out)
		self.stack = ConvStack(channels_out, layers=2, res_connect=res_connect)
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
				DownMod(32, 32, res_connect=True),
				DownMod(32, 64, res_connect=True),
				DownMod(64, 128, res_connect=True),
				ConvStack(128, 2, res_connect=True)
			],
			DECODER: [
				UpMod(128, 64, res_connect=True),
				UpMod(64, 32, res_connect=True),
				UpMod(32, 32, res_connect=True),
				BnConv(32, channels, activation=torch.sigmoid),
			]
		}
		self.sub_modules = nn.ModuleDict({key: nn.Sequential(*layers) for key, layers in self.layers.items()}) 
	def forward(self, x):
		x = self.sub_modules[ENCODER](x)	
		x = self.sub_modules[DECODER](x)	
		return x