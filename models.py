import torch
from torch import nn, cat 
from torch.nn import functional

SAME = 'same'
VALID = 'valid'

ENCODER = 'encoder'
DECODER = 'decoder'

class BnConv(nn.Module):
	def __init__(self, channels_in, channels_out, kernel_size=3, padding=SAME, res_connect=False, half_dilated = False, activation = torch.relu) -> None:
		super().__init__()
		#self.bn = nn.BatchNorm2d(channels_out)
		self.conv = nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, padding=padding)
		self.activation = activation
		self.res_connect = res_connect and channels_in == channels_out 
	def forward(self, x):
		#if self.res_connect:
		#	return self.activation(self.bn(self.conv(x))) + x
		#else:
		#	return self.activation(self.bn(self.conv(x)))
		if self.res_connect:
			return self.activation(self.conv(x)) + x
		else:
			return self.activation(self.conv(x))

class UpBnConv(nn.Module):
	def __init__(self, channels_in, channels_out, kernel_size=4, stride=2, padding=1, activation=torch.relu) -> None:
		super().__init__()
		#self.bn = nn.BatchNorm2d(channels_out)
		self.conv = nn.ConvTranspose2d(channels_in, channels_out, kernel_size=kernel_size, stride=stride, padding=padding)
		self.activation = activation
	def forward(self, x):
		#return self.activation(self.bn(self.conv(x)))
		return self.activation(self.conv(x))
class DownBnConv(nn.Module):
	def __init__(self, channels_in, channels_out, kernel_size=2, stride=2, padding=0, activation=torch.relu) -> None:
		super().__init__()
		#self.bn = nn.BatchNorm2d(channels_out)
		self.conv = nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, stride=stride, padding=padding)
		self.activation = activation
	def forward(self, x):
		#return self.activation(self.bn(self.conv(x)))
		return self.activation(self.conv(x))

class ConvStack(nn.Module):
	def __init__(self, channels_in, channels_out, layers, res_connect=False, half_dilated = False) -> None:
		super().__init__()
		self.first_layer = BnConv(channels_in, channels_out, res_connect=res_connect)
		self.layers = nn.ModuleList([BnConv(channels_out, channels_out, res_connect=res_connect) for _ in range(layers - 1)])
	def forward(self, x):
		x = self.first_layer(x)
		for layer  in self.layers:
			x = layer(x)
		return x


class DownMod(nn.Module):
	def __init__(self, channels_in, channels_out, res_connect=False,  half_dilated = False) -> None:
		super().__init__()
		self.stack = ConvStack(channels_in, channels_in, layers=2, res_connect=res_connect)
		self.down = DownBnConv(channels_in, channels_out) 
	def forward(self, x):
		x = self.stack(x)
		x = self.down(x)
		return x	
	
class UpMod(nn.Module):
	def __init__(self, channels_in, channels_out, res_connect=False, half_dilated = False) -> None:
		super().__init__()
		self.up = UpBnConv(channels_in, channels_out)
		self.stack = ConvStack(channels_out, channels_out, layers=2, res_connect=res_connect)
	def forward(self, x):
		x = self.up(x) 
		x = self.stack(x)
		return x	
	
#256 1024
#128 512
#64 256
#32 128
#16 64
#8 32
class Model1(nn.Module):
	def __init__(self, channels, sequence_size = 1) -> None:
		super().__init__()
		self.sequence_size = sequence_size
		self.sections = [ENCODER, DECODER]
		self.layers = {
			ENCODER: [
				BnConv(channels, 32),
				DownMod(32, 32, res_connect=True),
				DownMod(32, 64, res_connect=True),
				DownMod(64, 128, res_connect=True),
				ConvStack(128, 128, 2, res_connect=True)
			],
			DECODER: [
				UpMod(128, 64, res_connect=True),
				UpMod(64, 32, res_connect=True),
				UpMod(32, 32, res_connect=True),
				BnConv(32, channels, activation=torch.tanh),
			]
		}
		self.sub_modules = nn.ModuleDict({key: nn.Sequential(*layers) for key, layers in self.layers.items()}) 
	def forward(self, x):
		x = self.sub_modules[ENCODER](x)	
		x = self.sub_modules[DECODER](x)	
		return x
	


def concat(arr):
	return cat(arr, dim=1)
class Model2(nn.Module):
	def __init__(self, channels, sequence_size = 1) -> None:
		super().__init__()
		self.sequence_size = sequence_size
		self.drop = nn.Dropout2d(p=0.1)
		self.sections = [ENCODER, DECODER]
		self.layers = nn.ModuleDict({
			ENCODER: nn.ModuleList([
				BnConv(channels*sequence_size, 48, res_connect=False),
				ConvStack(48, 48, 2, res_connect=True),
				DownBnConv(48, 96),
				ConvStack(96, 96, 3, res_connect=True),
				DownBnConv(96, 192),
				ConvStack(192, 192, 3, res_connect=True),
			]),
			DECODER: nn.ModuleList([
				UpBnConv(192, 96),
				ConvStack(192, 96, 3, res_connect=True),
				UpBnConv(96, 48),
				ConvStack(96, 48, 2, res_connect=True),
				BnConv(48, channels, res_connect=False, activation=torch.tanh),
			])
		})
	def forward(self, x):
		x = concat(x)
		x = self.layers[ENCODER][0](x)	
		x1 = self.layers[ENCODER][1](x)	
		x = self.layers[ENCODER][2](x1)	
		x2 = self.layers[ENCODER][3](x)	
		x = self.layers[ENCODER][4](x2)	
		x = self.layers[DECODER][0](x)	
		x = self.layers[DECODER][1](concat([x2, x]))	
		x = self.layers[DECODER][2](x)	
		x = self.layers[DECODER][3](self.drop(concat([x1, x])))	
		x = self.layers[DECODER][4](x)	
		return x