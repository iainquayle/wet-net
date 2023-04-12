from glob import glob
from os.path import join
import random
import torch
from torch import nn, cat, squeeze, tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional
from torchvision.io import read_image 
import matplotlib.pyplot as plt

AIRMASS = "airmass"
SANDWICH = "sandwich"
MICROCLOUD = "microcloud"

IMAGE_DIMENSION = 1024

class WeatherDataset(Dataset):
	def __init__(self, root, set_names, matching_file_names, sequence_size=1, truth_sequence_size=1, truth_offset=0, indexing=1):
		super().__init__() 
		self.root = root
		self.truth_offset = truth_offset 
		self.indexing = indexing
		self.set_names = set_names 
		self.sequence_size = sequence_size if sequence_size > 0 else 1  
		self.truth_sequence_size = truth_sequence_size
		self.matching_file_names = matching_file_names 
		self.transforms = nn.Sequential(
			transforms.Resize(IMAGE_DIMENSION),
		)  
		self.mask = functional.resize(tensor([[[(i < 4 and j > 26) or (j > 30)  for i in range(32)] for j in range(32)]
		]), size=IMAGE_DIMENSION, interpolation=transforms.InterpolationMode.NEAREST)
	@staticmethod
	def new_from_files(root, set_names, sequence_size=1, truth_sequence_size=1, truth_offset=0, indexing=1):
		print("initializing dataset")
		#cached_offsets = [0] * len(set_names)
		get_file_name = lambda p: p[p.rindex("\\") + 1:]
		paths = [sorted(list(map(get_file_name, glob(join(root, key, "*.jpg"), recursive=True)))) for key in set_names]
		#for i in range(1, len(set_names)):
		#	while chached_offsets
		matching_file_names = [file_name for i, file_name in enumerate(paths[0]) if all([file_name in other_subset for other_subset in paths[1:]])]
		print("dataset initiliazed")
		return WeatherDataset(root, set_names, matching_file_names,sequence_size, truth_sequence_size, truth_offset, indexing)
	def split_set(self, ratio):
		new_set = WeatherDataset(self.root, self.set_names, self.matching_file_names[int(len(self.matching_file_names) * ratio):], self.sequence_size, self.truth_sequence_size,self.truth_offset, self.indexing)
		self.matching_file_names = self.matching_file_names[:int(len(self.matching_file_names) * ratio)]
		return new_set	
	def __len__(self):
		return int(len(self.matching_file_names) / self.indexing) - self.sequence_size + 1 - self.truth_offset - self.truth_sequence_size
	def __getitem__(self, index):
		images = []
		for i in range(self.sequence_size):
			image = cat([read_image(join(self.root, set_name, self.matching_file_names[(index + i) * self.indexing])) for set_name in self.set_names])
			images.append(self.transforms(image / 255.0).masked_fill_(self.mask, 0.0))
		truth_images = []
		for i in range(self.truth_sequence_size):
			image = cat([read_image(join(self.root, set_name, self.matching_file_names[(self.sequence_size + index + i + self.truth_offset) * self.indexing])) for set_name in self.set_names])
			truth_images.append(self.transforms(image / 255.0).masked_fill_(self.mask, 0.0))
		return images, truth_images 
	def get_single(self, index):
		images = cat([read_image(join(self.root, set_name, self.matching_file_names[index])) for set_name in self.set_names])
		truth_images = cat([read_image(join(self.root, set_name, self.matching_file_names[index + self.truth_offset])) for set_name in self.set_names])
		return self.transforms(images / 255.0).masked_fill_(self.mask, 0.0),	self.transforms(truth_images / 255.0).masked_fill_(self.mask, 0.0)
		pass
	
class TrainingTransforms(nn.Module):
	def __init__(self, image_dimension_out) -> None:
		super().__init__()
		self.x = 0
		self.y = 0
		self.out_dim = image_dimension_out
	def forward(self, x):
		x = functional.crop(x, top=self.y, left=self.x, height=self.out_dim, width=self.out_dim)
		return x
	def scramble(self):
		self.x = random.randrange(IMAGE_DIMENSION - self.out_dim)
		self.y = random.randrange(IMAGE_DIMENSION - self.out_dim)

def plot_images(images):
	plt.figure(figsize=(15, 10))
	for index, image in enumerate(images):
		image =	functional.resize(image.detach(), 256) 
		plt.subplot(1, len(images), index + 1)
		plt.imshow(squeeze(image).permute(1, 2, 0))
		plt.axis('off')
	plt.show()

def plot_from_dataset(dataset, index, times):
	_, axs = plt.subplots(ncols=2, nrows=times, squeeze=False)
	for i in range(times):
		image_tuple = dataset[i + index]
		axs[i][0].imshow(image_tuple[0].permute(1, 2, 0))
		axs[i][1].imshow(image_tuple[0].permute(1, 2, 0))
	plt.tight_layout()
	plt.show()