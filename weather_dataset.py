from glob import glob
from os.path import join
from torch import nn, cat, squeeze, tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional
from torchvision.io import read_image 
import matplotlib.pyplot as plt

AIRMASS = "airmass"
SANDWICH = "sandwich"
MICROCLOUD = "microcloud"

class WeatherDataset(Dataset):
	def __init__(self, root, set_names, sequence_size = 1):
		super().__init__() 
		self.root = root
		self.set_names = set_names 
		self.sequence_size = sequence_size if sequence_size > 0 else 1  
		paths = [sorted(list(map(lambda p: p[p.rindex("\\") + 1:], glob(join(root, key, "*.jpg"), recursive=True)))) for key in set_names]
		self.matching_file_names = [file_name for i, file_name in enumerate(paths[0]) if all([file_name in other_subset for other_subset in paths[1:]])]
		self.transforms = nn.Sequential(
			transforms.Resize(1024),
			#transforms.Normalize(0.5, 0.5),			
		)  
		self.mask = functional.resize(tensor([[[(i < 4 and j > 26) or (j > 30)  for i in range(32)] for j in range(32)]
		]), size=1024, interpolation=transforms.InterpolationMode.NEAREST)
	@staticmethod
	def new(root, sets, sequence_size):
		pass
	@staticmethod
	def new_from(root, sets, sequence_size, paths):
		pass
	def split_set(self, ratio):
		pass
	def __len__(self):
		return len(self.matching_file_names) - self.sequence_size + 1 - 1
	def __getitem__(self, index):
		images = cat([read_image(join(self.root, set_name, self.matching_file_names[index])) for set_name in self.set_names])
		truth_images = cat([read_image(join(self.root, set_name, self.matching_file_names[index + 1])) for set_name in self.set_names])
		return self.transforms(images / 255.0).masked_fill_(self.mask, 0.0),	self.transforms(truth_images / 255.0).masked_fill_(self.mask, 0.0)
	

def plot_image(img):
	img = img.detach()
	image_plot = plt.imshow(squeeze(img).permute(1, 2, 0))
	plt.show()

def plot(dataset, index, times):
	_, axs = plt.subplots(ncols=2, nrows=times, squeeze=False)
	for i in range(times):
		image_tuple = dataset[i + index]
		axs[i][0].imshow(image_tuple[0].permute(1, 2, 0))
		axs[i][1].imshow(image_tuple[0].permute(1, 2, 0))
	plt.tight_layout()
	plt.show()