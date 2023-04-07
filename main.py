from weather_dataset import WeatherDataset, plot
from torch.utils.data import DataLoader
from models import Model1
import torchvision
import random

AIRMASS = "airmass"
SANDWICH = "sandwich"
MICROCLOUD = "microcloud"


if __name__ == '__main__':
	#dataset = WeatherDataset("data", [AIRMASS, SANDWICH, MICROCLOUD], 1)	
	print("init dataset")
	dataset = WeatherDataset("data", [AIRMASS], 1)	
	print("init dataloader")
	dataloader = DataLoader(dataset=dataset, batch_size=1)
	print("init model")
	model = Model1(3)
	img, _ = next(iter(dataloader)) 
	print("run model")
	img = model(img)
	print(img.size())
	#plot(dataset, 0, 1)

	for i in range(20):
		print(dataset.matching_file_names[i])
