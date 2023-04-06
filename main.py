from weather_dataset import WeatherDataset, plot
import torchvision
import random

AIRMASS = "airmass"
SANDWICH = "sandwich"
MICROCLOUD = "microcloud"


if __name__ == '__main__':
	#dataset = WeatherDataset("data", [AIRMASS, SANDWICH, MICROCLOUD], 1)	
	dataset = WeatherDataset("data", [AIRMASS], 1)	
	plot(dataset, 0, 1)

	for i in range(20):
		print(dataset.matching_file_names[i])
