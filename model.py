import glob
import os
import torchvision.transforms as transforms 
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
import pandas

USE_AMP = True

CUDA = 'cuda'
CPU = 'cpu'

ADAM = 'adam'
SGD = 'sgd'

OPTIMIZER = SGD 

TRAIN = 'train'
TEST = 'sample'
EVAL = 'eval'

MODE = EVAL 

HEAVY = 'heavy'
MEDIUM = 'medium'
LITE = 'lite'

AUGMENTATION = HEAVY 

INPUT_IMAGE_SIZE = 480 

NEW_MODEL = False 
FREEZE_FEATURES = False

ADAM_LR = 0.0001
SGD_LR = 0.01
GAMMA = 0.96

EPOCHS = 1000
BATCH_SIZE = 32 
LOADER_WORKERS = 2 
PERSITANT_WORKERS = LOADER_WORKERS > 0
data_path = 'data/Assignment 3 Dataset/'
read_path = './PreTrain/effv2_m_4.pt'
save_path = './PreTrain/effv2_m_4.pt'
preictions_save_path = 'predictions.csv'

AIRMASS = "airmass"
SANDWICH = "sandwich"
MICROCLOUD = "microcloud"

def get_file_name(path):
	return path[path.rindex("/") + 1, -1]
class WeatherDataset(Dataset):
	def __init__(self, root, sets, sequence_size):
		super().__init__() 
		self.root = root
		self.sets = sets
		self.sequence_size = sequence_size
		self.paths = {}
		for name in sets:
			self.paths[name] = glob.glob(f"{root}/{name}/*.jpg", recursive=True).sort() 	
		indices_forward = self.match_paths_forward(0)
		indices_backward = self.match_paths_backward(-1)
		for key in self.paths:
			self.paths = self.paths[indices_forward[key], indices_backward[key]]
	def match_paths_forward(self, index):
		indices = {}
		largest_name = "0"
		for key, paths in self.paths.items():
			indices[key] = index 
			if paths[index] > largest_name:
				largest_name = paths[index]
		for key in self.paths:
			while self.paths[key][indices[key]] < largest_name :			
				indices[key] += 1	
		return indices 
	def match_paths_backward(self, index):
		indices = {}
		smallest_name = "111111111111111111111111"
		for key, paths in self.paths.items():
			indices[key] = index 
			if paths[index] < smallest_name:
				smallest_name = paths[index]
		for key in self.paths:
			while self.paths[key][indices[key]] > smallest_name:			
				indices[key] -= 1	
		return indices
	@staticmethod
	def new(root, sets, sequence_size):
		pass
	@staticmethod
	def new_from(root, sets, sequence_size, paths):
		pass
	def split_set(self, ratio):
		pass
	def __len__(self):
		smallest_length = len(self.paths[self.sets[0]]) 
		for name in self.sets:
			if len(self.paths[name]) < smallest_length:
				smallest_length = len(self.paths[name])
		return smallest_length - self.sequence_size - 1 
	def __getitem__(self, index):
		images_in = [] 
		for i in range(len(self.sets)):
			images_in += self.paths[self.sets[i]][index]
		images_out = [] 
		for i in range(len(self.sets)):
			images_out += self.paths[self.sets[i]][index]
		return self.standard_transforms(image.type(torch.float32)) / 255.0, self.data_frame.iloc[i][1]
	

def plot(dataset, index, times):
    fig, axs = plt.subplots(ncols=times, squeeze=False)
    for i in range(times):
        image_tuple = dataset[i + index]
        axs[0][i].imshow(image_tuple[0].permute(1, 2, 0))
        axs[0][i].set(title=image_tuple[1], xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
	torch.manual_seed(0)

	#TODO: looking into torch amp and maybe able to fit effnet v2 m on gpu

	device = torch.device(CUDA if torch.cuda.is_available() else CPU)
	#use_amp = use_amp and device == CUDA
	print(device == CUDA)
	print(USE_AMP)
	print(device)

	batched_augmentation_transforms = None
	if AUGMENTATION == HEAVY:
		batched_augmentation_transforms = nn.Sequential(
         transforms.RandomAdjustSharpness(0.4, 0.2),
			transforms.ElasticTransform(alpha=40.0),
         transforms.RandomPerspective(distortion_scale=0.35, p=0.35),
			transforms.RandomErasing(p=0.3),
			transforms.RandomErasing(p=0.3),
			transforms.RandomAffine(degrees=(0, 30), translate=(0.0, 0.5), scale=(0.7, 1.0), shear=(0.0, 0.4)),
			transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.2, 0.2)),
			transforms.RandomHorizontalFlip(p=0.5),
         transforms.RandomGrayscale(p=0.2),
		)
	elif AUGMENTATION == MEDIUM:
		batched_augmentation_transforms = nn.Sequential(
			transforms.RandomAffine(degrees=(0, 20), translate=(0.0, 0.2), scale=(0.8, 1.0), shear=(0.0, 0.2)),
			transforms.ColorJitter(brightness=(0.90, 1.10), contrast=(0.9, 1.1), saturation=(0.9, 1.1), hue=(-0.10, 0.10)),
			transforms.ElasticTransform(alpha=8.0),
			transforms.RandomHorizontalFlip(p=0.5),
		)
	else:
		batched_augmentation_transforms = nn.Sequential(
			transforms.RandomHorizontalFlip(0.5),
		)
	


	batched_standard_transforms =	None
	model = None
	if MODEL == RES34:
		model = TransferModel()
		batched_standard_transforms = nn.Sequential(
			transforms.Resize(256),
			transforms.CenterCrop(224),	
		)
	elif MODEL == EFFV2:
		model = EffTransferModel()
		batched_standard_transforms = nn.Sequential(
			transforms.Resize(INPUT_IMAGE_SIZE),
			transforms.CenterCrop(INPUT_IMAGE_SIZE),	
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		)
	else:
		model = NewModel()	
		batched_standard_transforms = nn.Sequential(
			transforms.Resize(512),
			transforms.CenterCrop(512),	
		)

	if not NEW_MODEL:
		model.load_state_dict(torch.load(read_path, map_location='cpu'))


	if MODE == TRAIN:
		train_dataset = AssignmentCarsDataset(data_path, TRAIN)
		train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=LOADER_WORKERS, persistent_workers=PERSITANT_WORKERS, pin_memory=True)

		criterion = nn.CrossEntropyLoss()
		optimizer = None
		if OPTIMIZER == ADAM:
			optimizer = optim.Adam(model.parameters(), lr=ADAM_LR, weight_decay=0.00002)
		else:
			optimizer = optim.SGD(model.parameters(), lr=SGD_LR, momentum=0.9, weight_decay=0.00002)
		scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=GAMMA, verbose=True)
		#scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.5, verbose=True)
		scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

		model = model.to(device)
		batched_augmentation_transforms = batched_augmentation_transforms.to(device)
		batched_standard_transforms = batched_standard_transforms.to(device)
		model.train()

		print(device)

		best_accuracy = 0.0
		best_loss = 10000000.0
		for epoch in range(EPOCHS):
			print(f"epoch: {epoch}")
			total_loss = 0
			total_sample = 0
			correct = 0
			for batch_index, (images, labels) in enumerate(train_loader):
				optimizer.zero_grad(set_to_none=True)
				if device == CUDA:
					torch.cuda.empty_cache()
				images = images.to(device)
				labels = labels.to(device)
				with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=USE_AMP):					
					images = batched_augmentation_transforms(images)
					images = batched_standard_transforms(images)
					outputs = model(images)
					loss = criterion(outputs, labels)
				scaler.scale(loss).backward()
				scaler.step(optimizer)
				scaler.update()
				#loss.backward()
				#optimizer.step()
				total_loss += loss.item() * images.size(0) 
				total_sample += images.size(0) 
				correct += (torch.argmax(outputs, dim=1, keepdim=False) == labels).sum().item()
			scheduler.step()
			accuracy = correct / total_sample
			loss = total_loss / total_sample
			if loss < best_loss:
				print("saved")
				torch.save(model.state_dict(), save_path)
				best_loss = loss
				best_accuracy = accuracy
			print(f"Training: Accuracy: {accuracy}, Loss: {loss}")
			

		torch.save(model.state_dict(), save_path)
		
	if MODE == EVAL:
		test_dataset = AssignmentCarsDataset(data_path, TEST)
		test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

		model = model.to(device)
		batched_standard_transforms = batched_standard_transforms.to(device)

		model.eval()
		for i, (image, _) in enumerate(test_loader):
			image = image.to(device)
			image = batched_standard_transforms(image)
			outputs = model(image)
			predicted = torch.argmax(outputs, dim=1).item()
			test_dataset.data_frame.at[i, 'Predicted'] = predicted
			print(test_dataset.data_frame.iloc[i])
		
		test_dataset.data_frame.to_csv(preictions_save_path, index=False)