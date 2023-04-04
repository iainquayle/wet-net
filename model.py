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

class WeatherDataset(Dataset):
	def __init__(self, root):
		super().__init__() 
		self.root = root
		self.paths = {
			AIRMASS: glob.glob(f"{root}/{AIRMASS}/*.jpg", recursive=True), 	
			SANDWICH: glob.glob(f"{root}/{SANDWICH}/*.jpg", recursive=True), 	
			MICROCLOUD: glob.glob(f"{root}/{MICROCLOUD}/*.jpg", recursive=True), 	
		}
		indeces = [0] * len(self.paths)
		
	def __len__(self):
		len(self.airmass_paths)
	def __getitem__(self, i):
		image = torchvision.io.read_image(os.path.join(self.root, self.data_frame.iloc[i][0]))
		return self.standard_transforms(image.type(torch.float32)) / 255.0, self.data_frame.iloc[i][1]
	

class AssignmentCarsDataset(Dataset):
	def __init__(self, root, set):
		super().__init__() 
		self.root = root
		self.set = set
		self.data_frame = pandas.read_csv(os.path.join(root, f"{set}.csv"))
		self.standard_transforms = nn.Sequential(
			transforms.Resize(512),
			transforms.CenterCrop(size=(512 - 128, 512 + 64)),
			transforms.Pad(padding=256),
			transforms.CenterCrop(512 + 64),
		)
	def __len__(self):
		return len(self.data_frame)
	def __getitem__(self, i):
		image = torchvision.io.read_image(os.path.join(self.root, self.data_frame.iloc[i][0]))
		return self.standard_transforms(image.type(torch.float32)) / 255.0, self.data_frame.iloc[i][1]


def plot(dataset, index, times):
    fig, axs = plt.subplots(ncols=times, squeeze=False)
    for i in range(times):
        image_tuple = dataset[i + index]
        axs[0][i].imshow(image_tuple[0].permute(1, 2, 0))
        axs[0][i].set(title=image_tuple[1], xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.tight_layout()
    plt.show()

class Head512(nn.Module):
	def __init__(self, in_channels, out_channels) -> None:
		super().__init__()	
		self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=int(out_channels / 2), kernel_size=2, stride=2,  padding='valid')
		self.relu_1 = nn.ReLU()
		self.conv_2 = nn.Conv2d(in_channels=int(out_channels / 2), out_channels=out_channels, kernel_size=2, stride=2,  padding='valid')
		self.relu_2 = nn.ReLU()
	def forward(self, x):
		x = self.conv_1(x)
		x = self.relu_1(x)
		x = self.conv_2(x)
		x = self.relu_2(x)
		return x
class Head256(nn.Module):
	def __init__(self, in_channels, out_channels) -> None:
		super().__init__()	
		self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=int(out_channels / 3), kernel_size=3,  padding='same')
		self.relu_1 = nn.ReLU()
		self.conv_2 = nn.Conv2d(in_channels=int(out_channels / 3), out_channels=int(out_channels / 2), kernel_size=2, stride=2,  padding='valid')
		self.relu_2 = nn.ReLU()
		self.conv_3 = nn.Conv2d(in_channels=int(out_channels / 2), out_channels=out_channels, kernel_size=3,  padding='same')
		self.relu_3 = nn.ReLU()
	def forward(self, x):
		x = self.conv_1(x)
		x = self.relu_1(x)
		x = self.conv_2(x)
		x = self.relu_2(x)
		x = self.conv_3(x)
		x = self.relu_3(x)
		return x
class DownSample(nn.Module):
	def __init__(self, in_channels, out_channels) -> None:
		super().__init__()
		self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)
		self.conv_relu = nn.ReLU()
		self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
		self.pool_pointwise = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
		self.pool_relu = nn.ReLU()
	def forward(self, x):
		return torch.cat(tensors=(
			self.conv_relu(self.conv(x)),
			self.pool_relu(self.pool_pointwise(self.pool(x)))	
		), dim=1)
class ConvDownSample(nn.Module):
	def __init__(self, in_channels, out_channels) -> None:
		super().__init__()
		self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)
		self.conv_relu = nn.ReLU()
	def forward(self, x):
		return self.conv_relu(self.conv(x))
class Residual(nn.Module):
	def __init__(self, channels) -> None:
		super().__init__()
		self.conv_1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3,  padding='same')
		self.conv_2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3,  padding='same')
		self.conv_1x = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 1),  padding='same')
		self.conv_1y = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(1, 3),  padding='same')
		self.conv_2x = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 1),  padding='same')
		self.conv_2y = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(1, 3),  padding='same')
		self.conv_relu = nn.ReLU()
		self.batch_norm_1 = nn.BatchNorm2d(num_features=channels)
		self.batch_norm_2 = nn.BatchNorm2d(num_features=channels)
		self.final_batch_norm = nn.BatchNorm2d(num_features=channels)
		self.final_relu = nn.ReLU()
		self.dropout = nn.Dropout2d(p=0.1)
	def forward(self, x):
		y = self.conv_1(x)	
		#y = self.conv_1x(x)	
		#y = self.conv_1y(y)	
		y = self.batch_norm_1(y)	
		y = self.conv_relu(y)	
		y = self.conv_2(y)	
		#y = self.conv_2x(y)	
		#y = self.conv_2y(y)	
		y = self.batch_norm_2(y)	
		y = self.final_relu(y + x)
		#y = self.dropout(y)
		return y 

class TransferModel(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.pretrained_model = models.resnet34(pretrained=True) 
		num_features = self.pretrained_model.fc.in_features
		#for param in self.pretrained_model.parameters():
			#param.requires_grad = False
		self.pretrained_model.fc = nn.Sequential(
			nn.Linear(num_features, (num_features + 100) // 2),
			nn.ReLU(),
			nn.Dropout(0.1),
			nn.Linear((num_features + 100) // 2, 100),
			nn.Sigmoid()
		)
	def forward(self, x):
		return self.pretrained_model(x)	

class EffTransferModel(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.pretrained_model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT) 
		num_features = self.pretrained_model.classifier[-1].in_features
		#for param in self.pretrained_model.parameters():
		#	param.requires_grad = False
		new_classifier = list(self.pretrained_model.classifier.children())[:-1]
		new_classifier.extend([nn.Linear(num_features, 100)])	
		new_classifier = nn.Sequential(*new_classifier)
		#for param in new_classifier.parameters():
		#	param.requires_grad = True 
		self.pretrained_model.classifier = new_classifier 
	def forward(self, x):
		return self.pretrained_model(x)	


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