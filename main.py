from weather_dataset import WeatherDataset, TrainingTransforms, AIRMASS, MICROCLOUD, SANDWICH, plot_from_dataset, plot_images, plot_image_pairs
from torch.utils.data import DataLoader
from torch import optim, nn
from models import Model1
import torch
import torchvision
import random





USE_AMP = True

CUDA = 'cuda'
CPU = 'cpu'

DATA_SUBSETS = [AIRMASS]
SHUFFLE = True

LOSS = None

ADAM = 'adam'
SGD = 'sgd'
OPTIMIZER = ADAM 

ADAM_LR = 0.0002
SGD_LR = 0.01
GAMMA = 0.96

TRAIN = 'train'
EVAL = 'eval'
MODE = TRAIN 

INPUT_IMAGE_SIZE = 256 

NEW_MODEL = True 

EPOCHS = 10 
BATCH_SIZE = 16 
LOADER_WORKERS = 4 
PERSITANT_WORKERS = LOADER_WORKERS > 0

read_path = "model_saves\\model1_1" 
save_path = "model_saves\\model1_1"


def run_model():
	torch.manual_seed(0)

	device = torch.device(CUDA if torch.cuda.is_available() else CPU)
	use_amp = USE_AMP and device.type == CUDA
	print(device.type)
	print(USE_AMP)
	
	model = Model1(3 * len(DATA_SUBSETS))

	if not NEW_MODEL:
		model.load_state_dict(torch.load(read_path, map_location='cpu'))


	if MODE == TRAIN:
		train_dataset = WeatherDataset.new_from_files("data", DATA_SUBSETS, truth_sequence_size=1, truth_offset=1, stride=6)	
		valid_dataset = train_dataset.split_set(0.8)
		train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=LOADER_WORKERS, persistent_workers=PERSITANT_WORKERS, pin_memory=True)

		#criterion = nn.KLDivLoss()
		criterion = nn.MSELoss()
		
		optimizer = optim.Adam(model.parameters(), lr=ADAM_LR, weight_decay=0.00002) if OPTIMIZER == ADAM else optim.SGD(model.parameters(), lr=SGD_LR, momentum=0.9, weight_decay=0.00002)
		scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=GAMMA, verbose=True)
		#scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.5, verbose=True)
		scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)


		model = model.to(device)
		model.train()
		train_transforms = TrainingTransforms(INPUT_IMAGE_SIZE).to(device)

		best_loss = 10000000.0
		for epoch in range(EPOCHS):
			print(f"epoch: {epoch}")
			total_loss = 0
			total_sample = 0
			for batch_index, (images, truth_images) in enumerate(train_loader):
				if device == CUDA:
					torch.cuda.empty_cache()
				optimizer.zero_grad(set_to_none=True)
				train_transforms.scramble()
				images = [train_transforms(image.to(device)) for image in images]
				truth_images = [train_transforms(image.to(device)) for image in truth_images]
				for i in range(len(truth_images)):
					with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=USE_AMP):					
						outputs = model(images[-1])
						loss = criterion(outputs, truth_images[i])
						images.append(outputs.detach())
					scaler.scale(loss).backward()
					scaler.step(optimizer)
					scaler.update()
					total_loss += loss.item() * images[0].size(0) 
					total_sample += images[0].size(0) 
				print(f"batch: {batch_index}")
			scheduler.step()
			loss = total_loss / total_sample
			if loss < best_loss:
				print("saved")
				torch.save(model.state_dict(), save_path)
				best_loss = loss
			print(f"Training: Loss: {loss}")

			images = []
			demo_size = 4
			hold = valid_dataset.truth_sequence_size
			valid_dataset.truth_sequence_size = demo_size 
			image, truths = valid_dataset[0]
			image = image[-1][None, :].to(device)
			images.append(model(image).detach().to(CPU))
			for i in range(0, demo_size - 1):
				images.append(model(images[i].to(device)).detach().to(CPU))
			valid_dataset.truth_sequence_size = hold
			plot_image_pairs(list(zip(images, truths)))

			

		torch.save(model.state_dict(), save_path)
		
	if MODE == EVAL:
		pass

	
if __name__ == '__main__':
	run_model()