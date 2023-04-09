from weather_dataset import WeatherDataset, AIRMASS, MICROCLOUD, SANDWICH, plot, plot_image
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

ADAM_LR = 0.0001
SGD_LR = 0.01
GAMMA = 0.96

TRAIN = 'train'
EVAL = 'eval'
MODE = TRAIN 

INPUT_IMAGE_SIZE = 480 

NEW_MODEL = True 

EPOCHS = 10 
BATCH_SIZE = 32 
LOADER_WORKERS = 2 
PERSITANT_WORKERS = LOADER_WORKERS > 0

read_path = "model_saves\\" 
save_path = "model_saves\\model1_1"


def run_model():
	torch.manual_seed(0)

	device = torch.device(CUDA if torch.cuda.is_available() else CPU)
	#use_amp = use_amp and device == CUDA
	print(device == CUDA)
	print(USE_AMP)
	print(device)

	
	model = Model1(3)
	print(type(model.parameters()))	


	if not NEW_MODEL:
		model.load_state_dict(torch.load(read_path, map_location='cpu'))


	if MODE == TRAIN:
		train_dataset = WeatherDataset.new_from_files("data", DATA_SUBSETS)	
		train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=LOADER_WORKERS, persistent_workers=PERSITANT_WORKERS, pin_memory=True)

		criterion = nn.KLDivLoss()
		
		optimizer = optim.Adam(model.parameters(), lr=ADAM_LR, weight_decay=0.00002) if OPTIMIZER == ADAM else optim.SGD(model.parameters(), lr=SGD_LR, momentum=0.9, weight_decay=0.00002)
		scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=GAMMA, verbose=True)
		#scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.5, verbose=True)
		scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)


		model = model.to(device)
		model.train()

		best_loss = 10000000.0
		for epoch in range(EPOCHS):
			print(f"epoch: {epoch}")
			total_loss = 0
			total_sample = 0
			for batch_index, (images, truth_images) in enumerate(train_loader):
				optimizer.zero_grad(set_to_none=True)
				if device == CUDA:
					torch.cuda.empty_cache()
				images = images.to(device)
				truth_images = truth_images.to(device)
				with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=USE_AMP):					
					outputs = model(images)
					loss = criterion(outputs, truth_images)
				scaler.scale(loss).backward()
				scaler.step(optimizer)
				scaler.update()
				#loss.backward()
				#optimizer.step()
				total_loss += loss.item() * images.size(0) 
				total_sample += images.size(0) 
			scheduler.step()
			loss = total_loss / total_sample
			if loss < best_loss:
				print("saved")
				torch.save(model.state_dict(), save_path)
				best_loss = loss
			print(f"Training: Loss: {loss}")
			

		torch.save(model.state_dict(), save_path)
		
	if MODE == EVAL:
		pass

	
if __name__ == '__main__':
	run_model()