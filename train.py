#Basic Imports
import os
import cv2 
import torch 
import numpy
import numpy as np
import pandas as pd

#Custom Imports
#from models import RoadNet

#Torch Imports
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import transforms




# Hyper-parameter

num_epochs = 100
batch_size = 5



#Dataset


class OttawaDataset(Dataset):
	def __init__(self, csv_file, root_dir, transform = None):
		self.csv_data = pd.read_csv(csv_file)
		self.root_dir = root_dir
		self.transform = transform
		'''
		-csv_file:
		-> folder_name [each image is stored in seperate folder]
		-> original_img [name of original training image] - .tif

		-edge.png
		-segmentation.png
		-centerline.png
		'''

	def __len__(self):
		return len(self.csv_data)

	def __getitem__(self, index):

		dir_path = os.path.dirname(os.path.realpath(__file__))+'/'

		image_path = os.path.join(self.root_dir, str(self.csv_data.iloc[index,0]), self.csv_data.iloc[index,1])
		edge_path = os.path.join(self.root_dir, str(self.csv_data.iloc[index,0]), "edge.png")
		segmentation_path = os.path.join(self.root_dir, str(self.csv_data.iloc[index,0]), "segmentation.png")
		centerline_path = os.path.join(self.root_dir, str(self.csv_data.iloc[index,0]), "centerline.png")

		image = cv2.imread(dir_path+image_path)
		edge_img = cv2.imread(dir_path+edge_path)
		segmentation_img = cv2.imread(dir_path+segmentation_path)
		centerline_img = cv2.imread(dir_path+centerline_path)


		if self.transform:
			image = self.transform(image)




		return [image,edge_img,segmentation_img,centerline_img]






#Data Transform

class Resize(object):
	def __init__(self, output_size):
		self.output_size = output_size

	def __call__(self, img, edge_img, segmentation_img, centerline_img):
		H,W = image.shape

		print(H,W)
		#Get ratio to resize


		#Get resized dimentions

		#Apply to images and store 
		



class Crop(object):
	def __init__(self,output_size):
		self.output_size = output_size

	def __call__(self, img, edge_img, segmentation_img, centerline_img):
		#Above step makes sure that rest of images, edge, segent, centerline are all lined up with the original
		#Once done they can be cut into smaller parts

		#Divide images into chunks of data 



		#Reshaping Images


		#Converting to tensors

		# why the above 2 steps needed  ? 





class ToTensor(object): #Transforms from cv2 array data to tensor, always needed to do 
	def __init__(self):
		self.output_size = ""#output_size

	def __call__(self):
		""



data_transform = transforms.Compose([
        Resize(256),
        Crop(128),
        transforms.ToTensor(), # WHy need to convert tot tensor ? 
        transforms.normalize() # Why good idea to normalize data

    ])







# Data Loader

dataset = OttawaDataset(csv_file = "data/ottawa.csv", 
						root_dir = "data/Ottawa-Dataset/", 
						transform= data_transform)


print(dataset.__getitem__(1))
print(dataset.__len__())

train_loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True)

#train_set, test_set = torch.utils.data.random_split(dataset,[20000,5000])

#def get_mean_std(loader):





print(train_loader)

d_iter = iter(train_loader)
print(d_iter)
data = next(d_iter)
print(data)

cv2.imshow("image 1", my_image_1)
cv2.imshow("image 2", my_image_2)
cv2.waitKey(0)



#Learning Rate Scheduler, after optimizer update



'''


# Roadnet Model




model = Module()






# loss and optimizer

#Construct Loss function and optimizer, model.parameters() will contain lernable paramerters of 2 nn.Linear layers 
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameeters(), lr = 0.01)




#Strat Training Network

for epochs in range(num_epochs):
	losses = []
	for i in enumerate(data_loader):
		pass



'''





