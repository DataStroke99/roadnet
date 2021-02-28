import torch 
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
	def __init__(self, in_channels, out_channels,  selu=False, pooling = False, num_conv_l = 2, upsampling = None):

		super(Block, self).__init__()
		#layers = [] // Another way to arrange the layers, if not in the foward()
		self.upsampling = upsampling

		#Activation
		if selu:
			self.activation = nn.SELU()
		else:
			self.activation = nn.ReLU()


		# Convolution Layers
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride=1,padding=1)
		#layers.append(self.conv1)
		#layers.append(self.activation)
		self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
		#layers.append(self.conv2)
		


		# 3rd Conv Layer
		if num_conv_l > 2:
			self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride=1, padding=1)
			#layers.append(self.conv3)



		#Residue Side Conv Layer
		self.side_conv = nn.Conv2d(in_channels=out_channels,out_channels=1,kernel_size=1,stride=1, padding=0)
		


		#Pooling Layer
		if pooling:
			self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

		
		
		#self.layer = nn.Sequential(*layers)


	def foward(self,x):

		x = self.conv1(x)
		x = self.activation(x)

		x = self.conv2(x)
		x = self.activation(x)

		if num_conv_l > 2:
			x = self.conv3(x)
			x = self.activation(x)

		side_output = self.side_conv(x)

		x = self.max_pool(x)

		if self.upsampling is not None:
			side_output = F.interpolate(side_output,size=self.upsampling, mode='bilinear', align_corners=True)


		return x, side_output



		

	






class RoadNetModule(nn.Module):
	def __init__(self, input_shape=(128,128,3)):

		super(RoadNetModule, self).__init__()


		H,W,C = input_shape


		#Input Layer
		self.layer1 = Block(in_channels = C, out_channels = 64, selu=True, pooling=True)
		self.layer2 = Block(in_channels = 64, out_channels = 128, selu=True, pooling=True, upsampling=(H,W))

		self.layer3 = Block(in_channels = 128, out_channels = 256, selu=True, pooling=True, num_conv_l = 3, upsampling=(H,W))
		self.layer4 = Block(in_channels = 256, out_channels = 512, selu=True, pooling=True, num_conv_l = 3, upsampling=(H,W))
		self.layer5 = Block(in_channels = 512, out_channels = 512, selu=True, pooling=False , num_conv_l = 3, upsampling=(H,W))

		self.fusion = nn.Conv2d(in_channels= 5, out_channels=1, kernel_size=1, stride=1, padding=0)





	def foward(self,x):

		block1_output, block1_side = self.layer1(x)
		block2_output, block2_side = self.layer2(block1_output)
		block3_output, block3_side = self.layer3(block2_output)
		block4_output, block4_side = self.layer4(block3_output)
		block5_output, block5_side = self.layer5(block4_output)

		combine = torch.cat([block1_side, block2_side, block3_side, block4_side, block5_side], dim=1)
		out = self.fusion(combine)
		return block1_side, block2_side, block3_side, block4_side, block5_side, out








class RoadNetModule2(nn.Module):
	def __init__(self, input_shape=(128,128,4)):
		super(RoadNetModule2, self).__init__()

		H,W,C = input_shape

		self.layer1 = Block(in_channels = C, out_channels=32, selu = True, pooling = True)
		self.layer2 = Block(in_channels=32, out_channels=64, selu=True, pooling=True, upsampling=(H,W))
		self.layer3 = Block(in_channels=64, out_channels=128, selu=True, pooling=True, upsampling=(H,W))
		self.layer4 = Block(in_channels=128, out_channels=256, selu=True, pooling=False, upsampling=(H,W))

		self.fusion = nn.Conv2d(in_channels= 4, out_channels=1, kernel_size=1, stride=1, padding=0)

	def foward(self, x):

		block1_output, block1_side = self.layer1(x)
		block2_output, block2_side = self.layer2(block1_output)
		block3_output, block3_side = self.layer3(block2_output)
		block4_output, block4_side = self.layer4(block3_output)

		combine = torch.cat([block1_side, block2_side, block3_side, block4_side], dim=1)
		out = self.fusion(combine)
		return block1_side, block2_side, block3_side, block4_side, out





net = RoadNet()
print(net)





