import torch
import torch.nn as nn
import torch.functional as F 

from network import RoadNetModule, RoadNetModule2


class RoadNet(nn.Module):
	def __init__(self, input_shape=(128,128)):

		super(RoadNet,self).__init__()

		self.W = input_shape[0]
		self.H = input_shape[1]

		self.segment_net = RoadNetModule(input_shape=(self.H, self.W, 3))
		self.edge_net = RoadNetModule2(input_shape=(self.H, self.W, 4))
		self.centerline_net = RoadNetModule2(input_shape=(self.H, self.W, 4))




	def segment_foward(self, input):
		block1_side, block2_side, block3_side, block4_side, block5_side, out = self.segment_net(input)
		return [block1_side, block2_side, block3_side, block4_side, block5_side, out]

	def edge_foward(self, input):
		block1_side, block2_side, block3_side, block4_side, out = self.edge_net(input)
		return [block1_side, block2_side, block3_side, block4_side, out]


	def centerline_foward(self,input):
		block1_side, block2_side, block3_side, block4_side, out = self.centerline_net(input)
		return [block1_side, block2_side, block3_side, block4_side, out]


	def foward(self, input):
		segment = self.segment_foward(input)

		segment_out = segment[-1]
		input2 = torch.cat([input,segment_out], dim=1)

		centerlines = self.centerline_foward(input2)
		edges = self.edge_foward(input2)

		return segment, centerlines, edg






