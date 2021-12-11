import random
import numpy as np 
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn
import torch.nn.functional as F 
import utility as utility

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class MLP(nn.Module):
	def __init__(self,nodes, n_classes):
		super().__init__()
		self.input_layer = nn.Linear(nodes[0], nodes[1])
		self.hidden_layer_1 = nn.Linear(nodes[1], nodes[2])
		self.hidden_layer_2 = nn.Linear(nodes[2], nodes[3])
		self.hidden_layer_3 = nn.Linear(nodes[3], nodes[4])
		self.output_layer = nn.Linear(nodes[4], n_classes)
	
	def forward(self, X):
		X = self.input_layer(X)
		X = F.relu(X)
		X = self.hidden_layer_1(X)
		X = F.relu(X)
		X = self.hidden_layer_2(X)
		X = F.relu(X)
		X = self.hidden_layer_3(X)
		X = F.relu(X)
		X = self.output_layer(X)
		prob = F.softmax(X, dim=1)
		return prob


class LeNet5(nn.Module):
	def __init__(self,first_channel,n_classes):
		super().__init__()
		self.model = nn.Sequential(
			nn.Conv2d(in_channels=first_channel, out_channels = 6, kernel_size = 5, stride = 1),
			nn.Tanh(),
			nn.AvgPool2d(kernel_size=2),
			nn.Conv2d(in_channels=6, out_channels = 16, kernel_size = 5, stride = 1),
			nn.Tanh(),
			nn.AvgPool2d(kernel_size=2),
			nn.Conv2d(in_channels=16, out_channels = 120, kernel_size = 5, stride = 1),
			nn.Tanh(),
			nn.Flatten(),
			nn.Linear(in_features = 120, out_features = 84),
			nn.Tanh(),
			nn.Linear(in_features = 84, out_features = n_classes),
			nn.Softmax(dim = 1)
		)
	def forward(self,X):
		prob = self.model(X)
		return prob

if __name__ == '__main__':
	pass