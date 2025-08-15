import torch
import torch.nn as nn
import numpy as np
'''
Basic Neural Nets
'''
class basicNN(nn.Module):
	"""Fully-connected neural network."""
	def __init__(self,input_size,output_size,activation_fn,transform_fn=None):
		super().__init__()
		self.linear_u_1 = nn.Linear(input_size,1024)
		self.linear_u_2 = nn.Linear(1024,1024)
		self.linear_u_3 = nn.Linear(1024,1024)
		self.linear_u_4 = nn.Linear(1024,1024)
		self.linear_u_5 = nn.Linear(1024,1024)
		self.linear_u_6 = nn.Linear(1024,output_size)
		if activation_fn=='ReLU':
			self.activation_u= nn.ReLU()
		elif activation_fn=='tanh':
			self.activation_u= nn.Tanh()
		elif activation_fn=='SiLU':
			self.activation_u= nn.SiLU()

		self.transform_fn = transform_fn
		for m in [self.linear_u_1,self.linear_u_2,
				  self.linear_u_3,self.linear_u_4,
				  self.linear_u_5,self.linear_u_6]:
			torch.nn.init.xavier_uniform_(m.weight) 

	def forward(self, inputs):
		u = self.activation_u(self.linear_u_1(inputs))
		for m in [self.linear_u_2,self.linear_u_3,self.linear_u_4,self.linear_u_5]:
			u = self.activation_u(m(u))
		u = self.linear_u_6(u)
		if self.transform_fn is not None:
			u = self.transform_fn(u)
		return u
class sineNN(nn.Module):
	"""Fully-connected neural network."""
	def __init__(self,input_size,output_size,activation_fn,x_init,x_end,transform_fn=None):
		super().__init__()
		self.linear_u_1 = nn.Linear(input_size,1024)
		self.linear_u_2 = nn.Linear(1024,1024)
		self.linear_u_3 = nn.Linear(1024,1024)
		self.linear_u_4 = nn.Linear(1024,1024)
		self.linear_u_5 = nn.Linear(1024,1024)
		self.linear_u_6 = nn.Linear(1024,output_size)
		self.x_start = x_init
		self.x_end = x_end
		if activation_fn=='ReLU':
			self.activation_u= nn.ReLU()
		elif activation_fn=='tanh':
			self.activation_u= nn.Tanh()
		elif activation_fn=='SiLU':
			self.activation_u= nn.SiLU()

		self.transform_fn = transform_fn
		for m in [self.linear_u_1,self.linear_u_2,
				  self.linear_u_3,self.linear_u_4,
				  self.linear_u_5,self.linear_u_6]:
			torch.nn.init.xavier_uniform_(m.weight) 

	def forward(self, inputs):
		u = self.activation_u(self.linear_u_1(inputs))
		for m in [self.linear_u_2,self.linear_u_3,self.linear_u_4,self.linear_u_5]:
			u = self.activation_u(m(u))
		u = self.linear_u_6(u)*torch.sin(np.pi*inputs) + self.x_start*torch.cos(np.pi*inputs/2)+self.x_end*torch.sin(np.pi*inputs/2)
		return u
class basicNNp(nn.Module):
	"""Fully-connected neural network."""
	def __init__(self,input_size,output_size,activation_fn,transform_fn=None):
		super().__init__()
		self.linear_u_1 = nn.Linear(input_size,2)
		self.transform_fn = transform_fn
		for m in [self.linear_u_1]:
			torch.nn.init.xavier_uniform_(m.weight) 
	def forward(self, inputs):
		u = self.linear_u_1(inputs)
		return u
class resNN(nn.Module):
	"""Fully-connected neural network."""
	def __init__(self,input_size,output_size,activation_fn,transform_fn=None):
		super().__init__()
		self.linear_u_1 = nn.Linear(input_size,256)
		self.linear_u_2 = nn.Linear(256,256)
		self.linear_u_3 = nn.Linear(256,256)
		self.linear_u_4 = nn.Linear(256,256)
		self.linear_u_5 = nn.Linear(256,256)
		self.linear_u_6 = nn.Linear(256,output_size)
		if activation_fn=='ReLU':
			self.activation_u= nn.ReLU()
		elif activation_fn=='tanh':
			self.activation_u= nn.Tanh()
		elif activation_fn=='SiLU':
			self.activation_u= nn.SiLU()
		self.transform_fn = transform_fn
		for m in [self.linear_u_1,self.linear_u_2,
				  self.linear_u_3,self.linear_u_4,
				  self.linear_u_5,self.linear_u_6]:
			torch.nn.init.xavier_uniform_(m.weight) 
	def forward(self, inputs):
		u = self.activation_u(self.linear_u_1(inputs))
		for m in [self.linear_u_2,self.linear_u_3,self.linear_u_4,self.linear_u_5]:
			u = u+self.activation_u(m(u))
		u = self.linear_u_6(u)
		if self.transform_fn is not None:
			u = self.transform_fn(u)
		return u