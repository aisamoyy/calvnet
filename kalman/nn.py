import torch
import torch.nn as nn
from config import *
class basicNN(nn.Module):
	"""Fully-connected neural network."""
	def __init__(self,input_size,output_size,activation_fn,transform_fn=None):
		super().__init__()
		self.linear_u_1 = nn.Linear(input_size,512)
		self.linear_u_2 = nn.Linear(512,512)
		self.linear_u_3 = nn.Linear(512,512)
		self.linear_u_4 = nn.Linear(512,512)
		self.linear_u_5 = nn.Linear(512,512)
		self.linear_u_6 = nn.Linear(512,output_size)

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

class basicNNx(nn.Module):
	"""Fully-connected neural network."""
	def __init__(self,input_size,output_size,activation_fn,transform_fn=None):
		super().__init__()
		self.linear_u_1 = nn.Linear(input_size,512)
		self.linear_u_2 = nn.Linear(512,512)
		self.linear_u_3 = nn.Linear(512,512)
		self.linear_u_4 = nn.Linear(512,512)
		self.linear_u_5 = nn.Linear(512,512)
		self.linear_u_6 = nn.Linear(512,128)

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
		bz = inputs.shape[0]
		u = self.activation_u(self.linear_u_1(inputs))
		for m in [self.linear_u_2,self.linear_u_3,self.linear_u_4,self.linear_u_5]:
			u = self.activation_u(m(u))
		u = self.linear_u_6(u)
		if self.transform_fn is not None:
			u = self.transform_fn(u)
		pd = u.reshape(bz,4,32)
		pd_t = torch.transpose(pd,1,2)
		state = pd@pd_t
		return state.reshape(bz,16)

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
	
class basicNNxbc(nn.Module):
	"""Fully-connected neural network."""
	def __init__(self,input_size,output_size,activation_fn,transform_fn=None):
		super().__init__()
		self.linear_u_1 = nn.Linear(input_size,512)
		self.linear_u_2 = nn.Linear(512,512)
		self.linear_u_3 = nn.Linear(512,512)
		self.linear_u_4 = nn.Linear(512,512)
		self.linear_u_5 = nn.Linear(512,512)
		self.linear_u_6 = nn.Linear(512,128)
		self.sigmoid = nn.Sigmoid()
		self.init = x_start
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
		bz = inputs.shape[0]
		u = self.activation_u(self.linear_u_1(inputs))
		for m in [self.linear_u_2,self.linear_u_3,self.linear_u_4,self.linear_u_5]:
			u = self.activation_u(m(u))
		u = self.linear_u_6(u)
		if self.transform_fn is not None:
			u = self.transform_fn(u)
		pd = u.reshape(bz,4,32)
		pd_t = torch.transpose(pd,1,2)
		state = (pd@pd_t)
		state_reshape = state.reshape(bz,16)
		return state_reshape*(self.sigmoid(inputs)-0.5) + self.init
class basicNNp(nn.Module):
	"""Fully-connected neural network."""
	def __init__(self,input_size,output_size,activation_fn,transform_fn=None):
		super().__init__()
		self.linear_u_1 = nn.Linear(input_size,512)
		self.linear_u_2 = nn.Linear(512,512)
		self.linear_u_3 = nn.Linear(512,512)
		self.linear_u_4 = nn.Linear(512,512)
		self.linear_u_5 = nn.Linear(512,512)
		self.linear_u_6 = nn.Linear(512,output_size)
		self.end = p_end
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
		out_p = u*(T-inputs)+self.end
		return out_p