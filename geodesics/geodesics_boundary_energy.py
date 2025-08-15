import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
torch.manual_seed(2010)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = 1
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
class basicNNu(nn.Module):
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
		x = inputs[:,:3]
		u = self.activation_u(self.linear_u_1(inputs))
		for m in [self.linear_u_2,self.linear_u_3,self.linear_u_4,self.linear_u_5]:
			u = self.activation_u(m(u))
		u = self.linear_u_6(u)

		if self.transform_fn is not None:
			u = self.transform_fn(x,u)
		u = torch.nn.functional.normalize(u)
		return u
def get_normal_vector(x):
	return torch.nn.functional.normalize(x)
def projection_tangent_sphere(x,u):
	n = get_normal_vector(x)
	u_proj = u - torch.sum(u*n,1,keepdim=True)*n
	return u_proj
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
def sphere_boundary(x):
	return (torch.sum(x[:,:2]**2,1)-1)**2 + torch.sum(x[:,2:],1)#**2 + torch.sum(x[:,2:]**2,1)
def sphere_constraint(x):
	return torch.sum(x**2,1).unsqueeze(1)-1
def cylinder_constraint(x):
	return torch.sum(x[:,:2]**2,1).unsqueeze(1)-1
def hyperbolic_constraint(x):
	x1 = x[:,0]
	x2 = x[:,1]
	x3 = x[:,2]
	return torch.unsqueeze(x3-x1**2+x2**2,1)
def velocity_constraint(x):
	return torch.sum(x**2,1,keepdim=True)-1
# def newton_dynamic(x,u):
# 	'''
# 	Input:
# 	x is the state of dimension batch_size x (ndim x ndim)
# 	u is the control of dimension batch_size x (ndim x mdim)
# 	Output:
# 	f(x,u) of dimension batch_size x ndim x ndim
# 	'''
# 	return u
def hamiltonian(x,xdot,p,u):
	return energy_function(xdot)+torch.sum(p*newton_dynamic(x,u),1)
def energy_function(v):
	return torch.sum(v**2,1,keepdim=True)
def pde(t,net_x,net_p,manifold_constraint,loss):
	x = net_x(t)
	p = net_p(t)
	#u = net_u(torch.cat([x,p],1))
	bz,xdim = x.shape[0],x.shape[1]
	pdim = p.shape[1]
	all_zero = torch.zeros((bz,xdim)).to(device)
	all_zero1 = torch.zeros((bz,1)).to(device)
	dx_dt = torch.zeros((bz,xdim)).to(device)
	ddx_dtt = torch.zeros((bz,xdim)).to(device)

	for j in range(xdim):
		tangent = torch.zeros_like(x)
		tangent[:,j]=1
		dx_dt[:,j] = torch.autograd.grad(x, t, grad_outputs=tangent,create_graph=True)[0][:,0]
	for j in range(xdim):
		tangent = torch.zeros_like(x)
		tangent[:,j]=1
		ddx_dtt[:,j] = torch.autograd.grad(dx_dt, t, grad_outputs=tangent,create_graph=True)[0][:,0]
	f = manifold_constraint(x)
	df_dx = torch.autograd.grad(f.sum(),x,create_graph=True)[0]
	g = energy_function(dx_dt)
	dg_xdot = torch.autograd.grad(g.sum(),dx_dt,create_graph=True)[0]
	ddg_dxx = torch.autograd.grad(dg_xdot, dx_dt,grad_outputs=ddx_dtt, create_graph=True)[0]

	pde1 = loss(f,all_zero1)
	pde2 = loss(p*df_dx-ddg_dxx,all_zero)
	return pde1,pde2
# def pde_u(x_p,net):
# 	u = net(x_p)
# 	x = x_p[:,:3]
# 	p = x_p[:,3:6]
# 	H_clone = hamiltonian(x,p,u)
# 	pde3 = torch.mean(H_clone,0)
# 	return pde3

def pde_bc(t_terminal,net_x,net_p,pf,boundary_constraint,manifold_constraint,loss):
	x_tf = net_x(t_terminal)
	p_tf = net_p(t_terminal)
	bz,xdim = x_tf.shape[0],x_tf.shape[1]
	s = boundary_constraint(x_tf)
	ds_dx = torch.autograd.grad(s.sum(), x_tf, create_graph=True)[0]

	all_zero = torch.zeros(1).to(device)
	all_zero3 = torch.zeros((1,3)).to(device)
	dx_dt = torch.zeros((bz,xdim)).to(device)
	for j in range(xdim):
		tangent = torch.zeros_like(x_tf)
		tangent[:,j]=1
		dx_dt[:,j] = torch.autograd.grad(x_tf, t_terminal, grad_outputs=tangent,create_graph=True)[0][:,0]
	g = energy_function(dx_dt)
	dg_xdot = torch.autograd.grad(g.sum(),dx_dt,create_graph=True)[0]
	loss1 = loss(s,all_zero)
	loss2 = loss(pf*ds_dx+dg_xdot,all_zero3)
	return loss1,loss2

if __name__ == '__main__':
	net_x = basicNN(1,3,'tanh').to(device)
	net_p = basicNN(1,3,'tanh').to(device)
	min_time = torch.tensor([[T]],dtype=torch.float32, requires_grad=True, device=device)
	pf = torch.tensor([[1.0]],dtype=torch.float32, requires_grad=True, device=device)
	loss_angle=torch.nn.CosineEmbeddingLoss()
	optimizer_x = torch.optim.SGD(net_x.parameters(),lr=0.000005)
	optimizer_p = torch.optim.SGD(net_p.parameters(),lr=0.000005)
	optimizer_pf = torch.optim.SGD([pf], lr=0.00001)

	net_x.train()
	net_p.train()
	# Logging data 
	writer = SummaryWriter(log_dir="runs/sphere_boundary_energy",comment="LR=1e-3_CST=1.2_alpha=1.09")
	## Data from Boundary Conditions
	mse_cost_function = torch.nn.MSELoss() # Mean squared error
	x_init = np.array([[1/3,-1/3,np.sqrt(7)/3]])
	t_init = np.array([[0.0]])
	### Training / Fitting
	alpha = 10
	iterations = 2000001
	for epoch in range(iterations):
		optimizer_x.zero_grad()
		optimizer_p.zero_grad()
		optimizer_pf.zero_grad()
		# Loss based on boundary conditions
		pt_x_init = Variable(torch.from_numpy(x_init).float(), requires_grad=False).to(device)
		pt_t_init = Variable(torch.from_numpy(t_init).float(), requires_grad=False).to(device)
		net_init_x = net_x(pt_t_init)
		mse_init = mse_cost_function(net_init_x, pt_x_init)

		loss1,loss2 = pde_bc(min_time,net_x,net_p,pf,sphere_boundary,sphere_constraint,mse_cost_function)
		mse_bc = mse_init+loss1+loss2
		# Loss based on PDE
		t_collocation = np.random.uniform(low=0.0, high=T, size=(5000,1))
		pt_t_collocation = Variable(torch.from_numpy(t_collocation).float(), requires_grad=True).to(device)	
		pde1,pde2 = pde(pt_t_collocation,net_x,net_p,sphere_constraint,mse_cost_function)
		mse_f = pde1+pde2
		# Combining the loss functions
		loss = alpha*mse_bc+10*mse_f 
		loss.backward()
		optimizer_x.step()
		optimizer_p.step()
		optimizer_pf.step()
		with torch.autograd.no_grad():
			print("Epoch:{} Pde1:{} Pde2:{} BC:{}".format(epoch,pde1,pde2,mse_bc))
		writer.add_scalar("Loss PDE 1", pde1, epoch)
		writer.add_scalar("Loss PDE 2", pde2, epoch)
		writer.add_scalar("Init State", mse_init, epoch)
		writer.add_scalar("Terminal state", loss1, epoch)
		writer.add_scalar("Terminal velocity state", loss2, epoch)
		writer.add_scalar("P_f", pf, epoch)
		if epoch%5000==0:
			alpha*=1.05
			torch.save(net_x.state_dict(), 'sphere/model_x_sphere_extend_'+str(epoch)+'.pth')
			torch.save(net_p.state_dict(), 'sphere/model_p_sphere_extend_'+str(epoch)+'.pth')