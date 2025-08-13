import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
torch.manual_seed(2024)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

y0 = np.array([1.0,0.0])
T = 3.0
A = np.array([[0.0,1.0],[0.0,0.0]]).T
B = np.array([[0.0],[1.0]]).T
A_torch=torch.Tensor(A).to(device)
B_torch=torch.Tensor(B).to(device)
sqrt5 = np.sqrt(1)
is_zero = 0.05

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

class basicNNu(nn.Module):
	"""Fully-connected neural network."""
	def __init__(self,input_size,output_size,activation_fn,transform_fn=None):
		super().__init__()
		self.linear_u_1 = nn.Linear(input_size,512)
		self.linear_u_2 = nn.Linear(512,1024)
		self.linear_u_3 = nn.Linear(1024,1024)
		self.linear_u_4 = nn.Linear(1024,512)
		self.linear_u_5 = nn.Linear(512,512)
		self.linear_u_6 = nn.Linear(512,output_size)
		self.tanh = nn.Tanh()
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
		x = inputs[:,:2]
		norm_x = torch.norm(x,dim=1)-0.01
		threshold = torch.unsqueeze(torch.sigmoid(100*norm_x),dim=1)
		u = self.activation_u(self.linear_u_1(inputs))
		for m in [self.linear_u_2,self.linear_u_3,self.linear_u_4,self.linear_u_5]:
			u = self.activation_u(m(u))
		u = self.linear_u_6(u)
		u=threshold*self.tanh(u)
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

def newton_dynamic(x,u):
	'''
	Input:
	x is the state of dimension batch_size x (ndim x ndim)
	u is the control of dimension batch_size x (ndim x mdim)
	Output:
	f(x,u) of dimension batch_size x ndim x ndim
	'''
	return x@A + u@B

def newton_dynamic_torch(x,u):
	'''
	Input:
	x is the state of dimension batch_size x (ndim x ndim)
	u is the control of dimension batch_size x (ndim x mdim)
	Output:
	f(x,u) of dimension batch_size x ndim x ndim
	'''
	return x@A_torch + u@B_torch
	
def solution_bangbang(t,x):
	'''
	Input:
	x is the state of dimension batch_size x 2
	u is the control of dimension batch_size x 1
	Output:
	f(x,u) of dimension batch_size x 2
	'''
	if t<sqrt5:
		u_bangbang = -np.ones((1))
	elif t<2*sqrt5 and t>sqrt5:
		u_bangbang = np.ones((1))
	else:
		u_bangbang = np.zeros((1))
	return x@A + u_bangbang@B

def solution_p1(t):
	'''
	Input:
	x is the state of dimension batch_size x 2
	u is the control of dimension batch_size x 1
	Output:
	f(x,u) of dimension batch_size x 2
	'''
	gradient = -1/sqrt5
	return t*0+gradient

def solution_p2(t):
	'''
	Input:
	x is the state of dimension batch_size x 2
	u is the control of dimension batch_size x 1
	Output:
	f(x,u) of dimension batch_size x 2
	'''
	gradient = -1/sqrt5
	return t*gradient+1
def solution_position(t):
	'''
	Input:
	x is the state of dimension batch_size x 2
	u is the control of dimension batch_size x 1
	Output:
	f(x,u) of dimension batch_size x 2
	'''
	midpoint = sqrt5
	if t<midpoint:
		return starting_pos-0.5*t*t
	elif t>midpoint and t<=2*midpoint:
		return starting_pos-0.5*midpoint*midpoint-midpoint*(t-midpoint)+0.5*(t-midpoint)**2
	else:
		return t*0.0
def integrate_state(t,x,net_x,net_p,net_u):
	t_tensor = torch.tensor([[t]]).float()
	x_tensor = torch.tensor([x]).view(1,2).float()
	#x_tensor = net_x(torch.tensor(t_tensor))
	p = net_p(torch.tensor(t_tensor))
	xp = torch.cat((x_tensor,p),1)
	u=net_u(xp).detach().numpy()
	norm_x = np.linalg.norm(x)
	norm_u = np.linalg.norm(u)
	if norm_x<is_zero:
		return steady_state(x,u)
	return newton_dynamic(x,u)

def newton_pde(t,net_x,net_p,net_u,loss):
	x = net_x(t)
	p = net_p(t)
	u = net_u(torch.cat([x,p],1))
	bz = x.shape[0]
	all_zero = torch.zeros((bz,2)).to(device)
	dx_dt = torch.zeros((bz,2)).to(device)
	for j in range(2):
		tangent = torch.zeros_like(x)
		tangent[:,j]=1
		dx_dt[:,j] = torch.autograd.grad(x, t, grad_outputs=tangent,create_graph=True)[0][:,0]
	dp_dt = torch.zeros((bz,2)).to(device)
	for j in range(2):
		tangent = torch.zeros_like(p)
		tangent[:,j]=1
		dp_dt[:,j] = torch.autograd.grad(p, t, grad_outputs=tangent,create_graph=True)[0][:,0]
	p_clone = p.clone().detach()
	x_clone = x.clone().detach()
	# u_clone = u.clone().detach()
	f_xu = newton_dynamic_torch(x,u)

	H = torch.sum(p*f_xu,1)+1
	dH_dx = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
	H_clone = torch.sum(p_clone*newton_dynamic_torch(x_clone,u),1)
	pde3 = torch.mean(H_clone,0)
	pde1 = loss(dx_dt-f_xu,all_zero)
	pde2 = loss(dH_dx+dp_dt,all_zero)
	return pde1,pde2,pde3

def pde_u(x_p,net):
	u = net(x_p)
	x = x_p[:,:3]
	p = x_p[:,3:6]
	H_clone = torch.sum(p*newton_dynamic_torch(x,u),1)
	pde3 = torch.mean(H_clone,0)
	return pde3


def gen_truedata():
	t = np.linspace(0, T, 500)
	sol = integrate.solve_ivp(solution_bangbang, (0, T), y0,t_eval=t)
	sigma_true = np.transpose(sol.y,(1,0))
	return sigma_true

def gen_sampledata(start,end):
	t = np.linspace(start, end, 150)
	sol = integrate.solve_ivp(solution_bangbang, (0, T), y0,t_eval=t)
	sigma_true = np.transpose(sol.y,(2,0))
	t = t.reshape(150,1)
	return t,sigma_true

def gen_predicted():
	t = np.linspace(0, T, 500)
	t_input = t.reshape(500,1)
	sol = integrate.solve_ivp(integrate_state, (0, T),y0 ,t_eval=t,args=(model,))
	sigma_pred = np.transpose(sol.y,(1,0))	  
	return t,sigma_pred

def find_terminal_time(net_x):
	t_range = torch.linspace(2*sqrt5,T,100).view(100,1).to(device)
	x_predicted = net_x(t_range)
	for i in range(100):
		if torch.norm(x_predicted)<is_zero:
			return t_range[i]
	return None

def pde_p_bc(t_terminal,net_x,net_p,net_u,loss):
	x_tf = torch.zeros((1,2)).to(device)
	p_tf = net_p(t_terminal) 
	u_tf = torch.ones((1,1)).to(device)						 # 
	all_zero = torch.zeros((1,1)).to(device)
	f_x_uclone=newton_dynamic_torch(x_tf,u_tf)
	H = torch.sum(p_tf*f_x_uclone,1)+1 #torch.sigmoid(cst*torch.sum(x**2,1))
	return loss(H,all_zero)

if __name__ == '__main__':
	
	# Define model
	net_x = basicNN(1,2,'tanh').to(device)
	net_p = basicNN(1,2,'ReLU').to(device)#resNN(1,2,'ReLU').to(device)
	net_u = basicNNu(4,1,'tanh').to(device)
	min_time = torch.tensor([[3.0]],dtype=torch.float32, requires_grad=True, device="cuda")
	optimizer_t = torch.optim.SGD([min_time], lr=0.00005)
	net_x.train()
	net_p.train()
	net_u.train()
	# Logging data 

	writer = SummaryWriter(log_dir="../runs/min_time",comment="LR=1e-3_CST=1.2_alpha=1.09")

	## Data from Boundary Conditions
	mse_cost_function = torch.nn.MSELoss() # Mean squared error
	
	x_init = np.array([[1.0,0.0]])
	t_init = np.array([[0.0]])
	x_end = np.array([[0.0,0.0]])

	p_terminal = np.array([[2.0],[-2.0]])
	t_terminal	= np.array([[0.0],[T]])

	### Training / Fitting
	optimizer_x = torch.optim.SGD(net_x.parameters(),lr=0.001)
	optimizer_p_pretrain = torch.optim.SGD(net_p.parameters(),lr=0.001)
	optimizer_u = torch.optim.SGD(net_u.parameters(),lr=0.001)
	iterations = 3005001
	iter_u=50
	iter_x=1

	#Initializing network p
	for epoch in range(2000):
		optimizer_p_pretrain.zero_grad()
		pt_p_terminal = Variable(torch.from_numpy(p_terminal).float(), requires_grad=False).to(device)
		pt_t_terminal = Variable(torch.from_numpy(t_terminal).float(), requires_grad=False).to(device)
		net_terminal_p = net_p(pt_t_terminal)[:,1].view(2,1)
		mse_terminal_p = mse_cost_function(net_terminal_p, pt_p_terminal)
		mse_terminal_p.backward()
		optimizer_p_pretrain.step()
	del optimizer_p_pretrain
	# Training train state + u alternatively
	optimizer_p = torch.optim.SGD(net_p.parameters(),lr=0.001)
	cst = 1
	alpha = 1.5
	for epoch in range(iterations):
		#Training the second part
		for epoch_u in range(10):
			optimizer_u.zero_grad()
			# Loss based on PDE
			t_collocation = np.random.uniform(low=-3.0, high=3.0, size=(5000,4))
			pt_t_collocation = Variable(torch.from_numpy(t_collocation).float(), requires_grad=True).to(device)
			loss_u = pde_u(pt_t_collocation, net_u)
			loss_u.backward() 
			optimizer_u.step()
		for epoch_x in range(iter_x):
			optimizer_x.zero_grad()
			optimizer_p.zero_grad()
			optimizer_t.zero_grad()

			# Loss based on boundary conditions
			pt_x_init = Variable(torch.from_numpy(x_init).float(), requires_grad=False).to(device)
			pt_t_init = Variable(torch.from_numpy(t_init).float(), requires_grad=False).to(device)
			net_init_x = net_x(pt_t_init)
			mse_init = mse_cost_function(net_init_x, pt_x_init)

			pt_x_end = Variable(torch.from_numpy(x_end).float(), requires_grad=False).to(device)
			net_end_x = net_x(min_time)
			mse_end = mse_cost_function(net_end_x, pt_x_end)

			pde_p = pde_p_bc(min_time,net_x,net_p,net_u,mse_cost_function)
			mse_bc = mse_init+mse_end+pde_p
			
			# Loss based on PDE
			t_collocation = np.random.uniform(low=0.0, high=T, size=(3000,1))
			pt_t_collocation = Variable(torch.from_numpy(t_collocation).float(), requires_grad=True).to(device) 
			pde1,pde2,pde3 = newton_pde(pt_t_collocation, net_x,net_p,net_u,mse_cost_function,cst) 
			mse_f = pde1+pde2+pde3

			# Combining the loss functions
			loss = alpha*mse_bc+mse_f 
			loss.backward()
			optimizer_x.step()
			optimizer_p.step()
			optimizer_t.step()
		with torch.autograd.no_grad():
			print("Epoch:{} Pde1:{} Pde2:{} BC:{}".format(epoch,pde1,pde2,mse_bc))
		writer.add_scalar("Loss PDE 1", pde1, epoch)
		writer.add_scalar("Loss PDE 2", pde2, epoch)
		writer.add_scalar("Loss PDE 3", pde3, epoch)
		writer.add_scalar("Loss BC", mse_bc, epoch)
		writer.add_scalar("Min time", min_time, epoch)
		if epoch%10000==0:
			alpha*=1.05
			torch.save(net_x.state_dict(), 'model_x_bc_'+str(epoch)+'.pth')
			torch.save(net_p.state_dict(), 'model_p_bc_'+str(epoch)+'.pth')
			torch.save(net_u.state_dict(), 'model_u_bc_'+str(epoch)+'.pth')
