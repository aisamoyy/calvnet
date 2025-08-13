# kalman new 1
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from nn import *
from config import *
from utils import *
import os
import argparse
import numpy as np
# ================= Hyperparameters ====================
default_lr = 5e-6  # learning rate for net
default_iterations = 2000000 + 1
start_iteration = 0
activation_f = "tanh"  # ReLU tanh SiLU
# =====================================================


# ---------------------------
# Main training script with dynamic coefficient update
# ---------------------------
	
def kalman_pde(t,net_x,net_p,net_u,loss,i):
	x = net_x(t)
	p = net_p(t)
	u = net_u(x)
	bz = x.shape[0]
	all_zero = torch.zeros((bz,16)).to(device)
	dx_dt = torch.zeros((bz,16)).to(device)
	for j in range(16):
		tangent = torch.zeros_like(x)
		tangent[:,j]=1
		dx_dt[:,j] = torch.autograd.grad(x, t, grad_outputs=tangent,create_graph=True)[0][:,0]
	dp_dt = torch.zeros((bz,16)).to(device)
	for j in range(16):
		tangent = torch.zeros_like(p)
		tangent[:,j]=1
		dp_dt[:,j] = torch.autograd.grad(p, t, grad_outputs=tangent,create_graph=True)[0][:,0]
	x_clone = x.clone().detach()
	p_clone = p.clone().detach()
	u_clone = u.clone().detach()
	f_xu = kalman_dynamic_torch(x,u)
	Hx = torch.sum(p_clone*kalman_dynamic_torch(x,u_clone))
	Hu = torch.sum(p_clone*kalman_dynamic_torch(x_clone,u))
	dH_du = torch.autograd.grad(Hu.sum(), u, create_graph=True)[0]
	dH_dx = torch.autograd.grad(Hx.sum(), x, create_graph=True)[0]
	pde1 = i*loss(dx_dt-f_xu,all_zero)
	pde2 = i*loss(dH_dx+dp_dt,all_zero)
	pde3 = i*loss(dH_du,all_zero)
	return pde1,pde2,pde3


# _,sigma_true = gen_truedata()
# kalman_gain = gen_kalman_gain(sigma_true)
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Geodesic path solver on a manifold")
	parser.add_argument(
        "--lr", type=float, default=default_lr, help="Learning rate for net_x"
    )
	parser.add_argument(
		"--folder_name",
		type=str,
		default="1",
		help="Name of the folder to save the model",
	)
	parser.add_argument(
        "--iterations",
        type=int,
        default=default_iterations,
        help="Number of training iterations",
    )
	args = parser.parse_args()
	folder_name = args.folder_name
	os.makedirs(f"runs/kalman/{folder_name}", exist_ok=True)
	os.makedirs(f"model/{folder_name}", exist_ok=True)
	writer = SummaryWriter(log_dir=f"runs/kalman/{folder_name}", comment="")
	lr, iterations = args.lr, args.iterations
	# Define model
	net_x = basicNNxbc(1,16,'tanh').to(device)
	net_p = basicNNp(1,16,'tanh').to(device)
	net_u = basicNN(16,16,'tanh').to(device)
	net_x.train()
	net_p.train()
	net_u.train()
	# Logging data 

	## Data from Boundary Conditions
	mse_cost_function = torch.nn.MSELoss() # Mean squared error

	### Training / Fitting
	optimizer_x = torch.optim.SGD(net_x.parameters(),lr=0.00005)
	optimizer_p = torch.optim.SGD(net_p.parameters(),lr=0.00005)
	optimizer_u = torch.optim.SGD(net_u.parameters(),lr=0.00005)
	iterations = 1005001

	cst = 1
	alpha = 1
	f1,f2,f3 = 1.0,1.0,1.0
	for epoch in tqdm(range(start_iteration, iterations)):
		#Training the second part
		optimizer_x.zero_grad()
		optimizer_p.zero_grad()
		optimizer_u.zero_grad()

		# Loss based on boundary conditions
		net_init_x = net_x(t_start)
		mse_init = torch.sum((net_init_x-x_start)**2)
		
		net_end_p = net_p(t_end)
		mse_end = torch.sum((net_end_p-p_end)**2)
			
		# Loss based on PDE
		t_collocation = np.random.uniform(low=0.0, high=T, size=(7000,1))
		pt_t_collocation = Variable(torch.from_numpy(t_collocation).float(), requires_grad=True).to(device)	
		pde1,pde2,pde3 = kalman_pde(pt_t_collocation,net_x,net_p,net_u,mse_cost_function,cst) 
		mse_f = f1*pde1+f2*pde2+f3*pde3

		# Combining the loss functions
		loss =mse_f 
		loss.backward()
		optimizer_x.step()
		optimizer_p.step()
		optimizer_u.step()
		with torch.no_grad():
			if epoch % 1000 == 0 and epoch > 0:
				print(
                    f"Epoch:{epoch} mse_init:{mse_init.item():.6f} mse_end:{mse_end.item():.6f} "
                    f"pde1:{pde1.item():.6f} pde2:{pde2.item():.6f} pde3:{pde3.item():.6f}"
                )
		writer.add_scalar("Loss PDE 1", pde1, epoch)
		writer.add_scalar("Loss PDE 2", pde2, epoch)
		writer.add_scalar("Loss PDE 3", pde3, epoch)
		writer.add_scalar("Loss Init", mse_init, epoch)
		writer.add_scalar("Loss Terminal", mse_end, epoch)
		#Dynamic coefficient update every 5000 epochs.
		if epoch % 1000 == 0 and epoch > 10000:
			loss_vals = [
                pde1.item(),
                pde2.item(),
				pde3.item()
            ]
			idx_max, idx_min = np.argmax(loss_vals), np.argmin(loss_vals)
			if loss_vals[idx_max] > 10 * loss_vals[idx_min]:
				coeffs = [f1, f2,f3]
				transfer = 0.5 * coeffs[idx_min]
				coeffs[idx_min] *= 0.5
				coeffs[idx_max] += transfer
				f1, f2,f3 = coeffs
				print(
                    f"Dynamic update at epoch {epoch}:  f1={f1:.4f}, f2={f2:.4f}, f3={f3:.4f}"
					)
				writer.add_scalar("Coefficient f1", f1, epoch)
				writer.add_scalar("Coefficient f2", f2, epoch)
				writer.add_scalar("Coefficient f3", f3, epoch)
		if epoch%10000==0:
			torch.save(net_x.state_dict(), f"model/{folder_name}/x_+{epoch}.pth")
			torch.save(net_p.state_dict(), f"model/{folder_name}/p_+{epoch}.pth")
			torch.save(net_u.state_dict(), f"model/{folder_name}/u_+{epoch}.pth")
