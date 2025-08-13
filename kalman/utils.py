from scipy import integrate
from config import *

def gen_truedata(num=500,end=T):
	t = np.linspace(0, end, num)
	sol = integrate.solve_ivp(solution_kalman, (0, end), y0,t_eval=t)
	sigma_true = np.transpose(sol.y,(1,0))
	return t,sigma_true

def gen_sampledata(start,end):
	t = np.linspace(start, end, 150)
	sol = integrate.solve_ivp(solution_kalman, (0, T), y0,t_eval=t)
	sigma_true = np.transpose(sol.y,(1,0))
	t = t.reshape(150,1)
	return t,sigma_true
def gen_predicted(net_x,net_p,net_u,num=500,T=5.0):
    t = np.linspace(0, T, num)
    t_input = t.reshape(num,1)
    sol = integrate.solve_ivp(integrate_state, (0, T),y0 ,t_eval=t,args=(net_x,net_p,net_u,))
    sigma_pred = np.transpose(sol.y,(1,0))
    return sol.t,sigma_pred
def gen_kalman_gain(true_state):
	true_state_reshape = true_state.reshape(-1,4,4)
	kalman_gain = true_state_reshape@invR
	return kalman_gain.reshape(-1,16)
def solution_kalman(t,sigma):
	sigma = sigma.reshape(1,4,4)
	matrix_sol = A@sigma + sigma@A.T+B@Q@B.T-sigma@np.linalg.inv(R)@sigma
	return matrix_sol.reshape(16)
	
def integrate_state(t,x,net_x,net_p,net_u):
    x_tensor = torch.tensor([x]).view(1,16).float()
    u=net_u(x_tensor).detach().numpy()
    return kalman_dynamic(x,u)
def kalman_dynamic(x,u):
	'''
	Input:
	x is the state of dimension batch_size x (ndim x ndim)
	u is the control of dimension batch_size x (ndim x mdim)
	A is the state transition matrix of dimension ndim x ndim
	B is the input matrix of dimension ndim x cdim
	C is the measurement matrix of dimension mdim x ndim
	Q is the input noise covariance of dimension cdim x cdim
	R is the measurement noise covariance of dimension mdim x mdim

	Output:
	f(x,u) of dimension batch_size x ndim x ndim
	'''
	if len(x.shape)<3:
		x=x.reshape(-1,4,4)
	if len(u.shape)<3:
		u=u.reshape(-1,4,4)
	A_GC = A-u # dimension bz x ndim x ndim
	A_GCT = np.transpose(A_GC, (0, 2, 1))
	out = A_GC@x+ x@A_GCT+B@Q@B.T+ u@R@np.transpose(u, (0, 2, 1))
	return out.reshape(-1,16)

def kalman_dynamic_torch(x,u):
	'''
	Input:
	x is the state of dimension batch_size x (ndim x ndim)
	u is the control of dimension batch_size x (ndim x mdim)
	A is the state transition matrix of dimension ndim x ndim
	B is the input matrix of dimension ndim x cdim
	C is the measurement matrix of dimension mdim x ndim
	Q is the input noise covariance of dimension cdim x cdim
	R is the measurement noise covariance of dimension mdim x mdim

	Output:
	f(x,u) of dimension batch_size x ndim x ndim
	'''
	x = x.reshape(-1,4,4)
	u = u.reshape(-1,4,4)
	A_GC = A_torch-u # dimension bz x ndim x ndim
	A_GCT = torch.transpose(A_GC,  2, 1)
	out = A_GC@x+ x@A_GCT+B_torch@Q_torch@B_torch.T+ u@R_torch@torch.transpose(u, 2,1)
	return out.reshape(-1,16)