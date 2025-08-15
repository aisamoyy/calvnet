import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from nn import *
from utils import *
torch.manual_seed(2010)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = 1

# ================= Hyperparameters ====================
default_lr_x = 5e-6  # learning rate for net_x
default_lr_p = 5e-6  # learning rate for net_p
default_lr_pf = 1e-6  # learning rate for pf
default_iterations = 1000000 + 1
start_iteration = 0
activation_f = "tanh"  # ReLU tanh SiLU
# =====================================================

# ---------------------------
# Network and helper function definitions
# ---------------------------

t_init = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=False, device=device)
t_end = torch.tensor([[T]], dtype=torch.float32, requires_grad=False, device=device)

def pde(t, net_x, net_p, manifold_constraint):
    x = net_x(t)
    p = net_p(t)
    bz, xdim = x.shape

    # Compute time derivatives using the chain rule.
    dx_dt = torch.cat(
        [
            torch.autograd.grad(
                x,
                t,
                grad_outputs=torch.eye(xdim, device=device)[j].expand_as(x),
                create_graph=True,
            )[0][:, 0:1]
            for j in range(xdim)
        ],
        dim=1,
    )
    ddx_dtt = torch.cat(
        [
            torch.autograd.grad(
                dx_dt,
                t,
                grad_outputs=torch.eye(xdim, device=device)[j].expand_as(x),
                create_graph=True,
            )[0][:, 0:1]
            for j in range(xdim)
        ],
        dim=1,
    )

    f = manifold_constraint(x)
    df_dx = torch.autograd.grad(f.sum(), x, create_graph=True)[0]
    g = energy_function(dx_dt)
    dg_xdot = torch.autograd.grad(g.sum(), dx_dt, create_graph=True)[0]
    ddg_dxx = torch.autograd.grad(
        dg_xdot, dx_dt, grad_outputs=ddx_dtt, create_graph=True
    )[0]

    pde1 = torch.mean(f**2)
    pde2 = torch.mean((p * df_dx - ddg_dxx)**2)
    return pde1, pde2

def pde_bc(
    net_x, net_p, pf, boundary_constraint, manifold_constraint
):
    x_tf = net_x(t_terminal)
    bz, xdim = x_tf.shape
    s = boundary_constraint(x_tf)
    ds_dx = torch.autograd.grad(s.sum(), x_tf, create_graph=True)[0]
    dx_dt = torch.cat(
        [
            torch.autograd.grad(
                x_tf,
                t_terminal,
                grad_outputs=torch.eye(xdim, device=device)[j].expand_as(x_tf),
                create_graph=True,
            )[0][:, 0:1]
            for j in range(xdim)
        ],
        dim=1,
    )
    g = energy_function(dx_dt)
    dg_xdot = torch.autograd.grad(g.sum(), dx_dt, create_graph=True)[0]
    loss1 = torch.mean(s**2)
    loss2 = torch.sum((pf * ds_dx + dg_xdot)**2)
    return loss1, loss2


# ---------------------------
# Main training script with dynamic coefficient update
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Geodesic path solver on a manifold")
    parser.add_argument(
        "--lr_x", type=float, default=default_lr_x, help="Learning rate for net_x"
    )
    parser.add_argument(
        "--lr_p", type=float, default=default_lr_p, help="Learning rate for net_p"
    )
    parser.add_argument(
        "--lr_pf", type=float, default=default_lr_pf, help="Learning rate for pf"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=default_iterations,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--folder_name",
        type=str,
        default="1",
        help="Name of the folder to save the model",
    )
    parser.add_argument(
        "--manifold",
        type=str,
        default=None,
        help="manifold list: sphere, torus, cylinder, ellipsoid, hyperbolic",
    )
    parser.add_argument(
        "--bc",
        type=str,
        default=False,
        help="False: terminal points given, True: terminal boundary given",
    )
    args = parser.parse_args()
    iterations = args.iterations
    manifold = args.manifold
    bc = args.bc
    if manifold == "torus":
        manifold_constraint = torus_constraint
        x_init = torch.tensor([[0, 1.5, 1.0]]).float().to(device)
        x_end = torch.tensor([[0, 1.5, 1.0]]).float().to(device)
    elif manifold == "sphere":
        manifold_constraint = sphere_constraint
        x_init = torch.tensor([[0.0, 1.0, 0.0]]).float().to(device)
        x_end = torch.tensor([[1.0, 0.0, 0.0]]).float().to(device)
    elif manifold == "cylinder":
        manifold_constraint = cylinder_constraint
        x_init = torch.tensor([[0, 1.0, 0.0]]).float().to(device)
        x_end = torch.tensor([[0, 1.0, 0.0]]).float().to(device)
    elif manifold == "ellipsoid":
        manifold_constraint = ellipsoid_constraint
        x_init = torch.tensor([[1.0, 0.0, 0.0]]).float().to(device)
        x_end = torch.tensor([[0, 1.0, np.sqrt(27.0/4)]]).float().to(device)
    elif manifold == "hyperbolic":
        manifold_constraint = hyperbolic_constraint
        x_init = torch.tensor([[1.0, 0.0, 0.0]]).float().to(device)
        x_end = torch.tensor([[0.0, 1.0, 0.5]]).float().to(device)
    elif manifold == "helicoid":
        manifold_constraint = helicoid_constraint
        v = 0.5
        c=1
        z0 = 0.1
        z1 = 3*np.pi/4
        x_init = torch.tensor([[v*np.cos(z0), v*np.sin(z0), z0]]).float().to(device)
        x_end = torch.tensor([[v*np.cos(z1), v*np.sin(z1), z1]]).float().to(device)
    if bc:
        net_x = sineNN1(1, 3, activation_f,x_init).to(device)
        boundary_constraint = equator_boundary
    else:
        net_x = sineNN2(1, 3, activation_f,x_init,x_end).to(device)
    net_p = basicNN(1, 3, activation_f).to(device)

    optimizer_x = torch.optim.SGD(net_x.parameters(), lr=default_lr_x)
    optimizer_p = torch.optim.SGD(net_p.parameters(), lr=default_lr_p)

    net_x.train()
    net_p.train()

    # Initialize coefficients for loss components (they must sum to 1)
    if bc:
        f1, f2, f3,f4 = 0.25, 0.25,0.25,0.25
    else:
        f1, f2 = 0.5,0.5
    folder_name = args.folder_name
    os.makedirs(f"runs/{folder_name}", exist_ok=True)
    writer = SummaryWriter(log_dir=f"runs/{folder_name}/", comment="")

    for epoch in tqdm(range(start_iteration, iterations)):
        optimizer_x.zero_grad()
        optimizer_p.zero_grad()

        net_x_init=net_x(t_init)
        mse_init = torch.sum((net_x_init-x_init)**2)

        net_x_end = net_x(t_end)
        mse_end = torch.sum((net_x_end - x_end) ** 2)

        t_collocation = np.random.uniform(0.0, T, size=(5000, 1)).astype(np.float32)
        pt_t_collocation = torch.from_numpy(t_collocation).to(device)
        pt_t_collocation.requires_grad_()  # Ensure t requires grad
        pde1, pde2 = pde(pt_t_collocation, net_x, net_p, manifold_constraint)
        if bc:
            pde3,pde4 = pde_bc(net_x, net_p, pf, boundary_constraint, manifold_constraint)
            loss = f1 * pde1 + f2 * pde2 + f3 * pde3 + f4*pde4
        loss.backward()
        optimizer_x.step()
        optimizer_p.step()

        with torch.no_grad():
            if epoch % 1000 == 0 and epoch > 0:
                print(
                    f"Epoch:{epoch} mse_init:{mse_init.item():.6f} mse_end:{mse_end.item():.6f} "
                    f"pde1:{pde1.item():.6f} pde2:{pde2.item():.6f} Combined Loss:{loss.item():.6f}"
                )
        writer.add_scalar("Loss mse_init", mse_init, epoch)
        writer.add_scalar("Loss mse_end", mse_end, epoch)
        writer.add_scalar("Loss PDE 1", pde1, epoch)
        writer.add_scalar("Loss PDE 2", pde2, epoch)
        writer.add_scalar("Loss Combined", loss, epoch)

        # Dynamic coefficient update every 1000 epochs.
        if epoch % 1000 == 0 and epoch > 0:
            if bc:
                loss_vals = [
                    pde1.item(),
                    pde2.item(),
                    pde3.item(),
                    pde4.item(),
                ]
            else:
                loss_vals = [
                    pde1.item(),
                    pde2.item()
                ]
            idx_max, idx_min = np.argmax(loss_vals), np.argmin(loss_vals)
            if loss_vals[idx_max] > 10 * loss_vals[idx_min]:
                coeffs = [f1, f2]
                transfer = 0.5 * coeffs[idx_min]
                coeffs[idx_min] *= 0.5
                coeffs[idx_max] += transfer
                f1, f2 = coeffs
                print(
                    f"Dynamic update at epoch {epoch}: f1={f1:.4f}, f2={f2:.4f}"
                )
                writer.add_scalar("Coefficient f1", f1, epoch)
                writer.add_scalar("Coefficient f2", f2, epoch)

        # Save checkpoints every 5000 epochs.
        if epoch % 5000 == 0 and epoch > 0:
            os.makedirs(f"{folder_name}", exist_ok=True)
            torch.save(net_x.state_dict(), f"{folder_name}/x_{epoch}.pth")
            torch.save(net_p.state_dict(), f"{folder_name}/p_{epoch}.pth")
