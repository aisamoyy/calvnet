import torch
import torch.nn as nn
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
def sphere_boundary(x):
    return (torch.sum(x[:, :2] ** 2, 1) - 1) ** 2 + torch.sum(
        x[:, 2:], 1
    )  # **2 + torch.sum(x[:,2:]**2,1)


def sphere_constraint(x):
    return torch.sum(x**2, 1).unsqueeze(1) - 1  # (n,) to (n,1)


def cylinder_constraint(x):
    return torch.sum(x[:, :2] ** 2, 1).unsqueeze(1) - 1


def ellipsoid_constraint(x):
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    return torch.unsqueeze((x1**2) / 1.0 + (x2**2) / 4.0 + (x3**2) / 9.0 - 1, 1)


def hyperbolic_constraint(x):
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    return torch.unsqueeze(x3 - x1**2 + x2**2, 1)


def torus_constraint(x, R=2.0, r=1.0):
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    return torch.unsqueeze((torch.sqrt(x1**2 + x2**2) - R) ** 2 + x3**2 - r**2, 1)

def velocity_constraint(x):
    return torch.sum(x**2, 1, keepdim=True) - 1


def energy_function(v):
    return torch.sum(v**2, 1, keepdim=True)
def helicoid_constraint(x, c=1.0):
    """
    Implicit function for helicoid surface:
        F(x, y, z) = x sin(z/c) - y cos(z/c) = 0
    Returns:
        F(x, y, z)
    """
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    return x1 * torch.sin(x3 / c) - x2 * torch.cos(x3 / c)

class Manifold:

    def _draw(self, X, Y, Z, path = None, points=[], geodesics_points=[], show = False):
        fig = go.Figure()
        fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.5))
        if len(points) > 0:
            for scatter in points:
                x, y, z = scatter
                fig.add_trace(go.Scatter3d(
                    x=[x], y=[y], z=[z],
                    mode='markers',
                    marker=dict(color='red', size=4),
                ))
        if len(geodesics_points) > 0:
            for scatter in geodesics_points:
                x, y, z = scatter
                fig.add_trace(go.Scatter3d(
                    x=[x], y=[y], z=[z],
                    mode='markers',
                    marker=dict(color='blue', size=2),
                ))
        max_range = max(np.ptp(X), np.ptp(Y), np.ptp(Z))
        x_mid = np.mean([np.min(X), np.max(X)])
        y_mid = np.mean([np.min(Y), np.max(Y)])
        z_mid = np.mean([np.min(Z), np.max(Z)])

        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title='X',
                    range=[x_mid - max_range/2, x_mid + max_range/2],
                ),
                yaxis=dict(
                    title='Y',
                    range=[y_mid - max_range/2, y_mid + max_range/2],
                ),
                zaxis=dict(
                    title='Z',
                    range=[z_mid - max_range/2, z_mid + max_range/2],
                ),
                aspectmode='cube'
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        fig.update_layout(showlegend=False)
        if path:
            fig.write_image(path)
        if show:
            fig.show()

class Sphere(Manifold):

    def __init__(self, r):
        self.r = r

    def uv_to_xyz(self, u, v):
        x = self.r * np.sin(u) * np.cos(v)
        y = self.r * np.sin(u) * np.sin(v)
        z = self.r * np.cos(u)
        return x, y, z

    def sampling_points(self, num_points):
        u = np.random.uniform(0, np.pi, num_points)  # Polar angle
        v = np.random.uniform(0, 2 * np.pi, num_points)  # Azimuthal angle
        x, y, z = self.uv_to_xyz(u, v)
        return list(zip(x, y, z))

    def plot(self, points = [], geodesics_points = [], path=None, show = False):
        u = np.linspace(0, np.pi, 50)
        v = np.linspace(0, 2 * np.pi, 100)
        U, V = np.meshgrid(u, v)
        X, Y, Z = self.uv_to_xyz(U, V)
        self._draw(X, Y, Z, path, points, geodesics_points, show = show)

    def constraint(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        return torch.unsqueeze((x1**2 + x2**2 + x3**2)**(1/2) - self.r, 1)

class Torus(Manifold):

    def __init__(self, R, r):
        self.R = R
        self.r = r

    def uv_to_xyz(self, u, v):
        x = (self.R + self.r * np.cos(v)) * np.cos(u)
        y = (self.R + self.r * np.cos(v)) * np.sin(u)
        z = self.r * np.sin(v)
        return x, y, z

    def sampling_points(self, num_points):
        u = np.random.uniform(0, 2 * np.pi, num_points)
        v = np.random.uniform(0, 2 * np.pi, num_points)
        x, y, z = self.uv_to_xyz(u, v)
        return list(zip(x, y, z))

    def plot(self, points = [], geodesics_points = [], path=None, show = False):
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, 2 * np.pi, 50)
        U, V = np.meshgrid(u, v)
        X, Y, Z = self.uv_to_xyz(U, V)
        self._draw(X, Y, Z, path, points, geodesics_points, show = show)

    def constraint(self, x):
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = x[:,2]

        eps = 1e-8
        radius = torch.sqrt(x1*x1 + x2*x2 + eps)

        return torch.unsqueeze(
            (radius - self.R)**2 + x3**2 - self.r**2,
            1
        )

class Ellipsoid(Manifold):

    def __init__(self, a=1.0, b=2.0, c=3.0):
        self.a = a
        self.b = b
        self.c = c

    def uv_to_xyz(self, u, v):
        x = self.a * np.sin(u) * np.cos(v)
        y = self.b * np.sin(u) * np.sin(v)
        z = self.c * np.cos(u)
        return x, y, z

    def sampling_points(self, num_points):
        u = np.random.uniform(0, np.pi, num_points)
        v = np.random.uniform(0, 2 * np.pi, num_points)
        x, y, z = self.uv_to_xyz(u, v)
        return list(zip(x, y, z))

    def plot(self, points = [], geodesics_points = [], path=None, show = False):
        u = np.linspace(0, np.pi, 50)
        v = np.linspace(0, 2 * np.pi, 100)
        U, V = np.meshgrid(u, v)
        X, Y, Z = self.uv_to_xyz(U, V)
        self._draw(X, Y, Z, path, points, geodesics_points, show = show)

    def constraint(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        return torch.unsqueeze((x1/self.a)**2 + (x2/self.b)**2 + (x3/self.c)**2 - 1, 1)

class Helicoid(Manifold):

    def __init__(self, c):
        self.c = c  # pitch factor

    def uv_to_xyz(self, u, v):
        x = u * np.cos(v)
        y = u * np.sin(v)
        z = self.c * v
        return x, y, z

    def sampling_points(self, num_points):
        u = np.random.uniform(-2, 2, num_points)  # controls radius
        v = np.random.uniform(0, 2 * np.pi, num_points)  # controls rotation (2 full turns)
        x, y, z = self.uv_to_xyz(u, v)
        return list(zip(x, y, z))

    def plot(self, points=[], geodesics_points=[], path=None, show=False):
        u = np.linspace(-2, 2, 50)  # width of helicoid
        v = np.linspace(0, np.pi, 100)  # number of rotations
        U, V = np.meshgrid(u, v)
        X, Y, Z = self.uv_to_xyz(U, V)
        self._draw(X, Y, Z, path, points, geodesics_points, show=show)

    def constraint(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        theta = torch.atan2(x2, x1)
        r = torch.sqrt(x1**2 + x2**2)
        return torch.unsqueeze(x3 - theta, 1)

class MobiusStrip(Manifold):

    def __init__(self, r):
        self.r = r  # radius of the strip

    def uv_to_xyz(self, u, v):
        # u controls position along strip (0 to 2π)
        # v controls position across strip (-1 to 1)
        x = (self.r + v * np.cos(u/2)) * np.cos(u)
        y = (self.r + v * np.cos(u/2)) * np.sin(u)
        z = v * np.sin(u/2)
        return x, y, z

    def sampling_points(self, num_points):
        u = np.random.uniform(0, 2 * np.pi, num_points)
        v = np.random.uniform(-1, 1, num_points)
        x, y, z = self.uv_to_xyz(u, v)
        return list(zip(x, y, z))

    def plot(self, points=[], geodesics_points=[], path=None, show=False):
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(-1, 1, 50)
        U, V = np.meshgrid(u, v)
        X, Y, Z = self.uv_to_xyz(U, V)
        self._draw(X, Y, Z, path, points, geodesics_points, show=show)

    def constraint(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        theta = torch.atan2(x2, x1)
        r = torch.sqrt(x1**2 + x2**2)
        v = x3 / torch.sin(theta/2)
        return torch.unsqueeze(r - (self.r + v * torch.cos(theta/2)), 1)