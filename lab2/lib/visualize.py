import numpy as np
from matplotlib import ticker

RATE = 20


def draw_trajectory(ax, trajectory, label=None, color='r', scale=(100, 100)):
    trajectory_x, trajectory_y = points_in_box(trajectory, scale)
    if label is None:
        ax.plot(trajectory_x, trajectory_y,  f",{color}-")
    else:
        ax.plot(trajectory_x, trajectory_y,  f",{color}-", label=label)

def draw_contour(ax, f, scale=(100, 100)):
    tx, ty = np.linspace(-scale[0], scale[0], RATE), np.linspace(-scale[1], scale[1], RATE)
    x, y = np.meshgrid(tx, ty)
    z = calculate_3d_grid(f, x, y)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.contourf(x, y, z, locator=ticker.LogLocator())

def points_in_box(points, scale):
    points_x = points[:, 0]
    points_y = points[:, 1]
    in_box = (np.abs(points_x) <= scale[0]) & (np.abs(points_y) <= scale[1])
    points_x = points_x[in_box]
    points_y = points_y[in_box]
    return points_x, points_y

def calculate_3d(f, x, y):
    z = np.zeros_like(x)
    for i in range(x.shape[0]):
        z[i] = f(np.array([x[i], y[i]]))
    return z

def calculate_3d_grid(f, x, y):
    z = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i][j] = f(np.array([x[i][j], y[i][j]]))
    return z

