import numpy as np

RATE = 20

def draw_3d_with_trajectory(ax, trajectory, f, scale=(100, 100)):
    draw_3d(ax, f, scale)
    draw_trajectory_3d(ax, trajectory, f, scale)

def draw_3d(ax, f, scale=(100, 100)):
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    tx, ty = np.linspace(-scale[0], scale[0], RATE), np.linspace(-scale[1], scale[1], RATE)
    x, y = np.meshgrid(tx, ty)
    z = calculate_3d_grid(f, x, y)

    ax.plot_surface(x,
            y,
            z,
            rstride=1,
            cstride=1,
            alpha=0.5,
            cmap='viridis',
            edgecolor='none')

def draw_trajectory_3d(ax, trajectory, f, scale=(100, 100)):
    trajectory_x, trajectory_y = points_in_box(trajectory, scale)
    trajectory_z = calculate_3d(f, trajectory_x, trajectory_y)
    ax.plot(trajectory_x, trajectory_y, calculate_3d(f, trajectory_x, trajectory_y),  f".r-")
    ax.plot(trajectory_x[0], trajectory_y[0], trajectory_z[0], f"og")
    ax.plot(trajectory_x[-1], trajectory_y[-1], trajectory_z[-1], f"ob")


def draw_contour_with_trajectory(ax, trajectory, f, scale=(100, 100)):
    draw_contour(ax, f, scale)
    draw_trajectory(ax, trajectory, scale)

def draw_trajectory(ax, trajectory, scale=(100, 100)):
    trajectory_x, trajectory_y = points_in_box(trajectory, scale)
    ax.plot(trajectory_x, trajectory_y,  f".r-")
    ax.plot(trajectory_x[0], trajectory_y[0], f"og")
    ax.plot(trajectory_x[-1], trajectory_y[-1], f"ob")


def draw_contour(ax, f, scale=(100, 100)):
    tx, ty = np.linspace(-scale[0], scale[0], RATE), np.linspace(-scale[1], scale[1], RATE)
    x, y = np.meshgrid(tx, ty)
    z = calculate_3d_grid(f, x, y)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.contourf(x, y, z)

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

