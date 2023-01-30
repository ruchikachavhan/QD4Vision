import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


x, y = np.linspace(0.5, 1, 100), np.linspace(0.5, 1, 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

x1, y1 = np.linspace(0., 0.5, 100), np.linspace(0.5, 1.0, 100)
X1, Y1 = np.meshgrid(x1, y1)
Z1 = np.zeros_like(X1)

x2, y2 = np.linspace(0.5, 1.0, 100), np.linspace(0., 0.5, 100)
X2, Y2 = np.meshgrid(x2, y2)
Z2 = np.zeros_like(X2)

x3, y3 = np.linspace(0.0, 0.5, 100), np.linspace(0.0, 0.5, 100)
X3, Y3 = np.meshgrid(x3, y3)
Z3 = np.zeros_like(X3)



# x4 = np.array([0.5])
# y4 = np.linspace(0.5, 1, 100)
# z4 = np.linspace(0.0, 1.0, 100)
# X4, Z4 = np.meshgrid(y4, z4)
# Y4 = np.ones_like(X4) * x4

# y5 = np.array([0.5])
# x5 = np.linspace(0.5, 1, 100)
# z5 = np.linspace(0.0, 1.0, 100)
# Y5, Z5 = np.meshgrid(x5, z5)
# X5 = np.ones_like(Y5) * y5

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.grid(True, linestyle='-.')
ax.set_axisbelow(True)


ax.plot_surface(X, Y, Z, color='green', alpha=0.5)
ax.plot_surface(X1, Y1, Z1, color='red', alpha=0.5)
ax.plot_surface(X2, Y2, Z2, color='blue', alpha=0.5)
ax.plot_surface(X3, Y3, Z3, color='lightgray', alpha=0.5)

# Generate data for the green cube
x_c, y_c = np.meshgrid(np.linspace(0.5, 1, 2), np.linspace(0.5, 1, 2))
z_c = np.ones_like(x_c) 
ax.plot_surface(x_c, y_c, z_c, color='green', alpha=0.5)
y_c, z_c = np.meshgrid(np.linspace(0.5, 1, 2), np.linspace(0.0, 1, 2))
x_c = np.ones_like(y_c) * 0.5
ax.plot_surface(x_c, y_c, z_c, color='green', alpha=0.5)
x_c, z_c = np.meshgrid(np.linspace(0.5, 1, 2), np.linspace(0.0, 1, 2))
y_c = np.ones_like(y_c)
ax.plot_surface(x_c, y_c, z_c, color='green', alpha=0.5)

# Generate data for the red cube
x_c, y_c = np.meshgrid(np.linspace(0.0, 0.5, 2), np.linspace(0.5, 1, 2))
z_c = np.ones_like(x_c) 
ax.plot_surface(x_c, y_c, z_c, color='red', alpha=0.5)
y_c, z_c = np.meshgrid(np.linspace(0.5, 1, 2), np.linspace(0.0, 1, 2))
x_c = np.zeros_like(y_c) * 0.5
ax.plot_surface(x_c, y_c, z_c, color='red', alpha=0.5)
x_c, z_c = np.meshgrid(np.linspace(0., 0.5, 2), np.linspace(0.0, 1, 2))
y_c = np.ones_like(y_c)
ax.plot_surface(x_c, y_c, z_c, color='red', alpha=0.5)

# Generate data for the blue cube
x_c, y_c = np.meshgrid(np.linspace(0.5, 1.0, 2), np.linspace(0.0, 0.5, 2))
z_c = np.ones_like(x_c) 
ax.plot_surface(x_c, y_c, z_c, color='blue', alpha=0.4)
y_c, z_c = np.meshgrid(np.linspace(0.0, 0.5, 2), np.linspace(0.0, 1, 2))
x_c = np.ones_like(y_c)
ax.plot_surface(x_c, y_c, z_c, color='blue', alpha=0.4)
x_c, z_c = np.meshgrid(np.linspace(0.5, 1.0, 2), np.linspace(0.0, 1, 2))
y_c = np.zeros_like(y_c) * 0.5
ax.plot_surface(x_c, y_c, z_c, color='blue', alpha=0.4)

# Generate data for the gray cube
x_c, y_c = np.meshgrid(np.linspace(0.0, 0.5, 2), np.linspace(0.0, 0.5, 2))
z_c = np.ones_like(x_c) 
ax.plot_surface(x_c, y_c, z_c, color='lightgray', alpha=0.4)
y_c, z_c = np.meshgrid(np.linspace(0.0, 0.5, 2), np.linspace(0.0, 1, 2))
x_c = np.ones_like(y_c) * 0.5
ax.plot_surface(x_c, y_c, z_c, color='lightgray', alpha=0.4)
x_c, z_c = np.meshgrid(np.linspace(0., 0.5, 2), np.linspace(0.0, 1, 2))
y_c = np.zeros_like(y_c) * 0.5
ax.plot_surface(x_c, y_c, z_c, color='lightgray', alpha=0.4)

ax.plot3D(np.arange(0, 1.1, 0.1), [0.5 for _ in range(11)], [1.0 for _ in range(11)], 'black', linestyle='--', linewidth=2)
ax.plot3D([0.5 for _ in range(11)], np.arange(0, 1.1, 0.1), [1.0 for _ in range(11)], 'black', linestyle='--', linewidth=2)
ax.scatter(0.7, 0.8, 0.95, c='black',  marker="x")
ax.scatter(0.75, 0.81, 0.96, c='black',  marker="x")
ax.scatter(0.82, 0.74, 0.94, c='black',  marker="x")
ax.scatter(0.86, 0.95, 0.95, c='black',  marker="x")
ax.scatter(0.91, 0.76, 0.97, c='black',  marker="x")

ax.scatter(0.9, 0.7, 0.97, c='black', marker="^")
ax.scatter(0.5, 0.9, 0.97, c='black', marker="^")
ax.scatter(0.6, 0.78, 0.97, c='black', marker="^")
ax.scatter(0.3, 0.8, 0.95, c='black', marker="^")
ax.scatter(0.2, 0.9, 0.95, c='black', marker="^")
ax.scatter(0.4, 0.7, 0.95, c='black', marker="^")
ax.scatter(0.8, 0.3, 0.95, c='black', marker="^")
ax.scatter(0.7, 0.2, 0.95, c='black', marker="^")
ax.scatter(0.5, 0.4, 0.95, c='black', marker="^")
ax.scatter(0.6, 0.1, 0.95, c='black', marker="^")
# ax.scatter(0.75, 0.81, 0.96, c='black')
# ax.scatter(0.82, 0.74, 0.94, c='black')
# ax.scatter(0.86, 0.95, 0.95, c='black')
# ax.scatter(0.91, 0.76, 0.97, c='black')


# Customize the tick positions
ax.set_xticks([0.0, 0.5, 1.0])
ax.set_yticks([0.0, 0.5, 1.0])
ax.set_zticks([0.0, 1.0])

ax.set_xlabel('Spatial Invariance')
ax.set_ylabel('Appearance Invariance')
ax.set_zlabel('Quality')
plt.savefig("teaser.png")
