import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import imageio

generator_matrix = np.array(
    [[1.0918, 0.0000, 0.0000], [-0.3679, 1.0263, 0.0000], [0.3688, 0.5092, 0.8924]]
)

# generator_matrix = np.array([[1, 1, -1], [-1, -1, -1], [-1, 1, 1]], dtype=np.float64)


def generate_lattice_points(generator_matrix, range_min=-3, range_max=3):
    indices = np.arange(range_min, range_max + 1)
    grid = np.array(np.meshgrid(indices, indices, indices)).reshape(3, -1).T
    lattice_points = grid @ generator_matrix
    return lattice_points.T


lattice_points = generate_lattice_points(generator_matrix)

r1 = np.array(generator_matrix[0])
r2 = np.array(generator_matrix[1])
r3 = np.array(generator_matrix[2])

# 用新的坐标轴来画格点，以便于更好的可视化

x_dir = (r1 + r2) / 2.0
y_dir = (r2 + r3) / 2.0
z_dir = (r1 + r3) / 2.0


rotation_matrix = np.array([x_dir, y_dir, z_dir]).T

R = np.linalg.inv(rotation_matrix)

lattice_points = R @ lattice_points

new_x_dir = R @ r1
new_y_dir = R @ r2
new_z_dir = R @ r3

theta1 = np.dot(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2))
theta2 = np.dot(r2, r3) / (np.linalg.norm(r2) * np.linalg.norm(r3))
theta3 = np.dot(r1, r3) / (np.linalg.norm(r1) * np.linalg.norm(r3))

print("basis r1:", r1)
print("basis r2:", r2)
print("basis r3:", r3)
print("cos(angle between r1 and r2) is:", theta1)
print("cos(angle between r2 and r3) is:", theta2)
print("cos(angle between r1 and r3) is:", theta3)


# 绘制晶格
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# 提取点坐标
x, y, z = lattice_points

# 绘制散点
ax.scatter(x, y, z, color="black", s=50, label="Lattice Points")

# 可视化基矢量
origin = np.zeros(3)

ax.quiver(*origin, *new_x_dir, color="r", normalize=False)
ax.quiver(*origin, *new_y_dir, color="g", normalize=False)
ax.quiver(*origin, *new_z_dir, color="b", normalize=False)


# 设置轴范围和标题
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.set_zlim([-3, 3])
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.set_title("3D Lattice")
# ax.grid(False)
# ax.axis("off")
plt.legend()
plt.show()


# 定义旋转函数
def update(frame):
    # 让旋转同时进行上下和左右旋转
    ax.view_init(elev=30 + 15 * np.sin(np.radians(frame)), azim=frame)


# 创建动画
ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50)

# 保存为GIF
ani.save("figures/3d_rotation_up_down.gif", writer="imagemagick")
