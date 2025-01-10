from ilc import iterative_lattice_construction
import matplotlib.pyplot as plt
import numpy as np


generator_matrix = np.array([[1.0729, 0.0000], [-0.5355, 0.9321]])


def generate_lattice_points(generator_matrix, range_min=-3, range_max=3):
    indices = np.arange(range_min, range_max + 1)
    grid = np.array(np.meshgrid(indices, indices)).reshape(2, -1).T
    lattice_points = grid @ generator_matrix
    return lattice_points.T


lattice_points = generate_lattice_points(generator_matrix)


r1 = np.array(generator_matrix[0])
r2 = np.array(generator_matrix[1])

theta = np.dot(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2))
print("basis r1:", r1)
print("basis r2:", r2)
print("cos(angle between r1 and r2) is:", theta)

plt.figure(figsize=(5, 5))

# 绘制行向量
plt.quiver(
    0,
    0,
    r1[0],
    r1[1],
    angles="xy",
    scale_units="xy",
    scale=1,
    color="r",
)
plt.quiver(
    0,
    0,
    r2[0],
    r2[1],
    angles="xy",
    scale_units="xy",
    scale=1,
    color="b",
)

x, y = lattice_points
plt.scatter(x, y, color="black", marker="o")

# 设置轴的范围
plt.xlim(-3, 3)
plt.ylim(-3, 3)

plt.axhline(0, color="black", linewidth=1)
plt.axvline(0, color="black", linewidth=1)
plt.grid(False)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
