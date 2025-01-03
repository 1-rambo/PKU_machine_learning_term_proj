import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ilc import iterative_lattice_construction

if __name__ == "__main__":

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    B_optimal = iterative_lattice_construction(3)
    #B_optimal = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    r1 = np.array(B_optimal[0])
    r2 = np.array(B_optimal[1])
    r3 = np.array(B_optimal[2])

    theta1 = np.dot(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2))
    theta2 = np.dot(r2, r3) / (np.linalg.norm(r2) * np.linalg.norm(r3))
    theta3 = np.dot(r1, r3) / (np.linalg.norm(r1) * np.linalg.norm(r3))
    print("basis r1:", r1)
    print("basis r2:", r2)
    print("basis r3:", r3)
    print("cos(angle between r1 and r2) is:", theta1)
    print("cos(angle between r2 and r3) is:", theta2)
    print("cos(angle between r1 and r3) is:", theta3)

    # 绘制矢量场
    ax.quiver(0, 0, 0,
            r1[0], r1[1], r1[2],
    )
    ax.quiver(0, 0, 0,
            r2[0], r2[1], r2[2],
    )
    ax.quiver(0, 0, 0,
            r3[0], r3[1], r3[2],
    )

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    ax.set_xlim(-1, 2)
    ax.set_ylim(-1, 2)
    ax.set_zlim(-1, 2)

    plt.show()