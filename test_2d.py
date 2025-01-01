from ilc import iterative_lattice_construction


if __name__ == "__main__":

    B_optimal = iterative_lattice_construction(2)

    import matplotlib.pyplot as plt
    import numpy as np

    r1 = np.array(B_optimal[0])
    r2 = np.array(B_optimal[1])

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

    # 设置轴的范围
    plt.xlim(-1, 5)
    plt.ylim(-1, 5)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(0, color="black", linewidth=1)
    plt.grid(True)
    plt.legend()
    plt.show()
