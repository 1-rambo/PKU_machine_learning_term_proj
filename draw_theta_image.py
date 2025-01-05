import numpy as np
import torch
import matplotlib.pyplot as plt
from utils import RED, ORTH, FTN
from tqdm import tqdm

if __name__ == "__main__":
    B = torch.tensor([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]], dtype=torch.float64)
    # B = torch.tensor([[ 1.0912,  0.0000,  0.0000],
    #     [-0.3627,  1.0277,  0.0000],
    #     [ 0.3675,  0.5113,  0.8918]])
    # B = torch.tensor([[0.5, 0.5 ,0],
    #                    [0.5, 0, 0.5],
    #                    [0, 0.5, 0.5]])
    # B = torch.tensor([[2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                    [0, 2, 0, 0, 0, 0, 0 ,0, 0, 0, 0, 0, 0, 0, 0],
    #                    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                    [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                    [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                    [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    #                    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
    #                    [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    #                    [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
    #                    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    #                    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    #                    [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    #                    [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    #                    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    #                    ])
    B = ORTH(RED(B))
    H = torch.inverse(B)
    U = FTN(H, 6)
    print(len(U))
    print(H)
    r2 = 0.0
    listx = []
    listy = []
    for i in range(551):
        listx.append(r2)
        r2 += 0.01
        listy.append(0)
    # print(listx)
    for i in U:
        # print(i, i @ B)
        dist = np.linalg.norm(i @ B)
        dist = dist * dist
        # print(dist)
        for j in range(550, -1, -1):
            # print(listx[j], dist)
            if listx[j] >= dist:
                listy[j] += 1
            else:
                break
    plt.plot(listx, listy)
    plt.show()