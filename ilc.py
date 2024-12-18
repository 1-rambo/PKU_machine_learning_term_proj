from utils import GRAN, URAN, RED, ORTH, CLP
import torch
from tqdm import tqdm

mu_0 = 0.005
ratio = 200
T = 1000000
T_r = 100


def iterative_lattice_construction(n):
    """
    Args:
        n(int): Dimension of lattice.

    Returns:
        torch.Tensor: Generator matrix of shape n*n
    """

    B = ORTH(RED(GRAN(n, n)))

    V = torch.prod(torch.diagonal(B))
    B = V ** (-1 / n) * B

    for t in tqdm(range(T)):

        mu = mu_0 * ratio ** (-t / (T - 1))
        z = URAN(n)
        y = CLP(B, z @ B)
        e = y @ B

        for i in range(n):
            for j in range(i):
                B[i, j] -= mu * y[i] * e[j]
            B[i, i] -= mu * (y[i] * e[i] - torch.norm(e) ** 2 / (n * B[i, i]))

        if t % T_r == T_r - 1:
            B = ORTH(RED(B))
            V = torch.prod(torch.diagonal(B))
            B = V ** (-1 / n) * B

    return B
