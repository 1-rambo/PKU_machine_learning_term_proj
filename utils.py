"""

A torch-cpu version of implementation of functions GRAN, URAN, CLP, RED, and ORTH

All docs are from the original paper to make sure a corresponding implementation and allows for easy debugging

"""

import torch
import numpy as np


def URAN(n) -> torch.Tensor:
    """
    The function URAN(n) returns n random real numbers,
    which are uniformly distributed in [0, 1) and independent of each other
    """
    return torch.rand(n)


def GRAN(n, m) -> torch.Tensor:
    """
    GRAN (n, m) returns an n * m
    matrix of random independent real numbers, each with a
    Gaussian zero-mean, unit-variance distribution.
    """
    return torch.randn(n, m)


def ORTH(B) -> torch.Tensor:
    """
    We implement this function by Cholesky-decomposing the Gram matrix A = BB^T
    """
    A = torch.matmul(B, B.T)
    L = torch.linalg.cholesky(A)
    return L


def RED(B) -> torch.Tensor:
    """
    A fast and popular algorithm for the purpose is
    the Lenstra-Lenstra-Lovasz algorithm, which we
    apply in this work.
    """
    B_np = B.detach().numpy()
    B_np = np.array(LLL(B_np), dtype=np.float32)

    return torch.tensor(B_np)


def CLP(B, x) -> torch.Tensor:
    """
    Not sure if the following implementation is correct.
    """
    ## TODO: implement CLP based on algorithm 5 in paper https://doi.org/10.1109/TIT.2011.2143830
    B_inv = torch.pinverse(B)
    y = B_inv @ x
    z = torch.round(y)
    return z @ B


#####################################################################
##     The hidden secrets of LLL algorithm used in function RED    ##
##   You shouldn't need to call the following functions directly   ##
##                       Reference:                                ##
##   https://blog.csdn.net/qq_44925830/article/details/125632110   ##
#####################################################################


from decimal import Decimal
import math


def Hadamard(v):
    n = len(v)
    detL = abs(np.linalg.det(v))
    detL = Decimal(str(detL))
    product = 1
    for i in range(n):
        product *= np.linalg.norm(v[i])
    product = Decimal(str(product))
    ab = detL / product
    res = math.pow(ab, 1 / n)
    return res


"""
随机生成一个v*v矩阵的优质基，
上限为N，
Hadamard下限是h
"""


def random_basis(N, v, h):
    res = np.random.randint(-N, N + 1, (v, v))
    while Hadamard(res) < h:
        res = np.random.randint(-N, N + 1, (v, v))
    print(Hadamard(res))
    return res


def orthogonal(m):
    n = np.shape(m)
    M = np.zeros(n, dtype=np.float64)
    n = n[0]
    M[0, :] = m[0, :]
    for i in range(1, n):
        M[i, :] = m[i, :]
        for j in range(0, i):
            u_ij = np.dot(m[i, :], M[j, :]) / (np.linalg.norm(M[j, :]) ** 2)
            M[i, :] -= u_ij * M[j, :]
    # print('H:' + str(Hadamard(M)))
    return M


def lll(v):
    n = np.shape(v)
    n = n[0]
    k = 2
    while k <= n:
        # print(k)
        V = orthogonal(v[0:k, :])
        for j in range(0, k - 1):
            u = np.dot(v[k - 1, :], V[j, :]) / (np.linalg.norm(V[j, :]) ** 2)
            v[k - 1, :] = v[k - 1, :] - np.round(u) * v[j, :]
        u = np.dot(v[k - 1, :], V[k - 2, :]) / (np.linalg.norm(V[k - 2, :]) ** 2)
        if np.linalg.norm(V[k - 1, :]) ** 2 >= (3 / 4 - (u**2)) * (
            np.linalg.norm(V[k - 2, :]) ** 2
        ):
            k += 1
        else:
            v[[k - 2, k - 1], :] = v[
                [k - 1, k - 2], :
            ]  # 注意在同一个矩阵中交换向量的写法
            k = max(k - 1, 2)
    return v


def LLL(v):

    a = lll(v)
    b = lll(a)
    while a.all() != b.all():

        a = b
        b = lll(b)

    return b