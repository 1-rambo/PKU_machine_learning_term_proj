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


def CLP(n, B, x) -> torch.Tensor:
    """
    Not sure if the following implementation is correct.
    """
    ## implement CLP based on algorithm 5 in paper https://doi.org/10.1109/TIT.2011.2143830
    # G : B, r : x
    C = float("inf")
    i = n
    d = torch.full((n,), n - 1)
    lamb = torch.zeros(n + 1)
    u = torch.zeros(n)
    p = torch.zeros(n)
    Delta = torch.zeros(n)
    result = torch.zeros(n)
    F = torch.zeros(n, n)
    F[n - 1] = x.clone()
    while True:
        while True:
            if i != 0:  ## i != 1
                i = i - 1
                for j in range(d[i], i, -1):
                    F[j - 1, i] = F[j, i] - u[j] * B[j, i]
                p[i] = F[i, i] / B[i, i]
                u[i] = torch.round(p[i])
                y = (p[i] - u[i]) * B[i, i]
                if y > 0:
                    Delta[i] = 1
                else:
                    Delta[i] = -1
                lamb[i] = lamb[i + 1] + y * y
            else:
                result = u.clone()
                C = float(lamb[0])
            if lamb[i] >= C:
                break
        # print(i, result)
        m = i
        # return result
        while True:
            if i == n - 1:
                return result
            else:
                i = i + 1

                u[i] = u[i] + Delta[i]

                if Delta[i] > 0:
                    Delta[i] = -Delta[i] - 1
                else:
                    Delta[i] = -Delta[i] + 1
                y = (p[i] - u[i]) * B[i, i]
                lamb[i] = lamb[i + 1] + y * y
            if lamb[i] < C:
                break
        for j in range(m, i):
            d[j] = i
        for j in range(m - 1, -1, -1):
            if d[j] < i:
                d[j] = i
            else:
                break


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


if __name__ == "__main__":

    """
    Unit test code for function CLP
    """

    # identity_mat = torch.eye(2)
    # x = torch.tensor([1.8, -0.2])
    # print(identity_mat)
    # print(x)

    # res = CLP(2, identity_mat, x)
    # print(res)
    # res = CLP(10, identity_mat, x)
    # print(res)

    B = torch.tensor([[1.0, 0.0], [-1.0, 1.0]])
    red_B = ORTH(RED(B))
    print(red_B)
    # x = torch.tensor([-0.0, 1.9])
    # res = CLP(2, B, x)
    # print(res @ B)

    pass
