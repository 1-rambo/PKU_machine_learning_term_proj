import numpy as np

T = 1000000   # the number of iterations
mu_0 = 0.005  # the initial step size
nu = 200      # ratio of the step size
Tr = 100      # reduction interval

def ILC(n: int):
    B = ORTH(RED(GRAN(n, n)))
    V = np.prod(np.diag(B))
    B = V ** (1 / n) * B
    exp = 0
    for t in range(T):
        exp += 1 / (T - 1)
        mu = mu_0 * (nu ** (-exp))
        z = URAN(n)
        y = z - CLB(B, z * B)
        e = y * B
        for i in range(n):
            for j in range(i):
                B[i][j] -= mu * y[i] * e[j]
            B[i][i] -= mu * (y[i] * e[j] - (np.linalg.norm(e) * np.linalg.norm(e))/(n * B[i][i]))
        if t % Tr == 0:
            B = ORTH(RED(B))
            V = np.prod(np.diag(B))
            B = V ** (1 / n) * B

    
if __name__ == "__main__":
    n = int(input("The dimension: "))
    AnsMatrix = ILC(n)
    print(AnsMatrix)
