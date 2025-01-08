#include <torch/torch.h>
#include <torch/csrc/autograd/variable.h>
#include <iostream>

torch::Tensor URAN(int n);
torch::Tensor GRAN(int n, int m);
torch::Tensor ORTH(torch::Tensor B);
torch::Tensor RED(torch::Tensor B);
torch::Tensor CLP(int n, torch::Tensor B, torch::Tensor x);

int main() {
    // for test
    torch::Tensor B = torch::randn({5, 5});
    torch::Tensor x = torch::randn({5});
    std::cout << "*1* Generated URAN (5): " << URAN(5) << std::endl;
    std::cout << "*2* Generated GRAN (3, 4): " << GRAN(3, 4) << std::endl;
    std::cout << "*3* Orthogonal matrix from B: " << ORTH(B) << std::endl;
    std::cout << "*4* Closest lattice point: " << CLP(5, B, x) << std::endl;
    return 0;
}

torch::Tensor URAN(int n){
    return torch::rand(n);
}

torch::Tensor GRAN(int n, int m) {
    return torch::randn({n, m});
}

torch::Tensor ORTH(torch::Tensor B) {
    torch::Tensor A = torch::matmul(B, B.transpose(0, 1));
    return torch::cholesky(A);
}

torch::Tensor RED(torch::Tensor B) {
    // TODO
    return torch::zeros({3, 4});
}

int sign(float x) {
    return (x > 0) ? 1 : -1;
}

torch::Tensor CLP(int n, torch::Tensor B, torch::Tensor x) {
    float C = std::numeric_limits<float>::infinity();
    int i = n;
    torch::Tensor d = torch::full({n}, n - 1);
    torch::Tensor lamb = torch::zeros({n + 1});
    torch::Tensor u = torch::zeros({n});
    torch::Tensor p = torch::zeros({n});
    torch::Tensor Delta = torch::zeros({n});
    torch::Tensor result = torch::zeros({n});
    torch::Tensor F = torch::zeros({n, n});
    F[n - 1] = x.clone();

    while (true) {
        while (true) {
            if (i != 0) {
                i = i - 1;
                for (int j = d[i].item<int>(); j > i; j--) {
                    F[j - 1][i] = F[j][i] - u[j] * B[j][i];
                }
                p[i] = F[i][i] / B[i][i];
                u[i] = torch::round(p[i]);
                float y = (p[i].item<float>() - u[i].item<float>()) * B[i][i].item<float>();
                Delta[i] = sign(y);
                lamb[i] = lamb[i + 1] + y * y;
            } else {
                result = u.clone();
                C = lamb[0].item<float>();
            }
            if (lamb[i].item<float>() >= C) {
                break;
            }
        }
        int m = i;
        while (true) {
            if (i == n - 1) {
                return result;
            } else {
                i = i + 1;
                u[i] = u[i] + Delta[i];
                Delta[i] = -Delta[i] - 1;
                float y = (p[i].item<float>() - u[i].item<float>()) * B[i][i].item<float>();
                lamb[i] = lamb[i + 1] + y * y;
            }
            if (lamb[i].item<float>() < C) {
                break;
            }
        }
        for (int j = m; j < i; j++) {
            d[j] = i;
        }
        for (int j = m - 1; j >= 0; j--) {
            if (d[j].item<int>() < i) {
                d[j] = i;
            } else {
                break;
            }
        }
    }
}

torch::Tensor LLL(){
    // TODO
    return torch::zeros({3, 4});
}