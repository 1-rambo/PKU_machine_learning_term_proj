#include <torch/torch.h>
#include <torch/csrc/autograd/variable.h>
#include <iostream>
#include <cmath>

extern torch::Tensor GRAN(int, int);
extern torch::Tensor URAN(int);
extern torch::Tensor RED(torch::Tensor);
extern torch::Tensor ORTH(torch::Tensor);
extern torch::Tensor CLP(int, torch::Tensor, torch::Tensor);

const double mu_0 = 0.005;
const double ratio = 200;
const int T = 100000;
const int T_r = 100;

torch::Tensor iterative_lattice_construction(int n) {
    torch::Tensor B = ORTH(RED(GRAN(n, n)));
    
    torch::Tensor V = torch::prod(torch::diagonal(B));
    B = B * torch::pow(V, -1.0 / n);

    for (int t = 0; t < T; ++t) {
        printf("%d/100000\r", t);
        double mu = mu_0 * std::pow(ratio, -1.0 * t / (T - 1));
        torch::Tensor z = URAN(n);
        torch::Tensor y = z - CLP(n, B, torch::matmul(z, B));
        torch::Tensor e = torch::matmul(y, B);

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                B.index_put_({i, j}, B.index({i, j}) - mu * y[i].item<double>() * e[j].item<double>());
            }
            B.index_put_({i, i}, B.index({i, i}) - mu * (y[i].item<double>() * e[i].item<double>() - torch::norm(e).item<double>() * torch::norm(e).item<double>() / (n * B.index({i, i}).item<double>())));
        }
        
        // printf("-- Current: Dim = %d, t = %d\n", n, t);

        if (t % T_r == T_r - 1) {
            B = ORTH(RED(B));
            V = torch::prod(torch::diagonal(B));
            B = B * torch::pow(V, -1.0 / n);
        }
    }
    printf("/n");
    return B;
}
