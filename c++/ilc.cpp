#include <torch/torch.h>
#include <torch/csrc/autograd/variable.h>
#include <iostream>
#include <cmath>
#include <unistd.h>

// Parameters
const float mu_0 = 0.005;
const float ratio = 200;
const int T = 1000000;
const int T_r = 100;

extern torch::Tensor GRAN(int, int);
extern torch::Tensor URAN(int);
extern torch::Tensor RED(torch::Tensor);
extern torch::Tensor ORTH(torch::Tensor);
extern torch::Tensor CLP(int, torch::Tensor, torch::Tensor);

torch::Tensor init_tensor(int n);

torch::Tensor iterative_lattice_construction(int n) {
    torch::Tensor B = init_tensor(n);

    for (int t = 0; t < T; ++t) {
        printf("%d/%d\r", t, T); // Progress bar

        float mu = mu_0 * std::pow(ratio, -1.0 * t / (T - 1));
        torch::Tensor z = URAN(n);            
        torch::Tensor tmp = CLP(n, B, torch::matmul(z, B));
        torch::Tensor y = z - tmp;
        torch::Tensor e = torch::matmul(y, B);

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                B.index_put_({i, j}, B.index({i, j}) - mu * y[i].item<float>() * e[j].item<float>());
            }
            B.index_put_({i, i}, 
                B.index({i, i}) - mu * (y[i].item<float>() * e[i].item<float>()
                    - torch::norm(e).item<float>() * torch::norm(e).item<float>() / (n * B.index({i, i}).item<float>())));
        }

        if (t % T_r == T_r - 1) {
            B = ORTH(RED(B));
            torch::Tensor V = torch::prod(torch::diagonal(B));
            float scale_factor = std::pow(V.item<float>(), -1.0 / n); 
            B = B * scale_factor;
        }

    }
    printf("\n");
    return B;
}

torch::Tensor init_tensor(int n){
    torch::Tensor B = ORTH(RED(GRAN(n, n)));
    torch::Tensor V = torch::prod(torch::diagonal(B));
    float scale_factor = std::pow(V.item<float>(), -1.0 / n); 
    return B * scale_factor;
}