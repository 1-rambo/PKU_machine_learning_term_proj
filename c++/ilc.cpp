#include <torch/torch.h>
#include <torch/csrc/autograd/variable.h>
#include <iostream>
#include <cmath>
#include <unistd.h>

extern torch::Tensor GRAN(int, int);
extern torch::Tensor URAN(int);
extern torch::Tensor RED(torch::Tensor);
extern torch::Tensor ORTH(torch::Tensor);
extern torch::Tensor CLP(int, torch::Tensor, torch::Tensor);

const float mu_0 = 0.005;
const float ratio = 200;
const int T = 100000; // Set to 100000 to save time
const int T_r = 100;

torch::Tensor iterative_lattice_construction(int n) {
    torch::Tensor B = ORTH(RED(GRAN(n, n)));
    
    torch::Tensor V = torch::prod(torch::diagonal(B));
    B = B * torch::pow(V, -1.0 / n);
    // std::cout << B << std::endl;

    for (int t = 0; t < T; ++t) {
        printf("%d/%d\r", t, T); // Progress bar
        // for testing
        // printf("%d/100000\n", t); 
        // usleep(500000);

        float mu = mu_0 * std::pow(ratio, -1.0 * t / (T - 1));
        torch::Tensor z = URAN(n);
        torch::Tensor y = z - CLP(n, B, torch::matmul(z, B));
        torch::Tensor e = torch::matmul(y, B);

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                B.index_put_({i, j}, B.index({i, j}) - mu * y[i].item<float>() * e[j].item<float>());
            }
            B.index_put_({i, i}, 
                B.index({i, i}) - mu * (y[i].item<float>() * e[i].item<float>()
                    - torch::norm(e).item<float>() * torch::norm(e).item<float>() / (n * B.index({i, i}).item<float>())));
        }
        
        // printf("-- Current: Dim = %d, t = %d\n", n, t);

        if (t % T_r == T_r - 1) {
            B = ORTH(RED(B));
            V = torch::prod(torch::diagonal(B));
            B = B * torch::pow(V, -1.0 / n);
        }

        // std::cout << B << std::endl;
    }
    printf("/n");
    return B;
}
