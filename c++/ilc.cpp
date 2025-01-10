// #ifndef DEBUG
// #define DEBUG
// #endif

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
const int T = 100000; // Set to 1000000 for a full result
const int T_r = 100;

torch::Tensor init_tensor(int n){
    torch::Tensor B = ORTH(RED(GRAN(n, n)));
    torch::Tensor V = torch::prod(torch::diagonal(B));
    float scale_factor = std::pow(V.item<float>(), -1.0 / n); 

    #ifdef DEBUG
    std::cout << "B = ORTH(RED(GRAN(n, n))):\n" << B << std::endl;
    std::cout << "scale_factor = " << scale_factor << std::endl;
    std::cout << "V = torch::prod(torch::diagonal(B)):\n" << V << std::endl;
    #endif

    return B * scale_factor;
}

torch::Tensor iterative_lattice_construction(int n) {
    torch::Tensor B = init_tensor(n);

    #ifdef DEBUG    
    std::cout << "B normalized:\n" << B << std::endl;
    #endif

    for (int t = 0; t < T; ++t) {
        #ifdef DEBUG
        printf("\n%d\n", t); 
        // usleep(500000);
        #else
        printf("%d/%d\r", t, T); // Progress bar
        #endif
        float mu = mu_0 * std::pow(ratio, -1.0 * t / (T - 1));
        torch::Tensor z = URAN(n);            
        torch::Tensor tmp = CLP(n, B, torch::matmul(z, B));
        torch::Tensor y = z - tmp;
        torch::Tensor e = torch::matmul(y, B);

        #ifdef DEBUG
        std::cout << "z, tmp, y, e:\n" 
            << z << std::endl << tmp << std::endl << y << std::endl << e << std::endl;
        #endif

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                B.index_put_({i, j}, B.index({i, j}) - mu * y[i].item<float>() * e[j].item<float>());
            }
            B.index_put_({i, i}, 
                B.index({i, i}) - mu * (y[i].item<float>() * e[i].item<float>()
                    - torch::norm(e).item<float>() * torch::norm(e).item<float>() / (n * B.index({i, i}).item<float>())));
        }
        #ifdef DEBUG
        std::cout << "New B:\n" << B << std::endl;
        #endif        

        if (t % T_r == T_r - 1) {
            // torch::Tensor tmpB = ORTH(RED(B));
            // torch::Tensor tmpV = torch::prod(torch::diagonal(B));
            // B = torch::pow(V, -1.0 / n) * B;

            // B.copy_(ORTH(RED(B)));
            // V.copy_(torch::prod(torch::diagonal(B)));
            // B.copy_(B * torch::pow(V, -1.0 / n));
            
            B = ORTH(RED(B));
            #ifdef DEBUG
            std::cout << "B = ORTH(RED(B)):\n" << B << std::endl;
            #endif
            torch::Tensor V = torch::prod(torch::diagonal(B));
            float scale_factor = std::pow(V.item<float>(), -1.0 / n); 
            B = B * scale_factor;
            #ifdef DEBUG
            // std::cout << "V = torch::prod(torch::diagonal(B)):\n" << V << std::endl;
            printf("scale_factor = %.10f\n", scale_factor);
            std::cout << "Updated B:\n" << B << std::endl;
            #endif
        }

    }
    printf("/n");
    return B;
}
