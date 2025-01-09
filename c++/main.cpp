#include <torch/torch.h>
#include <torch/csrc/autograd/variable.h>
#include <iostream>
#include <ctime>

extern torch::Tensor iterative_lattice_construction(int n);
extern torch::Tensor RED(torch::Tensor);
extern torch::Tensor ORTH(torch::Tensor);
extern torch::Tensor CLP(int, torch::Tensor, torch::Tensor);

void printTime(){
    time_t now = time(0);
    char* currentTime = ctime(&now);
    std::cout << "Current Time:" << currentTime << std::endl;
}

const bool IS_TEST = 0; // Set to 1 for testing

int main(){
    // For basic CLP, ORTH and RED testing
    if(IS_TEST){
        torch::Tensor identity_mat = torch::eye(2);
        torch::Tensor x = torch::tensor({1.8, -0.2});

        std::cout << "Identity Matrix:\n" << identity_mat << std::endl;
        std::cout << "Tensor x:\n" << x << std::endl;

        torch::Tensor res_1 = CLP(2, identity_mat, x);
        std::cout << "Result of CLP(2, identity_mat, x):\n" << res_1 << std::endl;

        torch::Tensor B = torch::tensor({{1.0, 0.0}, {-1.0, 1.0}});
        torch::Tensor red_B = ORTH(RED(B));
        std::cout << "Result of ORTH(RED(B)):\n" << red_B << std::endl;

        torch::Tensor x2 = torch::tensor({-0.0, 1.9});
        torch::Tensor res_3 = CLP(2, B, x2);
        std::cout << "Result of  CLP(2, B, x):\n" << torch::matmul(res_3, B) << std::endl;
        return 0;
    }

    std::vector<torch::Tensor> vt;
    for (int dim = 2; dim <= 17; dim++){
        printf("*** Dim = %d ***\n", dim);
        printTime();
        torch::Tensor t = iterative_lattice_construction(dim);
        std::cout << t << std::endl;
        vt.push_back(t);
        printTime();
        printf("\n");
    }
    return 0;
}