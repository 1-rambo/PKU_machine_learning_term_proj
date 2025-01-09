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
    // For testing
    if(IS_TEST){
        torch::Tensor B = torch::tensor({{1.0, 0.0}, {-1.0, 1.0}});
        torch::Tensor red_B = ORTH(RED(B));
        std::cout << red_B << std::endl;

        torch::Tensor x = torch::tensor({-0.0, 1.9});
        auto res = CLP(2, B, x);
        std::cout << torch::matmul(res, B) << std::endl;
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