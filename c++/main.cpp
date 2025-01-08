#include <torch/torch.h>
#include <torch/csrc/autograd/variable.h>
#include <iostream>
#include <ctime>

extern torch::Tensor iterative_lattice_construction(int n);
void printTime(){
    time_t now = time(0);
    char* currentTime = ctime(&now);
    std::cout << "Current Time:" << currentTime << std::endl;
}

int main(){
    std::vector<torch::Tensor> vt;
    for (int dim = 2; dim <= 17; dim++){
        printf("*** Dim = %d ***\n", dim);
        printTime();
        torch::Tensor t = iterative_lattice_construction(dim);
        std::cout << t << std::endl;
        vt.push_back(t);
        printTime();
    }
}