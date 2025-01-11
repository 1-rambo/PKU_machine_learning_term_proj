#include <torch/torch.h>
#include <torch/csrc/autograd/variable.h>
#include <iostream>
#include <ctime>

extern torch::Tensor iterative_lattice_construction(int n);

void printTime(bool IS_BEGIN);

int main(){
    std::vector<torch::Tensor> vt;
    for (int dim = 2; dim <= 18; dim++){
        printf("*** Dim = %d ***\n", dim);
        printTime(1);

        torch::Tensor t = iterative_lattice_construction(dim);
        std::cout << t << std::endl;

        printTime(0);

        printf("\n");
    }
    return 0;
}

void printTime(bool IS_BEGIN){
    time_t now = time(0);
    char* currentTime = ctime(&now);
    if(IS_BEGIN)
        std::cout << "START TIME: ";
    else 
        std::cout << "END TIME: ";
    std::cout << currentTime << std::endl;
}