# Steps to run the code:
# 1. Install CMake (https://cmake.org/)
# 2. Download LibTorch package (https://pytorch.org/get-started/locally/)
# 3. Download Eigen package (https://eigen.tuxfamily.org/)
# 4. Configure CMakeLists.txt

cmake_minimum_required(VERSION 3.15)
project(c++)

add_compile_options(-DCMAKE_PREFIX_PATH)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(c++ main.cpp ilc.cpp utils.cpp)

# set(CMAKE_PREFIX_PATH YOURLIBTORCHPATH/libtorch)
set(CMAKE_PREFIX_PATH /Users/zlx/Downloads/libtorch)

find_package(Torch REQUIRED)

# include_directories("YOURLIBTORCHPATH/libtorch/include")
# include_directories("YOURLIBTORCHPATH/libtorch/include/torch/csrc/api/include")
include_directories("/Users/zlx/Downloads/libtorch/include")
include_directories("/Users/zlx/Downloads/libtorch/include/torch/csrc/api/include")

include_directories("/usr/local/include")
include_directories("/usr/include")
include_directories("/usr/include/eigen3")

target_link_libraries(c++ "${TORCH_LIBRARIES}")
set_property(TARGET c++ PROPERTY CXX_STANDARD 17)