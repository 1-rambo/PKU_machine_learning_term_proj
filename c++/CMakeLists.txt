cmake_minimum_required(VERSION 3.15)
project(rambo)

add_compile_options(-DCMAKE_PREFIX_PATH)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(ramboo main.cpp ilc.cpp)
add_executable(test utils.cpp)

# set(CMAKE_PREFIX_PATH YOURLIBTORCHPATH/libtorch)
set(CMAKE_PREFIX_PATH /Users/zlx/Downloads/libtorch)

find_package(Torch REQUIRED)


# include_directories("YOURLIBTORCHPATH/libtorch/include")
# include_directories("YOURLIBTORCHPATH/libtorch/include/torch/csrc/api/include")
include_directories("/Users/zlx/Downloads/libtorch/include")
include_directories("/Users/zlx/Downloads/libtorch/include/torch/csrc/api/include")

target_link_libraries(ramboo "${TORCH_LIBRARIES}")
set_property(TARGET ramboo PROPERTY CXX_STANDARD 17)

target_link_libraries(test "${TORCH_LIBRARIES}")
set_property(TARGET test PROPERTY CXX_STANDARD 17)