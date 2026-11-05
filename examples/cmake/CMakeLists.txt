cmake_minimum_required(VERSION 3.25)
project(torchpairwise_consumer LANGUAGES CXX)

find_package(torchpairwise CONFIG REQUIRED)
add_executable(kernel_product main.cpp)
target_link_libraries(kernel_product PRIVATE torchpairwise::torchpairwise)
