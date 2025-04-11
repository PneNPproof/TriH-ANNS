cmake_minimum_required(VERSION 3.20)
project(trih LANGUAGES CXX CUDA)

# Set CUDA architecture (adjust based on your GPU)
set(CMAKE_CUDA_ARCHITECTURES 80)  # Example: 80 for Ampere architecture

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(trih)