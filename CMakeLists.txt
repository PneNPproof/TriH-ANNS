cmake_minimum_required(VERSION 3.18)
project(trih LANGUAGES CXX CUDA)

# Set CUDA architecture (adjust based on your GPU)
set(CMAKE_CUDA_ARCHITECTURES 80)  # Example: 80 for Ampere architecture

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_FLAGS  "-fopenmp")
set(CMAKE_BUILD_TYPE Release)

# Find RAFT package
# find_package(RAFT REQUIRED)

# # Add executable (replace "main.cu" with your source file)
# add_executable(raft_example test_raft/test_argmin.cu)

# # Link RAFT library
# target_link_libraries(raft_example PRIVATE 
#                         raft::raft
#                         cublasLt
#                         cudart
#                      )

# # Ensure CUDA compilation for device code
# set_target_properties(raft_example PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
#     POSITION_INDEPENDENT_CODE ON
# )

# # Add executable for selectk_example
# add_executable(selectk_example test_raft/test_select_k.cu)  # Replace with your source file

# # Link RAFT library for selectk_example
# target_link_libraries(selectk_example PRIVATE 
#                         raft::raft
#                         cublasLt
#                         cudart
#                      )

# # Ensure CUDA compilation for device code for selectk_example
# set_target_properties(selectk_example PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
#     POSITION_INDEPENDENT_CODE ON
# )

# # Add executable for selectk_fp16_example
# add_executable(selectk_fp16_example test_raft/test_selectk_fp16.cu)

# # Link RAFT library for selectk_fp16_example
# target_link_libraries(selectk_fp16_example PRIVATE 
#                         raft::raft
#                         cublasLt
#                         cudart
#                      )

# # Ensure CUDA compilation for device code for selectk_fp16_example
# set_target_properties(selectk_fp16_example PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
#     POSITION_INDEPENDENT_CODE ON
# )

# # Add executable for select_example
# add_executable(select_example select.cu)

# # Ensure CUDA compilation for device code for select_example
# set_target_properties(select_example PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
#     POSITION_INDEPENDENT_CODE ON
# )

add_compile_options(
    "-O3"
    "-mavx512f"         # 启用AVX-512基础指令集
    "-march=native"     # 针对本地CPU架构优化
    "-ffast-math"       # 启用快速数学优化
)


add_subdirectory(trih)