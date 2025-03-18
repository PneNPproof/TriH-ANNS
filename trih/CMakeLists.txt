# cmake_minimum_required(VERSION 3.23)
# project(pca LANGUAGES CXX CUDA)

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EIGEN3_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/eigen-3.4.0 CACHE PATH "Path to Eigen3 headers")
include_directories(${EIGEN3_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/gpu ${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(/usr/include/hdf5/serial)
# set(CMAKE_CUDA_ARCHITECTURES 80)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)

link_directories("/usr/lib/x86_64-linux-gnu/hdf5/serial")



set(TARGET_NAME pca)

add_executable(${TARGET_NAME} 
                src/main.cu 
                src/pca.cpp 
                src/search.cpp 
                src/gist.cpp 
                src/shuffle.cpp 
                src/utils.cpp 
                src/gpu_pca_anns.cu 
                gpu/l2mm.cu 
                gpu/reduce_min.cu
                gpu/rerank.cu
                src/rerank.cpp
                )

target_include_directories(${TARGET_NAME} PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    /usr/include
)
set(CUDART_LIBRARY cudart)
set(CUBLASLT_LIBRARY cublasLt)
set(CUBLASLT_LIBRARY cublas)

find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CUBLASLT_LIBRARY cublasLt ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CUBLAS_LIBRARY cublas ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

target_link_libraries(${TARGET_NAME} PRIVATE hdf5 pthread)

target_link_libraries(${TARGET_NAME} PRIVATE
    ${CUDART_LIBRARY}
    ${CUBLASLT_LIBRARY}
    ${CUBLAS_LIBRARY}
)
