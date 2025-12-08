# TriH-ANNS CMake Configuration
# Builds the Triangle Inequality based Hierarchical Approximate Nearest Neighbor Search system
# Supports both CPU and GPU acceleration with optimized compiler flags for performance

# Configure Eigen3 library for linear algebra operations
set(EIGEN3_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/eigen-3.4.0 CACHE PATH "Path to Eigen3 headers")

# Include directories for all project components
include_directories(
    ${EIGEN3_INCLUDE_DIR}                                    # Eigen library for PCA computation
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu                          # CUDA GPU kernels
    ${CMAKE_CURRENT_SOURCE_DIR}/src                          # Core source files
    ${CMAKE_CURRENT_SOURCE_DIR}/include                      # Public headers
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/util/include           # Utility headers
)
include_directories(/usr/include/hdf5/serial)                # HDF5 library for dataset loading

# Configure output directories for organized builds
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)

# Set C++17 standard for both host and device code
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)  # Enable separate compilation for CUDA

# Find OpenMP for CPU parallelization
find_package(OpenMP REQUIRED)

# CUDA-specific compiler flags for modern GPU features
add_compile_options(
  "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>"              # Enable relaxed constexpr
  "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-deprecated-declarations>" # Suppress deprecation warnings
)

# CPU optimization flags for maximum performance
add_compile_options(
    "$<$<COMPILE_LANGUAGE:CXX>:-march=native>"   # Optimize for current CPU architecture
    "$<$<COMPILE_LANGUAGE:CXX>:-mtune=native>"   # Tune for current CPU
    "$<$<COMPILE_LANGUAGE:CXX>:-mavx512f>"       # Enable AVX-512 instructions
)

# Debug build configuration
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")  # Enable debug symbols
endif()

# Release build optimizations for maximum performance
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    # Aggressive CPU optimizations
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:CXX>:-ffast-math>"      # Fast math optimizations
        "$<$<COMPILE_LANGUAGE:CXX>:-funroll-loops>"   # Loop unrolling
    )
    # Aggressive GPU and CPU optimization levels
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>" # Fast math for CUDA
        "$<$<COMPILE_LANGUAGE:CXX>:-O3>"              # Maximum CPU optimization
        "$<$<COMPILE_LANGUAGE:CUDA>:-O3>"             # Maximum CUDA optimization
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-O3>"     # Maximum PTX optimization
    )
endif()

# Configure library search paths
link_directories("/usr/lib/x86_64-linux-gnu/hdf5/serial")

set(TARGET_NAME trih_anns)

# Define executable with all source files
add_executable(${TARGET_NAME} 
                src/main.cu              # Main application entry point
                src/rerank.cpp           # CPU reranking algorithms
                src/pca.cpp             # PCA computation and projection
                src/search.cpp          # Core search algorithms
                src/shuffle.cpp         # Data shuffling utilities
                src/utils.cpp           # General utilities and helpers
                src/gpu_pca_anns.cu     # GPU worker class implementation
                gpu/l2mm.cu             # GPU L2 distance matrix kernels
                gpu/reduce_min.cu       # GPU reduction and top-k selection
                gpu/rerank.cu           # GPU-assisted reranking
                gpu/pca.cu              # GPU PCA projection kernels
                src/sq.cpp              # Scalar quantization implementation
                src/data.cpp            # Dataset loading functionality
                )

# Include CUDA toolkit headers
target_include_directories(${TARGET_NAME} PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Conditionally enable detailed logging for debugging
if(ENABLE_DETAILED_LOG)
    target_compile_definitions(${TARGET_NAME} PRIVATE DETAILED_LOG)
    message(STATUS "Detailed logging enabled.")
else()
    message(STATUS "Detailed logging disabled.")
endif()

# Optional PTX optimization flags (commented out for compatibility)
# target_compile_options(${TARGET_NAME} PRIVATE
#   "$<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-dlcm=ca>"  # Cache control
#   "$<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-dscm=cs>"   # Shared memory configuration
# )

# Find required CUDA libraries
set(CUDART_LIBRARY cudart)
set(CUBLASLT_LIBRARY cublasLt)
set(CUBLASLT_LIBRARY cublas)
set(CUDA_cusolver_LIBRARY cusolver)

find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CUBLAS_LIBRARY cublas ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

# Link essential libraries
target_link_libraries(${TARGET_NAME} PRIVATE hdf5 pthread)  # HDF5 for data loading, pthread for threading

# Link CUDA libraries for GPU acceleration
target_link_libraries(${TARGET_NAME} PRIVATE
    ${CUDART_LIBRARY}           # CUDA runtime
    ${CUBLAS_LIBRARY}           # cuBLAS for linear algebra
    ${CUDA_cusolver_LIBRARY}    # cuSOLVER for advanced operations
)

# Link OpenMP for CPU parallelization
target_link_libraries(${TARGET_NAME} PRIVATE OpenMP::OpenMP_CXX)

# Optional static linking configuration (commented out)
# set_target_properties(${TARGET_NAME} PROPERTIES
#     LINK_SEARCH_START_STATIC 1
#     LINK_SEARCH_END_STATIC 1
#     POSITION_INDEPENDENT_CODE OFF
# )
