set(EIGEN3_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/eigen-3.4.0 CACHE PATH "Path to Eigen3 headers")
include_directories(
    ${EIGEN3_INCLUDE_DIR} 
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu 
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/util/include
)
include_directories(/usr/include/hdf5/serial)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

find_package(OpenMP REQUIRED)

add_compile_options(
  "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-deprecated-declarations>"
)

add_compile_options(
    "$<$<COMPILE_LANGUAGE:CXX>:-march=native>"
    "$<$<COMPILE_LANGUAGE:CXX>:-mtune=native>"
    "$<$<COMPILE_LANGUAGE:CXX>:-mavx512f>"
)


if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:CXX>:-ffast-math>"
        "$<$<COMPILE_LANGUAGE:CXX>:-funroll-loops>"
    )
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>"
        "$<$<COMPILE_LANGUAGE:CXX>:-O3>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-O3>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-O3>"
    )
endif()

link_directories("/usr/lib/x86_64-linux-gnu/hdf5/serial")

set(TARGET_NAME trih_anns)

add_executable(${TARGET_NAME} 
                src/main.cu
                src/rerank.cpp 
                src/pca.cpp 
                src/search.cpp 
                src/shuffle.cpp 
                src/utils.cpp 
                src/gpu_pca_anns.cu 
                gpu/l2mm.cu 
                gpu/reduce_min.cu
                gpu/rerank.cu
                gpu/pca.cu
                src/sq.cpp
                src/data.cpp
                )

target_include_directories(${TARGET_NAME} PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    # /usr/include
)

# Conditionally add the compile definition to your target
if(ENABLE_DETAILED_LOG)
    target_compile_definitions(${TARGET_NAME} PRIVATE DETAILED_LOG)
    message(STATUS "Detailed logging enabled.")
else()
    message(STATUS "Detailed logging disabled.")
endif()

# target_compile_options(${TARGET_NAME} PRIVATE
#   "$<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-dlcm=ca>"  # 缓存控制
#   "$<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-dscm=cs>"   # 共享内存配置
# )


set(CUDART_LIBRARY cudart)
set(CUBLASLT_LIBRARY cublasLt)
set(CUBLASLT_LIBRARY cublas)
set(CUDA_cusolver_LIBRARY cusolver)

find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
# find_library(CUBLASLT_LIBRARY cublasLt ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CUBLAS_LIBRARY cublas ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

target_link_libraries(${TARGET_NAME} PRIVATE hdf5 pthread)

target_link_libraries(${TARGET_NAME} PRIVATE
    ${CUDART_LIBRARY}
    # ${CUBLASLT_LIBRARY}
    ${CUBLAS_LIBRARY}
    ${CUDA_cusolver_LIBRARY}
)

target_link_libraries(${TARGET_NAME} PRIVATE OpenMP::OpenMP_CXX)



# set_target_properties(${TARGET_NAME} PROPERTIES
#     LINK_SEARCH_START_STATIC 1
#     LINK_SEARCH_END_STATIC 1
#     POSITION_INDEPENDENT_CODE OFF
# )
