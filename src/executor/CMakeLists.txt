file(GLOB files
    moe_gemm.cu
    activations.cu
)

if(SM_NUM STREQUAL "90a" OR SM_NUM STREQUAL "100a")
    list(APPEND files aok_grouped_gemm.cu)
endif()


set(LIB_NAME "executor")
add_library(${LIB_NAME} ${EPS_INTERNAL_LIB_TYPE} ${files})
set_target_properties(${LIB_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_include_directories(${LIB_NAME} PUBLIC ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/3rdparty)
target_include_directories(${LIB_NAME} PUBLIC ${CUTLASS_DIR}/include)

set(deps common eps_aok_grouped_gemm)

if(SM_NUM STREQUAL "90a" OR SM_NUM STREQUAL "100a")
    target_include_directories(${LIB_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/src/gemm)
    target_link_directories(${LIB_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/src/gemm/aok_grouped_gemm/include)
endif()

target_link_libraries(${LIB_NAME} PUBLIC ${deps})
target_compile_options(${LIB_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
# target_compile_options(${LIB_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
# target_compile_options(${LIB_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)

set(binding_files
    bindings/silu_ops.cu
    bindings/bindings.cpp
)
if(SM_NUM STREQUAL "90a" OR SM_NUM STREQUAL "100a")
    list(APPEND binding_files bindings/aok_grouped_gemm_ops.cu)
endif()
add_library(executor_ SHARED ${binding_files})

target_link_libraries(executor_ PUBLIC
    executor
    torch::py_limited
    Python::Module
    CUDA::cuda_driver
    CUDA::cudart
)
target_include_directories(executor_ PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(executor_ PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../python/eps/executor
    CUDA_SEPARABLE_COMPILATION ON
)
