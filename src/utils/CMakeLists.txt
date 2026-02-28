file(GLOB files *.cu *.cuh)

set(LIB_NAME "utils_")

add_library(${LIB_NAME} SHARED
    ${files}
    bindings/th_rms_norm.cu
    bindings/th_topk_sigmoid.cu
    bindings/bindings.cpp
)

target_link_libraries(${LIB_NAME} PUBLIC
    ${EPS_MSCCLPP_TARGET}
    common
    torch::py_limited
    Python::Module
    CUDA::cuda_driver
    CUDA::cudart
)
target_include_directories(${LIB_NAME} SYSTEM PRIVATE ${PROJECT_SOURCE_DIR} ${repo-mscclpp_SOURCE_DIR}/include)

target_compile_options(${LIB_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
set_target_properties(${LIB_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON POSITION_INDEPENDENT_CODE ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(${LIB_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../python/eps/utils
    CUDA_SEPARABLE_COMPILATION ON
)
