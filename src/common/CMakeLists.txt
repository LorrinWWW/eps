add_library(common ${EPS_INTERNAL_LIB_TYPE}
    debug.cu
    cuda_fp8_utils.cu
)
set_target_properties(common PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
