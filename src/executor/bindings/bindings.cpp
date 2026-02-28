#include <torch/library.h>

#include "bindings/aok_grouped_gemm_ops.h"
#include "bindings/silu_ops.h"

#include "src/registration.h"

TORCH_LIBRARY(executor, m) {
  #if defined(IS_HOPPER) || defined(IS_BLACKWELL)
  executor::register_aok_gemm_ops(m);
  #endif
  executor::register_silu_ops(m);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
