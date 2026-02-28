#pragma once

#include <memory>
#include "hopper_grouped_gemm_kernel.h"

namespace aok {

template <typename InputType, typename OutputType, typename GroupType>
class HopperGmm {
public:
  HopperGmm();

  ~HopperGmm();

  void Run(const InputType        *A,
           const InputType        *B,
           OutputType              *C,
           const GroupType  *total_rows_before_expert,
           int            total_rows,
           int64_t        gemm_n,
           int64_t        gemm_k,
           int            num_experts,
           void           *ws,
           size_t         workspace_size,
           int            sm_count,
           cudaStream_t   st);
  
  void RunV2(const InputType        *A,
           const InputType        *B,
           OutputType              *C,
           const GroupType  *total_rows_before_expert,
           int            total_rows,
           int64_t        gemm_m,
           int64_t        gemm_n,
           int64_t        gemm_k,
           int            num_experts,
           void           *ws,
           size_t         workspace_size,
           int            sm_count,
           cudaStream_t   st);

  size_t GetWorkSpace(int64_t gemm_n,
                      int64_t gemm_k,
                      int64_t total_rows,
                      int num_experts,
                      int sm_count);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};


} // namespace aok