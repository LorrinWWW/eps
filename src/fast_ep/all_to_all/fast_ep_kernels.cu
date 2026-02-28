#include "src/fast_ep/all_to_all/fast_ep_kernels.cuh"

#include <algorithm>

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "src/common/cuda_utils.cuh"
#include "src/scheduler/utils.cuh"
#include "src/communication/alltoall_runner.cuh"

namespace fast_ep {

// NOTE: a hack number for distinguishing zero-computation tokens from the original ones
constexpr int ZC_MARK = 1000000;

template<int N>
__device__ __forceinline__ void WrapExclusiveSum(int r[N + 1], int lane_id) {
  #pragma unroll
  for (int n = 0; n < N; n++) {
    r[n+1] += r[n];
  }

  int v = r[N];
  #pragma unroll
  for (int offset = 1; offset < 32; offset *= 2) {
    int n = __shfl_up_sync(0xFFFFFFFF, v, offset);
    if (lane_id >= offset) v += n;
  }

  int prev = v - r[N];
  #pragma unroll
  for (int n = 0; n < N + 1; n++) {
    r[n] += prev;
  }
}

template<int NUM_EXPERTS>
__global__ void expandPlanKernel(ExpandPlan p) {
  extern __shared__ int smem[]; // p.num_experts

  int tid = threadIdx.x;
  for (int i = tid; i < p.num_experts; i += blockDim.x) {
    smem[i] = 0;
  }

  __syncthreads();

  for (int i = tid; i < p.num_tokens * p.topk; i += blockDim.x) {
    int expert_index = p.expert_indices[i];
    // non-zero computation experts
    if (expert_index >= 0) {
      atomicAdd(&smem[expert_index], 1);
    }
  }

  __syncthreads();

  if (threadIdx.x / 32 == 0) {
    int lane_id = threadIdx.x % 32;

    int r[NUM_EXPERTS / 32 + 1];
    r[0] = 0;
    #pragma unroll
    for (int n = 0; n < NUM_EXPERTS / 32; n++) {
      int k = NUM_EXPERTS / 32 * lane_id + n;
      r[n+1] = smem[k];
    }
    WrapExclusiveSum<NUM_EXPERTS / 32>(r, lane_id);

    if (lane_id == 0) {
      p.exclusive_sum[0] = 0;
    }
    #pragma unroll
    for (int n = 0; n < NUM_EXPERTS / 32; n++) {
      int k = NUM_EXPERTS / 32 * lane_id + n;
      smem[k] = r[n+1];
      p.exclusive_sum[k + 1] = r[n + 1];
    }
  }
  __syncthreads();

  for (int i = tid; i < p.num_tokens * p.topk; i += blockDim.x) {
    int expert_index = p.expert_indices[i];
    if (expert_index < 0) {
      int token_idx = i / p.topk;
      // NOTE: for zero-computation experts
      p.mapping[i] = token_idx + ZC_MARK;
    }
  }

  if (tid < p.num_experts) {
    for (int i = 0; i < p.num_tokens * p.topk; ++i) {
      int expert_index = p.expert_indices[i];
      if (expert_index < 0) continue;
      if (expert_index == tid) {
        p.mapping[i] = smem[expert_index] - 1;
        smem[expert_index] -= 1;
      }
    }
  }
}

__global__ void blockExpandPlanKernel(ExpandPlan p) {
  extern __shared__ int smem[]; // p.num_experts

  int tid = threadIdx.x;
  for (int i = tid; i < p.num_experts; i += blockDim.x) {
    smem[i] = 0;
  }

  __syncthreads();

  for (int i = tid; i < p.num_tokens * p.topk; i += blockDim.x) {
    int expert_index = p.expert_indices[i];
    // non-zero computation experts
    if (expert_index >= 0) {
      atomicAdd(&smem[expert_index], 1);
    }
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    int accum = 0;
    p.exclusive_sum[0] = accum;
    for (int i = 0; i < p.num_experts; i++) {
      accum += smem[i];
      smem[i] = accum;
      p.exclusive_sum[i + 1] = accum;
    }
  }

  __syncthreads();

  for (int i = tid; i < p.num_tokens * p.topk; i += blockDim.x) {
    int expert_index = p.expert_indices[i];
    if (expert_index < 0) {
      int token_idx = i / p.topk;
      // NOTE: for zero-computation experts
      p.mapping[i] = token_idx + ZC_MARK;
    }
  }

  if (tid < p.num_experts) {
    for (int i = 0; i < p.num_tokens * p.topk; ++i) {
      int expert_index = p.expert_indices[i];
      if (expert_index < 0) continue;
      if (expert_index == tid) {
        p.mapping[i] = smem[expert_index] - 1;
        smem[expert_index] -= 1;
      }
    }
  }
}

void expandPlan(ExpandPlan p, cudaStream_t stream) {
  int block_size = std::min<int>(512, p.num_tokens * p.topk);
  block_size = std::max<int>(block_size, p.num_experts);

  if (p.num_experts == 512) {
    expandPlanKernel<512><<<1, block_size, sizeof(int) * p.num_experts, stream>>>(p);
  }
  else if (p.num_experts == 256) {
    expandPlanKernel<256><<<1, block_size, sizeof(int) * p.num_experts, stream>>>(p);
  }
  else if (p.num_experts == 128) {
    expandPlanKernel<128><<<1, block_size, sizeof(int) * p.num_experts, stream>>>(p);
  }
  else if (p.num_experts == 32) {
    expandPlanKernel<32><<<1, block_size, sizeof(int) * p.num_experts, stream>>>(p);
  } else {
    blockExpandPlanKernel<<<1, block_size, sizeof(int) * p.num_experts, stream>>>(p);
  }
}

template <typename T, int BLOCK_SIZE, int HIDDEN_SIZE>
__global__ void expandKernelVectorized(ExpandParams<T> p) {
  assert(BLOCK_SIZE == blockDim.x);

  using VectorizedType = VectorizedType<T>;
  constexpr int ItemsPerVec = VectorizedType::ItemsPerVec;
  static_assert(HIDDEN_SIZE % ItemsPerVec == 0);

  for (int i = blockIdx.x; i < p.plan.num_tokens * p.plan.topk; i += gridDim.x) {
    int expand_idx = p.plan.mapping[i];
    // skip zero-computation experts
    if (expand_idx >= ZC_MARK) continue;
    const T* src = p.src + p.cols * (i / p.plan.topk);
    T* dst = p.dst + p.cols * p.plan.mapping[i];

    #pragma unroll
    for (int k = threadIdx.x; k < HIDDEN_SIZE / ItemsPerVec; k += BLOCK_SIZE)
    {
      reinterpret_cast<VectorizedType *>(dst)[k] = reinterpret_cast<const VectorizedType *>(src)[k];
    }
  }
}

template <typename T>
void expand(ExpandParams<T> p, cudaStream_t stream) {
  if (p.plan.num_tokens > 0) {
    int num_blocks = p.plan.num_tokens * p.plan.topk;
    int block_size = 512;

    BLOCK_SIZE_SWITCH(block_size,
      HIDDEN_SIZE_SWITCH(p.cols,
        [&]() {
          expandKernelVectorized<T, BLOCK_SIZE, HIDDEN_SIZE><<<num_blocks, BLOCK_SIZE, 0, stream>>>(p);
        })
    )();
  }
}

template
void expand(ExpandParams<float> p, cudaStream_t stream);

template
void expand(ExpandParams<half> p, cudaStream_t stream);

#ifdef ENABLE_BF16
template
void expand(ExpandParams<__nv_bfloat16> p, cudaStream_t stream);
#endif

template
void expand(ExpandParams<__nv_fp8_e4m3> p, cudaStream_t stream);

template <typename T, typename ScaleType, int BLOCK_SIZE, int HIDDEN_SIZE>
__global__ void reduceKernel(__grid_constant__ const ReduceParams<T, ScaleType> p) {
  assert(BLOCK_SIZE == blockDim.x);

  using AccumType = float;
  using PackedAccumType = float2;
  constexpr int ItemsPerPack = 2;

  using VectorizedType = VectorizedType<T>;
  using PackedType = typename PackedType<T>::type;
  constexpr int ItemsPerVec = VectorizedType::ItemsPerVec;
  constexpr int VecsPerThread = CEILDIV(HIDDEN_SIZE, BLOCK_SIZE * ItemsPerVec);
  constexpr int PacksPerVec = ItemsPerVec / ItemsPerPack;

  static_assert(HIDDEN_SIZE % ItemsPerVec == 0);

  extern __shared__ char smem_[]; /*topk*/
  ScaleType* scales = reinterpret_cast<ScaleType*>(smem_);

  PackedAccumType thread_result[VecsPerThread][PacksPerVec];
  auto clear = [&thread_result]() {
#pragma unroll
    for (int i = 0; i < VecsPerThread; ++i) {
#pragma unroll
      for (int j = 0; j < PacksPerVec; ++j)
        thread_result[i][j] = {};
    }
  };

  auto reduce = [&thread_result](VectorizedType thread_value, int i, ScaleType scale) {
    auto accum_scale = static_cast<AccumType>(scale);
    auto packed_thread_value = reinterpret_cast<PackedType*>(&thread_value);
#pragma unroll
    for (int j = 0; j < PacksPerVec; ++j) {
      thread_result[i][j] = thread_result[i][j] + cuda_cast<PackedAccumType>(packed_thread_value[j]) * accum_scale;
    }
  };

  for (int token_idx = blockIdx.x; token_idx < p.plan.num_tokens; token_idx += gridDim.x) {
    __syncthreads();
    if (threadIdx.x == 0) {
      for (int j = 0; j < p.plan.topk; j++) {
        scales[j] = p.expert_scales[token_idx * p.plan.topk + j];
      }
    }
    __syncthreads();

    clear();

    for (int topk_idx = 0; topk_idx < p.plan.topk; topk_idx++) {
      int expanded_row_idx = p.plan.mapping[token_idx * p.plan.topk + topk_idx];
      const T* src;
      if (expanded_row_idx >= ZC_MARK) {
        int raw_token_idx = expanded_row_idx - ZC_MARK;
        src = p.raw_tokens + HIDDEN_SIZE * raw_token_idx;
      } else {
        src = p.src + HIDDEN_SIZE * expanded_row_idx;
      }

      ScaleType scale = scales[topk_idx];

#pragma unroll
      for (int k = threadIdx.x; k < HIDDEN_SIZE / ItemsPerVec; k += BLOCK_SIZE) {
        VectorizedType thread_value = ((VectorizedType*)src)[k];
        reduce(thread_value, k / BLOCK_SIZE, scale);
      }
    }

    T* dst = p.dst + HIDDEN_SIZE * token_idx;
    const T* shared = p.shared + HIDDEN_SIZE * token_idx;
#pragma unroll
    for (int k = threadIdx.x; k < HIDDEN_SIZE / ItemsPerVec; k += BLOCK_SIZE) {
      if (p.shared == nullptr) {
#pragma unroll
        for (int n = 0; n < PacksPerVec; n++) {
          reinterpret_cast<PackedType*>(dst)[k * PacksPerVec + n] =
              cuda_cast<PackedType>(thread_result[k / BLOCK_SIZE][n]);
        }
      } else {
#pragma unroll
        for (int n = 0; n < PacksPerVec; n++) {
          reinterpret_cast<PackedType*>(dst)[k * PacksPerVec + n] = cuda_cast<PackedType>(
              thread_result[k / BLOCK_SIZE][n] +
              cuda_cast<PackedAccumType>(reinterpret_cast<const PackedType*>(shared)[k * PacksPerVec + n]));
        }
      }
    }
  }
}

template<typename T, typename ScaleType>
void reduce(ReduceParams<T, ScaleType> p, cudaStream_t stream) {
  if (p.plan.num_tokens > 0) {
    int num_blocks = p.plan.num_tokens;
    int block_size = 512;

    BLOCK_SIZE_SWITCH(block_size, HIDDEN_SIZE_SWITCH(p.cols, [&]() {
                        if constexpr (!std::is_same_v<T, float>) {
                          reduceKernel<T, ScaleType, BLOCK_SIZE, HIDDEN_SIZE>
                              <<<num_blocks, BLOCK_SIZE, sizeof(ScaleType) * p.plan.topk, stream>>>(p);
                        } else {
                          throw std::runtime_error("Not implemented");
                        }
                      }))();
  }
}

template
void reduce(ReduceParams<float, float> p, cudaStream_t stream);

template
void reduce(ReduceParams<half, float> p, cudaStream_t stream);

#ifdef ENABLE_BF16
template
void reduce(ReduceParams<__nv_bfloat16, float> p, cudaStream_t stream);
#endif

__device__ __inline__ bool is_same_node(int my_rank, int remote_rank, int num_ranks_per_node) {
  return (my_rank / num_ranks_per_node) == (remote_rank / num_ranks_per_node);
}

__global__ void allGatherPlanKernel(AllGatheredPlan p) {
  int remote_rank = blockIdx.x;

  if (is_same_node(p.my_rank, remote_rank, p.num_ranks_per_node)) {
    p.comm_buff.smChan_put(
        remote_rank,
        p.my_rank * (p.num_experts + 1) * sizeof(int),
        0,
        (p.num_experts + 1) * sizeof(int),
        threadIdx.x,
        blockDim.x,
        4);
    __syncthreads();

    if (threadIdx.x == 0) {
      p.comm_buff.smChan_signal(remote_rank);
      p.comm_buff.smChan_wait(remote_rank);
    }
  } else {
    if (threadIdx.x == 0) {
      p.comm_buff.proxyChan_putWithSignal(
          remote_rank,
          p.my_rank * (p.num_experts + 1) * sizeof(int),
          0,
          (p.num_experts + 1) * sizeof(int));
      p.comm_buff.proxyChan_wait(remote_rank);
    }
  }
}

__device__ __forceinline__ void recvPlan(AllGatheredPlan p) {
  int lane_id = threadIdx.x % 32;
  if (lane_id == 0) {
    int local_num_experts = p.num_experts / p.ep_world_size;

    for (int i = 0; i < p.ep_world_size; i++) {
      int remote_recv_plan = 0;
      int begin_expert = i * local_num_experts;
      int end_expert = (i + 1) * local_num_experts;
      for (int j = 0; j < p.my_rank; j++) {
        int* exclusive_sum = p.all_gathered + (p.num_experts + 1) * j;
        remote_recv_plan += __ldg(exclusive_sum + end_expert) - __ldg(exclusive_sum + begin_expert);
      }
      p.remote_recv_plan[i] = remote_recv_plan;
    }
  }
}

__device__ __forceinline__ void sendPlan(AllGatheredPlan p) {
  int lane_id = threadIdx.x % 32;
  if (lane_id == 0) {
    int local_num_experts = p.num_experts / p.ep_world_size;
    int local_send_plan = 0;
    int begin_expert = p.my_rank * local_num_experts;
    int end_expert = (p.my_rank + 1) * local_num_experts;
    p.local_send_plan[0] = local_send_plan;

    for (int i = 0; i < p.ep_world_size; i++) {
        int* exclusive_sum = p.all_gathered + (p.num_experts + 1) * i;
        local_send_plan += __ldg(exclusive_sum + end_expert) - __ldg(exclusive_sum + begin_expert);
        p.local_send_plan[i + 1] = local_send_plan;
    }
  }
}

template<int NUM_EXPERTS>
__device__ __forceinline__ void arrangePlanDst(AllGatheredPlan p) {
  int local_num_experts = p.num_experts / p.ep_world_size;
  int ep_world_size = p.ep_world_size;

  int r_rows[NUM_EXPERTS / 32 + 1];  // [ep_world_size * local_num_experts / 32 + 1]
  int lane_id = threadIdx.x % 32;

  r_rows[0] = 0;
  #pragma unroll
  for (int n = 0; n < NUM_EXPERTS / 32; n++) {
    int k = NUM_EXPERTS / 32 * lane_id + n;
    int i = k / ep_world_size;
    int j = k % ep_world_size;

    int expert_position = (p.num_experts + 1) * j + p.my_rank * local_num_experts + i;
    int rows = __ldg(p.all_gathered + expert_position + 1) - __ldg(p.all_gathered + expert_position);
    r_rows[n+1] = rows;
  }

  WrapExclusiveSum<NUM_EXPERTS / 32>(r_rows, lane_id);

  if (lane_id == 0) {
    p.gemm_plan[0] = 0;
  }
  #pragma unroll
  for (int n = 0; n < NUM_EXPERTS / 32; n++) {
    int k = NUM_EXPERTS / 32 * lane_id + n;
    int i = k / ep_world_size;
    int j = k % ep_world_size;

    p.arrange_plan.dst[k] = r_rows[n];
    if (j == ep_world_size - 1) {
      p.gemm_plan[i + 1] = r_rows[n + 1];
    }
  }
}

template<int NUM_EXPERTS>
__device__ __forceinline__ void  arrangePlanSrc(AllGatheredPlan p) {
  int local_num_experts = p.num_experts / p.ep_world_size;
  int ep_world_size = p.ep_world_size;

  int r_rows[NUM_EXPERTS / 32 + 1];  // [local_num_experts * ep_world_size / 32 + 1]
  int lane_id = threadIdx.x % 32;

  r_rows[0] = 0;
  #pragma unroll
  for (int n = 0; n < NUM_EXPERTS / 32; n++) {
    int k = NUM_EXPERTS / 32 * lane_id + n;
    int i = k % local_num_experts;
    int j = k / local_num_experts;

    int expert_position = (p.num_experts + 1) * j + p.my_rank * local_num_experts + i;
    int rows = __ldg(p.all_gathered + expert_position + 1) - __ldg(p.all_gathered + expert_position);
    r_rows[n+1] = rows;

    p.arrange_plan.rows[i * ep_world_size + j] = rows;
  }

  WrapExclusiveSum<NUM_EXPERTS / 32>(r_rows, lane_id);

  #pragma unroll
  for (int n = 0; n < NUM_EXPERTS / 32; n++) {
    int k = NUM_EXPERTS / 32 * lane_id + n;
    int i = k % local_num_experts;
    int j = k / local_num_experts;

    p.arrange_plan.src[i * ep_world_size + j] = r_rows[n];
  }
}

template<int NUM_EXPERTS>
__global__ void planKernel(AllGatheredPlan p) {
  int warp_id = threadIdx.x / 32;

  if (warp_id == 0) {
    recvPlan(p);
  }

  if (warp_id == 1) {
    sendPlan(p);
  }

  if (warp_id == 2) {
    arrangePlanSrc<NUM_EXPERTS>(p);
  }

  if (warp_id == 3) {
    arrangePlanDst<NUM_EXPERTS>(p);
  }
}

__global__ void blockAllGatherPlanKernel(AllGatheredPlan p) {
  if (threadIdx.x < p.ep_world_size) {
    int remote_rank = threadIdx.x;

    if (is_same_node(p.my_rank, remote_rank, p.num_ranks_per_node)) {
      p.comm_buff.smChan_put(remote_rank,
          p.my_rank * (p.num_experts + 1) * sizeof(int),
          0,
          (p.num_experts + 1) * sizeof(int),
          0,
          1,
          4);
      p.comm_buff.smChan_signal(remote_rank);
      p.comm_buff.smChan_wait(remote_rank);
    } else {
      p.comm_buff.proxyChan_putWithSignal(remote_rank,
          p.my_rank * (p.num_experts + 1) * sizeof(int),
          0,
          (p.num_experts + 1) * sizeof(int));
      p.comm_buff.proxyChan_wait(remote_rank);
    }
  }

  __syncthreads();

  int local_num_experts = p.num_experts / p.ep_world_size;

  if (threadIdx.x == 0) {
    for (int i = 0; i < p.ep_world_size; i++) {
      int remote_recv_plan = 0;
      int begin_expert = i * local_num_experts;
      int end_expert = (i + 1) * local_num_experts;
      for (int j = 0; j < p.my_rank; j++) {
        int* exclusive_sum = p.all_gathered + (p.num_experts + 1) * j;
        remote_recv_plan += exclusive_sum[end_expert] - exclusive_sum[begin_expert];
      }
      p.remote_recv_plan[i] = remote_recv_plan;
    }
  }

  if (threadIdx.x == 1) {
    int local_send_plan = 0;
    int begin_expert = p.my_rank * local_num_experts;
    int end_expert = (p.my_rank + 1) * local_num_experts;
    p.local_send_plan[0] = local_send_plan;
    for (int i = 0; i < p.ep_world_size; i++) {
        int* exclusive_sum = p.all_gathered + (p.num_experts + 1) * i;
        local_send_plan += exclusive_sum[end_expert] - exclusive_sum[begin_expert];
        p.local_send_plan[i + 1] = local_send_plan;
    }
  }

  if (threadIdx.x == 2) {
    int accum = 0;
    p.gemm_plan[0] = accum;
    for (int i = 0; i < local_num_experts; i++) {
      for (int j = 0; j < p.ep_world_size; j++) {
        int expert_position = (p.num_experts + 1) * j + p.my_rank * local_num_experts + i;
        int rows = p.all_gathered[expert_position + 1] - p.all_gathered[expert_position];
        p.arrange_plan.dst[i * p.ep_world_size + j] = accum;
        accum += rows;
      }
      p.gemm_plan[i + 1] = accum;
    }
  }

  if (threadIdx.x == 3) {
    int accum = 0;
    for (int j = 0; j < p.ep_world_size; j++) {
      for (int i = 0; i < local_num_experts; i++) {
        p.arrange_plan.src[i * p.ep_world_size + j] = accum;

        int expert_position = (p.num_experts + 1) * j + p.my_rank * local_num_experts + i;
        int rows = p.all_gathered[expert_position + 1] - p.all_gathered[expert_position];
        p.arrange_plan.rows[i * p.ep_world_size + j] = rows;
        accum += rows;
      }
    }
  }
}

void allGatherPlan(AllGatheredPlan p, cudaStream_t stream) {
  if (p.num_experts == 512) {
    allGatherPlanKernel<<<p.ep_world_size, p.num_experts + 1, 0, stream>>>(p);
    planKernel<512><<<1, 128, 0, stream>>>(p);
  }
  else if (p.num_experts == 256) {
    allGatherPlanKernel<<<p.ep_world_size, p.num_experts + 1, 0, stream>>>(p);
    planKernel<256><<<1, 128, 0, stream>>>(p);
  }
  else if (p.num_experts == 128) {
    allGatherPlanKernel<<<p.ep_world_size, p.num_experts + 1, 0, stream>>>(p);
    planKernel<128><<<1, 128, 0, stream>>>(p);
  }
  else if (p.num_experts == 32) {
    allGatherPlanKernel<<<p.ep_world_size, p.num_experts + 1, 0, stream>>>(p);
    planKernel<32><<<1, 128, 0, stream>>>(p);
  } else {
    blockAllGatherPlanKernel<<<1, std::max<int>(4, p.ep_world_size), 0, stream>>>(p);
  }
}

// prepare() methods moved to fast_ep_kernels.cuh for CUDA 13 nvlink compatibility

template<typename T, int BLOCK_SIZE, int HIDDEN_SIZE>
__global__ void arrangeKernel(ArrangeParams<T> p) {
  using VectorizedType = VectorizedType<T>;
  constexpr int ItemsPerVec = VectorizedType::ItemsPerVec;
  static_assert(HIDDEN_SIZE % ItemsPerVec == 0);

  int num_blocks_per_problem = CEILDIV(gridDim.x, p.num_experts / p.ep_world_size * p.ep_world_size);
  int problem_idx = blockIdx.x / num_blocks_per_problem;
  int block_offset = blockIdx.x % num_blocks_per_problem;

  int num_rows = p.plan.rows[problem_idx];
  int64_t src_offset = p.plan.src[problem_idx];
  int64_t dst_offset = p.plan.dst[problem_idx];

  for (int i = block_offset; i < num_rows; i += num_blocks_per_problem) {
    const T* src = p.src + (src_offset + i) * HIDDEN_SIZE;
    T* dst = p.dst + (dst_offset + i) * HIDDEN_SIZE;

    #pragma unroll
    for (int k = threadIdx.x; k < HIDDEN_SIZE / ItemsPerVec; k += BLOCK_SIZE)
    {
      reinterpret_cast<VectorizedType *>(dst)[k] = reinterpret_cast<const VectorizedType *>(src)[k];
    }
  }
}

template<typename T>
void arrange(ArrangeParams<T> p, cudaStream_t stream) {
  int block_size = 512;
  int num_problems = p.num_experts / p.ep_world_size * p.ep_world_size;
  int num_blocks = CEILDIV(p.num_tokens_hint, num_problems) * num_problems;

  BLOCK_SIZE_SWITCH(block_size,
    HIDDEN_SIZE_SWITCH(p.cols,
      [&]() {
        arrangeKernel<T, BLOCK_SIZE, HIDDEN_SIZE><<<num_blocks, BLOCK_SIZE, 0, stream>>>(p);
      })
  )();
}

template
void arrange(ArrangeParams<float> p, cudaStream_t stream);

template
void arrange(ArrangeParams<half> p, cudaStream_t stream);

#ifdef ENABLE_BF16
template
void arrange(ArrangeParams<__nv_bfloat16> p, cudaStream_t stream);
#endif

}

template 
void eps::All2All<float, fast_ep::DispatchAll2AllParams<float>>(fast_ep::DispatchAll2AllParams<float> p, cudaStream_t stream);

template 
void eps::All2All<float, fast_ep::CombineAll2AllParams<float>>(fast_ep::CombineAll2AllParams<float> p, cudaStream_t stream);

template 
void eps::All2All<half, fast_ep::DispatchAll2AllParams<half>>(fast_ep::DispatchAll2AllParams<half> p, cudaStream_t stream);

template 
void eps::All2All<half, fast_ep::CombineAll2AllParams<half>>(fast_ep::CombineAll2AllParams<half> p, cudaStream_t stream);


#ifdef ENABLE_BF16
template 
void eps::All2All<__nv_bfloat16, fast_ep::DispatchAll2AllParams<__nv_bfloat16>>(fast_ep::DispatchAll2AllParams<__nv_bfloat16> p, cudaStream_t stream);

template 
void eps::All2All<__nv_bfloat16, fast_ep::CombineAll2AllParams<__nv_bfloat16>>(fast_ep::CombineAll2AllParams<__nv_bfloat16> p, cudaStream_t stream);
#endif

template 
void eps::All2All<__nv_fp8_e4m3, fast_ep::DispatchAll2AllParams<__nv_fp8_e4m3>>(fast_ep::DispatchAll2AllParams<__nv_fp8_e4m3> p, cudaStream_t stream);

template 
void eps::All2All<__nv_fp8_e4m3, fast_ep::CombineAll2AllParams<__nv_fp8_e4m3>>(fast_ep::CombineAll2AllParams<__nv_fp8_e4m3> p, cudaStream_t stream);
