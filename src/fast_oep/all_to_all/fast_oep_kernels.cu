#include "src/fast_oep/all_to_all/fast_oep_kernels.cuh"

#include <algorithm>

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "src/common/cuda_utils.cuh"
#include "src/scheduler/utils.cuh"
#include "src/communication/alltoall_runner.cuh"

namespace fast_oep {

__global__ void arrangeAndPlanKernel(ArrangePlan p) {
  extern __shared__ int smem[]; // world_size

  int tid = threadIdx.x;
  for (int i = tid; i < p.world_size; i += blockDim.x) {
    smem[i] = 0;
  }
  __syncthreads();

  for (int i = tid; i < p.num_tokens; i += blockDim.x) {
    int id = p.ids[i];

    if (id >= 0) {
      int dst_rank = id / p.size_per_rank;
      atomicAdd(&smem[dst_rank], 1);
    }
  }
  __syncthreads();

  if (tid == 0) {
    for (int i = 1; i < p.world_size; i++) {
      smem[i] += smem[i-1];
    }

    p.exclusive_sum[0] = 0;
    for (int i = 0; i < p.world_size; i++) {
      p.exclusive_sum[i+1] = smem[i];
    }
  }
  __syncthreads();

  for (int i = tid; i < p.num_tokens; i += blockDim.x) {
    int id = p.ids[i];

    if (id >= 0) {
      int dst_rank = id / p.size_per_rank;
      int pos = atomicAdd(&smem[dst_rank], -1) - 1;

      p.mapping[i] = pos;
      p.arranged_ids[pos] = id;
    } else {
      p.mapping[i] = -1;
    }
  }
}

void arrangePlan(ArrangePlan p, cudaStream_t stream) {
  arrangeAndPlanKernel<<<1, 1024, sizeof(int) * p.world_size, stream>>>(p);
}

__device__ __inline__ bool is_same_node(int my_rank, int remote_rank, int num_ranks_per_node) {
  return (my_rank / num_ranks_per_node) == (remote_rank / num_ranks_per_node);
}

__global__ void allGatherPlanKernel(AllGatheredPlan p) {
  int remote_rank = blockIdx.x;

  if (is_same_node(p.my_rank, remote_rank, p.num_ranks_per_node)) {
    p.comm_buff.smChan_put(
        remote_rank,
        p.my_rank * (p.world_size + 1) * sizeof(int),
        0,
        (p.world_size + 1) * sizeof(int),
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
          p.my_rank * (p.world_size + 1) * sizeof(int),
          0,
          (p.world_size + 1) * sizeof(int));
      p.comm_buff.proxyChan_wait(remote_rank);
    }
  }
}

__device__ __forceinline__ void recvPlan(AllGatheredPlan p) {
  int lane_id = threadIdx.x % 32;
  if (lane_id == 0) {

    for (int i = 0; i < p.world_size; i++) {
      int remote_recv_plan = 0;

      for (int j = 0; j < p.my_rank; j++) {
        int* exclusive_sum = p.all_gathered + (p.world_size + 1) * j;
        remote_recv_plan += __ldg(exclusive_sum + i + 1) - __ldg(exclusive_sum + i);
      }
      p.remote_recv_plan[i] = remote_recv_plan;
    }
  }
}

__device__ __forceinline__ void sendPlan(AllGatheredPlan p) {
  int lane_id = threadIdx.x % 32;
  if (lane_id == 0) {
    int local_send_plan = 0;

    p.local_send_plan[0] = local_send_plan;

    for (int i = 0; i < p.world_size; i++) {
        int* exclusive_sum = p.all_gathered + (p.world_size + 1) * i;
        local_send_plan += __ldg(exclusive_sum + p.my_rank + 1) - __ldg(exclusive_sum + p.my_rank);
        p.local_send_plan[i + 1] = local_send_plan;
    }
  }
}

__global__ void planKernel(AllGatheredPlan p) {
  int warp_id = threadIdx.x / 32;

  if (warp_id == 0) {
    recvPlan(p);
  }

  if (warp_id == 1) {
    sendPlan(p);
  }
}

void allGatherPlan(AllGatheredPlan p, cudaStream_t stream) {
  allGatherPlanKernel<<<p.world_size, p.world_size + 1, 0, stream>>>(p);
  planKernel<<<1, 64, 0, stream>>>(p);
}

template<typename T, int BLOCK_SIZE, int64_t EMBED_DIM>
__global__ void lookupKernel(LookupParams<T> p) {
  using VectorizedType = VectorizedType<T>;
  constexpr int ItemsPerVec = VectorizedType::ItemsPerVec;
  static_assert(EMBED_DIM % ItemsPerVec == 0);

  int num_rows = p.local_send_plan[p.world_size];
  for (int i = blockIdx.x; i < num_rows; i += gridDim.x) {
    int local_id = p.ids[i] % p.size_per_rank;
    const T* src = p.embed_table + local_id * EMBED_DIM;
    T* dst = p.dst + i * EMBED_DIM;

    #pragma unroll
    for (int k = threadIdx.x; k < EMBED_DIM / ItemsPerVec; k += BLOCK_SIZE)
    {
      reinterpret_cast<VectorizedType *>(dst)[k] = reinterpret_cast<const VectorizedType *>(src)[k];
    }
  }
}

template<typename T>
void lookup(LookupParams<T> p, cudaStream_t stream) {
  int block_size = 256;

  BLOCK_SIZE_SWITCH(block_size,
    EMBED_DIM_SWITCH(p.embed_dim,
      [&]() {
        lookupKernel<T, BLOCK_SIZE, EMBED_DIM><<<p.num_tokens_hint, BLOCK_SIZE, 0, stream>>>(p);
      })
  )();
}

template<typename T, int BLOCK_SIZE, int64_t EMBED_DIM, bool DO_PERMUTE>
__global__ void scatterKernel(ScatterParams<T> p) {
  using VectorizedType = VectorizedType<T>;
  constexpr int ItemsPerVec = VectorizedType::ItemsPerVec;
  static_assert(EMBED_DIM % ItemsPerVec == 0);

  for (int i = blockIdx.x; i < p.num_tokens; i += gridDim.x) {
    if (p.mapping[i] >= 0) {
      const T* src = p.src + p.mapping[i] * EMBED_DIM;
      T* dst;
      if constexpr (!DO_PERMUTE) {
        dst = p.dst + i * EMBED_DIM;
      } else {
        int batch = p.num_tokens / p.n_grams;
        dst = p.dst + (i / p.n_grams + i % p.n_grams * batch) * EMBED_DIM;
      }

      #pragma unroll
      for (int k = threadIdx.x; k < EMBED_DIM / ItemsPerVec; k += BLOCK_SIZE)
      {
        reinterpret_cast<VectorizedType *>(dst)[k] = reinterpret_cast<const VectorizedType *>(src)[k];
      }
    }
  }
}

template<typename T>
void scatter(ScatterParams<T> p, cudaStream_t stream) {
  if (p.num_tokens <= 0) return;

  int block_size = 256;

  BLOCK_SIZE_SWITCH(block_size,
    EMBED_DIM_SWITCH(p.embed_dim,
      [&]() {
        if (p.do_permute) {
          scatterKernel<T, BLOCK_SIZE, EMBED_DIM, true><<<p.num_tokens, BLOCK_SIZE, 0, stream>>>(p);
        } else {
          scatterKernel<T, BLOCK_SIZE, EMBED_DIM, false><<<p.num_tokens, BLOCK_SIZE, 0, stream>>>(p);
        }
      })
  )();
}


template
void lookup(LookupParams<float> p, cudaStream_t stream);

template
void lookup(LookupParams<half> p, cudaStream_t stream);

#ifdef ENABLE_BF16
template
void lookup(LookupParams<__nv_bfloat16> p, cudaStream_t stream);
#endif

template
void scatter(ScatterParams<float> p, cudaStream_t stream);

template
void scatter(ScatterParams<half> p, cudaStream_t stream);

#ifdef ENABLE_BF16
template
void scatter(ScatterParams<__nv_bfloat16> p, cudaStream_t stream);
#endif

}

template 
void eps::All2All<int32_t, fast_oep::DispatchAll2AllParams<int32_t>>(fast_oep::DispatchAll2AllParams<int32_t> p, cudaStream_t stream);

template 
void eps::All2All<float, fast_oep::CombineAll2AllParams<float>>(fast_oep::CombineAll2AllParams<float> p, cudaStream_t stream);

template 
void eps::All2All<half, fast_oep::CombineAll2AllParams<half>>(fast_oep::CombineAll2AllParams<half> p, cudaStream_t stream);


#ifdef ENABLE_BF16
template 
void eps::All2All<__nv_bfloat16, fast_oep::CombineAll2AllParams<__nv_bfloat16>>(fast_oep::CombineAll2AllParams<__nv_bfloat16> p, cudaStream_t stream);
#endif
