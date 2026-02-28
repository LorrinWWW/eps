#pragma once

#include <mscclpp/core.hpp>
#include <mscclpp/proxy_channel.hpp>
#include <mscclpp/sm_channel.hpp>

#include "src/scheduler/common.cuh"
#include "src/communication/alltoall_kernels.cuh"

using namespace eps;

namespace fast_oep {

struct ArrangePlan {
    int64_t size_per_rank;
    int64_t num_tokens;
    int64_t world_size;

    const int* ids; /*[num_tokens]*/
    int* arranged_ids; /*[num_tokens]*/
    int* exclusive_sum; /*world_size + 1*/
    int* mapping; /*num_tokens*/
};
void arrangePlan(ArrangePlan p, cudaStream_t stream);


struct AllGatheredPlan {
    int64_t my_rank;
    int64_t num_ranks_per_node;
    int64_t world_size;

    int* all_gathered; /* world_size * (world_size + 1) */
    int* remote_recv_plan; /* world_size */ // for dispatch alltoall
    int* local_send_plan; /* world_size + 1 */ // for combine alltoall

    CommBuff<int> comm_buff;
};
void allGatherPlan(AllGatheredPlan p, cudaStream_t stream);


template<typename T>
struct LookupParams {
  int64_t size_per_rank;
  int64_t world_size;
  int64_t num_tokens_hint;
  int64_t embed_dim;

  int32_t* ids;
  int32_t* local_send_plan;

  const T* embed_table;
  T* dst;
};
template<typename T>
void lookup(LookupParams<T> p, cudaStream_t stream);


template<typename T>
struct ScatterParams {
  const int32_t* mapping;
  int64_t num_tokens;
  int64_t embed_dim;
  int64_t n_grams;
  bool do_permute;

  const T* src;
  T* dst;
};

template<typename T>
void scatter(ScatterParams<T> p, cudaStream_t stream);

template<typename T>
struct DispatchAll2AllParams {
    int64_t my_rank;
    int64_t num_ranks_per_node;
    int64_t world_size;

    int* exclusive_sum; /* (world_size + 1) */
    int* remote_recv_plan; /* world_size */

    CommBuff<T> comm_buff;

    int64_t cols = 1;
    __device__ All2AllParams<T> prepare(int remote_rank) {
        int recv_start = this->remote_recv_plan[remote_rank];
        int send_start = this->exclusive_sum[remote_rank];
        int send_end = this->exclusive_sum[remote_rank + 1];
        return All2AllParams<T>{
            .my_rank = this->my_rank,
            .remote_rank = remote_rank,
            .num_ranks_per_node = this->num_ranks_per_node,
            .world_size = this->world_size,
            .comm_buff = this->comm_buff,
            .send_start = send_start,
            .send_end = send_end,
            .recv_start = recv_start,
            .cols = this->cols
        };
    }
};

template<typename T>
struct CombineAll2AllParams {
    int64_t my_rank;
    int64_t num_ranks_per_node;
    int64_t world_size;
    int64_t cols;

    int* all_gathered; /* world_size * (world_size + 1) */
    int* local_send_plan; /* world_size + 1*/

    CommBuff<T> comm_buff;

    __device__ All2AllParams<T> prepare(int remote_rank) {
        int recv_start = this->all_gathered[(this->world_size + 1) * remote_rank + this->my_rank];
        int send_start = this->local_send_plan[remote_rank];
        int send_end = this->local_send_plan[remote_rank + 1];
        return All2AllParams<T>{
            .my_rank = this->my_rank,
            .remote_rank = remote_rank,
            .num_ranks_per_node = this->num_ranks_per_node,
            .world_size = this->world_size,
            .comm_buff = this->comm_buff,
            .send_start = send_start,
            .send_end = send_end,
            .recv_start = recv_start,
            .cols = this->cols
        };
    }
};

}