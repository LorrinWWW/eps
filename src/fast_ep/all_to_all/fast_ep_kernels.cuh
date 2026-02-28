#pragma once

#include <mscclpp/core.hpp>
#include <mscclpp/proxy_channel.hpp>
#include <mscclpp/sm_channel.hpp>

#include "src/scheduler/gemm_executor_interface.cuh"
#include "src/scheduler/common.cuh"
#include "src/communication/alltoall_kernels.cuh"

using namespace eps;

namespace fast_ep {

struct ExpandPlan {
    int64_t num_tokens;
    int64_t topk;
    int64_t num_experts;

    const int* expert_indices; /*[num_tokens, topk]*/
    int* exclusive_sum; /*num_experts + 1*/
    int* mapping; /*num_tokens * topk*/
};

template<typename T>
struct ExpandParams {
    const T* src;
    T* dst;
    int64_t cols;

    ExpandPlan plan;
};

void expandPlan(ExpandPlan p, cudaStream_t stream);

template <typename T>
void expand(ExpandParams<T> p, cudaStream_t stream);

template<typename T, typename ScaleType>
struct ReduceParams {
    const T* src;
    T* dst;
    const T* shared;

    const T* raw_tokens;
    const ScaleType* expert_scales;

    int64_t cols;

    ExpandPlan plan;
};

template<typename T, typename ScaleType>
void reduce(ReduceParams<T, ScaleType> p, cudaStream_t stream);

struct ArrangePlan {
  int* src; /* [(num_experts / ep_world_size), ep_world_size] */
  int* rows; /* [(num_experts / ep_world_size), ep_world_size] */
  int* dst; /* [(num_experts / ep_world_size), ep_world_size] */
};

template<typename T>
struct ArrangeParams {
  int64_t ep_world_size;
  int64_t num_experts;
  int64_t cols;

  const T* src;
  T* dst;
  ArrangePlan plan;

  int64_t num_tokens_hint;
};

template<typename T>
void arrange(ArrangeParams<T> p, cudaStream_t stream);

struct AllGatheredPlan {
    int64_t my_rank;
    int64_t num_ranks_per_node;
    int64_t ep_world_size;
    int64_t num_experts; // Global

    int* all_gathered; /* ep_world_size * (p.num_experts + 1) */
    int* remote_recv_plan; /* ep_world_size */ // for dispatch alltoall
    int* local_send_plan; /* ep_world_size + 1 */ // for combine alltoall
    int* gemm_plan; /* (num_experts / ep_world_size) + 1 */

    ArrangePlan arrange_plan;

    CommBuff<int> comm_buff;
};

void allGatherPlan(AllGatheredPlan p, cudaStream_t stream);

template<typename T>
struct DispatchAll2AllParams {
    int64_t my_rank;
    int64_t num_ranks_per_node;
    int64_t world_size;
    int64_t num_experts; // Global
    int64_t cols;

    int* exclusive_sum; /* (p.num_experts + 1) */
    int* remote_recv_plan; /* ep_world_size */

    CommBuff<T> comm_buff;

    __device__ eps::All2AllParams<T> prepare(int remote_rank) {
        int local_num_experts = this->num_experts / this->world_size;
        int recv_start = this->remote_recv_plan[remote_rank];
        int send_start = this->exclusive_sum[local_num_experts * remote_rank];
        int send_end = this->exclusive_sum[local_num_experts * (remote_rank + 1)];
        return eps::All2AllParams<T>{
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
    int64_t num_experts; // Global
    int64_t cols;

    int* all_gathered; /* ep_world_size * (p.num_experts + 1) */
    int* local_send_plan; /* ep_world_size + 1*/

    CommBuff<T> comm_buff;

    __device__ eps::All2AllParams<T> prepare(int remote_rank) {
        int local_num_experts = this->num_experts / this->world_size;
        int recv_start = this->all_gathered[(this->num_experts + 1) * remote_rank + this->my_rank * local_num_experts];
        int send_start = this->local_send_plan[remote_rank];
        int send_end = this->local_send_plan[remote_rank + 1];
        return eps::All2AllParams<T>{
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