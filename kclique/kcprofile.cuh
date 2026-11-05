#pragma once

#include "kcprofile_defs.cuh"

#if KCGPU_PROFILE

#ifdef KCPROFILE_DEFINE_SYMBOLS
__device__ unsigned long long *kcp_working_time;
__device__ int *kcp_working_idx;
__device__ int *kcp_worker_state;
__device__ unsigned long long *kcp_sm_node_visits;
__device__ unsigned long long *kcp_sm_node_cycles;
#else
extern __device__ unsigned long long *kcp_working_time;
extern __device__ int *kcp_working_idx;
extern __device__ int *kcp_worker_state;
extern __device__ unsigned long long *kcp_sm_node_visits;
extern __device__ unsigned long long *kcp_sm_node_cycles;
#endif

__device__ __forceinline__ int kcp_worker_id(int wx, int num_partitions) {
  if (blockIdx.x >= KCPROFILE_MAX_BLOCKS) {
    return -1;
  }
  const int worker_id = (int)blockIdx.x * num_partitions + wx;
  return (worker_id < KCPROFILE_MAX_WORKERS) ? worker_id : -1;
}

__device__ __forceinline__ int kcp_worker_slot_id(int sm_id, int level_ptr,
                                                  int slots_per_sm) {
  if (sm_id < 0 || sm_id >= KCPROFILE_MAX_SMS || level_ptr < 0 ||
      level_ptr >= slots_per_sm) {
    return -1;
  }
  const int worker_id = sm_id * slots_per_sm + level_ptr;
  return (worker_id < KCPROFILE_MAX_WORKERS) ? worker_id : -1;
}

__device__ __forceinline__ unsigned long long kcp_profile_timestamp_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ void kcp_profile_mark_working(int worker_id,
                                                         int lx) {
  if (worker_id < 0 || worker_id >= KCPROFILE_MAX_WORKERS || lx != 0) {
    return;
  }
  if (kcp_worker_state[worker_id] == 1) {
    return;
  }
  const int idx = kcp_working_idx[worker_id];
  if (idx < KCPROFILE_MAX_SAMPLES) {
    kcp_working_time[static_cast<size_t>(worker_id) * KCPROFILE_MAX_SAMPLES +
                     idx] = kcp_profile_timestamp_ns();
  }
  kcp_working_idx[worker_id] = idx + 1;
  kcp_worker_state[worker_id] = 1;
}

__device__ __forceinline__ void kcp_profile_mark_not_working(int worker_id,
                                                             int lx) {
  if (worker_id < 0 || worker_id >= KCPROFILE_MAX_WORKERS || lx != 0) {
    return;
  }
  if (kcp_worker_state[worker_id] == 0) {
    return;
  }
  const int idx = kcp_working_idx[worker_id];
  if (idx < KCPROFILE_MAX_SAMPLES) {
    kcp_working_time[static_cast<size_t>(worker_id) * KCPROFILE_MAX_SAMPLES +
                     idx] = kcp_profile_timestamp_ns();
  }
  kcp_working_idx[worker_id] = idx + 1;
  kcp_worker_state[worker_id] = 0;
}

__device__ __forceinline__ void
kcp_profile_note_node_visit(int sm_id, int lx, unsigned long long node_cycles) {
  if (lx != 0 || sm_id < 0 || sm_id >= KCPROFILE_MAX_SMS) {
    return;
  }
  atomicAdd(&kcp_sm_node_visits[sm_id], 1ULL);
  atomicAdd(&kcp_sm_node_cycles[sm_id], node_cycles);
}

__device__ __forceinline__ void
kcp_profile_add_node_visits(int sm_id, unsigned long long node_visits,
                            unsigned long long node_cycles) {
  if (sm_id < 0 || sm_id >= KCPROFILE_MAX_SMS || node_visits == 0ULL) {
    return;
  }
  atomicAdd(&kcp_sm_node_visits[sm_id], node_visits);
  if (node_cycles != 0ULL) {
    atomicAdd(&kcp_sm_node_cycles[sm_id], node_cycles);
  }
}

#else

__device__ __forceinline__ int kcp_worker_id(int wx, int num_partitions) {
  (void)wx;
  (void)num_partitions;
  return -1;
}

__device__ __forceinline__ int kcp_worker_slot_id(int sm_id, int level_ptr,
                                                  int slots_per_sm) {
  (void)sm_id;
  (void)level_ptr;
  (void)slots_per_sm;
  return -1;
}

__device__ __forceinline__ unsigned long long kcp_profile_timestamp_ns() {
  return 0ULL;
}

__device__ __forceinline__ void kcp_profile_mark_working(int worker_id,
                                                         int lx) {
  (void)worker_id;
  (void)lx;
}

__device__ __forceinline__ void kcp_profile_mark_not_working(int worker_id,
                                                             int lx) {
  (void)worker_id;
  (void)lx;
}

__device__ __forceinline__ void
kcp_profile_note_node_visit(int sm_id, int lx, unsigned long long node_cycles) {
  (void)sm_id;
  (void)lx;
  (void)node_cycles;
}

__device__ __forceinline__ void
kcp_profile_add_node_visits(int sm_id, unsigned long long node_visits,
                            unsigned long long node_cycles) {
  (void)sm_id;
  (void)node_visits;
  (void)node_cycles;
}

#endif
