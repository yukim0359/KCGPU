#pragma once

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <limits>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

#ifndef KCGPU_PROFILE
#define KCGPU_PROFILE 0
#endif

#if KCGPU_PROFILE

#include "kcprofile_defs.cuh"

struct KcpProfileBuffers {
  unsigned long long *d_working = nullptr;
  int *d_idx = nullptr;
  int *d_state = nullptr;
  unsigned long long *d_node_visits = nullptr;
  unsigned long long *d_node_cycles = nullptr;
  int device_id = -1;
};

static inline KcpProfileBuffers &kcp_profile_buffers() {
  static KcpProfileBuffers buffers;
  return buffers;
}

static inline bool kcp_profile_ensure_buffers(int device_id) {
  KcpProfileBuffers &buffers = kcp_profile_buffers();
  if (buffers.d_working != nullptr) {
    if (buffers.device_id != device_id) {
      cudaSetDevice(device_id);
    }
    return true;
  }

  cudaSetDevice(device_id);
  const size_t working_bytes = static_cast<size_t>(KCPROFILE_MAX_WORKERS) *
                               KCPROFILE_MAX_SAMPLES *
                               sizeof(unsigned long long);
  const size_t worker_meta_bytes =
      static_cast<size_t>(KCPROFILE_MAX_WORKERS) * sizeof(int);
  const size_t sm_bytes =
      static_cast<size_t>(KCPROFILE_MAX_SMS) * sizeof(unsigned long long);

  if (cudaMalloc(&buffers.d_working, working_bytes) != cudaSuccess ||
      cudaMalloc(&buffers.d_idx, worker_meta_bytes) != cudaSuccess ||
      cudaMalloc(&buffers.d_state, worker_meta_bytes) != cudaSuccess ||
      cudaMalloc(&buffers.d_node_visits, sm_bytes) != cudaSuccess ||
      cudaMalloc(&buffers.d_node_cycles, sm_bytes) != cudaSuccess) {
    std::fprintf(stderr, "KCGPU profile: cudaMalloc failed: %s\n",
                 cudaGetErrorString(cudaGetLastError()));
    return false;
  }

  cudaMemcpyToSymbol(kcp_working_time, &buffers.d_working,
                     sizeof(buffers.d_working));
  cudaMemcpyToSymbol(kcp_working_idx, &buffers.d_idx, sizeof(buffers.d_idx));
  cudaMemcpyToSymbol(kcp_worker_state, &buffers.d_state,
                     sizeof(buffers.d_state));
  cudaMemcpyToSymbol(kcp_sm_node_visits, &buffers.d_node_visits,
                     sizeof(buffers.d_node_visits));
  cudaMemcpyToSymbol(kcp_sm_node_cycles, &buffers.d_node_cycles,
                     sizeof(buffers.d_node_cycles));

  buffers.device_id = device_id;
  return true;
}

static inline void kcp_ensure_dir(const char *path) {
  struct stat st {};
  if (stat(path, &st) == -1) {
    mkdir(path, 0755);
  }
}

static inline const char *kcp_profile_dir() {
  const char *dir = std::getenv("KCGPU_PROFILE_DIR");
  return (dir && dir[0]) ? dir : "./profile";
}

static inline std::string kcp_profile_tag() {
  const char *tag = std::getenv("KCGPU_PROFILE_TAG");
  return (tag && tag[0]) ? std::string("_") + tag : std::string();
}

static inline void kcp_profile_reset(int device_id) {
  if (!kcp_profile_ensure_buffers(device_id)) {
    std::fprintf(stderr,
                 "KCGPU profile: reset skipped (buffer allocation failed)\n");
    return;
  }

  KcpProfileBuffers &buffers = kcp_profile_buffers();
  cudaSetDevice(device_id);
  cudaMemset(buffers.d_working, 0,
             static_cast<size_t>(KCPROFILE_MAX_WORKERS) *
                 KCPROFILE_MAX_SAMPLES * sizeof(unsigned long long));
  cudaMemset(buffers.d_idx, 0,
             static_cast<size_t>(KCPROFILE_MAX_WORKERS) * sizeof(int));
  cudaMemset(buffers.d_state, 0,
             static_cast<size_t>(KCPROFILE_MAX_WORKERS) * sizeof(int));
  cudaMemset(buffers.d_node_visits, 0,
             static_cast<size_t>(KCPROFILE_MAX_SMS) *
                 sizeof(unsigned long long));
  cudaMemset(buffers.d_node_cycles, 0,
             static_cast<size_t>(KCPROFILE_MAX_SMS) *
                 sizeof(unsigned long long));
}

static inline int kcp_device_clock_khz(int device_id) {
  int clock_khz = 0;
  cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, device_id);
  return clock_khz;
}

static inline void kcp_profile_export_csv(int device_id, const char *app_name) {
  if (!kcp_profile_ensure_buffers(device_id)) {
    std::fprintf(stderr,
                 "KCGPU profile: export aborted (buffer allocation failed)\n");
    return;
  }

  KcpProfileBuffers &buffers = kcp_profile_buffers();
  cudaSetDevice(device_id);
  std::vector<unsigned long long> h_working(
      static_cast<size_t>(KCPROFILE_MAX_WORKERS) * KCPROFILE_MAX_SAMPLES, 0ULL);
  std::vector<int> h_idx(static_cast<size_t>(KCPROFILE_MAX_WORKERS), 0);
  std::vector<unsigned long long> h_node_visits(
      static_cast<size_t>(KCPROFILE_MAX_SMS), 0ULL);
  std::vector<unsigned long long> h_node_cycles(
      static_cast<size_t>(KCPROFILE_MAX_SMS), 0ULL);

  cudaMemcpy(h_working.data(), buffers.d_working,
             h_working.size() * sizeof(unsigned long long),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_idx.data(), buffers.d_idx, h_idx.size() * sizeof(int),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_node_visits.data(), buffers.d_node_visits,
             h_node_visits.size() * sizeof(unsigned long long),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_node_cycles.data(), buffers.d_node_cycles,
             h_node_cycles.size() * sizeof(unsigned long long),
             cudaMemcpyDeviceToHost);

  unsigned long long min_timestamp_ns =
      std::numeric_limits<unsigned long long>::max();
  for (int worker = 0; worker < KCPROFILE_MAX_WORKERS; ++worker) {
    const int samples = std::min(h_idx[worker], KCPROFILE_MAX_SAMPLES);
    for (int i = 0; i < samples; ++i) {
      const unsigned long long t =
          h_working[static_cast<size_t>(worker) * KCPROFILE_MAX_SAMPLES + i];
      if (t == 0ULL) {
        continue;
      }
      min_timestamp_ns = std::min(min_timestamp_ns, t);
    }
  }
  if (min_timestamp_ns == std::numeric_limits<unsigned long long>::max()) {
    min_timestamp_ns = 0ULL;
  }

  const std::string tag = kcp_profile_tag();
  const char *out_dir = kcp_profile_dir();
  kcp_ensure_dir(out_dir);

  char timeline_path[512];
  std::snprintf(timeline_path, sizeof(timeline_path),
                "%s/%s_warp_timeline_working%s.csv", out_dir, app_name,
                tag.c_str());
  FILE *timeline = std::fopen(timeline_path, "w");
  if (!timeline) {
    std::fprintf(stderr, "KCGPU profile: failed to open %s (%s)\n",
                 timeline_path, std::strerror(errno));
    return;
  }
  std::fprintf(timeline, "warp_id,time_index,timestamp_ns,relative_time_ms,"
                         "state,state_description\n");

  for (int worker = 0; worker < KCPROFILE_MAX_WORKERS; ++worker) {
    const int samples = std::min(h_idx[worker], KCPROFILE_MAX_SAMPLES);
    for (int i = 0; i < samples; ++i) {
      const unsigned long long timestamp_ns =
          h_working[static_cast<size_t>(worker) * KCPROFILE_MAX_SAMPLES + i];
      if (timestamp_ns == 0ULL) {
        continue;
      }
      const int state = (i % 2 == 0) ? 1 : 0;
      const char *state_desc = (state == 1) ? "Working" : "NotWorking";
      const double relative_ms =
          static_cast<double>(timestamp_ns - min_timestamp_ns) / 1000000.0;
      std::fprintf(timeline, "%d,%d,%llu,%.6f,%d,%s\n", worker, i, timestamp_ns,
                   relative_ms, state, state_desc);
    }
  }
  std::fclose(timeline);

  char stats_path[512];
  std::snprintf(stats_path, sizeof(stats_path),
                "%s/%s_warp_statistics_working%s.csv", out_dir, app_name,
                tag.c_str());
  FILE *stats = std::fopen(stats_path, "w");
  if (stats) {
    std::fprintf(stats,
                 "warp_id,total_samples,recorded_samples,dropped_samples,"
                 "first_activity_ms,last_activity_ms,duration_ms\n");
    for (int worker = 0; worker < KCPROFILE_MAX_WORKERS; ++worker) {
      const int total_samples = h_idx[worker];
      const int samples = std::min(h_idx[worker], KCPROFILE_MAX_SAMPLES);
      if (samples == 0) {
        std::fprintf(stats, "%d,0,0,0,0.0,0.0,0.0\n", worker);
        continue;
      }
      unsigned long long first = std::numeric_limits<unsigned long long>::max();
      unsigned long long last = 0ULL;
      for (int i = 0; i < samples; ++i) {
        const unsigned long long timestamp_ns =
            h_working[static_cast<size_t>(worker) * KCPROFILE_MAX_SAMPLES + i];
        if (timestamp_ns == 0ULL) {
          continue;
        }
        first = std::min(first, timestamp_ns);
        last = std::max(last, timestamp_ns);
      }
      const double first_ms =
          static_cast<double>(first - min_timestamp_ns) / 1000000.0;
      const double last_ms =
          static_cast<double>(last - min_timestamp_ns) / 1000000.0;
      const int dropped_samples = std::max(0, total_samples - samples);
      std::fprintf(stats, "%d,%d,%d,%d,%.6f,%.6f,%.6f\n", worker, total_samples,
                   samples, dropped_samples, first_ms, last_ms,
                   last_ms - first_ms);
    }
    std::fclose(stats);
  }

  char sm_path[512];
  std::snprintf(sm_path, sizeof(sm_path), "%s/%s_sm_balance%s.csv", out_dir,
                app_name, tag.c_str());
  FILE *sm_file = std::fopen(sm_path, "w");
  if (sm_file) {
    unsigned long long total_nodes = 0ULL;
    unsigned long long total_cycles = 0ULL;
    int active_sms = 0;
    for (int sm = 0; sm < KCPROFILE_MAX_SMS; ++sm) {
      if (h_node_visits[sm] > 0ULL) {
        ++active_sms;
        total_nodes += h_node_visits[sm];
        total_cycles += h_node_cycles[sm];
      }
    }
    const double avg_nodes =
        (active_sms > 0)
            ? static_cast<double>(total_nodes) / static_cast<double>(active_sms)
            : 0.0;
    const double avg_cycles = (active_sms > 0)
                                  ? static_cast<double>(total_cycles) /
                                        static_cast<double>(active_sms)
                                  : 0.0;

    std::fprintf(sm_file,
                 "sm_id,node_visits,node_visit_time_ns,node_ratio_to_avg,"
                 "time_ratio_to_avg,avg_time_ns_per_node\n");
    for (int sm = 0; sm < KCPROFILE_MAX_SMS; ++sm) {
      const unsigned long long nodes = h_node_visits[sm];
      const unsigned long long cycles = h_node_cycles[sm];
      const double node_ratio =
          (avg_nodes > 0.0) ? static_cast<double>(nodes) / avg_nodes : 0.0;
      const double cycle_ratio =
          (avg_cycles > 0.0) ? static_cast<double>(cycles) / avg_cycles : 0.0;
      const double avg_per_node =
          (nodes > 0ULL)
              ? static_cast<double>(cycles) / static_cast<double>(nodes)
              : 0.0;
      std::fprintf(sm_file, "%d,%llu,%llu,%.6f,%.6f,%.6f\n", sm, nodes, cycles,
                   node_ratio, cycle_ratio, avg_per_node);
    }
    std::fclose(sm_file);
  }

  std::printf("KCGPU profile exported:\n");
  std::printf("  %s\n", timeline_path);
  std::printf("  %s\n", stats_path);
  std::printf("  %s\n", sm_path);
  std::fflush(stdout);
}

#else

static inline void kcp_profile_reset(int device_id) { (void)device_id; }

static inline void kcp_profile_export_csv(int device_id, const char *app_name) {
  (void)device_id;
  (void)app_name;
}

#endif
