#pragma once

#include <c10/core/Device.h>
#include <string>

namespace c10d::symmetric_memory {

// Cache of NCCL/RCCL device communicators (ncclDevComm) keyed by
// (device, group_name, key). Implemented in NCCLSymmetricMemory.cu, the one
// TU where <nccl_device.h> is includable: RCCL's version instantiates device
// builtins that do not exist in host-only compiles, so the type cannot leak
// into headers consumed by ProcessGroupNCCL.cpp and friends.
//
// Entries are erased when the owning process group tears down
// (release_nccl_devcomms_for_group), so a recreated group can never observe
// a communicator built for its predecessor. Erased entries are reclaimed by
// the communicator itself; survivors are destroyed at process exit.
//
// Returns a pointer to a cache-owned ncclDevComm; callers compiled against
// <nccl_device.h> cast it to ncclDevComm*.
void* get_or_create_nccl_devcomm(
    const c10::Device& device,
    const std::string& group_name,
    const std::string& key,
    int lsa_barrier_count,
    bool lsa_multimem);

// Identity-safe teardown: erase only the device communicators owned by `comm`
// (the host ncclComm_t, passed as void* so this header stays free of NCCL
// types). A stale producer whose comm was already replaced by a successor
// under the same group name becomes a no-op, so it cannot wipe the successor.
void release_nccl_devcomms_for_group(
    const c10::Device& device,
    const std::string& group_name,
    void* comm);

// Record, at communicator-init time, whether the RCCL symmetric-memory window
// preconditions (NCCL_CUMEM_ENABLE / NCCL_WIN_ENABLE) were set. RCCL samples
// these env vars inside ncclCommInitRank, i.e. before symm_mem is requested, so
// the value cannot be re-derived at rendezvous (the environment may have
// changed). The producing backend calls this right after comm creation; the
// ROCm rendezvous path enforces the recorded snapshot. No-op off ROCm.
void note_rccl_symm_precondition(
    const c10::Device& device,
    const std::string& group_name,
    bool ok);

} // namespace c10d::symmetric_memory
