#pragma once

#include <c10/core/Device.h>
#include <cstddef>
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
// Copies the cached ncclDevComm into caller-owned storage while holding the
// cache lock. The opaque output keeps ncclDevComm out of host-only headers and
// prevents teardown from invalidating a cache-owned reference at kernel launch.
void get_or_create_nccl_devcomm(
    const c10::Device& device,
    const std::string& group_name,
    const std::string& key,
    int lsa_barrier_count,
    bool lsa_multimem,
    void* devcomm_out,
    size_t devcomm_size);

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
//
// Keyed by the host communicator (`comm`, an ncclComm_t passed as void*), not
// by group name: rendezvous looks the snapshot up by the comm it resolves, so a
// later comm reusing a group name -- including a producer that never records a
// snapshot -- misses and falls back to the live environment instead of
// inheriting a destroyed group's value.
void note_rccl_symm_precondition(void* comm, bool ok);

// Drop the snapshot recorded for `comm`. The owning backend calls this when the
// communicator tears down so a reused ncclComm_t pointer cannot inherit a dead
// comm's value. No-op off ROCm.
void forget_rccl_symm_precondition(void* comm);

// Invalidate and remove only PAIs whose peer mappings were created by `comm`.
// The communicator-independent ncclMemAlloc blocks and PAIs for other groups
// remain reusable. The owning backend calls this before ncclCommDestroy/Abort.
// No-op off ROCm. Implemented in NCCLSymmetricMemory.cu.
void invalidate_symm_mem_for_comm(
    const c10::Device& device,
    const std::string& group_name,
    void* comm);

} // namespace c10d::symmetric_memory
