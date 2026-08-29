// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#include "prefetch/expert_residency.h"

#include <cuda_runtime_api.h>
#include <algorithm>

std::shared_ptr<ExpertResidencyManager> kExpertResidencyManager = nullptr;
std::unique_ptr<ExpertResidencyClient> kDemandResidencyClient = nullptr;
std::unique_ptr<ExpertResidencyClient> kPrefetchResidencyClient = nullptr;

bool TopologyMoveToHostOps::MoveToHost(const NodePtr& node) {
  if (!node) return false;
  node->SetDevice(node->default_host);
  return true;
}

void InitExpertResidency() {
  if (kExpertResidencyManager != nullptr) return;
  kExpertResidencyManager = std::make_shared<ExpertResidencyManager>(
      std::make_shared<TopologyMoveToHostOps>());
  kDemandResidencyClient = std::make_unique<ExpertResidencyClient>(
      kExpertResidencyManager, AdmissionSource::DEMAND);
  kPrefetchResidencyClient = std::make_unique<ExpertResidencyClient>(
      kExpertResidencyManager, AdmissionSource::PREFETCH);
}

void ConfigureExpertResidencyCapacityFromTopology() {
  if (kExpertResidencyManager == nullptr || kTopologyHandle == nullptr) return;
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  for (int device = 0; device < device_count; ++device) {
    const std::int64_t limit =
        kTopologyHandle->GetSparseCacheLimit(CUDA_DEVICE(device));
    if (limit > 0) {
      kExpertResidencyManager->ConfigureCapacity(device, limit);
    }
  }
}

ExpertResidencyManager::ExpertResidencyManager(
    std::shared_ptr<ResidencyTransferOps> transfer_ops)
    : transfer_ops_(std::move(transfer_ops)) {}

ExpertResidencyManager::GpuState& ExpertResidencyManager::StateFor(int gpu_id) {
  return gpu_states_[gpu_id];
}

const ExpertResidencyManager::GpuState* ExpertResidencyManager::FindState(
    int gpu_id) const {
  auto it = gpu_states_.find(gpu_id);
  return it == gpu_states_.end() ? nullptr : &it->second;
}

ResidencyEntry* ExpertResidencyManager::FindEntry(GpuState& state,
                                                  std::size_t node_id) {
  for (auto& entry : state.entries) {
    if (entry.node && entry.node->id == node_id) return &entry;
  }
  return nullptr;
}

const ResidencyEntry* ExpertResidencyManager::FindEntry(
    const GpuState& state, std::size_t node_id) const {
  for (const auto& entry : state.entries) {
    if (entry.node && entry.node->id == node_id) return &entry;
  }
  return nullptr;
}

std::uint32_t ExpertResidencyManager::NodeLeaseCount(
    std::size_t node_id) const {
  std::uint32_t count = 0;
  for (const auto& kv : leases_) {
    if (kv.second.node && kv.second.node->id == node_id) ++count;
  }
  return count;
}

NodePtr ExpertResidencyManager::SelectVictim(const GpuState& state,
                                             std::int64_t needed_bytes) const {
  NodePtr victim;
  std::size_t best_last_access = static_cast<std::size_t>(-1);
  for (const auto& entry : state.entries) {
    if (!entry.node) continue;
    if (entry.lease_count > 0 || NodeLeaseCount(entry.node->id) > 0) continue;
    if (protected_ids_.count(entry.node->id) > 0) continue;
    if (entry.bytes < needed_bytes) continue;
    std::size_t last_access = entry.node->last_access_time;
    if (!victim || last_access < best_last_access) {
      victim = entry.node;
      best_last_access = last_access;
    }
  }
  return victim;
}

bool ExpertResidencyManager::ConfigureCapacity(int gpu_id,
                                               std::int64_t capacity_bytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (capacity_bytes < 0) return false;
  for (const auto& kv : pending_tickets_) {
    if (kv.second.gpu_id == gpu_id) return false;
  }
  GpuState& state = StateFor(gpu_id);
  if (capacity_bytes < state.resident_bytes) return false;
  state.configured = true;
  state.capacity_bytes = capacity_bytes;
  return true;
}

bool ExpertResidencyManager::IsCapacityConfigured(int gpu_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const GpuState* state = FindState(gpu_id);
  return state != nullptr && state->configured;
}

std::int64_t ExpertResidencyManager::CapacityBytes(int gpu_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const GpuState* state = FindState(gpu_id);
  return state != nullptr ? state->capacity_bytes : 0;
}

ResidencyTicket ExpertResidencyManager::BeginAdmission(const NodePtr& incoming,
                                                       int gpu_id,
                                                       ExpertPhase phase,
                                                       AdmissionMode mode,
                                                       AdmissionSource source) {
  std::lock_guard<std::mutex> lock(mutex_);
  ResidencyTicket ticket;
  ticket.incoming = incoming;
  ticket.gpu_id = gpu_id;
  ticket.phase = phase;
  ticket.source = source;
  ticket.outcome = AdmissionOutcome::REJECTED;
  ticket.valid = false;

  if (!incoming) return ticket;
  const GpuState* const_state = FindState(gpu_id);
  if (const_state == nullptr || !const_state->configured) return ticket;

  GpuState& state = StateFor(gpu_id);
  if (FindEntry(state, incoming->id) != nullptr) {
    ticket.outcome = AdmissionOutcome::ALREADY_RESIDENT;
    ticket.valid = false;
    return ticket;
  }

  const std::int64_t bytes = incoming->byte_size;
  const std::int64_t free_bytes = state.capacity_bytes - state.resident_bytes;

  ticket.id = next_ticket_id_++;
  if (bytes <= free_bytes) {
    ticket.outcome = AdmissionOutcome::ADMIT;
    ticket.valid = true;
    pending_tickets_[ticket.id] = ticket;
    return ticket;
  }

  const std::int64_t needed = bytes - free_bytes;
  NodePtr victim = SelectVictim(state, needed);
  if (!victim) {
    if (mode == AdmissionMode::TRANSIENT_ON_PRESSURE) {
      ticket.outcome = AdmissionOutcome::TRANSIENT;
      ticket.transient = true;
      ticket.valid = true;
      pending_tickets_[ticket.id] = ticket;
      return ticket;
    }
    ticket.outcome = AdmissionOutcome::REJECTED;
    ticket.valid = false;
    return ticket;
  }

  ticket.reserved_victim = victim;
  ticket.outcome = AdmissionOutcome::ADMIT;
  ticket.valid = true;
  ResidencyEntry* victim_entry = FindEntry(state, victim->id);
  if (victim_entry) victim_entry->lease_count += 1;
  pending_tickets_[ticket.id] = ticket;
  return ticket;
}

bool ExpertResidencyManager::EvictReserved(const ResidencyTicket& ticket) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = pending_tickets_.find(ticket.id);
  if (it == pending_tickets_.end() || !it->second.valid) return false;
  ResidencyTicket& pending = it->second;
  if (!pending.reserved_victim) return false;

  GpuState& state = StateFor(pending.gpu_id);
  ResidencyEntry* victim_entry = FindEntry(state, pending.reserved_victim->id);
  if (victim_entry == nullptr) return false;

  if (transfer_ops_) {
    if (!transfer_ops_->MoveToHost(pending.reserved_victim)) return false;
  }
  state.resident_bytes -= victim_entry->bytes;
  const std::size_t victim_id = pending.reserved_victim->id;
  state.entries.erase(std::remove_if(state.entries.begin(), state.entries.end(),
                                     [victim_id](const ResidencyEntry& entry) {
                                       return entry.node &&
                                              entry.node->id == victim_id;
                                     }),
                      state.entries.end());
  pending.reserved_victim = nullptr;
  return true;
}

bool ExpertResidencyManager::CommitAdmission(const ResidencyTicket& ticket) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = pending_tickets_.find(ticket.id);
  if (it == pending_tickets_.end() || !it->second.valid) return false;
  ResidencyTicket pending = it->second;
  pending_tickets_.erase(it);

  if (pending.reserved_victim) {
    GpuState& state = StateFor(pending.gpu_id);
    ResidencyEntry* victim_entry =
        FindEntry(state, pending.reserved_victim->id);
    if (victim_entry && victim_entry->lease_count > 0) {
      victim_entry->lease_count -= 1;
    }
    return false;
  }

  if (pending.transient) return true;

  GpuState& state = StateFor(pending.gpu_id);
  if (FindEntry(state, pending.incoming->id) != nullptr) return true;
  ResidencyEntry entry;
  entry.node = pending.incoming;
  entry.bytes = pending.incoming->byte_size;
  entry.lease_count = 0;
  state.entries.push_back(entry);
  state.resident_bytes += entry.bytes;
  return true;
}

bool ExpertResidencyManager::AbortAdmission(const ResidencyTicket& ticket) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = pending_tickets_.find(ticket.id);
  if (it == pending_tickets_.end()) return false;
  ResidencyTicket pending = it->second;
  pending_tickets_.erase(it);

  if (pending.reserved_victim) {
    GpuState& state = StateFor(pending.gpu_id);
    ResidencyEntry* victim_entry =
        FindEntry(state, pending.reserved_victim->id);
    if (victim_entry && victim_entry->lease_count > 0) {
      victim_entry->lease_count -= 1;
    }
  }
  return true;
}

std::uint64_t ExpertResidencyManager::AcquireLease(const NodePtr& node,
                                                   LeaseKind kind) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!node) return 0;
  LeaseRecord record;
  record.id = next_lease_id_++;
  record.node = node;
  record.kind = kind;
  leases_[record.id] = record;
  return record.id;
}

bool ExpertResidencyManager::ReleaseLease(std::uint64_t lease_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = leases_.find(lease_id);
  if (it == leases_.end()) return false;
  leases_.erase(it);
  return true;
}

void ExpertResidencyManager::ReplaceProtectedCandidates(
    const NodePtrList& candidates) {
  std::lock_guard<std::mutex> lock(mutex_);
  protected_ids_.clear();
  for (const auto& node : candidates) {
    if (node) protected_ids_.insert(node->id);
  }
}

void ExpertResidencyManager::RecordAccess(const NodePtr& node,
                                          ExpertPhase phase, bool hit) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!node) return;
  ExpertPolicyMetadata& meta = access_metadata_[node->id];
  const std::uint64_t sequence = ++access_sequence_;
  const ExpertPhase effective = EffectivePhase(phase);
  if (effective == ExpertPhase::PREFILL) {
    meta.prefill_accesses += 1;
    meta.last_prefill_sequence = sequence;
  } else {
    meta.decode_accesses += 1;
    meta.last_decode_sequence = sequence;
  }
  (void)hit;
}

ExpertPolicyStats ExpertResidencyManager::Snapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);
  ExpertPolicyStats stats;
  std::int64_t total_resident_bytes = 0;
  std::int64_t total_resident_count = 0;
  std::int64_t total_capacity_bytes = 0;
  for (const auto& kv : gpu_states_) {
    total_resident_bytes += kv.second.resident_bytes;
    total_resident_count += static_cast<std::int64_t>(kv.second.entries.size());
    total_capacity_bytes += kv.second.capacity_bytes;
  }
  stats["resident_bytes"] = total_resident_bytes;
  stats["resident_count"] = total_resident_count;
  stats["capacity_bytes"] = total_capacity_bytes;
  stats["active_leases"] = static_cast<std::int64_t>(leases_.size());
  stats["pending_tickets"] = static_cast<std::int64_t>(pending_tickets_.size());
  return stats;
}

std::int64_t ExpertResidencyManager::ResidentBytes(int gpu_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const GpuState* state = FindState(gpu_id);
  return state != nullptr ? state->resident_bytes : 0;
}

std::size_t ExpertResidencyManager::ResidentCount(int gpu_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const GpuState* state = FindState(gpu_id);
  return state != nullptr ? state->entries.size() : 0;
}
