// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#include "prefetch/expert_residency.h"

#include <algorithm>
#include <set>
#include <stdexcept>

#include <cuda_runtime_api.h>

namespace {

bool EventComplete(cudaEvent_t event) {
  if (event == nullptr) return true;
  return cudaEventQuery(event) == cudaSuccess;
}

ResidencyVariant CanonicalVariant(const NodePtr& node) {
  ResidencyVariant variant;
  if (node) {
    variant.key.logical_expert_key = node->id;
    variant.key.format = ExpertFormat::BF16;
    variant.key.generation = 0;
    variant.node = node;
    variant.payload_bytes = node->byte_size;
    variant.aligned_bytes = node->byte_size;
  }
  return variant;
}

}  // namespace

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

void ResetExpertResidency() {
  kDemandResidencyClient.reset();
  kPrefetchResidencyClient.reset();
  kExpertResidencyManager.reset();
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

ResidencyEntry* ExpertResidencyManager::FindEntry(
    GpuState& state, const ResidencyVariantKey& key) {
  for (auto& entry : state.entries) {
    if (entry.key == key) return &entry;
  }
  return nullptr;
}

const ResidencyEntry* ExpertResidencyManager::FindEntry(
    const GpuState& state, const ResidencyVariantKey& key) const {
  for (const auto& entry : state.entries) {
    if (entry.key == key) return &entry;
  }
  return nullptr;
}

ResidencyEntry* ExpertResidencyManager::FindEntry(
    const ResidencyVariantKey& key, int* gpu_id) {
  for (auto& state_pair : gpu_states_) {
    if (auto* entry = FindEntry(state_pair.second, key)) {
      if (gpu_id != nullptr) *gpu_id = state_pair.first;
      return entry;
    }
  }
  return nullptr;
}

std::uint32_t ExpertResidencyManager::NodeLeaseCount(
    std::size_t node_id) const {
  std::uint32_t count = 0;
  for (const auto& lease : leases_) {
    if (lease.second.node && lease.second.node->id == node_id) ++count;
  }
  return count;
}

bool ExpertResidencyManager::IsEvictable(const ResidencyEntry& entry,
                                         std::int64_t needed_bytes) const {
  if (!entry.node || entry.state != ResidencyState::ACTIVE) return false;
  if (entry.lease_count > 0 || NodeLeaseCount(entry.node->id) > 0) {
    return false;
  }
  if (protected_ids_.count(entry.node->id) > 0) return false;
  if (std::find(protected_variants_.begin(), protected_variants_.end(),
                entry.key) != protected_variants_.end()) {
    return false;
  }
  return entry.bytes >= needed_bytes;
}

NodePtr ExpertResidencyManager::SelectVictim(const GpuState& state,
                                             std::int64_t needed_bytes) const {
  NodePtr victim;
  std::size_t best_last_access = static_cast<std::size_t>(-1);
  for (const auto& entry : state.entries) {
    if (!IsEvictable(entry, needed_bytes)) continue;
    if (!victim || entry.node->last_access_time < best_last_access) {
      victim = entry.node;
      best_last_access = entry.node->last_access_time;
    }
  }
  return victim;
}

std::optional<ExpertCacheKey> ExpertResidencyManager::KeyOf(
    const NodePtr& node) const {
  if (!node) return std::nullopt;
  auto it = node_keys_.find(node->id);
  if (it == node_keys_.end()) return std::nullopt;
  return it->second;
}

int ExpertResidencyManager::RegisteredLayerCount() const {
  std::unordered_set<int> layers;
  for (const auto& node_key : node_keys_) {
    layers.insert(ExpertCacheKeyLayer(node_key.second));
  }
  return static_cast<int>(layers.size());
}

std::optional<std::size_t> ExpertResidencyManager::DenseSlot(
    ExpertCacheKey key) const {
  const auto layer = static_cast<std::int64_t>(key >> 32);
  const auto expert = static_cast<std::int64_t>(key & 0xffffffffu);
  if (layer > max_registered_layer_ || expert > max_registered_expert_) {
    return std::nullopt;
  }
  return static_cast<std::size_t>(layer * (max_registered_expert_ + 1) +
                                  expert);
}

EvictionPolicy* ExpertResidencyManager::PolicyFor(GpuState& state,
                                                  std::int64_t expert_bytes) {
  if (!eviction_policy_active_.load(std::memory_order_relaxed)) return nullptr;
  if (state.policy) return state.policy.get();
  if (expert_bytes <= 0) return nullptr;
  EvictionPolicyOptions options = eviction_options_;
  if (options.num_layers <= 0) options.num_layers = RegisteredLayerCount();
  state.policy = CreateEvictionPolicy(eviction_policy_name_, options);
  if (!state.policy) return nullptr;
  // Experts of one model share a size: capacity in experts = bytes / size.
  state.policy_expert_bytes = expert_bytes;
  state.policy->SetCapacity(static_cast<std::size_t>(
      std::max<std::int64_t>(1, state.capacity_bytes / expert_bytes)));
  // Experts already resident enter as cold admissions.
  state.policy_key_entries.clear();
  state.pending_demand.clear();
  for (auto& entry : state.entries) {
    if (!entry.policy_key) entry.policy_key = KeyOf(entry.node);
    if (entry.policy_key &&
        ++state.policy_key_entries[*entry.policy_key] == 1) {
      state.policy->OnAdmit(*entry.policy_key, true);
    }
  }
  return state.policy.get();
}

void ExpertResidencyManager::TrackEntryAdded(
    GpuState& state, const std::optional<ExpertCacheKey>& key,
    AdmissionSource source) {
  if (!state.policy || !key) return;
  const bool demanded = state.pending_demand.erase(*key) > 0;
  if (++state.policy_key_entries[*key] == 1) {
    state.policy->OnAdmit(*key,
                          source == AdmissionSource::PREFETCH && !demanded);
  }
}

void ExpertResidencyManager::TrackEntryRemoved(
    GpuState& state, const std::optional<ExpertCacheKey>& key) {
  if (!state.policy || !key) return;
  auto it = state.policy_key_entries.find(*key);
  if (it == state.policy_key_entries.end()) return;
  if (--it->second == 0) {
    state.policy_key_entries.erase(it);
    state.policy->OnEvict(*key);
  }
}

NodePtr ExpertResidencyManager::SelectPolicyVictim(GpuState& state,
                                                   ExpertCacheKey incoming,
                                                   bool need_room,
                                                   std::int64_t needed_bytes) {
  // Index the evictable entries by dense (layer, expert) slot so the policy's
  // eligibility checks are O(1) without per-call hashing.
  auto& slots = state.evictable_slots;
  auto& used = state.evictable_used;
  const auto dense_size =
      static_cast<std::size_t>(std::max(max_registered_layer_ + 1, 0)) *
      static_cast<std::size_t>(std::max(max_registered_expert_ + 1, 0));
  if (slots.size() < dense_size) slots.assign(dense_size, nullptr);
  used.clear();
  for (const auto& entry : state.entries) {
    if (!entry.policy_key || !IsEvictable(entry, needed_bytes) ||
        !NodeIdle(entry.node)) {
      continue;
    }
    if (auto slot = DenseSlot(*entry.policy_key)) {
      slots[*slot] = &entry;
      used.push_back(*slot);
    }
  }
  // Nothing evictable (e.g. every resident is in flight): skip the policy so
  // a retried admission does not advance its state (clock hands, promotions).
  if (used.empty()) {
    return need_room ? SelectIdleVictim(state, needed_bytes) : nullptr;
  }
  const auto chosen = state.policy->SelectVictim(
      incoming, need_room, [this, &slots](ExpertCacheKey key) {
        const auto slot = DenseSlot(key);
        return slot && slots[*slot] != nullptr;
      });
  NodePtr victim = chosen ? slots[*DenseSlot(*chosen)]->node : nullptr;
  for (std::size_t slot : used) slots[slot] = nullptr;
  if (victim) {
    counters_["policy_victims"] += 1;
    if (!need_room) counters_["policy_requested_evictions"] += 1;
    return victim;
  }
  if (!need_room) return nullptr;
  // Residents the policy does not know: fall back to the built-in choice
  // rather than stall the admission.
  NodePtr fallback = SelectIdleVictim(state, needed_bytes);
  if (fallback) counters_["policy_fallback_victims"] += 1;
  return fallback;
}

bool ExpertResidencyManager::NodeIdle(const NodePtr& node) {
  return node->exec_state.load(std::memory_order_acquire) ==
             NodeExecState::IDLE &&
         node->pending_dispatches.load(std::memory_order_acquire) == 0;
}

NodePtr ExpertResidencyManager::SelectIdleVictim(
    const GpuState& state, std::int64_t needed_bytes) const {
  NodePtr victim;
  std::size_t best_last_access = static_cast<std::size_t>(-1);
  for (const auto& entry : state.entries) {
    if (!IsEvictable(entry, needed_bytes) || !NodeIdle(entry.node)) continue;
    if (!victim || entry.node->last_access_time < best_last_access) {
      victim = entry.node;
      best_last_access = entry.node->last_access_time;
    }
  }
  return victim;
}

bool ExpertResidencyManager::ConfigureCapacity(int gpu_id,
                                               std::int64_t capacity_bytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (capacity_bytes < 0) return false;
  for (const auto& ticket : pending_tickets_) {
    if (ticket.second.gpu_id == gpu_id) return false;
  }
  GpuState& state = StateFor(gpu_id);
  const auto charged = state.resident_bytes + state.transition_reserved_bytes +
                       state.workspace_bytes;
  if (capacity_bytes < charged) return false;
  state.configured = true;
  state.capacity_bytes = capacity_bytes;
  if (state.policy && state.policy_expert_bytes > 0) {
    state.policy->SetCapacity(static_cast<std::size_t>(
        std::max<std::int64_t>(1, capacity_bytes / state.policy_expert_bytes)));
  }
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

void ExpertResidencyManager::RegisterVariant(const ResidencyVariant& variant) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!variant.node || variant.payload_bytes < 0 ||
      variant.aligned_bytes <= 0 || variant.workspace_bytes < 0 ||
      variant.payload_bytes > variant.aligned_bytes) {
    return;
  }
  auto duplicate =
      std::find_if(registered_variants_.begin(), registered_variants_.end(),
                   [&variant](const ResidencyVariant& existing) {
                     return existing.key == variant.key;
                   });
  if (duplicate == registered_variants_.end()) {
    registered_variants_.push_back(variant);
  }
}

std::optional<ResidencyVariant> ExpertResidencyManager::RegisteredVariant(
    const ResidencyVariantKey& key) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto found = std::find_if(
      registered_variants_.begin(), registered_variants_.end(),
      [&key](const ResidencyVariant& variant) { return variant.key == key; });
  return found == registered_variants_.end()
             ? std::nullopt
             : std::optional<ResidencyVariant>(*found);
}

ResidencyTicket ExpertResidencyManager::BeginAdmission(const NodePtr& incoming,
                                                       int gpu_id,
                                                       ExpertPhase phase,
                                                       AdmissionMode mode,
                                                       AdmissionSource source) {
  return BeginAdmission(CanonicalVariant(incoming), gpu_id, phase, mode,
                        source);
}

ResidencyTicket ExpertResidencyManager::BeginAdmission(
    const ResidencyVariant& incoming, int gpu_id, ExpertPhase phase,
    AdmissionMode mode, AdmissionSource source) {
  std::lock_guard<std::mutex> lock(mutex_);
  ResidencyTicket ticket;
  ticket.kind = ResidencyTransactionKind::ADMISSION;
  ticket.incoming = incoming.node;
  ticket.incoming_variant = incoming;
  ticket.gpu_id = gpu_id;
  ticket.phase = phase;
  ticket.source = source;

  const GpuState* found = FindState(gpu_id);
  if (!incoming.node || incoming.aligned_bytes <= 0 || found == nullptr ||
      !found->configured) {
    return ticket;
  }
  GpuState& state = StateFor(gpu_id);
  if (FindEntry(state, incoming.key) != nullptr) {
    ticket.outcome = AdmissionOutcome::ALREADY_RESIDENT;
    return ticket;
  }
  for (const auto& entry : state.entries) {
    if (entry.state == ResidencyState::ACTIVE &&
        entry.key.logical_expert_key == incoming.key.logical_expert_key) {
      return ticket;
    }
  }

  const auto requested = incoming.aligned_bytes + incoming.workspace_bytes;
  const auto charged = state.resident_bytes + state.transition_reserved_bytes +
                       state.workspace_bytes;
  const auto free_bytes = state.capacity_bytes - charged;
  ticket.id = next_ticket_id_++;
  // Only registered experts are policy keys; other nodes (dense weights
  // prefetched through the same manager) keep the built-in victim choice and
  // must not size the policy.
  std::optional<ExpertCacheKey> incoming_key;
  EvictionPolicy* policy = nullptr;
  if (eviction_policy_active_.load(std::memory_order_relaxed)) {
    incoming_key = KeyOf(incoming.node);
    if (incoming_key) policy = PolicyFor(state, incoming.aligned_bytes);
  }
  const bool need_room = requested > free_bytes;
  if (need_room || (policy && policy->WantsVictim(*incoming_key))) {
    const std::int64_t needed = need_room ? requested - free_bytes : 0;
    NodePtr victim =
        policy ? SelectPolicyVictim(state, *incoming_key, need_room, needed)
               : SelectVictim(state, needed);
    if (!victim && need_room) {
      if (mode == AdmissionMode::TRANSIENT_ON_PRESSURE) {
        ticket.outcome = AdmissionOutcome::TRANSIENT;
        ticket.transient = true;
        ticket.valid = true;
        pending_tickets_[ticket.id] = ticket;
      }
      return ticket;
    }
    if (victim) {
      ticket.reserved_victim = victim;
      if (auto* victim_entry = FindEntry(state, victim->id)) {
        ticket.reserved_victim_key = victim_entry->key;
        ++victim_entry->lease_count;
      }
    }
  }

  ticket.outcome = AdmissionOutcome::ADMIT;
  ticket.reserved_aligned_bytes = incoming.aligned_bytes;
  ticket.reserved_workspace_bytes = incoming.workspace_bytes;
  ticket.valid = true;
  if (!ticket.reserved_victim_key.has_value()) {
    state.transition_reserved_bytes += ticket.reserved_aligned_bytes;
    state.workspace_bytes += ticket.reserved_workspace_bytes;
  }
  pending_tickets_[ticket.id] = ticket;
  return ticket;
}

ResidencyTicket ExpertResidencyManager::BeginTransition(
    const ResidencyVariantKey& current, const ResidencyVariant& incoming,
    int gpu_id, ExpertPhase phase, AdmissionSource source) {
  std::lock_guard<std::mutex> lock(mutex_);
  ResidencyTicket ticket;
  ticket.kind = ResidencyTransactionKind::TRANSITION;
  ticket.incoming = incoming.node;
  ticket.incoming_variant = incoming;
  ticket.replaced_key = current;
  ticket.gpu_id = gpu_id;
  ticket.phase = phase;
  ticket.source = source;

  const GpuState* found = FindState(gpu_id);
  if (!incoming.node || incoming.aligned_bytes <= 0 || found == nullptr ||
      !found->configured ||
      current.logical_expert_key != incoming.key.logical_expert_key) {
    return ticket;
  }
  GpuState& state = StateFor(gpu_id);
  ResidencyEntry* current_entry = FindEntry(state, current);
  if (current_entry == nullptr ||
      current_entry->state != ResidencyState::ACTIVE)
    return ticket;
  for (const auto& pending : pending_tickets_) {
    if (pending.second.kind == ResidencyTransactionKind::TRANSITION &&
        pending.second.replaced_key.has_value() &&
        pending.second.replaced_key->logical_expert_key ==
            current.logical_expert_key) {
      return ticket;
    }
  }
  const auto requested = incoming.aligned_bytes + incoming.workspace_bytes;
  const auto charged = state.resident_bytes + state.transition_reserved_bytes +
                       state.workspace_bytes;
  if (charged + requested > state.capacity_bytes) return ticket;

  ticket.id = next_ticket_id_++;
  ticket.outcome = AdmissionOutcome::ADMIT;
  ticket.reserved_aligned_bytes = incoming.aligned_bytes;
  ticket.reserved_workspace_bytes = incoming.workspace_bytes;
  ticket.valid = true;
  ++current_entry->lease_count;
  state.transition_reserved_bytes += ticket.reserved_aligned_bytes;
  state.workspace_bytes += ticket.reserved_workspace_bytes;
  pending_tickets_[ticket.id] = ticket;
  return ticket;
}

bool ExpertResidencyManager::EvictReserved(const ResidencyTicket& ticket) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto pending_it = pending_tickets_.find(ticket.id);
  if (pending_it == pending_tickets_.end() || !pending_it->second.valid ||
      !pending_it->second.reserved_victim_key.has_value()) {
    return false;
  }
  ResidencyTicket& pending = pending_it->second;
  GpuState& state = StateFor(pending.gpu_id);
  ResidencyEntry* victim = FindEntry(state, *pending.reserved_victim_key);
  if (victim == nullptr || victim->lease_count != 1) return false;
  // Under an eviction policy the victim must still be idle: it may have been
  // dispatched since BeginAdmission picked it.
  if (state.policy && victim->policy_key && !NodeIdle(victim->node)) {
    counters_["policy_victims_dispatched"] += 1;
    return false;
  }
  if (transfer_ops_ && !transfer_ops_->MoveToHost(victim->node)) return false;
  state.resident_bytes -= victim->bytes;
  const auto key = victim->key;
  const auto victim_policy_key = victim->policy_key;
  state.entries.erase(std::remove_if(state.entries.begin(), state.entries.end(),
                                     [&key](const ResidencyEntry& entry) {
                                       return entry.key == key;
                                     }),
                      state.entries.end());
  TrackEntryRemoved(state, victim_policy_key);
  pending.reserved_victim = nullptr;
  pending.reserved_victim_key.reset();
  state.transition_reserved_bytes += pending.reserved_aligned_bytes;
  state.workspace_bytes += pending.reserved_workspace_bytes;
  return true;
}

bool ExpertResidencyManager::CommitTransaction(
    const ResidencyTicket& transaction) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto pending_it = pending_tickets_.find(transaction.id);
  if (pending_it == pending_tickets_.end() || !pending_it->second.valid)
    return false;
  ResidencyTicket pending = pending_it->second;
  GpuState& state = StateFor(pending.gpu_id);
  if (pending.reserved_victim_key.has_value()) {
    if (auto* victim = FindEntry(state, *pending.reserved_victim_key)) {
      if (victim->lease_count > 0) --victim->lease_count;
    }
    pending_tickets_.erase(pending_it);
    return false;
  }
  if (pending.transient) {
    if (auto key = KeyOf(pending.incoming_variant.node)) {
      state.pending_demand.erase(*key);
    }
    pending_tickets_.erase(pending_it);
    return true;
  }

  if (pending.kind == ResidencyTransactionKind::TRANSITION &&
      pending.replaced_key.has_value()) {
    ResidencyEntry* replaced = FindEntry(state, *pending.replaced_key);
    if (replaced == nullptr || replaced->state != ResidencyState::ACTIVE) {
      pending_tickets_.erase(pending_it);
      state.transition_reserved_bytes -= pending.reserved_aligned_bytes;
      state.workspace_bytes -= pending.reserved_workspace_bytes;
      return false;
    }
    if (replaced->lease_count > 0) --replaced->lease_count;
    replaced->state = ResidencyState::RETIRING;
  }

  ResidencyEntry entry;
  entry.node = pending.incoming_variant.node;
  entry.bytes = pending.incoming_variant.aligned_bytes;
  entry.key = pending.incoming_variant.key;
  entry.payload_bytes = pending.incoming_variant.payload_bytes;
  entry.state = ResidencyState::ACTIVE;
  entry.policy_key = KeyOf(entry.node);
  state.entries.push_back(entry);
  state.resident_bytes += entry.bytes;
  state.transition_reserved_bytes -= pending.reserved_aligned_bytes;
  TrackEntryAdded(state, entry.policy_key, pending.source);
  if (pending.reserved_workspace_bytes > 0) {
    workspace_records_[pending.id] = WorkspaceRecord{
        pending.id, pending.incoming_variant.key, pending.gpu_id,
        pending.reserved_workspace_bytes, nullptr};
  }
  pending_tickets_.erase(pending_it);
  return true;
}

bool ExpertResidencyManager::CommitAdmission(const ResidencyTicket& ticket) {
  return CommitTransaction(ticket);
}

bool ExpertResidencyManager::AbortTransaction(
    const ResidencyTicket& transaction) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto pending_it = pending_tickets_.find(transaction.id);
  if (pending_it == pending_tickets_.end()) return false;
  ResidencyTicket pending = pending_it->second;
  GpuState& state = StateFor(pending.gpu_id);
  const bool reservation_charged =
      pending.kind == ResidencyTransactionKind::TRANSITION ||
      !pending.reserved_victim_key.has_value();
  if (pending.reserved_victim_key.has_value()) {
    if (auto* victim = FindEntry(state, *pending.reserved_victim_key)) {
      if (victim->lease_count > 0) --victim->lease_count;
    }
  }
  if (pending.kind == ResidencyTransactionKind::TRANSITION &&
      pending.replaced_key.has_value()) {
    if (auto* replaced = FindEntry(state, *pending.replaced_key)) {
      if (replaced->lease_count > 0) --replaced->lease_count;
    }
  }
  if (reservation_charged) {
    state.transition_reserved_bytes -= pending.reserved_aligned_bytes;
    state.workspace_bytes -= pending.reserved_workspace_bytes;
  }
  pending_tickets_.erase(pending_it);
  return true;
}

bool ExpertResidencyManager::AbortAdmission(const ResidencyTicket& ticket) {
  return AbortTransaction(ticket);
}

std::uint64_t ExpertResidencyManager::AcquireLease(const NodePtr& node,
                                                   LeaseKind kind) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!node) return 0;
  for (auto& state : gpu_states_) {
    if (auto* entry = FindEntry(state.second, node->id)) {
      LeaseRecord record{next_lease_id_++, node, entry->key, kind};
      ++entry->lease_count;
      leases_[record.id] = record;
      return record.id;
    }
  }
  return 0;
}

std::uint64_t ExpertResidencyManager::AcquireLease(
    const ResidencyVariantKey& key, LeaseKind kind) {
  std::lock_guard<std::mutex> lock(mutex_);
  int gpu_id = -1;
  ResidencyEntry* entry = FindEntry(key, &gpu_id);
  if (entry == nullptr) return 0;
  LeaseRecord record{next_lease_id_++, entry->node, key, kind};
  ++entry->lease_count;
  leases_[record.id] = record;
  return record.id;
}

bool ExpertResidencyManager::ReleaseLease(std::uint64_t lease_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto lease_it = leases_.find(lease_id);
  if (lease_it == leases_.end()) return false;
  int gpu_id = -1;
  if (auto* entry = FindEntry(lease_it->second.key, &gpu_id)) {
    if (entry->lease_count > 0) --entry->lease_count;
  }
  leases_.erase(lease_it);
  return true;
}

bool ExpertResidencyManager::RecordLastUse(const ResidencyVariantKey& key,
                                           cudaEvent_t event) {
  std::lock_guard<std::mutex> lock(mutex_);
  int gpu_id = -1;
  auto* entry = FindEntry(key, &gpu_id);
  if (entry == nullptr) return false;
  entry->last_use_event = event;
  return true;
}

bool ExpertResidencyManager::RecordWorkspaceUse(std::uint64_t transaction_id,
                                                cudaEvent_t event) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = workspace_records_.find(transaction_id);
  if (it == workspace_records_.end()) return false;
  it->second.completion_event = event;
  return true;
}

std::size_t ExpertResidencyManager::ReapWorkspace(int gpu_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::size_t reaped = 0;
  for (auto it = workspace_records_.begin(); it != workspace_records_.end();) {
    if (it->second.gpu_id == gpu_id &&
        EventComplete(it->second.completion_event)) {
      StateFor(gpu_id).workspace_bytes -= it->second.bytes;
      it = workspace_records_.erase(it);
      ++reaped;
    } else {
      ++it;
    }
  }
  return reaped;
}

bool ExpertResidencyManager::RequestRetirement(const ResidencyVariantKey& key) {
  std::lock_guard<std::mutex> lock(mutex_);
  int gpu_id = -1;
  auto* entry = FindEntry(key, &gpu_id);
  if (entry == nullptr || entry->state != ResidencyState::ACTIVE) return false;
  entry->state = ResidencyState::RETIRING;
  return true;
}

std::size_t ExpertResidencyManager::ReapRetired(int gpu_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  GpuState& state = StateFor(gpu_id);
  std::size_t reaped = 0;
  for (auto it = state.entries.begin(); it != state.entries.end();) {
    if (it->state == ResidencyState::RETIRING && it->lease_count == 0 &&
        EventComplete(it->last_use_event) &&
        (!transfer_ops_ || transfer_ops_->MoveToHost(it->node))) {
      state.resident_bytes -= it->bytes;
      const auto policy_key = it->policy_key;
      it = state.entries.erase(it);
      TrackEntryRemoved(state, policy_key);
      ++reaped;
    } else {
      ++it;
    }
  }
  return reaped;
}

std::vector<ResidencyEntry> ExpertResidencyManager::ResidentGenerations(
    int gpu_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const GpuState* state = FindState(gpu_id);
  return state == nullptr ? std::vector<ResidencyEntry>{} : state->entries;
}

std::size_t ExpertResidencyManager::ResidentGenerationCount(int gpu_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const GpuState* state = FindState(gpu_id);
  return state == nullptr ? 0 : state->entries.size();
}

std::optional<ResidencyVariantKey> ExpertResidencyManager::ActiveGeneration(
    int gpu_id, std::uint64_t logical_expert_key) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const GpuState* state = FindState(gpu_id);
  if (state == nullptr) return std::nullopt;
  for (const auto& entry : state->entries) {
    if (entry.state == ResidencyState::ACTIVE &&
        entry.key.logical_expert_key == logical_expert_key) {
      return entry.key;
    }
  }
  return std::nullopt;
}

void ExpertResidencyManager::ReplaceProtectedVariants(
    const std::vector<ResidencyVariantKey>& candidates) {
  std::lock_guard<std::mutex> lock(mutex_);
  protected_variants_ = candidates;
}

void ExpertResidencyManager::ReplaceProtectedCandidates(
    const NodePtrList& candidates) {
  std::lock_guard<std::mutex> lock(mutex_);
  protected_ids_.clear();
  for (const auto& node : candidates) {
    if (node) protected_ids_.insert(node->id);
  }
}

void ExpertResidencyManager::ConfigurePolicy(const PhasePolicyConfig& config) {
  std::lock_guard<std::mutex> lock(mutex_);
  config_ = config;
  counters_["enabled"] = config.enabled ? 1 : 0;
  managed_residency_.store(config_.enabled || eviction_policy_name_ != "legacy",
                           std::memory_order_release);
}

bool ExpertResidencyManager::PolicyEnabled() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return config_.enabled;
}

bool ExpertResidencyManager::ConfigureEvictionPolicy(
    const std::string& name, const EvictionPolicyOptions& options) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!pending_tickets_.empty()) return false;
  const bool active = name != "legacy";
  if (active && !CreateEvictionPolicy(name, options)) return false;
  eviction_policy_name_ = name;
  eviction_options_ = options;
  for (auto& gpu : gpu_states_) {
    gpu.second.policy.reset();
    gpu.second.policy_expert_bytes = 0;
    gpu.second.policy_key_entries.clear();
    gpu.second.pending_demand.clear();
  }
  counters_["eviction_policy_active"] = active ? 1 : 0;
  eviction_policy_active_.store(active, std::memory_order_release);
  managed_residency_.store(config_.enabled || active,
                           std::memory_order_release);
  return true;
}

std::string ExpertResidencyManager::EvictionPolicyName() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return eviction_policy_name_;
}

bool ExpertResidencyManager::ManagedResidency() const {
  return managed_residency_.load(std::memory_order_acquire);
}

void ExpertResidencyManager::RegisterExpertKey(std::size_t node_id, int layer,
                                               int expert) {
  if (layer < 0 || expert < 0) return;
  std::lock_guard<std::mutex> lock(mutex_);
  node_keys_[node_id] = MakeExpertCacheKey(layer, expert);
  max_registered_layer_ = std::max(max_registered_layer_, layer);
  max_registered_expert_ = std::max(max_registered_expert_, expert);
}

void ExpertResidencyManager::RecordRouted(const NodePtr& node, int gpu_id) {
  if (!eviction_policy_active_.load(std::memory_order_acquire)) return;
  std::lock_guard<std::mutex> lock(mutex_);
  if (!node || gpu_id < 0) return;
  const auto key = KeyOf(node);
  if (!key) return;
  GpuState& state = StateFor(gpu_id);
  EvictionPolicy* policy = PolicyFor(state, node->byte_size);
  if (policy == nullptr) return;
  const bool resident = state.policy_key_entries.count(*key) > 0;
  if (!resident) state.pending_demand.insert(*key);
  policy->OnAccess(*key, resident);
  counters_[resident ? "policy_routed_hits" : "policy_routed_misses"] += 1;
}

AdmissionMode ExpertResidencyManager::AdmissionFor(ExpertPhase phase) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return EffectivePhase(phase) == ExpertPhase::PREFILL
             ? config_.prefill_admission
             : config_.decode_admission;
}

std::uint32_t ExpertResidencyManager::StarvationLimit() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return config_.starvation_limit;
}

void ExpertResidencyManager::RecordStarvationPromotion() {
  std::lock_guard<std::mutex> lock(mutex_);
  counters_["starvation_promotions"] += 1;
}

void ExpertResidencyManager::RecordAccess(const NodePtr& node,
                                          ExpertPhase phase, bool hit) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!node) return;
  ExpertPolicyMetadata& meta = access_metadata_[node->id];
  const std::uint64_t sequence = ++access_sequence_;
  const ExpertPhase effective = EffectivePhase(phase);
  if (effective == ExpertPhase::PREFILL) {
    ++meta.prefill_accesses;
    meta.last_prefill_sequence = sequence;
  } else {
    ++meta.decode_accesses;
    meta.last_decode_sequence = sequence;
  }
  (void)hit;
}

ExpertPolicyStats ExpertResidencyManager::Snapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);
  ExpertPolicyStats stats = counters_;
  std::int64_t policy_slots = 0;
  for (const auto& gpu : gpu_states_) {
    if (gpu.second.policy) {
      policy_slots += static_cast<std::int64_t>(gpu.second.policy->Capacity());
    }
  }
  stats["eviction_policy_slots"] = policy_slots;
  std::int64_t resident_bytes = 0;
  std::int64_t resident_payload_bytes = 0;
  std::int64_t capacity_bytes = 0;
  std::int64_t transition_reserved_bytes = 0;
  std::int64_t workspace_bytes = 0;
  std::int64_t resident_generations = 0;
  std::int64_t retiring_generations = 0;
  std::set<std::pair<int, std::uint64_t>> resident_experts;
  for (const auto& gpu : gpu_states_) {
    resident_bytes += gpu.second.resident_bytes;
    capacity_bytes += gpu.second.capacity_bytes;
    transition_reserved_bytes += gpu.second.transition_reserved_bytes;
    workspace_bytes += gpu.second.workspace_bytes;
    resident_generations +=
        static_cast<std::int64_t>(gpu.second.entries.size());
    for (const auto& entry : gpu.second.entries) {
      resident_payload_bytes += entry.payload_bytes;
      if (entry.state == ResidencyState::ACTIVE) {
        resident_experts.emplace(gpu.first, entry.key.logical_expert_key);
      } else if (entry.state == ResidencyState::RETIRING) {
        ++retiring_generations;
      }
    }
  }
  stats["resident_bytes"] = resident_bytes;
  stats["resident_payload_bytes"] = resident_payload_bytes;
  stats["alignment_padding_bytes"] = resident_bytes - resident_payload_bytes;
  stats["resident_count"] = static_cast<std::int64_t>(resident_experts.size());
  stats["resident_experts"] =
      static_cast<std::int64_t>(resident_experts.size());
  stats["resident_generations"] = resident_generations;
  stats["retiring_generations"] = retiring_generations;
  stats["capacity_bytes"] = capacity_bytes;
  stats["transition_reserved_bytes"] = transition_reserved_bytes;
  stats["workspace_bytes"] = workspace_bytes;
  stats["registered_variants"] =
      static_cast<std::int64_t>(registered_variants_.size());
  stats["active_leases"] = static_cast<std::int64_t>(leases_.size());
  for (const auto& name :
       {"demand", "prefetch", "transfer", "execution", "transition"}) {
    stats[std::string("leases_") + name] = 0;
  }
  for (const auto& lease : leases_) {
    const char* name = "demand";
    switch (lease.second.kind) {
      case LeaseKind::PREFETCH:
        name = "prefetch";
        break;
      case LeaseKind::TRANSFER:
        name = "transfer";
        break;
      case LeaseKind::EXECUTION:
        name = "execution";
        break;
      case LeaseKind::TRANSITION:
        name = "transition";
        break;
      case LeaseKind::DEMAND:
        break;
    }
    ++stats[std::string("leases_") + name];
  }
  stats["pending_tickets"] = static_cast<std::int64_t>(pending_tickets_.size());
  stats["pending_transactions"] =
      static_cast<std::int64_t>(pending_tickets_.size());
  stats["peak_accounted_bytes"] =
      resident_bytes + transition_reserved_bytes + workspace_bytes;
  return stats;
}

std::int64_t ExpertResidencyManager::ResidentBytes(int gpu_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const GpuState* state = FindState(gpu_id);
  return state == nullptr ? 0 : state->resident_bytes;
}

std::size_t ExpertResidencyManager::ResidentCount(int gpu_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const GpuState* state = FindState(gpu_id);
  if (state == nullptr) return 0;
  std::set<std::uint64_t> logical_experts;
  for (const auto& entry : state->entries) {
    if (entry.state == ResidencyState::ACTIVE) {
      logical_experts.insert(entry.key.logical_expert_key);
    }
  }
  return logical_experts.size();
}
