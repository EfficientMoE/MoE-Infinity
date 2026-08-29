// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "model/model_topology.h"
#include "prefetch/expert_policy.h"

enum class AdmissionSource : std::uint8_t { DEMAND = 0, PREFETCH = 1 };

enum class AdmissionOutcome : std::uint8_t {
  ADMIT = 0,
  ALREADY_RESIDENT = 1,
  TRANSIENT = 2,
  REJECTED = 3
};

enum class LeaseKind : std::uint8_t { DEMAND = 0, PREFETCH = 1, TRANSFER = 2 };

class ResidencyTransferOps {
 public:
  virtual ~ResidencyTransferOps() = default;
  virtual bool MoveToHost(const NodePtr& node) = 0;
};

struct ResidencyTicket {
  std::uint64_t id = 0;
  NodePtr incoming;
  NodePtr reserved_victim;
  int gpu_id = -1;
  ExpertPhase phase = ExpertPhase::MIXED;
  AdmissionSource source = AdmissionSource::DEMAND;
  AdmissionOutcome outcome = AdmissionOutcome::REJECTED;
  bool transient = false;
  bool valid = false;
};

struct ResidencyEntry {
  NodePtr node;
  std::int64_t bytes = 0;
  std::uint32_t lease_count = 0;
};

struct LeaseRecord {
  std::uint64_t id = 0;
  NodePtr node;
  LeaseKind kind = LeaseKind::DEMAND;
};

using ExpertPolicyStats = std::unordered_map<std::string, std::int64_t>;

class ExpertResidencyManager {
 public:
  explicit ExpertResidencyManager(
      std::shared_ptr<ResidencyTransferOps> transfer_ops);

  bool ConfigureCapacity(int gpu_id, std::int64_t capacity_bytes);
  bool IsCapacityConfigured(int gpu_id) const;
  std::int64_t CapacityBytes(int gpu_id) const;

  ResidencyTicket BeginAdmission(const NodePtr& incoming, int gpu_id,
                                 ExpertPhase phase, AdmissionMode mode,
                                 AdmissionSource source);
  bool EvictReserved(const ResidencyTicket& ticket);
  bool CommitAdmission(const ResidencyTicket& ticket);
  bool AbortAdmission(const ResidencyTicket& ticket);

  std::uint64_t AcquireLease(const NodePtr& node, LeaseKind kind);
  bool ReleaseLease(std::uint64_t lease_id);

  void ReplaceProtectedCandidates(const NodePtrList& candidates);
  void RecordAccess(const NodePtr& node, ExpertPhase phase, bool hit);

  ExpertPolicyStats Snapshot() const;
  std::int64_t ResidentBytes(int gpu_id) const;
  std::size_t ResidentCount(int gpu_id) const;

 private:
  struct GpuState {
    bool configured = false;
    std::int64_t capacity_bytes = 0;
    std::int64_t resident_bytes = 0;
    std::vector<ResidencyEntry> entries;
  };

  GpuState& StateFor(int gpu_id);
  const GpuState* FindState(int gpu_id) const;
  ResidencyEntry* FindEntry(GpuState& state, std::size_t node_id);
  const ResidencyEntry* FindEntry(const GpuState& state,
                                  std::size_t node_id) const;
  NodePtr SelectVictim(const GpuState& state, std::int64_t needed_bytes) const;
  std::uint32_t NodeLeaseCount(std::size_t node_id) const;

  mutable std::mutex mutex_;
  std::shared_ptr<ResidencyTransferOps> transfer_ops_;
  std::unordered_map<int, GpuState> gpu_states_;
  std::unordered_map<std::uint64_t, ResidencyTicket> pending_tickets_;
  std::unordered_map<std::uint64_t, LeaseRecord> leases_;
  std::unordered_map<std::size_t, ExpertPolicyMetadata> access_metadata_;
  std::unordered_set<std::size_t> protected_ids_;
  std::uint64_t next_ticket_id_ = 1;
  std::uint64_t next_lease_id_ = 1;
  std::uint64_t access_sequence_ = 0;
};

class ExpertResidencyClient {
 public:
  ExpertResidencyClient(std::shared_ptr<ExpertResidencyManager> manager,
                        AdmissionSource source)
      : manager_(std::move(manager)), source_(source) {}

  ResidencyTicket BeginAdmission(const NodePtr& incoming, int gpu_id,
                                 ExpertPhase phase, AdmissionMode mode) {
    return manager_->BeginAdmission(incoming, gpu_id, phase, mode, source_);
  }

  std::shared_ptr<ExpertResidencyManager> manager() const { return manager_; }

 private:
  std::shared_ptr<ExpertResidencyManager> manager_;
  AdmissionSource source_;
};

class TopologyMoveToHostOps final : public ResidencyTransferOps {
 public:
  bool MoveToHost(const NodePtr& node) override;
};

extern std::shared_ptr<ExpertResidencyManager> kExpertResidencyManager;
extern std::unique_ptr<ExpertResidencyClient> kDemandResidencyClient;
extern std::unique_ptr<ExpertResidencyClient> kPrefetchResidencyClient;

void InitExpertResidency();
void ConfigureExpertResidencyCapacityFromTopology();
