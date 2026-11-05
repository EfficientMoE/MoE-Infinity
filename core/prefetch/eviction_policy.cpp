// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#include "prefetch/eviction_policy.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <iterator>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace {

using Key = ExpertCacheKey;
using EligibleFn = EvictionPolicy::EligibleFn;

// Insertion-ordered key set with O(1) append / move-to-back / erase, mirroring
// the OrderedDict operations of the Python reference (front = oldest).
class OrderedKeys {
 public:
  using const_iterator = std::list<Key>::const_iterator;

  bool Contains(Key key) const { return index_.count(key) > 0; }
  bool Empty() const { return keys_.empty(); }
  std::size_t Size() const { return keys_.size(); }
  Key Front() const { return keys_.front(); }
  const_iterator begin() const { return keys_.begin(); }
  const_iterator end() const { return keys_.end(); }

  // Append if absent (an existing key keeps its position, like od[k] = None).
  void Set(Key key) {
    if (!Contains(key)) PushBack(key);
  }
  void PushBack(Key key) { index_[key] = keys_.insert(keys_.end(), key); }
  // Append, or move to the back if present (od[k] = None; move_to_end(k)).
  void SetBack(Key key) {
    if (Contains(key)) {
      MoveToBack(key);
    } else {
      PushBack(key);
    }
  }
  void MoveToBack(Key key) {
    auto it = index_.find(key);
    if (it == index_.end()) return;
    keys_.splice(keys_.end(), keys_, it->second);
  }
  bool Erase(Key key) {
    auto it = index_.find(key);
    if (it == index_.end()) return false;
    keys_.erase(it->second);
    index_.erase(it);
    return true;
  }
  Key PopFront() {
    const Key key = keys_.front();
    Erase(key);
    return key;
  }

 private:
  std::list<Key> keys_;
  std::unordered_map<Key, std::list<Key>::iterator> index_;
};

std::optional<Key> FirstEligible(const OrderedKeys& keys,
                                 const EligibleFn& eligible) {
  for (Key key : keys) {
    if (eligible(key)) return key;
  }
  return std::nullopt;
}

// Python's round() (half to even) under the default rounding mode.
std::size_t RoundHalfEven(double value) {
  return static_cast<std::size_t>(std::nearbyint(value));
}

// ---------------------------------------------------------------------------
// Recency
// ---------------------------------------------------------------------------
class LRUPolicy final : public EvictionPolicy {
 public:
  const char* Name() const override { return "lru"; }
  void OnAccess(Key key, bool resident) override {
    if (resident) order_.MoveToBack(key);
  }
  std::optional<Key> SelectVictim(Key, bool,
                                  const EligibleFn& eligible) override {
    return FirstEligible(order_, eligible);
  }
  void OnEvict(Key key) override { order_.Erase(key); }
  void OnAdmit(Key key, bool) override { order_.SetBack(key); }

 private:
  OrderedKeys order_;  // front = least recently used
};

// Victim = smallest (score, tie) among eligible residents; shared by the
// frequency-style policies whose Python reference scans the resident set.
template <typename RankFn>
std::optional<Key> MinRanked(const std::unordered_set<Key>& resident,
                             const EligibleFn& eligible, RankFn rank) {
  std::optional<Key> best;
  decltype(rank(Key{})) best_rank{};
  for (Key key : resident) {
    if (!eligible(key)) continue;
    const auto r = rank(key);
    if (!best || r < best_rank || (r == best_rank && key < *best)) {
      best = key;
      best_rank = r;
    }
  }
  return best;
}

class LFUPolicy final : public EvictionPolicy {
 public:
  const char* Name() const override { return "lfu"; }
  void OnAccess(Key key, bool) override {
    ++freq_[key];
    last_[key] = static_cast<std::int64_t>(clock_++);
  }
  std::optional<Key> SelectVictim(Key, bool,
                                  const EligibleFn& eligible) override {
    return MinRanked(resident_, eligible, [this](Key key) {
      auto f = freq_.find(key);
      auto l = last_.find(key);
      return std::make_pair(f == freq_.end() ? 0 : f->second,
                            l == last_.end() ? -1 : l->second);
    });
  }
  void OnEvict(Key key) override { resident_.erase(key); }
  void OnAdmit(Key key, bool) override { resident_.insert(key); }

 private:
  std::unordered_map<Key, std::int64_t> freq_;  // lifetime, kept on eviction
  std::unordered_map<Key, std::int64_t> last_;
  std::uint64_t clock_ = 0;
  std::unordered_set<Key> resident_;
};

class LFUKPolicy final : public EvictionPolicy {
 public:
  explicit LFUKPolicy(std::size_t window) : window_size_(window) {}
  const char* Name() const override { return "lfu_k"; }
  void OnAccess(Key key, bool) override {
    ++freq_[key];
    last_[key] = static_cast<std::int64_t>(clock_++);
    window_.push_back(key);
    if (window_.size() > window_size_) {
      const Key expired = window_.front();
      window_.pop_front();
      auto it = freq_.find(expired);
      if (it != freq_.end() && --it->second <= 0) freq_.erase(it);
    }
  }
  std::optional<Key> SelectVictim(Key, bool,
                                  const EligibleFn& eligible) override {
    return MinRanked(resident_, eligible, [this](Key key) {
      auto f = freq_.find(key);
      auto l = last_.find(key);
      return std::make_pair(f == freq_.end() ? 0 : f->second,
                            l == last_.end() ? -1 : l->second);
    });
  }
  void OnEvict(Key key) override { resident_.erase(key); }
  void OnAdmit(Key key, bool) override { resident_.insert(key); }

 private:
  std::size_t window_size_;
  std::deque<Key> window_;
  std::unordered_map<Key, std::int64_t> freq_;
  std::unordered_map<Key, std::int64_t> last_;
  std::uint64_t clock_ = 0;
  std::unordered_set<Key> resident_;
};

class LHUPolicy final : public EvictionPolicy {
 public:
  explicit LHUPolicy(double alpha) : alpha_(alpha) {}
  const char* Name() const override { return "lhu"; }
  void OnAccess(Key key, bool) override {
    double& heat = heat_[key];
    heat = heat * alpha_ + 1.0;
    last_[key] = static_cast<std::int64_t>(clock_++);
  }
  std::optional<Key> SelectVictim(Key, bool,
                                  const EligibleFn& eligible) override {
    return MinRanked(resident_, eligible, [this](Key key) {
      auto h = heat_.find(key);
      auto l = last_.find(key);
      return std::make_pair(h == heat_.end() ? 0.0 : h->second,
                            l == last_.end() ? -1 : l->second);
    });
  }
  void OnEvict(Key key) override { resident_.erase(key); }
  void OnAdmit(Key key, bool) override { resident_.insert(key); }

 private:
  double alpha_;
  std::unordered_map<Key, double> heat_;
  std::unordered_map<Key, std::int64_t> last_;
  std::uint64_t clock_ = 0;
  std::unordered_set<Key> resident_;
};

// LRU-K (O'Neil et al., SIGMOD'93): evict the largest backward K-distance;
// keys with fewer than K references go first, least recent first.
class LRUKPolicy final : public EvictionPolicy {
 public:
  explicit LRUKPolicy(std::size_t k) : k_(k) {}
  const char* Name() const override { return "lru_k"; }
  void OnAccess(Key key, bool) override {
    auto& history = history_[key];
    history.push_back(static_cast<std::int64_t>(clock_++));
    if (history.size() > k_) history.pop_front();
  }
  std::optional<Key> SelectVictim(Key, bool,
                                  const EligibleFn& eligible) override {
    return MinRanked(resident_, eligible, [this](Key key) {
      auto it = history_.find(key);
      if (it == history_.end() || it->second.empty()) {
        return std::make_pair<std::int64_t, std::int64_t>(-1, -1);
      }
      const auto& h = it->second;
      return std::make_pair(h.size() >= k_ ? h.front() : -1, h.back());
    });
  }
  void OnEvict(Key key) override { resident_.erase(key); }
  void OnAdmit(Key key, bool) override { resident_.insert(key); }

 private:
  std::size_t k_;
  std::unordered_map<Key, std::deque<std::int64_t>> history_;
  std::uint64_t clock_ = 0;
  std::unordered_set<Key> resident_;
};

// ---------------------------------------------------------------------------
// ARC (Megiddo & Modha, FAST'03)
// ---------------------------------------------------------------------------
class ARCPolicy final : public EvictionPolicy {
 public:
  const char* Name() const override { return "arc"; }

  void OnAccess(Key key, bool resident) override {
    if (resident) {
      if (t1_.Erase(key)) {
        t2_.PushBack(key);
      } else {
        t2_.MoveToBack(key);
      }
      return;
    }
    const double c = static_cast<double>(capacity_);
    drop_t1_for_.erase(key);
    if (b1_.Contains(key)) {
      p_ = std::min(c, p_ + std::max(static_cast<double>(b2_.Size()) /
                                         static_cast<double>(b1_.Size()),
                                     1.0));
      pending_[key] = Origin::kB1;
    } else if (b2_.Contains(key)) {
      p_ = std::max(0.0, p_ - std::max(static_cast<double>(b1_.Size()) /
                                           static_cast<double>(b2_.Size()),
                                       1.0));
      pending_[key] = Origin::kB2;
    } else {
      pending_[key] = Origin::kNew;
      const std::size_t l1 = t1_.Size() + b1_.Size();
      if (l1 >= capacity_) {
        if (t1_.Size() < capacity_) {
          if (!b1_.Empty()) b1_.PopFront();
        } else {
          drop_t1_for_.insert(key);  // T1 is full: its LRU leaves, no ghost
        }
      } else {
        const std::size_t total = l1 + t2_.Size() + b2_.Size();
        if (total >= capacity_ && total >= 2 * capacity_ && !b2_.Empty()) {
          b2_.PopFront();
        }
      }
    }
  }

  std::optional<Key> SelectVictim(Key incoming, bool,
                                  const EligibleFn& eligible) override {
    auto pending = pending_.find(incoming);
    const Origin origin =
        pending == pending_.end() ? Origin::kNew : pending->second;
    bool from_t1 = true;
    const bool drop = origin == Origin::kNew && drop_t1_for_.count(incoming);
    if (!drop) {
      const bool in_b2 = origin == Origin::kB2;
      const double t1 = static_cast<double>(t1_.Size());
      from_t1 = !t1_.Empty() && ((in_b2 && t1 == p_) || t1 > p_);
      if (t2_.Empty()) from_t1 = true;
    }
    if (auto key = FirstEligible(from_t1 ? t1_ : t2_, eligible)) {
      if (drop) drop_on_evict_.insert(*key);
      return key;
    }
    return FirstEligible(from_t1 ? t2_ : t1_, eligible);
  }

  void OnEvict(Key key) override {
    if (t1_.Erase(key)) {
      if (drop_on_evict_.erase(key) == 0) b1_.PushBack(key);
    } else if (t2_.Erase(key)) {
      drop_on_evict_.erase(key);
      b2_.PushBack(key);
    }
  }

  void OnAdmit(Key key, bool) override {
    Origin origin = Origin::kNew;
    auto pending = pending_.find(key);
    if (pending != pending_.end()) {
      origin = pending->second;
      pending_.erase(pending);
    }
    drop_t1_for_.erase(key);
    drop_on_evict_.erase(key);
    if (origin == Origin::kB1 && b1_.Erase(key)) {
      t2_.PushBack(key);
    } else if (origin == Origin::kB2 && b2_.Erase(key)) {
      t2_.PushBack(key);
    } else {
      b1_.Erase(key);
      b2_.Erase(key);
      t1_.Set(key);
    }
  }

 private:
  enum class Origin { kNew, kB1, kB2 };
  OrderedKeys t1_, t2_, b1_, b2_;  // front = LRU end
  double p_ = 0.0;
  std::unordered_map<Key, Origin> pending_;  // demand misses not yet admitted
  std::unordered_set<Key> drop_t1_for_;
  std::unordered_set<Key> drop_on_evict_;
};

// ---------------------------------------------------------------------------
// 2Q (Johnson & Shasha, VLDB'94)
// ---------------------------------------------------------------------------
class TwoQPolicy final : public EvictionPolicy {
 public:
  const char* Name() const override { return "2q"; }
  void SetCapacity(std::size_t slots) override {
    EvictionPolicy::SetCapacity(slots);
    kin_ = std::max<std::size_t>(
        1, static_cast<std::size_t>(0.25 * static_cast<double>(slots)));
    kout_ = std::max<std::size_t>(
        1, static_cast<std::size_t>(0.5 * static_cast<double>(slots)));
  }
  void OnAccess(Key key, bool resident) override {
    if (resident) am_.MoveToBack(key);  // a hit in A1in is not reordered
  }
  std::optional<Key> SelectVictim(Key, bool,
                                  const EligibleFn& eligible) override {
    const bool from_a1in = a1in_.Size() > kin_ || am_.Empty();
    if (auto key = FirstEligible(from_a1in ? a1in_ : am_, eligible)) {
      return key;
    }
    return FirstEligible(from_a1in ? am_ : a1in_, eligible);
  }
  void OnEvict(Key key) override {
    if (a1in_.Erase(key)) {
      a1out_.Set(key);
      if (a1out_.Size() > kout_) a1out_.PopFront();
    } else {
      am_.Erase(key);
    }
  }
  void OnAdmit(Key key, bool) override {
    if (a1out_.Erase(key)) {
      am_.SetBack(key);
    } else {
      a1in_.Set(key);
    }
  }

 private:
  std::size_t kin_ = 1;
  std::size_t kout_ = 1;
  OrderedKeys a1in_, a1out_, am_;
};

// ---------------------------------------------------------------------------
// LIRS (Jiang & Zhang, SIGMETRICS'02)
// ---------------------------------------------------------------------------
class LIRSPolicy final : public EvictionPolicy {
 public:
  const char* Name() const override { return "lirs"; }
  void SetCapacity(std::size_t slots) override {
    EvictionPolicy::SetCapacity(slots);
    const std::size_t hirs = std::max<std::size_t>(
        1, RoundHalfEven(0.01 * static_cast<double>(slots)));
    l_lirs_ = slots > hirs ? slots - hirs : 0;
  }

  void OnAccess(Key key, bool resident) override {
    if (!resident) return;
    if (lir_.count(key)) {
      const bool was_bottom = !stack_.Empty() && stack_.Front() == key;
      stack_.MoveToBack(key);
      if (was_bottom) Prune();
    } else if (stack_.Contains(key) && !lir_.empty()) {
      stack_.MoveToBack(key);
      queue_.Erase(key);
      lir_.insert(key);
      DemoteBottomLir();
    } else {
      stack_.SetBack(key);
      queue_.SetBack(key);
    }
  }

  std::optional<Key> SelectVictim(Key, bool,
                                  const EligibleFn& eligible) override {
    if (auto key = FirstEligible(queue_, eligible)) return key;
    for (Key key : stack_) {  // no resident HIR is evictable: oldest LIR
      if (lir_.count(key) && eligible(key)) return key;
    }
    return std::nullopt;
  }

  void OnEvict(Key key) override {
    if (queue_.Erase(key)) return;  // stays in the stack as non-resident HIR
    if (lir_.erase(key)) {
      stack_.Erase(key);
      Prune();
    }
  }

  void OnAdmit(Key key, bool) override {
    if (lir_.size() < l_lirs_) {  // warm-up: fill the LIR set first
      lir_.insert(key);
      stack_.SetBack(key);
      return;
    }
    if (stack_.Contains(key) && !lir_.empty()) {
      stack_.MoveToBack(key);
      lir_.insert(key);
      DemoteBottomLir();
    } else {
      stack_.SetBack(key);
      queue_.Set(key);
    }
  }

 private:
  void Prune() {
    while (!stack_.Empty() && lir_.count(stack_.Front()) == 0) {
      stack_.PopFront();
    }
  }
  void DemoteBottomLir() {
    if (stack_.Empty()) return;
    const Key bottom = stack_.Front();
    lir_.erase(bottom);
    stack_.Erase(bottom);
    queue_.Set(bottom);
    Prune();
  }

  std::size_t l_lirs_ = 0;
  OrderedKeys stack_;  // recency stack, front = bottom
  OrderedKeys queue_;  // resident HIR keys, front = next victim
  std::unordered_set<Key> lir_;
};

// ---------------------------------------------------------------------------
// S3-FIFO (Yang et al., SOSP'23, Algorithm 1)
// ---------------------------------------------------------------------------
class S3FIFOPolicy final : public EvictionPolicy {
 public:
  const char* Name() const override { return "s3fifo"; }
  void SetCapacity(std::size_t slots) override {
    EvictionPolicy::SetCapacity(slots);
    s_cap_ = std::max<std::size_t>(
        1, static_cast<std::size_t>(0.1 * static_cast<double>(slots)));
    m_cap_ = slots > s_cap_ ? slots - s_cap_ : 0;
  }
  void OnAccess(Key key, bool resident) override {
    if (!resident) return;
    auto it = freq_.find(key);
    if (it != freq_.end() && it->second < 3) ++it->second;
  }
  // A small-queue pass that promoted into a full main queue already evicted
  // from main and still owes the small-queue eviction.
  bool WantsVictim(Key) const override { return in_small_pass_; }

  std::optional<Key> SelectVictim(Key, bool need_room,
                                  const EligibleFn& eligible) override {
    if (!in_small_pass_) {
      if (!(small_.Size() >= s_cap_ || main_.Empty())) {
        return need_room ? MainVictim(eligible) : std::nullopt;
      }
      in_small_pass_ = true;
    }
    for (auto it = small_.begin(); it != small_.end();) {
      const Key key = *it++;
      if (freq_[key] > 1) {
        freq_[key] = 0;
        small_.Erase(key);
        main_.PushBack(key);
        if (main_.Size() > m_cap_) {
          if (auto victim = MainVictim(eligible)) return victim;
        }
        continue;
      }
      if (eligible(key)) {
        in_small_pass_ = false;
        return key;
      }
    }
    in_small_pass_ = false;  // small queue exhausted without a victim
    return need_room ? MainVictim(eligible) : std::nullopt;
  }

  void OnEvict(Key key) override {
    if (small_.Erase(key)) {
      freq_.erase(key);
      ghost_.Set(key);
      if (ghost_.Size() > m_cap_) ghost_.PopFront();
    } else if (main_.Erase(key)) {
      freq_.erase(key);
    }
  }

  void OnAdmit(Key key, bool) override {
    freq_[key] = 0;
    if (ghost_.Erase(key)) {
      main_.Set(key);
    } else {
      small_.Set(key);
    }
  }

 private:
  // Main FIFO with lazy reinsertion: a key with a nonzero counter is moved to
  // the head with the counter decremented. Ineligible keys are skipped.
  std::optional<Key> MainVictim(const EligibleFn& eligible) {
    const std::size_t limit = 4 * main_.Size() + 1;
    for (std::size_t step = 0; step < limit && !main_.Empty(); ++step) {
      const Key key = main_.Front();
      int& freq = freq_[key];
      if (freq > 0) {
        --freq;
        main_.MoveToBack(key);
      } else if (eligible(key)) {
        return key;
      } else {
        main_.MoveToBack(key);
      }
    }
    return std::nullopt;
  }

  std::size_t s_cap_ = 1;
  std::size_t m_cap_ = 0;
  bool in_small_pass_ = false;
  OrderedKeys small_, main_, ghost_;  // front = oldest
  std::unordered_map<Key, int> freq_;
};

// ---------------------------------------------------------------------------
// SIEVE (Zhang et al., NSDI'24)
// ---------------------------------------------------------------------------
class SIEVEPolicy final : public EvictionPolicy {
 public:
  const char* Name() const override { return "sieve"; }
  void OnAccess(Key key, bool resident) override {
    if (!resident) return;
    auto it = index_.find(key);
    if (it != index_.end()) visited_[key] = true;
  }
  std::optional<Key> SelectVictim(Key, bool,
                                  const EligibleFn& eligible) override {
    if (queue_.empty()) return std::nullopt;
    auto it = HandIterator();
    const std::size_t limit = 2 * queue_.size() + 1;
    for (std::size_t step = 0; step < limit; ++step) {
      bool& visited = visited_[*it];
      if (visited) {
        visited = false;
      } else if (eligible(*it)) {
        pending_victim_ = *it;
        pending_hand_ = it == queue_.begin()
                            ? std::nullopt
                            : std::optional<Key>(*std::prev(it));
        return *it;
      }
      it = it == queue_.begin() ? std::prev(queue_.end()) : std::prev(it);
    }
    return std::nullopt;
  }
  void OnEvict(Key key) override {
    auto found = index_.find(key);
    if (found == index_.end()) return;
    if (pending_victim_ == key) {
      hand_ = pending_hand_;
      pending_victim_.reset();
    } else if (hand_ == key) {
      hand_ = found->second == queue_.begin()
                  ? std::nullopt
                  : std::optional<Key>(*std::prev(found->second));
    }
    if (hand_ == key) hand_.reset();
    queue_.erase(found->second);
    index_.erase(found);
    visited_.erase(key);
  }
  void OnAdmit(Key key, bool) override {
    if (index_.count(key)) return;
    index_[key] = queue_.insert(queue_.begin(), key);  // head = newest
    visited_[key] = false;
  }

 private:
  std::list<Key>::iterator HandIterator() {
    if (hand_) {
      auto it = index_.find(*hand_);
      if (it != index_.end()) return it->second;
    }
    return std::prev(queue_.end());  // tail = oldest
  }

  std::list<Key> queue_;  // front = head (newest), back = tail (oldest)
  std::unordered_map<Key, std::list<Key>::iterator> index_;
  std::unordered_map<Key, bool> visited_;
  std::optional<Key> hand_;
  std::optional<Key> pending_victim_;
  std::optional<Key> pending_hand_;
};

// ---------------------------------------------------------------------------
// W-TinyLFU (Einziger, Friedman & Manes, ACM ToS 2017)
// ---------------------------------------------------------------------------
class WTinyLFUPolicy final : public EvictionPolicy {
 public:
  const char* Name() const override { return "w_tinylfu"; }
  void SetCapacity(std::size_t slots) override {
    EvictionPolicy::SetCapacity(slots);
    w_cap_ = std::max<std::size_t>(
        1, RoundHalfEven(0.01 * static_cast<double>(slots)));
    m_cap_ = slots > w_cap_ ? slots - w_cap_ : 0;
    p_cap_ = static_cast<std::size_t>(0.8 * static_cast<double>(m_cap_));
    reset_every_ = std::max<std::size_t>(1, 10 * slots);
  }
  void OnAccess(Key key, bool resident) override {
    ++freq_[key];
    if (++ops_ >= reset_every_) {
      ops_ = 0;
      for (auto it = freq_.begin(); it != freq_.end();) {
        it->second >>= 1;
        it = it->second == 0 ? freq_.erase(it) : std::next(it);
      }
    }
    if (!resident) return;
    if (window_.Contains(key)) {
      window_.MoveToBack(key);
    } else if (protected_.Contains(key)) {
      protected_.MoveToBack(key);
    } else if (probation_.Erase(key)) {
      protected_.PushBack(key);
      if (protected_.Size() > p_cap_) probation_.Set(protected_.PopFront());
    }
  }
  std::optional<Key> SelectVictim(Key, bool,
                                  const EligibleFn& eligible) override {
    std::optional<Key> preferred, alternate;
    if (!window_.Empty()) {
      const Key candidate = window_.Front();
      const OrderedKeys& segment = probation_.Empty() ? protected_ : probation_;
      if (segment.Empty()) {
        preferred = candidate;
      } else {
        const Key victim = segment.Front();
        if (Freq(candidate) > Freq(victim)) {
          preferred = victim;
          alternate = candidate;
        } else {
          preferred = candidate;
          alternate = victim;
        }
      }
    }
    for (const auto& choice : {preferred, alternate}) {
      if (choice && eligible(*choice)) return choice;
    }
    for (const OrderedKeys* keys : {&probation_, &protected_, &window_}) {
      if (auto key = FirstEligible(*keys, eligible)) return key;
    }
    return std::nullopt;
  }
  void OnEvict(Key key) override {
    if (!window_.Erase(key) && !probation_.Erase(key)) protected_.Erase(key);
  }
  void OnAdmit(Key key, bool) override {
    window_.Set(key);
    if (window_.Size() > w_cap_) probation_.Set(window_.PopFront());
  }

 private:
  std::int64_t Freq(Key key) const {
    auto it = freq_.find(key);
    return it == freq_.end() ? 0 : it->second;
  }

  std::size_t w_cap_ = 1, m_cap_ = 0, p_cap_ = 0, reset_every_ = 1;
  std::size_t ops_ = 0;
  OrderedKeys window_, probation_, protected_;  // front = LRU
  std::unordered_map<Key, std::int64_t> freq_;
};

// ---------------------------------------------------------------------------
// Per-layer LRU (Mixtral-Offloading): fixed slots per MoE layer.
// ---------------------------------------------------------------------------
class LayerLRUPolicy final : public EvictionPolicy {
 public:
  explicit LayerLRUPolicy(int num_layers) : num_layers_(num_layers) {}
  const char* Name() const override { return "layer_lru"; }
  void OnAccess(Key key, bool resident) override {
    const int layer = ExpertCacheKeyLayer(key);
    max_layer_ = std::max(max_layer_, layer);
    if (!resident) return;
    auto it = layers_.find(layer);
    if (it != layers_.end()) it->second.MoveToBack(key);
  }
  // One eviction from the incoming expert's own layer when that layer is at
  // its quota; further room comes from the most over-quota layer.
  bool WantsVictim(Key incoming) const override {
    if (quota_evicted_for_ == incoming) return false;
    const int layer = ExpertCacheKeyLayer(incoming);
    auto it = layers_.find(layer);
    return it != layers_.end() && !it->second.Empty() &&
           static_cast<std::int64_t>(it->second.Size()) >=
               std::max<std::int64_t>(1, Quota(layer));
  }
  std::optional<Key> SelectVictim(Key incoming, bool need_room,
                                  const EligibleFn& eligible) override {
    if (WantsVictim(incoming)) {
      auto it = layers_.find(ExpertCacheKeyLayer(incoming));
      if (auto key = FirstEligible(it->second, eligible)) {
        quota_evicted_for_ = incoming;
        return key;
      }
    }
    if (!need_room) return std::nullopt;
    // Oldest entry of the most over-quota layer (first in insertion order).
    std::vector<std::pair<std::int64_t, int>> ranked;
    for (int layer : layer_order_) {
      const auto& keys = layers_.at(layer);
      if (keys.Empty()) continue;
      ranked.emplace_back(static_cast<std::int64_t>(keys.Size()) - Quota(layer),
                          layer);
    }
    std::stable_sort(
        ranked.begin(), ranked.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });
    for (const auto& entry : ranked) {
      if (auto key = FirstEligible(layers_.at(entry.second), eligible)) {
        return key;
      }
    }
    return std::nullopt;
  }
  void OnEvict(Key key) override {
    auto it = layers_.find(ExpertCacheKeyLayer(key));
    if (it != layers_.end()) it->second.Erase(key);
  }
  void OnAdmit(Key key, bool) override {
    const int layer = ExpertCacheKeyLayer(key);
    max_layer_ = std::max(max_layer_, layer);
    if (quota_evicted_for_ == key) quota_evicted_for_.reset();
    auto it = layers_.find(layer);
    if (it == layers_.end()) {
      it = layers_.emplace(layer, OrderedKeys{}).first;
      layer_order_.push_back(layer);
    }
    it->second.Set(key);
  }

 private:
  std::int64_t Quota(int layer) const {
    const std::int64_t nl = num_layers_ > 0 ? num_layers_ : max_layer_ + 1;
    if (nl <= 0) return static_cast<std::int64_t>(capacity_);
    const auto cap = static_cast<std::int64_t>(capacity_);
    const std::int64_t base = cap / nl;
    const std::int64_t extra = cap % nl;
    return base + ((layer % nl) < extra ? 1 : 0);
  }

  int num_layers_;
  int max_layer_ = -1;
  std::optional<Key> quota_evicted_for_;
  std::unordered_map<int, OrderedKeys> layers_;
  std::vector<int> layer_order_;  // first-admission order (Python dict order)
};

// ---------------------------------------------------------------------------
// LCNU: layer-cycle next-use (benchmarks/cache_simulator/research/
// decode_policy_study.md). Evict the farthest expected next use:
//   d(L) + layers * (1 / p - 1)
// d(L) = layer-steps until layer L runs again; p = beta * ema + (1 - beta) *
// smoothed routing frequency, both per layer visit.
//
// Between two visits of a layer the second term of its experts is frozen, so
// the best candidate of every layer is cached and only recomputed after the
// layer runs or its residents change (as in the Python reference). A cached
// candidate that is not eligible makes the search scan that layer only.
// ---------------------------------------------------------------------------
class LCNUPolicy final : public EvictionPolicy {
 public:
  explicit LCNUPolicy(const EvictionPolicyOptions& options)
      : num_layers_(options.num_layers),
        alpha_(options.lcnu_alpha),
        beta_(options.lcnu_beta),
        prior_(options.lcnu_prior),
        log_keep_(std::log1p(-options.lcnu_alpha)) {}

  const char* Name() const override { return "lcnu"; }

  void OnAccess(Key key, bool) override {
    const int layer = ExpertCacheKeyLayer(key);
    if (layer > max_layer_) {
      max_layer_ = layer;
      if (num_layers_ <= 0) cand_.clear();  // the inferred cycle grew
    }
    if (!has_cur_ || layer != cur_) {
      has_cur_ = true;
      cur_ = layer;
      ++visits_[layer];
      cand_.erase(layer);
      used_now_.clear();
    }
    const std::int64_t visit = visits_[layer];
    auto it = stats_.find(key);
    if (it == stats_.end()) {
      stats_[key] =
          Stats{(1.0 - alpha_) * prior_ + alpha_, 1, visit - 1, visit};
    } else {
      Stats& s = it->second;
      if (visit > s.last) {  // one routing event per layer visit
        const auto gap = static_cast<double>(visit - s.last);
        s.ema = s.ema * std::exp(log_keep_ * gap) + alpha_;
        s.count += 1;
      }
      s.last = visit;
    }
    used_now_.insert(key);
  }

  std::optional<Key> SelectVictim(Key incoming, bool,
                                  const EligibleFn& eligible) override {
    const std::int64_t cycle = std::max<std::int64_t>(
        1, num_layers_ > 0 ? num_layers_ : max_layer_ + 1);
    const int cur = has_cur_ ? cur_ : ExpertCacheKeyLayer(incoming);
    std::optional<Key> best;
    double best_score = 0.0;
    const auto consider = [&](double score, Key key) {
      if (!best || score > best_score || (score == best_score && key < *best)) {
        best = key;
        best_score = score;
      }
    };
    for (const auto& layer_keys : by_layer_) {
      const int layer = layer_keys.first;
      const auto& keys = layer_keys.second;
      if (keys.empty()) continue;
      if (layer == cur) {
        for (Key key : keys) {
          if (!eligible(key)) continue;
          const double d =
              used_now_.count(key) ? static_cast<double>(cycle) : 0.0;
          consider(d + Base(key, layer, cycle), key);
        }
        continue;
      }
      const double d =
          static_cast<double>(((layer - cur) % cycle + cycle) % cycle);
      const Candidate& cand = LayerCandidate(layer, keys, cycle);
      const double score = d + cand.base;
      // The cached candidate bounds every resident of the layer.
      if (best &&
          (score < best_score || (score == best_score && cand.key > *best))) {
        continue;
      }
      if (eligible(cand.key)) {
        consider(score, cand.key);
        continue;
      }
      for (Key key : keys) {
        if (eligible(key)) consider(d + Base(key, layer, cycle), key);
      }
    }
    return best;
  }

  void OnEvict(Key key) override {
    const int layer = ExpertCacheKeyLayer(key);
    auto it = by_layer_.find(layer);
    if (it == by_layer_.end() || it->second.erase(key) == 0) return;
    cand_.erase(layer);
  }

  void OnAdmit(Key key, bool) override {
    const int layer = ExpertCacheKeyLayer(key);
    if (by_layer_[layer].insert(key).second) cand_.erase(layer);
  }

 private:
  struct Stats {
    double ema;
    std::int64_t count;
    std::int64_t first;
    std::int64_t last;
  };

  struct Candidate {
    double base;
    Key key;
  };

  // cycle * (1 / p - 1): expected layer-steps beyond the next visit.
  double Base(Key key, int layer, std::int64_t cycle) const {
    return static_cast<double>(cycle) * (1.0 / Probability(key, layer) - 1.0);
  }

  const Candidate& LayerCandidate(int layer,
                                  const std::unordered_set<Key>& keys,
                                  std::int64_t cycle) {
    auto it = cand_.find(layer);
    if (it != cand_.end()) return it->second;
    Candidate cand{0.0, 0};
    bool found = false;
    for (Key key : keys) {
      const double base = Base(key, layer, cycle);
      if (!found || base > cand.base || (base == cand.base && key < cand.key)) {
        cand = Candidate{base, key};
        found = true;
      }
    }
    return cand_.emplace(layer, cand).first->second;
  }

  double Probability(Key key, int layer) const {
    auto it = stats_.find(key);
    if (it == stats_.end()) return prior_;  // prefetched, never routed
    const Stats& s = it->second;
    auto v = visits_.find(layer);
    const std::int64_t visits = v == visits_.end() ? 0 : v->second;
    const double recent =
        s.ema * std::exp(log_keep_ * static_cast<double>(visits - s.last));
    const double freq = (static_cast<double>(s.count) + 4.0 * prior_) /
                        (static_cast<double>(visits - s.first) + 4.0);
    const double p = beta_ * recent + (1.0 - beta_) * freq;
    return p > 1e-12 ? p : 1e-12;
  }

  int num_layers_;
  double alpha_, beta_, prior_, log_keep_;
  int max_layer_ = -1;
  bool has_cur_ = false;
  int cur_ = 0;
  std::unordered_map<int, std::int64_t> visits_;
  std::unordered_map<Key, Stats> stats_;
  std::unordered_set<Key> used_now_;
  std::unordered_map<int, std::unordered_set<Key>> by_layer_;  // residents
  std::unordered_map<int, Candidate> cand_;  // valid until the layer changes
};

}  // namespace

const std::vector<std::string>& EvictionPolicyNames() {
  static const std::vector<std::string> names = {
      "lru",  "lfu",    "lfu_k", "lhu",       "lru_k",     "arc", "2q",
      "lirs", "s3fifo", "sieve", "w_tinylfu", "layer_lru", "lcnu"};
  return names;
}

std::unique_ptr<EvictionPolicy> CreateEvictionPolicy(
    const std::string& name, const EvictionPolicyOptions& options) {
  if (name == "lru") return std::make_unique<LRUPolicy>();
  if (name == "lfu") return std::make_unique<LFUPolicy>();
  if (name == "lfu_k") return std::make_unique<LFUKPolicy>(64);
  if (name == "lhu") return std::make_unique<LHUPolicy>(0.5);
  if (name == "lru_k") return std::make_unique<LRUKPolicy>(2);
  if (name == "arc") return std::make_unique<ARCPolicy>();
  if (name == "2q") return std::make_unique<TwoQPolicy>();
  if (name == "lirs") return std::make_unique<LIRSPolicy>();
  if (name == "s3fifo") return std::make_unique<S3FIFOPolicy>();
  if (name == "sieve") return std::make_unique<SIEVEPolicy>();
  if (name == "w_tinylfu") return std::make_unique<WTinyLFUPolicy>();
  if (name == "layer_lru") {
    return std::make_unique<LayerLRUPolicy>(options.num_layers);
  }
  if (name == "lcnu") return std::make_unique<LCNUPolicy>(options);
  return nullptr;
}

std::size_t SimulateEvictionPolicy(const std::string& name,
                                   const std::vector<ExpertCacheKey>& accesses,
                                   std::size_t capacity,
                                   const EvictionPolicyOptions& options) {
  auto policy = CreateEvictionPolicy(name, options);
  if (!policy) return 0;
  policy->SetCapacity(capacity);
  std::unordered_set<ExpertCacheKey> resident;
  const auto eligible = [&resident](ExpertCacheKey key) {
    return resident.count(key) > 0;
  };
  std::size_t hits = 0;
  for (ExpertCacheKey key : accesses) {
    const bool hit = resident.count(key) > 0;
    policy->OnAccess(key, hit);
    if (hit) {
      ++hits;
      continue;
    }
    if (capacity == 0) continue;
    while (true) {
      const bool need_room = resident.size() >= capacity;
      if (!need_room && !policy->WantsVictim(key)) break;
      auto victim = policy->SelectVictim(key, need_room, eligible);
      if (!victim) break;
      resident.erase(*victim);
      policy->OnEvict(*victim);
    }
    resident.insert(key);
    policy->OnAdmit(key, false);
  }
  return hits;
}
