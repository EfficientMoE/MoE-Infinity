#pragma once

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <list>
#include <unordered_map>
#include <utility>
#include <vector>

template <typename KeyType, typename ValueType>
class LFUCache {
 private:
  struct Node {
    KeyType key;
    ValueType value;
    int freq;
    std::uint64_t touch;
    Node(KeyType k, ValueType v, std::uint64_t touch_)
        : key(std::move(k)), value(std::move(v)), freq(1), touch(touch_) {}
  };

  int capacity;
  int minFreq;
  std::uint64_t counter;
  std::unordered_map<KeyType, typename std::list<Node>::iterator> keyNodeMap;
  std::unordered_map<int, std::list<Node>> freqListMap;

 public:
  LFUCache(int capacity) : capacity(capacity), minFreq(0), counter(0) {}

  ValueType get(KeyType key) {
    if (!keyNodeMap.count(key)) {
      throw std::range_error("Key not found");
    }

    auto node_it = keyNodeMap[key];
    int freq = node_it->freq;
    auto& old_list = freqListMap[freq];

    Node node = std::move(*node_it);
    old_list.erase(node_it);

    if (old_list.empty()) {
      freqListMap.erase(freq);
      if (minFreq == freq) {
        minFreq += 1;
      }
    }

    node.freq += 1;
    node.touch = ++counter;
    auto& new_list = freqListMap[node.freq];
    new_list.push_front(std::move(node));
    keyNodeMap[key] = new_list.begin();

    return keyNodeMap[key]->value;
  }

  void put(KeyType key, ValueType value) {
    if (capacity == 0) return;

    if (keyNodeMap.count(key)) {
      auto node_it = keyNodeMap[key];
      node_it->value = std::move(value);
      get(key);  // update the node's frequency
      return;
    }

    if (keyNodeMap.size() == static_cast<size_t>(capacity)) {
      auto& list = freqListMap[minFreq];
      auto evict_it = std::prev(list.end());
      keyNodeMap.erase(evict_it->key);
      list.pop_back();
      if (list.empty()) {
        freqListMap.erase(minFreq);
      }
    }

    minFreq = 1;
    Node newNode(std::move(key), std::move(value), ++counter);
    auto& list = freqListMap[minFreq];
    list.push_front(std::move(newNode));
    keyNodeMap[list.begin()->key] = list.begin();
  }

  void reset() {
    if (keyNodeMap.empty()) {
      minFreq = 0;
      return;
    }

    std::vector<Node> nodes;
    nodes.reserve(keyNodeMap.size());
    for (auto& freq_pair : freqListMap) {
      for (auto& node : freq_pair.second) {
        node.freq = 1;
        nodes.push_back(node);
      }
    }

    std::sort(nodes.begin(), nodes.end(), [](const Node& a, const Node& b) {
      return a.touch > b.touch;  // most recent first
    });

    freqListMap.clear();
    keyNodeMap.clear();

    auto& list = freqListMap[1];
    for (auto& node : nodes) {
      list.push_back(std::move(node));
    }

    for (auto it = list.begin(); it != list.end(); ++it) {
      keyNodeMap[it->key] = it;
    }

    minFreq = 1;
  }
};
