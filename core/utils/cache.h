#pragma once

#include <iostream>
#include <unordered_map>
#include <list>
#include <utility>

template <typename KeyType, typename ValueType>
class LFUCache {
 private:
  struct Node {
    KeyType key;
    ValueType value;
    int freq;
    Node(KeyType k, ValueType v) : key(k), value(v), freq(1) {}
  };

  int capacity;
  int minFreq;
  std::unordered_map<KeyType, typename std::list<Node>::iterator> keyNodeMap;
  std::unordered_map<int, std::list<Node>> freqListMap;

 public:
  LFUCache(int capacity) : capacity(capacity), minFreq(0) {}

  ValueType get(KeyType key) {
    if (!keyNodeMap.count(key)) {
      throw std::range_error("Key not found");
    }

    auto node = keyNodeMap[key];
    int freq = node->freq;
    freqListMap[freq].erase(node);
    if (freqListMap[freq].empty()) {
      freqListMap.erase(freq);
      if (minFreq == freq) minFreq += 1;
    }

    node->freq += 1;
    freqListMap[node->freq].push_front(*node);
    keyNodeMap[key] = freqListMap[node->freq].begin();

    return node->value;
  }

  void put(KeyType key, ValueType value) {
    if (capacity == 0) return;

    if (keyNodeMap.count(key)) {
      auto node = keyNodeMap[key];
      node->value = value;
      get(key);  // update the node's frequency
      return;
    }

    if (keyNodeMap.size() == capacity) {
      auto node = freqListMap[minFreq].back();
      keyNodeMap.erase(node.key);
      freqListMap[minFreq].pop_back();
      if (freqListMap[minFreq].empty()) {
        freqListMap.erase(minFreq);
      }
    }

    minFreq = 1;
    Node newNode(key, value);
    freqListMap[minFreq].push_front(newNode);
    keyNodeMap[key] = freqListMap[minFreq].begin();
  }

  void reset() {
    for (auto& freq_pair : freqListMap) {
      for (auto& node : freq_pair.second) {
        node.freq = 1;  // reset frequency to 1
        freqListMap[1].push_back(node);
      }
      freq_pair.second.clear();
    }
    freqListMap.erase(++freqListMap.begin(), freqListMap.end());
    minFreq = 1;
  }
};
