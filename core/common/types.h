// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#pragma once

#include <cstdint>
#include <utility>
#include <string>
#include <type_traits>

typedef std::uint32_t TensorID;
typedef std::size_t HashID;
typedef std::size_t NodeID;
typedef std::uint64_t GraphID;
typedef std::uint64_t RequestID;

#define KB 1024
#define MB (KB * KB)
#define GB (KB * KB * KB)

#define DELETE_COPY_AND_ASSIGN(classname)          \
  classname(const classname&) = delete;            \
  classname& operator=(const classname&) = delete; \
  classname(classname&&) = delete;                 \
  classname& operator=(classname&&) = delete;

#define STATIC_GET_INSTANCE(classname)                          \
  static classname* GetInstance() {                             \
    static std::once_flag flag;                                 \
    static classname* instance = nullptr;                       \
    std::call_once(flag, []() { instance = new classname(); }); \
    return instance;                                            \
  }

template <typename T>
struct DoNothingDeleter {
  void operator()(T* ptr) const {}
};

// Helper to get the Nth type from a parameter pack
template <size_t N, typename... Args>
struct GetNthType;

template <typename First, typename... Rest>
struct GetNthType<0, First, Rest...> {
  using type = First;
};

template <size_t N, typename First, typename... Rest>
struct GetNthType<N, First, Rest...> {
  using type = typename GetNthType<N - 1, Rest...>::type;
};

template <size_t N, typename... Args>
using GetNthType_t = typename GetNthType<N, Args...>::type;

// Compile-time integer square root
template <int N>
struct ConstexprSqrt {
  static constexpr int compute(int low = 1, int high = N) {
    if (low == high) return low;
    int mid = (low + high + 1) / 2;
    return (mid * mid > N) ? compute(low, mid - 1) : compute(mid, high);
  }
  static constexpr int value = compute();
};

// Round to multiple helper
template <int N, int Multiple>
struct RoundToMultiple {
  static constexpr int value = ((N + Multiple - 1) / Multiple) * Multiple;
};

// A constexpr function to convert any const T* pointer to void*
template <typename T>
constexpr void* pointer_to_void(const T* ptr) {
  return const_cast<void*>(reinterpret_cast<const void*>(
      ptr));  // Cast to void* while preserving constness
}

// Helper macros to generate enum and string mappings
#define ENUM_ENTRY_COMMA(value, EnumType) value,
#define ENUM_CASE(value, EnumType) \
  case EnumType::value:            \
    return #value;
#define STRING_CASE(value, EnumType) \
  if (s == #value) return EnumType::value;

// General enum to string conversion using SFINAE
template <typename E>
constexpr auto enum_to_string(E e) noexcept
    -> std::enable_if_t<std::is_enum_v<E>, const char*> {
  // This will be specialized for each enum type
  return "Unknown";
}

// General string to enum conversion
template <typename E>
constexpr auto string_to_enum(const std::string& s) noexcept
    -> std::enable_if_t<std::is_enum_v<E>, E> {
  // This will be specialized for each enum type
  return static_cast<E>(0);  // Default to first enum value
}

// Macro to define enum class, enum to string, and string to enum functions
#define DEFINE_ENUM_CLASS(EnumType, ENUM_VALUES)                             \
  enum class EnumType { ENUM_VALUES(ENUM_ENTRY_COMMA, EnumType) Unknown };   \
                                                                             \
  /* Enum to string function */                                              \
  constexpr const char* EnumType##ToString(EnumType v) {                     \
    switch (v) {                                                             \
      ENUM_VALUES(ENUM_CASE, EnumType)                                       \
      default:                                                               \
        return "Unknown";                                                    \
    }                                                                        \
  }                                                                          \
                                                                             \
  /* String to enum function */                                              \
  EnumType StringTo##EnumType(const std::string& s) {                        \
    ENUM_VALUES(STRING_CASE, EnumType)                                       \
    return EnumType::Unknown;                                                \
  }                                                                          \
                                                                             \
  /* Specialize generic template functions for this enum type */             \
  template <>                                                                \
  constexpr auto enum_to_string<EnumType>(EnumType e) noexcept               \
      -> const char* {                                                       \
    return EnumType##ToString(e);                                            \
  }                                                                          \
                                                                             \
  template <>                                                                \
  auto string_to_enum<EnumType>(const std::string& s) noexcept -> EnumType { \
    return StringTo##EnumType(s);                                            \
  }
