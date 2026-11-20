# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

# Standalone parity + perf harness for core/parallel/expert_drop_select.cc
# against the Python reference. It needs no gtest/torch at run time; the
# fixture comes from gen_expert_drop_fixture.py (Python with torch).
add_executable(test_expert_drop_select
  ${PROJECT_SOURCE_DIR}/tests/cpp/test_expert_drop_select.cc
  ${PROJECT_SOURCE_DIR}/core/parallel/expert_drop_select.cc)
set(_drop_fixture ${CMAKE_CURRENT_BINARY_DIR}/expert_drop_fixture.txt)
add_test(NAME expert_drop_select_fixture
  COMMAND ${Python3_EXECUTABLE}
          ${PROJECT_SOURCE_DIR}/tests/cpp/gen_expert_drop_fixture.py
          ${_drop_fixture})
set_tests_properties(expert_drop_select_fixture PROPERTIES
  FIXTURES_SETUP expert_drop_fixture)
add_test(NAME expert_drop_select COMMAND test_expert_drop_select ${_drop_fixture})
set_tests_properties(expert_drop_select PROPERTIES
  FIXTURES_REQUIRED expert_drop_fixture)

add_executable(test_expert_timing_lifecycle test_expert_timing_lifecycle.cpp)
target_link_libraries(test_expert_timing_lifecycle PRIVATE
  archer_core GTest::gtest_main CUDA::cudart ${TORCH_LIBRARIES}
  ${Python3_LIBRARIES})
target_include_directories(test_expert_timing_lifecycle PRIVATE
  ${PROJECT_SOURCE_DIR}/core ${TORCH_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS}
  ${Python3_INCLUDE_DIRS})
include(GoogleTest)
gtest_discover_tests(test_expert_timing_lifecycle)

add_executable(test_event_pool test_event_pool.cpp)
target_link_libraries(test_event_pool PRIVATE
  archer_core GTest::gtest_main CUDA::cudart ${TORCH_LIBRARIES}
  ${Python3_LIBRARIES})
target_include_directories(test_event_pool PRIVATE
  ${PROJECT_SOURCE_DIR}/core ${TORCH_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS}
  ${Python3_INCLUDE_DIRS})
gtest_discover_tests(test_event_pool)
