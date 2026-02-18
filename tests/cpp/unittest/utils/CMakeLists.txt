cmake_minimum_required(VERSION 3.10)
project(UtilsTests)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# core/ headers are 4 levels up from tests/cpp/unittest/utils/
include_directories(${CMAKE_SOURCE_DIR}/../../../../core)

add_executable(test_lfu_cache test_lfu_cache.cpp)
target_link_libraries(test_lfu_cache ${GTEST_LIBRARIES} pthread)

add_executable(test_simple_object_pool test_simple_object_pool.cpp)
target_link_libraries(test_simple_object_pool ${GTEST_LIBRARIES} pthread)

enable_testing()
add_test(NAME LFUCacheTests COMMAND test_lfu_cache)
add_test(NAME SimpleObjectPoolTests COMMAND test_simple_object_pool)
