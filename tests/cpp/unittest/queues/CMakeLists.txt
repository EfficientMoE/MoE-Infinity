cmake_minimum_required(VERSION 3.10)
project(QueueTests)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add GoogleTest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Include core headers (project root is 4 levels up from tests/cpp/unittest/queues/)
message("CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
include_directories(${CMAKE_SOURCE_DIR}/../../../../core)

# Add test executables  (sources live in the same directory as this CMakeLists)
add_executable(test_lockfree_queue test_lockfree_queue.cpp)
add_executable(test_threadsafe_queue test_threadsafe_queue.cpp)

# Link GoogleTest and pthread
target_link_libraries(test_lockfree_queue ${GTEST_LIBRARIES} pthread)
target_link_libraries(test_threadsafe_queue ${GTEST_LIBRARIES} pthread)

enable_testing()
add_test(NAME LockFreeQueueTests COMMAND test_lockfree_queue)
add_test(NAME ThreadSafeQueueTests COMMAND test_threadsafe_queue)
