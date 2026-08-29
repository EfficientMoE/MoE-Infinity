find_package(GTest REQUIRED)

add_executable(test_expert_policy test_expert_policy.cpp)
target_link_libraries(test_expert_policy PRIVATE archer_core GTest::gtest_main pthread)

add_test(NAME ExpertPolicyTests COMMAND test_expert_policy)
