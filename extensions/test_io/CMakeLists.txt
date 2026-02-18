find_library(torch_python_LIBRARY torch_python
  PATHS "${TORCH_INSTALL_PREFIX}/lib")

Python3_add_library(test_io_module MODULE WITH_SOABI
  ${CMAKE_SOURCE_DIR}/tests/docker/test_io_module.cpp
)

# --whole-archive forces all object files from the static lib into the shared
# library, preventing "undefined symbol" errors at Python import time.
# archer_core is listed twice: once wrapped with --whole-archive (for complete
# symbol inclusion), and once as a CMake target (to propagate PUBLIC include
# dirs and transitive link libraries). The second occurrence is a linker no-op.
target_link_libraries(test_io_module PRIVATE
  -Wl,--whole-archive $<TARGET_FILE:archer_core> -Wl,--no-whole-archive
  archer_core
  ${torch_python_LIBRARY})

target_compile_options(test_io_module PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wno-reorder -fPIC -fopenmp -march=native>
)

install(TARGETS test_io_module
  LIBRARY DESTINATION "${CMAKE_SOURCE_DIR}/tests/docker"
  COMPONENT TestRuntime
)
