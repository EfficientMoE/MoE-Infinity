cmake_minimum_required(VERSION 3.10)
project(LockFreeQueueTests)

add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0) # needed for torch backward compatibility

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#
# Attempt to find the python package that uses the same python executable as
# `EXECUTABLE` and is one of the `SUPPORTED_VERSIONS`.
#
macro(find_python_from_executable EXECUTABLE SUPPORTED_VERSIONS)
  file(REAL_PATH ${EXECUTABLE} EXECUTABLE)
  set(Python3_EXECUTABLE ${EXECUTABLE})
  find_package(Python3 COMPONENTS Interpreter Development.Module)

  if(NOT Python3_FOUND)
    message(FATAL_ERROR "Unable to find python matching: ${EXECUTABLE}.")
  endif()

  set(_VER "${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}")
  set(_SUPPORTED_VERSIONS_LIST ${SUPPORTED_VERSIONS} ${ARGN})

  if(NOT _VER IN_LIST _SUPPORTED_VERSIONS_LIST)
    message(FATAL_ERROR
      "Python version (${_VER}) is not one of the supported versions: "
      "${_SUPPORTED_VERSIONS_LIST}.")
  endif()

  message(STATUS "Found python matching: ${EXECUTABLE}.")
endmacro()


#
# Run `EXPR` in python.  The standard output of python is stored in `OUT` and
# has trailing whitespace stripped.  If an error is encountered when running
# python, a fatal message `ERR_MSG` is issued.
#
function(run_python OUT EXPR ERR_MSG)
  execute_process(
    COMMAND
    "${Python3_EXECUTABLE}" "-c" "${EXPR}"
    OUTPUT_VARIABLE PYTHON_OUT
    RESULT_VARIABLE PYTHON_ERROR_CODE
    ERROR_VARIABLE PYTHON_STDERR
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(NOT PYTHON_ERROR_CODE EQUAL 0)
    message(FATAL_ERROR "${ERR_MSG}: ${PYTHON_STDERR}")
  endif()

  set(${OUT} ${PYTHON_OUT} PARENT_SCOPE)
endfunction()

# Run `EXPR` in python after importing `PKG`. Use the result of this to extend
# `CMAKE_PREFIX_PATH` so the torch cmake configuration can be imported.
macro(append_cmake_prefix_path PKG EXPR)
  run_python(_PREFIX_PATH
    "import ${PKG}; print(${EXPR})" "Failed to locate ${PKG} path")
  list(APPEND CMAKE_PREFIX_PATH ${_PREFIX_PATH})
endmacro()

# Add include directories and link for CUTLASS
set(CUTLASS_DIR $ENV{HOME}/cutlass)
message(STATUS "Using CUTLASS from: ${CUTLASS_DIR}")

if(DEFINED ENV{CONDA_PREFIX})
  set(CONDA_PREFIX_PATH $ENV{CONDA_PREFIX})
  message(STATUS "Conda environment path: ${CONDA_PREFIX_PATH}")
else()
  message(WARNING "CONDA_PREFIX is not set. Make sure your Conda environment is activated.")
endif()

set(CONDA_INCLUDE_DIRS ${CONDA_PREFIX_PATH}/include)
set(CONDA_LINK_DIRS ${CONDA_PREFIX_PATH}/lib)

find_package(PythonInterp REQUIRED)

string(REGEX MATCH "([0-9]+)\\.([0-9]+)\\..*" _ ${PYTHON_VERSION_STRING})
set(Python_VERSION ${CMAKE_MATCH_1}.${CMAKE_MATCH_2})
message(STATUS "Python version: ${Python_VERSION}")

# Supported python versions.  These versions will be searched in order, the
# first match will be selected.  These should be kept in sync with setup.py.
set(PYTHON_SUPPORTED_VERSIONS "3.8" "3.9" "3.10" "3.11")

set(PYTHON_EXECUTABLE_PATH ${CONDA_PREFIX_PATH}/bin/python)
message(STATUS "Using Python executable: ${PYTHON_EXECUTABLE_PATH}")
find_python_from_executable(${PYTHON_EXECUTABLE_PATH} "${PYTHON_SUPPORTED_VERSIONS}")
append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")

# add torch
find_package(Torch REQUIRED)

find_library(torch_python_LIBRARY torch_python PATHS
  "${TORCH_INSTALL_PREFIX}/lib")
message(STATUS "torch_python_LIBRARY: ${torch_python_LIBRARY}")

include_directories(
    ${CUTLASS_DIR}/include
    ${CUTLASS_DIR}/tools/util/include
    ${CUTLASS_DIR}/examples/13_two_tensor_op_fusion
    ${CUTLASS_DIR}/examples/common
    ${CONDA_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/../../core
)

# Link CUTLASS library
link_directories(
    ${CUTLASS_DIR}/build/tools/library
    ${CONDA_LINK_DIRS}
)

# Add CUDA kernel compilation
find_package(CUDA REQUIRED)

# set cuda architecture
set(CUDA_ARCHITECTURES 86)

# set nvcc flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -G -lineinfo -rdynamic -O3 -gencode arch=compute_86,code=sm_86 -Xcompiler -fopenmp")

set(SRC_LIST
    test_uvm_kernel.cu
    test_fused_mlp.cu
    # test_single_gemm_tiled.cu
    test_load_tile.cu
    test_load_tile_templated.cu
    test_tile_size.cu
    test_autosize_tileload.cu
    test_autosize_tileload_stage.cu
    test_autotune_blocksize.cu

)

set(TORCH_SRC_LIST
  test_topk_softmax.cu
  test_expert_fusion.cu
  test_expert_fusion_v2.cu
  # tests_masked_select.cu
  test_fused_mlp_wmma.cu
)

file(GLOB KERNEL_SRC "${CMAKE_SOURCE_DIR}/../../core/kernel/*.cu")
message(STATUS "Using kernel source files: ${KERNEL_SRC}, ${CMAKE_SOURCE_DIR}")

FOREACH(SRC ${SRC_LIST})
  get_filename_component(SRC_NAME ${SRC} NAME_WE)
  add_executable(${SRC_NAME} ${SRC})
  target_link_libraries(${SRC_NAME} cutlass ${CUDA_LIBRARIES})
ENDFOREACH()


FOREACH(SRC ${TORCH_SRC_LIST})
  get_filename_component(SRC_NAME ${SRC} NAME_WE)
  IF(${SRC_NAME} STREQUAL "test_topk_softmax")
    add_executable(${SRC_NAME} ${SRC} ${KERNEL_SRC})
  ELSE()
    add_executable(${SRC_NAME} ${SRC})
  ENDIF()
  target_link_libraries(${SRC_NAME} cutlass ${CUDA_LIBRARIES})

  target_link_libraries(${SRC_NAME} ${torch_python_LIBRARY} ${Python3_LIBRARIES} ${TORCH_LIBRARIES})
  target_include_directories(${SRC_NAME} PRIVATE ${CONDA_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})

  # IF(${SRC_NAME} STREQUAL "test_autosize_tileload")
  #   target_link_libraries(${SRC_NAME} thrust)
  # ENDIF()
ENDFOREACH()
