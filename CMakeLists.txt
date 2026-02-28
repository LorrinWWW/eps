cmake_minimum_required(VERSION 3.15 FATAL_ERROR) # for PyTorch extensions, version should be greater than 3.13
project(eps LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include(FindCUDA)

CUDA_SELECT_NVCC_ARCH_FLAGS(NVCC_FLAGS_AUTO)

# Extract compute capability using regex
string(REGEX MATCH "compute_([0-9]+)" _ ${NVCC_FLAGS_AUTO})
set(COMPUTE_CAPABILITY "${CMAKE_MATCH_1}")

set(SM_NUM "90a" CACHE STRING "Target SM architecture (e.g. 90a, 100a)")
if(DEFINED ENV{EPS_SM_NUM} AND NOT "$ENV{EPS_SM_NUM}" STREQUAL "")
  set(SM_NUM "$ENV{EPS_SM_NUM}" CACHE STRING "Target SM architecture (e.g. 90a, 100a)" FORCE)
endif()
if(SM_NUM STREQUAL "90a")
  add_definitions(-DIS_HOPPER)
elseif(SM_NUM STREQUAL "100a")
  add_definitions(-DIS_BLACKWELL)
endif()

option(WITH_NVSHMEM "Build with nvshmem" OFF)
option(EPS_STATIC_INTERNAL_LIBS "Statically link internal libs (mscclpp, common, quant, executor)" OFF)
if(EPS_STATIC_INTERNAL_LIBS)
    set(EPS_INTERNAL_LIB_TYPE STATIC)
    set(EPS_MSCCLPP_TARGET mscclpp_static)
else()
    set(EPS_INTERNAL_LIB_TYPE SHARED)
    set(EPS_MSCCLPP_TARGET mscclpp)
endif()

# Print the detected SM_NUM
message(STATUS "Detected SM_NUM: ${SM_NUM}")
# CMake 3.15 does not understand suffixed arch (e.g. 90a) and will emit sm_90.
# Avoid CMake-generated gencode when SM_NUM has a suffix and rely on manual -gencode below.
if(SM_NUM MATCHES "a$")
  set(CMAKE_CUDA_ARCHITECTURES "" CACHE STRING "CUDA arch list (empty to avoid CMake gencode)" FORCE)
else()
  set(CMAKE_CUDA_ARCHITECTURES ${SM_NUM} CACHE STRING "CUDA arch list" FORCE)
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall,-Wextra -Wno-unused-parameter")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_${SM_NUM},code=\\\"sm_${SM_NUM},compute_${SM_NUM}\\\"")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")


string(REPLACE "." ";" CUDA_VERSION_PARTS ${CMAKE_CUDA_COMPILER_VERSION})
list(GET CUDA_VERSION_PARTS 0 CUDA_VERSION_MAJOR)
list(GET CUDA_VERSION_PARTS 1 CUDA_VERSION_MINOR)
list(APPEND CUTLASS_CUDA_CLANG_FLAGS -D__CUDACC_VER_MAJOR__=${CUDA_VERSION_MAJOR} -D__CUDACC_VER_MINOR__=${CUDA_VERSION_MINOR})
if(${CUDA_VERSION_MAJOR} VERSION_GREATER_EQUAL "11")
  add_definitions("-DENABLE_BF16")
  add_definitions("-DENABLE_FP8")
  message("CUDA_VERSION ${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR} is greater or equal than 11.0, enable -DENABLE_BF16 flag")
endif()

add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)

include(FetchContent)
FetchContent_Declare(
  repo-mscclpp
  GIT_REPOSITORY https://github.com/meituan-longcat/mscclpp.git
  GIT_TAG        feature/eps
)
set(MSCCLPP_BUILD_TESTS OFF CACHE BOOL "Disable mscclpp tests" FORCE)
FetchContent_MakeAvailable(repo-mscclpp)
# Fix stack buffer overflow in mscclpp bootstrap: MAX_IF_NAME_SIZE (16) is too
# small for SocketToString() output which writes a full IP<port> string.
execute_process(COMMAND sed -i "s/MAX_IF_NAME_SIZE 16/MAX_IF_NAME_SIZE 32/g"
    ${repo-mscclpp_SOURCE_DIR}/src/include/socket.h
    ${repo-mscclpp_SOURCE_DIR}/src/include/endpoint.hpp)
set(repo-mscclpp_SOURCE_DIR ${repo-mscclpp_SOURCE_DIR} CACHE INTERNAL "")

find_package(Python3 COMPONENTS Interpreter Development)
include_directories(${Python3_INCLUDE_DIRS})

execute_process(COMMAND "python3" "-c" "from __future__ import print_function; import os; import torch; print(os.path.dirname(torch.__file__),end='');"
                RESULT_VARIABLE _PYTHON_SUCCESS
                OUTPUT_VARIABLE TORCH_DIR)
if (NOT _PYTHON_SUCCESS MATCHES 0)
    message(FATAL_ERROR "Torch config Error.")
endif()
list(APPEND CMAKE_PREFIX_PATH ${TORCH_DIR})
find_package(Torch REQUIRED)
execute_process(COMMAND "python3" "-c" "from __future__ import print_function; import torch; print(torch.__version__,end='');"
RESULT_VARIABLE _PYTHON_SUCCESS
OUTPUT_VARIABLE TORCH_VERSION)

add_definitions(-DTORCH_CUDA=1)

# === Configuration options ===
# set(CMAKE_CUDA_ARCHITECTURES 90a CACHE STRING "CUDA architecture to target")

# === CMake configuration ===
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)


# === Dependencies ===
include(FetchContent)
find_package(CUDAToolkit REQUIRED)  # Modern replacement for find_package(CUDA)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(Torch REQUIRED)
if(WITH_NVSHMEM)
find_package(NVSHMEM REQUIRED)
endif()

# Create imported target for PyTorch
add_library(torch_imported INTERFACE)
add_library(torch::py_limited ALIAS torch_imported)
target_include_directories(torch_imported SYSTEM INTERFACE ${TORCH_INCLUDE_DIRS})
# NOTE(lequn): We don't link against all ${TORCH_LIBRARIES} because we use py_limited_api.
# See: https://github.com/pytorch/pytorch/blob/9017becf1d895999a1c819c9d35b8139c090e7a9/torch/utils/cpp_extension.py#L1256-L1270
target_link_libraries(torch_imported INTERFACE c10 torch torch_cpu c10_cuda torch_cuda CUDA::cudart)

# === Compiler and linker flags ===
add_compile_options(-Wno-deprecated-declarations)
# add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)
# add_compile_definitions(Py_LIMITED_API=0x03090000)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# CUDA-specific compile options function
function(set_cuda_compile_options target)
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--threads=32 -O3>)
endfunction()

add_subdirectory(src/common)

add_subdirectory(src/communication)

add_subdirectory(src/scheduler)

add_subdirectory(src/fast_ep)

add_subdirectory(src/fast_oep)

add_subdirectory(src/quant)

add_subdirectory(src/executor)

add_subdirectory(src/gemm)

add_subdirectory(src/utils)

message(STATUS "========Build mode: ${CMAKE_BUILD_TYPE}========")

string(TOLOWER "${CMAKE_BUILD_TYPE}" cmake_build_type_lower)
if(cmake_build_type_lower STREQUAL "debug")
  message(STATUS "========Set ENABLE_DEBUG========")
  add_definitions(-DENABLE_DEBUG)
else()
  message(STATUS "========Unset ENABLE_DEBUG========")
endif()

# add_subdirectory(test/src)
