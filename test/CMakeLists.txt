cmake_minimum_required(VERSION 3.0)

set(CMAKE_C_COMPILER ${NCC})
set(CMAKE_CXX_COMPILER ${NCXX})

if (NOT X86)
  set(CMAKE_CXX_FLAGS "-O4 -fdiag-vector=0")
endif()

#set(CMAKE_CXX_FLAGS "-D_DEBUG")
set(CMAKE_CXX_FLAGS "-D_DEBUG -g")

set(VEDNN_PATH ${CMAKE_SOURCE_DIR}/libs/vednn CACHE PATH "path to vednn")

add_executable(test01 
  test01.cc # test01.cc have to be first???
  batch_matmul.cc
  matmul.cc
  biasAdd.cc
  conv2d.cc
  pad.cc
  reduction.cc
  relu.cc 
  test_binary_ops.cc)

target_link_libraries(test01 PRIVATE 
  -static
  vetfkernel
  vml
  ${CLANG_RUNTIME}
  -fopenmp
  -pthread
  -ldl)

add_executable(bench bench.cc)
target_link_libraries(bench PRIVATE
  -static
  vetfkernel
  vml
  ${CLANG_RUNTIME}
  -fopenmp
  -pthread
  -ldl)

if (USE_VEDNN)
  target_compile_definitions(bench PRIVATE USE_VEDNN)
endif()
