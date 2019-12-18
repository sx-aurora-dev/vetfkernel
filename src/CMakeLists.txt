cmake_minimum_required(VERSION 3.0)

option(USE_INTRINSIC "Use intrinsic" ON)

set(CMAKE_C_COMPILER ${NCC})
set(CMAKE_CXX_COMPILER ${NCXX})

#set(CMAKE_CXX_FLAGS "-O4 -fdiag-vector=0 -fdiag-parallel=0 -report-all -assembly-list")
set(CMAKE_CXX_FLAGS "-O4 -fdiag-vector=0 -fdiag-parallel=0 -assembly-list")

#set(CMAKE_CXX_FLAGS "-D_DEBUG")

set(SOURCES
  avgpooling.cc
  libvetfkernel.cc
  conv2d.cc
  conv2d_backprop_filter.cc
  conv2d_backprop_input.cc
  depthwise_conv.cc
  maxpooling.cc
  maxpooling_backprop.cc
  ops.cc
  ve_ops.cc
  unary_ops.cc
  binary_ops.cc
  reduction_ops.cc
  argmax.cc
  training_ops.cc
  dense_update_functor.cc
  gather_functor.cc
  segment_reduction_ops.cc
  sparse_xent_ops.cc
  cwise_ops_gradients.cc
  fused_batch_norm.cc
  pad.cc
  relu.cc
  reverse.cc
  revision.h)

add_library(vetfkernel STATIC
  ${SOURCES}
  $<TARGET_OBJECTS:vetfkernel_intrinsic>)

target_compile_options(vetfkernel PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>)
if (USE_INTRINSIC)
        target_compile_definitions(vetfkernel PRIVATE LIBVETF_INTRINSIC)
endif()

if (USE_PROFILE)
  target_compile_definitions(vetfkernel PRIVATE USE_PROFILE)
endif()

target_link_libraries(vetfkernel PRIVATE vednn_openmp)
target_link_libraries(vetfkernel PRIVATE vml)
target_link_libraries(vetfkernel PRIVATE blas_sequential)
target_include_directories(vetfkernel PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_include_directories(vetfkernel PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

add_custom_target(check_revision
        ALL
        DEPENDS revision.h.in
        BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/revision.h
        COMMAND ${CMAKE_COMMAND} -DTOPDIR=${CMAKE_SOURCE_DIR} -P ${CMAKE_SOURCE_DIR}/cmake/revision.cmake)

add_dependencies(vetfkernel check_revision)
