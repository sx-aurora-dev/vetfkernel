cmake_minimum_required(VERSION 3.0)

option(USE_INTRINSIC "Use intrinsic" ON)

if (USE_INTRINSIC)
  add_subdirectory(intrinsic)
endif()

set(CMAKE_C_COMPILER ${NCC})
set(CMAKE_CXX_COMPILER ${NCXX})

if (NOT X86)
  #set(CMAKE_CXX_FLAGS "-O4 -fdiag-vector=0 -fdiag-parallel=0 -report-all -assembly-list")
  set(CMAKE_CXX_FLAGS_RELEASE "-O4 -fdiag-vector=0 -fdiag-parallel=0 -assembly-list")
endif()

#set(CMAKE_CXX_FLAGS "-D_DEBUG")

set(SOURCES
  avgpooling.cc
  libvetfkernel.cc
  batch_matmul.cc
  ops.cc
  ve_ops.cc
  unary_ops.cc
  binary_ops.cc
  reduction_ops.cc
  argmax.cc
  training_ops.cc
  dense_update_functor.cc
  gather_functor.cc
  scatter_functor.cc
  segment_reduction_ops.cc
  sparse_xent_ops.cc
  cwise_ops_gradients.cc
  fused_batch_norm.cc
  pad.cc
  relu.cc
  reverse.cc
  fill_functor.cc
  einsum_functor.cc
  broadcast_to.cc
  revision.h)

add_library(vetfkernel STATIC)
target_sources(vetfkernel PRIVATE ${SOURCES})
if (USE_INTRINSIC)
  target_sources(vetfkernel PRIVATE $<TARGET_OBJECTS:vetfkernel_intrinsic>)
  target_compile_definitions(vetfkernel PRIVATE LIBVETF_INTRINSIC)
endif()
if (USE_VEDNN)
  target_sources(vetfkernel PRIVATE
    conv2d.cc
    conv2d_backprop_filter.cc
    conv2d_backprop_input.cc
    depthwise_conv.cc
    maxpooling.cc
    )
  target_compile_definitions(vetfkernel PRIVATE USE_VEDNN)
  target_link_libraries(vetfkernel PRIVATE vednn_openmp)
endif()

target_compile_options(vetfkernel PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>)

if (USE_PROFILE)
  target_compile_definitions(vetfkernel PRIVATE USE_PROFILE)
endif()

target_link_libraries(vetfkernel PRIVATE vml)
target_include_directories(vetfkernel PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_include_directories(vetfkernel PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

if (NOT X86)
  cmake_policy(SET CMP0079 NEW) # this is required to use blas_sequential
  target_link_libraries(vml PUBLIC blas_sequential)
  target_link_libraries(vml PUBLIC -lveio)
endif()

add_custom_target(check_revision
        ALL
        DEPENDS revision.h.in
        BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/revision.h
        COMMAND ${CMAKE_COMMAND} -DTOPDIR=${CMAKE_SOURCE_DIR} -P ${CMAKE_SOURCE_DIR}/cmake/revision.cmake)

add_dependencies(vetfkernel check_revision)
