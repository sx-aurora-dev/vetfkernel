cmake_minimum_required(VERSION 3.0)

option(USE_INTRINSIC "Use intrinsic" ON)

set(CMAKE_C_COMPILER ${NCC})
set(CMAKE_CXX_COMPILER ${NCXX})

#set(CMAKE_CXX_FLAGS "-O4 -fdiag-vector=0 -fdiag-parallel=0 -report-all -assembly-list")
set(CMAKE_CXX_FLAGS "-O4 -fdiag-vector=0 -fdiag-parallel=0 -assembly-list")

#set(CMAKE_CXX_FLAGS "-D_DEBUG")

set(VEDNN_PATH ${CMAKE_SOURCE_DIR}/libs/vednn CACHE PATH "path to vednn")

set(SOURCES
  asl.cc
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
  relu.cc
  reverse.cc
  revision.h
  vml.cc)

add_library(vetfkernel STATIC
  ${SOURCES}
  $<TARGET_OBJECTS:vetfkernel_intrinsic>)

target_compile_options(vetfkernel PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>)
if (USE_INTRINSIC)
        target_compile_definitions(vetfkernel PRIVATE LIBVETF_INTRINSIC)
endif()

target_include_directories(vetfkernel PRIVATE ${VEDNN_PATH}/src)
target_include_directories(vetfkernel PRIVATE /opt/nec/ve/nlc/1.0.0/include ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/intrinsic)
target_include_directories(vetfkernel PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

add_custom_target(check_revision
        ALL
        DEPENDS revision.h.in
        BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/revision.h
        COMMAND ${CMAKE_COMMAND} -DTOPDIR=${CMAKE_SOURCE_DIR} -P ${CMAKE_SOURCE_DIR}/cmake/revision.cmake)

add_dependencies(vetfkernel check_revision)

# Obosolete
#add_library(vetfkernel_oldintrin STATIC
#  ${SOURCES}
#  $<TARGET_OBJECTS:vetfkernel_old_intrinsic>)
#
#target_compile_options(vetfkernel_oldintrin PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>)
#if (USE_INTRINSIC)
#        target_compile_definitions(vetfkernel_oldintrin PRIVATE LIBVETF_INTRINSIC)
#endif()
#
#target_include_directories(vetfkernel_oldintrin PRIVATE ${VEDNN_PATH}/src)
#target_include_directories(vetfkernel_oldintrin PRIVATE /opt/nec/ve/nlc/1.0.0/include ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/intrinsic)
#target_include_directories(vetfkernel_oldintrin PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
#
#add_dependencies(vetfkernel_oldintrin check_revision)

