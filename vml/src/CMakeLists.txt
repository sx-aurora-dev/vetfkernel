cmake_minimum_required(VERSION 3.0)

option(USE_INTRINSIC "Use intrinsic" ON)

if (USE_INTRINSIC)
add_subdirectory(intrinsic)
endif()

set(CMAKE_C_COMPILER ${NCC})
set(CMAKE_CXX_COMPILER ${NCXX})

if (NOT X86)
  #set(CMAKE_CXX_FLAGS "-O4 -fdiag-vector=0 -fdiag-parallel=0 -report-all -assembly-list")
  set(CMAKE_CXX_FLAGS "-O4 -fdiag-vector=0 -fdiag-parallel=0 -assembly-list")
endif()

#set(CMAKE_CXX_FLAGS "-D_DEBUG")

set(VEDNN_PATH ${CMAKE_SOURCE_DIR}/libs/vednn CACHE PATH "path to vednn")

add_library(vml STATIC
  avgpooling.cc
  batch_matmul.cc
  biasAdd.cc
  binary_ops.cc
  pad.cc
  randomUniform.cc
  reduction.cc
  relu.cc
  tile.cc
  training_ops.cc
  unary_ops.cc
  vml.cc)

if (NOT X86)
  target_sources(vml PRIVATE asl.cc)
endif()

if (USE_INTRINSIC)
  target_sources(vml PRIVATE $<TARGET_OBJECTS:vml_intrinsic>)
endif()

if (USE_VEDNN)
  target_sources(vml PRIVATE
    conv2d.cc)
  target_link_libraries(vml PUBLIC vednn_openmp)
endif()

target_compile_options(vml PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>)
if (USE_INTRINSIC)
  target_compile_definitions(vml PRIVATE LIBVETF_INTRINSIC)
endif()

if (USE_PROFILE)
  target_compile_definitions(vml PRIVATE USE_PROFILE)
endif()

if (NOT X86)
  #target_link_libraries(vml PUBLIC blas_sequential)
  target_link_libraries(vml PUBLIC asl_sequential)
endif()
target_include_directories(vml PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_include_directories(vml PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_include_directories(vml PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include/vml)
target_include_directories(vml PRIVATE /opt/nec/ve/nlc/1.0.0/include)
