cmake_minimum_required(VERSION 3.0)

add_subdirectory(intrinsic)

option(USE_INTRINSIC "Use intrinsic" ON)

set(CMAKE_C_COMPILER ${NCC})
set(CMAKE_CXX_COMPILER ${NCXX})

#set(CMAKE_CXX_FLAGS "-O4 -fdiag-vector=0 -fdiag-parallel=0 -report-all -assembly-list")
set(CMAKE_CXX_FLAGS "-O4 -fdiag-vector=0 -fdiag-parallel=0 -assembly-list")

#set(CMAKE_CXX_FLAGS "-D_DEBUG")

set(VEDNN_PATH ${CMAKE_SOURCE_DIR}/libs/vednn CACHE PATH "path to vednn")

add_library(vml STATIC
  asl.cc
  avgpooling.cc
  binary_ops.cc
  conv2d.cc
  pad.cc
  randomUniform.cc
  reduction.cc
  relu.cc
  tile.cc
  training_ops.cc
  unary_ops.cc
  vml.cc
  $<TARGET_OBJECTS:vml_intrinsic>)

target_compile_options(vml PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>)
if (USE_INTRINSIC)
        target_compile_definitions(vml PRIVATE LIBVETF_INTRINSIC)
endif()

target_link_libraries(vml PUBLIC vednn_openmp)
target_link_libraries(vml PUBLIC asl_sequential)
target_include_directories(vml PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_include_directories(vml PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_include_directories(vml PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include/vml)
target_include_directories(vml PRIVATE /opt/nec/ve/nlc/1.0.0/include)
