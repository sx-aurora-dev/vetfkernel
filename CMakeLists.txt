cmake_minimum_required(VERSION 3.0)

option(X86 "Build for x86" OFF)
option(USE_PROFILE "Enable profiler" OFF)
option(USE_VEDNN "Use vednn" ON)
option(USE_PREBUILT_VEDNN "Use prebuilt vednn" OFF)
option(BUILD_VEORUN_TF "Build veorun_tf" ON)

if (X86)
  message("x86 build")
  set(NCC "gcc" CACHE FILEPATH "Path of ncc")
  set(NCXX "g++" CACHE FILEPATH "Path of nc++")

  set(USE_INTRINSIC OFF)
  set(USE_VEDNN OFF)
  set(BUILD_VEORUN_TF OFF)
else()
  set(NCC "/opt/nec/ve/bin/ncc" CACHE FILEPATH "Path of ncc")
  set(NCXX "/opt/nec/ve/bin/nc++" CACHE FILEPATH "Path of nc++")
  set(LLVM_DIR "/opt/nec/nosupport/llvm-ve/lib/cmake/llvm" CACHE FILEPATH "Path of llvm-ve")
  set(NLC "/opt/nec/ve/nlc/2.0.0" CACHE FILEPATH "Path to nlc")

  find_package(LLVM REQUIRED CONFIG)

  add_library(blas_sequential STATIC IMPORTED)
  set_target_properties(blas_sequential PROPERTIES IMPORTED_LOCATION ${NLC}/lib/libblas_sequential.a)
  set_target_properties(blas_sequential PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${NLC}/include)

  add_library(asl_sequential STATIC IMPORTED)
  set_target_properties(asl_sequential PROPERTIES IMPORTED_LOCATION ${NLC}/lib/libasl_sequential.a)
  set_target_properties(asl_sequential PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${NLC}/include)

  set(CLANG_RUNTIME
    ${LLVM_INSTALL_PREFIX}/lib/clang/10.0.0/lib/linux/libclang_rt.builtins-ve.a)
endif()

message("llvm: ${LLVM_INSTALL_PREFIX}")
message("ncc:  ${NCC}")
message("nc++: ${NCXX}")
message("nlc: ${NLC}")
message("USE_INTRINSIC: ${USE_INTRINSIC}")
message("USE_VEDNN: ${USE_VEDNN}")

if (USE_VEDNN)
  if (USE_PREBUILT_VEDNN)
    add_library(vednn_openmp STATIC IMPORTED)
    set(VEDNN_DIR ${CMAKE_SOURCE_DIR}/extra/vednn)
    set_target_properties(vednn_openmp PROPERTIES IMPORTED_LOCATION ${VEDNN_DIR}/libvednn_openmp.a)
    set_target_properties(vednn_openmp PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${VEDNN_DIR}/include)
    message("Use prebuilt vednn in ${VEDNN_DIR}")
  else()
    add_subdirectory(libs/vednn EXCLUDE_FROM_ALL)
    set(VEDNN_DIR ${CMAKE_SOURCE_DIR}/libs/vednn)
    set_target_properties(vednn_openmp PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${VEDNN_DIR}/src)
    message("Build vednn in ${VEDNN_DIR}")
  endif()
endif()

add_subdirectory(vml)
add_subdirectory(src)
add_subdirectory(test)

if (BUILD_VEORUN_TF)
  add_custom_target(veorun_tf ALL
    COMMAND ${CMAKE_COMMAND} -E env CC=${NCC} CXX=${NCXX}
    CFLAGS="-no-proginf;-no-perfcnt;-fopenmp;-static"
    /opt/nec/ve/bin/mk_veorun_static -o veorun_tf
    --link-nosym=${CLANG_RUNTIME}
    --link-nosym=$<TARGET_FILE:vednn_openmp>
    --link-nosym=$<TARGET_FILE:vml>
    --link-nosym=$<TARGET_FILE:asl_sequential>
    --link-nosym=$<TARGET_FILE:blas_sequential>
    -lveio $<TARGET_FILE:vetfkernel>
    DEPENDS vednn_openmp vetfkernel vml)

  # for make clean
  set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES veorun_tf)
endif()
