cmake_minimum_required(VERSION 3.0)

set(NCC "/opt/nec/ve/bin/ncc" CACHE FILEPATH "Path of ncc")
set(NCXX "/opt/nec/ve/bin/nc++" CACHE FILEPATH "Path of nc++")
set(LLVM_DIR "/opt/nec/nosupport/llvm-ve/lib/cmake/llvm" CACHE FILEPATH "Path of llvm-ve")

find_package(LLVM REQUIRED CONFIG)

set(BLAS /opt/nec/ve/nlc/2.0.0/lib/libblas_sequential.a)
set(ASL /opt/nec/ve/nlc/2.0.0/lib/libasl_sequential.a)

message("llvm: ${LLVM_INSTALL_PREFIX}")
message("ncc:  ${NCC}")
message("nc++: ${NCXX}")
message("blas: ${BLAS}")
message("asl:  ${ASL}")

option(USE_PROFILE "Enable profiler" OFF)

add_subdirectory(libs/vednn EXCLUDE_FROM_ALL)
add_subdirectory(vml)
add_subdirectory(src/intrinsic)
add_subdirectory(src)
add_subdirectory(test)

set(CLANG_RUNTIME ${LLVM_INSTALL_PREFIX}/lib/clang/10.0.0/lib/linux/libclang_rt.builtins-ve.a)

add_custom_target(veorun_tf ALL
  COMMAND ${CMAKE_COMMAND} -E env CC=${NCC} CXX=${NCXX} CFLAGS="-no-proginf;-no-perfcnt;-fopenmp;-static" /opt/nec/ve/bin/mk_veorun_static -o veorun_tf --link-nosym=${BLAS} --link-nosym=${CLANG_RUNTIME} --link-nosym=libs/vednn/src/libvednn_openmp.a --link-nosym=vml/src/libvml.a --link-nosym=${ASL} -lveio src/libvetfkernel.a 
  DEPENDS vednn_openmp vetfkernel vml)

# for make clean
set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES veorun_tf)
