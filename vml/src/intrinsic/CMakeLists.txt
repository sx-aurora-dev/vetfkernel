cmake_minimum_required(VERSION 3.0)

set(CMAKE_C_COMPILER ${LLVM_INSTALL_PREFIX}/bin/clang)
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -target ve -O3 -fno-vectorize -fno-slp-vectorize -fno-crash-diagnostics")

add_library(vml_intrinsic OBJECT
  #	BiasAdd.c
  #	BiasAddGrad.c
  	Add.c
  	Sub.c
  	Mul.c
  	Div.c
  	Sqrt.c
  	Rsqrt.c
  	Square.c
  	Neg.c
  #	Transpose.c
  	ApplyAdam.c
  #	Sum.c
  #	Mean.c
  	Tile.c
  )
