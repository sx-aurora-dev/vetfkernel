cmake_minimum_required(VERSION 3.0)

set(CMAKE_C_COMPILER ${LLVM_INSTALL_PREFIX}/bin/clang)
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -target ve -O3 -fno-vectorize -fno-slp-vectorize -fno-crash-diagnostics")

add_library(vetfkernel_intrinsic OBJECT
	BiasAdd.c
	BiasAddGrad.c
	Add.c
	Sub.c
	Mul.c
	Div.c
	Sqrt.c
	Rsqrt.c
	Square.c
	Neg.c
	Transpose.c
	ApplyAdam.c
	Sum.c
	Mean.c
	Tile.c)


add_library(vetfkernel_old_intrinsic OBJECT
	libvetfkernel_BiasAdd.c
	libvetfkernel_BiasAddGrad.c
	libvetfkernel_Add.c
	libvetfkernel_Sub.c
	libvetfkernel_Mul.c
	libvetfkernel_Div.c
	libvetfkernel_Sqrt.c
	libvetfkernel_Rsqrt.c
	libvetfkernel_Square.c
	libvetfkernel_Neg.c
	libvetfkernel_Transpose.c
	libvetfkernel_ApplyAdam.c
	libvetfkernel_Sum.c
	libvetfkernel_Mean.c
	libvetfkernel_Tile.c)

