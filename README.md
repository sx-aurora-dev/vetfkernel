See [tensorflow/README_ve.md](https://github.com/sx-aurora-dev/tensorflow/blob/develop/README_ve.md)

# Prerequisites

- ncc/nc++
- llvm-ve

# build

## build with prebuilt vednn in extra/vednn directory

    % mkdir build
    % (cd build && cmake3 -DUSE_PREBUILT_VEDNN=ON ..)

## build vednn and vetfkernel

    % git clone <vednn> libs/vednn
    % mkdir build
    % (cd build && cmake3 ..)

## test

    % ./build/test/test
    % ./build/test/bench
    % python test/python/avgpoolgrad.py

or

    % ./test.sh [BUILD_DIR]


## profiler

1. cmake with -DUSE_PROFILE=ON
2. run with VML_PROFILE=conv2d,conv2d_backprop_input
   
