This repository includes machine learning kernels for Vector Engine and its wrapper for TensorFlow for SX-Aurora.

# Prerequisites
- ncc/nc++
- llvm-ve
- vednn
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

Then, run perf test

    % python perf.py [-e <BUILD_DIR>/test/bench] test

See [here](doc/perf.md) for details.

## profiler

1. cmake with -DUSE_PROFILE=ON
2. run with VML_PROFILE=conv2d,conv2d_backprop_input
   

## VML API Document

The document is generated into `doc/api` directory.

    % doxygen

See in your browser. If you need http server, try

    % python3 -m http.server
