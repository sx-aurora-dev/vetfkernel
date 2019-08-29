See [tensorflow/README_ve.md](https://github.com/sx-aurora-dev/tensorflow/blob/develop/README_ve.md)

# Prerequisites

- ncc/nc++
- llvm-ve

# build

   % mkdir build
   % (cd build && cmake3 -DNCC=/opt/nec/ve/bin/ncc-2.3.1 -DNCXX=/opt/nec/ve/bin/nc++-2.3.1 ..)


## test

    % ./build/test/test
    % ./build/test/bench
    % python test/python/avgpoolgrad.py

or

    % ./test.sh [BUILD_DIR]
