# Performance test

`perf.py` is a script to run benchmark and check a result with latest result in
database.

## Design

A benchmark program is `bench` that is build in `<build direcotry>/test/bench`.

A result of benchmark is a simple text file like below. '# ts ..' line is
timestamp and mandatory.

```
Add                                                                                 1.151 ms
Sub                                                                                 1.063 ms
(snip)
Conv2D-32x256x34x34-512x256x4x4-32x512x31x31-1x1-1x1-0x0-1-1                      114.551 ms
Conv2DBackpropInput-32x1024x16x16-1024x256x4x4-32x256x32x32-2x2-1x1-1x1-1-1       121.133 ms
# ts: 2019-12-11T16:39:36+0900
# hostname: aurora-ds01
# user: ishizaka
```

A database is group of result files typically stored in `prefdb` directory.
A database should be shared among team members through git.

## Typical usage

Run benchmark and compare result with latest result in a database. Benchmark
result is stored in `hoge` file.

```
% python perf.py -e build-2.4.1/test/bench test -o hoge
```

If ok, store a result in a database. filename in a database is automatically
created from timestamp of a result.

```
% python perf.py store hoge
```

Then add new result into git repository.

## Other usages

```
   # Run benchmark, check and write result to <filename>
   % python perf.py [-e executable] test [-o filename]
   
   # Store <filename> to database
   % python perf.py store <filename>

   # Show results in a database
   % python perf.py show [-l num]

   # Compare latest two results in a database
   % python perf.py check [-v]

   # Compare a result in <filename> with latest result in a database
   % python perf.py check [-v] -a <filename>

   # Compare two results
   % python perf.py check [-v] -a <filename> -b <filename>

   # Run benchmark and output result
   % python perf.py run
```

## Options

```
   -e <filename>    benchmark executable [build/test/bench]
   -d <directory>   directory of database [perfdb]
   -t <threshold>   threshold in percent [5]
   -o <filename>    filename to save a benchmark result
   -v               increses verbose level
```

## Example

```
% python perf.py -e build-2.4.1/test/bench test -o hoge
a: None (2019-12-11 17:00:37+09:00)
b: perfdb/2019-12-11T07:39:36+0000 (2019-12-11 16:39:36+09:00)
All Results:
benchmark                                                                        a        b        diff     diff%
Add                                                                                 1.061    1.151   -0.090   -7.819 %
Sub                                                                                 1.046    1.063   -0.017   -1.599 %
Mul                                                                                 0.265    0.267   -0.002   -0.749 %
Div                                                                                 1.497    1.465    0.032    2.184 %
Mean                                                                                0.767    0.773   -0.006   -0.776 %
Sum                                                                                 0.751    0.746    0.005    0.670 %
Neg                                                                                 0.178    0.178    0.000    0.000 %
Rsqrt                                                                               0.178    0.177    0.001    0.565 %
Sqrt                                                                                0.531    0.523    0.008    1.530 %
Square                                                                              0.179    0.180   -0.001   -0.556 %
BiasAdd(NHWC)                                                                       0.522    0.531   -0.009   -1.695 %
BiasAdd(NCHW)                                                                       0.158    0.158    0.000    0.000 %
BiasAddGrad(NHWC)                                                                   3.791    3.731    0.060    1.608 %
BiasAddGrad(NCHW)                                                                   0.330    0.331   -0.001   -0.302 %
Tile                                                                                0.066    0.066    0.000    0.000 %
Transpose(0231)                                                                     0.230    0.228    0.002    0.877 %
Transpose(0312)                                                                     0.435    0.417    0.018    4.317 %
ApplyAdam                                                                           0.629    0.626    0.003    0.479 %
Conv2D-32x256x34x34-512x256x4x4-32x512x31x31-1x1-1x1-0x0-1-1                      114.541  114.551   -0.010   -0.009 %
Conv2DBackpropInput-32x1024x16x16-1024x256x4x4-32x256x32x32-2x2-1x1-1x1-1-1       121.149  121.133    0.016    0.013 %
0 errors found
write result to hoge

% python perf.py -e build-2.4.1/test/bench store hoge
Copy hoge to perfdb/2019-12-11T08:00:37+0000
```
