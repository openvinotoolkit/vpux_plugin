# Google benchmarks

Micro benchmarks can be used to estimate the performance impact of individual functions.

Follow these simple steps to gather the performance data:
* Add another C++ source file to `src` directory.
* Compose a function. It must accept `benchmark::State&` parameter.
* Wrap the function into a `BENCHMARK` macro.
* Build the project with `ENABLE_NPU_MICRO_BENCHMARKS` option set to `ON`
* Run `npuMicroBenchmark` binary.

Possible output:
```sh
-------------------------------------------------------------
Benchmark                   Time             CPU   Iterations
-------------------------------------------------------------
BM_GetValues       1354994342 ns   1354837630 ns            1
BM_GetTmpBuff       136714917 ns    136682567 ns            4
BM_FuseMulAdd       138881573 ns    138875599 ns            4
BM_FuseMulAddDtype  126526909 ns    126527880 ns            4
```
