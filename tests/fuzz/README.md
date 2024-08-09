# VPU Fuzz Testing

This test suite contains fuzzing tests for VPU Plugin. These tests aim to identify potential issues by feeding random or unexpected inputs to the target code. Examples of potential issues can be crashes, memory leaks, buffer overflows etc.

To achieve this, [libFuzzer](https://llvm.org/docs/LibFuzzer.html) is utilized coupled with sanitizers, such as [AddressSanitizer](https://clang.llvm.org/docs/AddressSanitizer.html). libFuzzer is a coverage-based fuzzing tool, which will mutate the input data so that it increases the code coverage of the target. The relevant inputs will be stored in the given corpus directory so that it can be reused later.

## Prerequisites

The following components are necessary:

- `clang`
- `libFuzzer` (included with `clang`)

## Building fuzz tests

In order to build and run the fuzz tests, the plugin has to be built using `clang` compiler since `libFuzzer` depends on it. A debug build is recommended.

To build the tests, enable the `ENABLE_NPU_FUZZ_TESTS` option. Additionally, the `ENABLE_FUZZING` and `ENABLE_SANITIZER` options should be enabled as they will enable the fuzzing and address sanitizers, as well as the instrumentation necessary to analyze code coverage. If these two options are not enabled, the tests can still be build, but they will mostly function as crash reproducers (e.g. for debugging or resolving found issues).

The recommended way to build the project is by using the `npuFuzzingBuild` preset. Running the following from the root of the project will configure all the necessary values and perform the build:

```sh
OPENVINO_HOME=/path/to/openvino cmake --preset npuFuzzingBuild .
cd ./build-x86_64/Debug
cmake --build .
```

## Running fuzz tests

Each test has its own executable which can be found in the OpenVINO binaries directory. If the test contains a `seeds` directory, it is recommended to use it as a starting point for the search as they contain relevant inputs for the target and can decrease the total time spent in increasing the code coverage.

The tests can be executed as follows:

```sh
mkdir -p fuzz/CORPUS
cd fuzz

# Example: ../fuzz_pipeline_default_hw_mode CORPUS /path/to/vpux-plugin/tests/fuzz/src/pipeline_default_hw_mode/seeds
/path/to/fuzz/test/executable CORPUS /path/to/seeds
```

**Note:** in case you encounter an error saying `ASan runtime does not come first in initial library list; you should either link runtime to your application or manually preload it with LD_PRELOAD`, you may need to manually provide the path to the ASAN library via `LD_PRELOAD`:

```sh
# Example path for clang-18
LD_PRELOAD="/usr/lib/llvm-18/lib/clang/18/lib/linux/libclang_rt.asan-x86_64.so" /path/to/fuzz/test/executable CORPUS /path/to/seeds
```

Creating a dedicated directory in which to run the test is recommended as it will be populated with artifacts, such as crash reproducers. An empty corpus directory is also recommended when starting from the seeds or from scratch, since it will be populated with all the relevant found inputs. Using the seeds directory directly or an existing corpus is also possible (e.g. `/path/to/fuzz/test/executable CORPUS`), but please be aware that the content of the directory will be changed by the test run.

Relevant options when running a fuzz test:
- `-max_total_time=` can limit the time spent executing the test as it can run indefinitely by default
- `-jobs=` can select how many jobs to run until completion, as only one job will run by default
  - this option can be coupled with `-workers=` to specify how many jobs should run in parallel as by default it uses `min(jobs, NumberOfCpuCores()/2)`
- `rss_limit_mb=` can configure the memory usage limit, which is by default `2048MB`; setting it to zero will disable the limit

## Test code coverage

Each run of a fuzz test will create a raw profile file of code coverage. Once you have a corpus of inputs, you can pass them through the fuzz test again in order to generate the raw profiling file that includes the total code coverage for all inputs. This file can then be used to generate the code coverage report using the following commands:

```sh
llvm-profdata merge -sparse *.profraw -o default.profdata && \
llvm-cov show /path/to/fuzz/test/executable -instr-profile=default.profdata -format=html -output-dir=PROFILE
```

## Reproducing findings

Once an issue is identified during a test run, the test halts and stores the input data that reproduced the issue as a file. The issue can be reproduced by passing the reproducer to the test:

```sh
/path/to/fuzz/test/executable crash-050c579aadd58c2be20c0a302a3acfabe3a43062
```

It is not necessary for the binaries to be built using `ENABLE_FUZZING` in order to run the reproducer.
