# Prepare profiling LIT test

To prepare IR for [LIT test](tests/lit/utils/profiling-*.mlir.txt)

- choose any model(prefarable tiny), for example age-gender-recognition-retail-0013 and import IE IR using

```sh
vpux-translate --import-IE --mlir-print-debuginfo --vpux-profiling $PATH_TO_XML -o tmp.mlir
```

- Then shrink IR to few layers(e.g. to first 3 layers), adjust model outputs.
- Using `vpux-opt --default-hw-mode="--vpu-arch=NPU37XX profiling=true" --mlir-print-debuginfo tmp.mlir -o tmp-ex.mlir` generate IR for LIT test.

To generate `profiling-0.bin` run InferenceManagerDemo with blob from lit/utils/profiling-*.mlir.txt and copy generated profiling-0.bin form inference results.

Most likely the LIT test have to be updated with new values as the real-time measurements can vary from run to run.
