# MLIR Operational Tools

A collection of tools for working with verbose IR dumped from passes in VPUx compiler.
The goal of these tools is to automate common tasks in development flow and improve productivity.

### Example

Having an MLIR file as a compilation result of any particular model, you could just modify some operations, like move them or rotate in order, and starting from this amended pass continue the compilation process and finally get an ELF-file.
Given ELF-file could be used for launching on a real device to estimate performance improvements (or diminishing) regarding applied changes. Particularly, it might be used to evaluate necessity in implementing the `BatchUnrolling` for various operations.

## How-To

The additional utils `compile-tool` is required to embrace that goal. Additionally, NPU plugin *MUST* be configured with `-DENABLE_DEVELOPER_BUILD=ON`
The following commands were tested on *Ubuntu*. Please, don't hesitate to update this file if you have a chance to try these on different OS.

1. Compile a model and dump the pass you're interested in (i.e. `my-pass`):

    `IE_NPU_PRINT_AS_TEXTUAL_PIPELINE_FILE=pipeline.txt IE_NPU_IR_PRINTING_FILE=output_my_pass_full.mlir IE_NPU_PRINT_FULL_IR=1 IE_NPU_IR_PRINTING_ORDER=After IE_NPU_PRINT_FULL_CONSTANT=1 IE_NPU_IR_PRINTING_FILTER=.*my-pass.* compile_tool -m <YOUR model path> -d <YOUR NPU type> -c <YOUR cfg_file>`

    Please, ensure that `IE_NPU_PRINT_FULL_CONSTANT=1` was set and enabled. In general circumstances, the subsequent steps will require full presence of all constants in the processed MLIR

2. Determine which pass you are going to intervent in, for example the pass `my-pass` and extract a list of passes applied by compiler after that. It is usually a long list, thus we remembered that in the file `going_after.passes`:

    `mlir_passes_cutter.py -s my-pass -o 1 < pipeline.txt > going_after.passes`

    The option `-s` determines which pass we cut pipeline until (it's `my-pass`);
    The option `-o` tells the position in the pipeline which we want to apply it starting from the our pass `my-pass`. We want to apply the pipeline startig from the next pass, thus `-o` has value `1`

3. Having all necessarily preparations done, you are free to modify IR in the generated MLIR dump: `output_my_pass_full.mlir`

4. We can now apply pipeline from earlier to our modified IR to generate an ELF file:

```
vpux-opt --vpu-arch=<YOUR device> -pass-pipeline="`cat going_after.passes`" -o applied_pipeline.mlir output_my_pass_full.mlir
vpux-opt --vpu-arch=<YOUR device> --lower-VPUIP-to-ELF -o elf.mlir applied_pipeline.mlir
vpux-translate --vpu-arch=<YOUR device> --export-ELF -o model.blob elf.mlir
```

5. Ultimately, the output file `model.blob` is ready to be used in on device inference
