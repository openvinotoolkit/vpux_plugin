# ELF large memory requirement samples

These MLIR files are intended to be used to easily and quickly generate ELF binaries that require a large amount of memory to load (> 2 GB).

The resulting ELF binaries are not intended to be executed by the NPU. In the event that loading succeeds, execution will not be possible because no MappedInference is present.

## Test files
- baseline -> expected to pass
- Leon overflow -> expected to fail with Leon out of memory
- SHAVE overflow -> expected to fail with SHAVE out of memory

## Usage

> ./vpux-translate --vpu-arch=NPU37XX --export-ELF <input_mlir_file> > <output_elf_file>
