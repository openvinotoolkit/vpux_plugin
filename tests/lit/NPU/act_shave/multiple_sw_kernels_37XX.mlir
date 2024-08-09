//
// Copyright (C) 2021-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-translate --vpu-arch=%arch% --export-VPUIP -o %t %s
// RUN: flatc --raw-binary --json %vpuip_schema_file% -- %t
// RUN: FileCheck %s --input-file %basename_t.json
// RUN: rm %basename_t.json
// REQUIRES: arch-NPU37XX
//
// This file generates a blob with 3 activation shaves-kernels sigmoid + softmax + sigmoid
// demonstrate that the runtime and compiler cannot handle this.  It's also a lit test to help
// check for regressions in the VPUIP dialect.
//

module @Test attributes {VPU.arch = #VPU.arch_kind<NPU37XX>, VPU.compilationMode = #VPU.compilation_mode<ReferenceHW>} {

IE.MemoryResource 31457280 bytes of @DDR {VPU.bandwidth = 8 : i64, VPU.derateFactor = 6.000000e-01 : f64}
IE.ExecutorResource 1 of @DMA_NN
IE.TileResource 1 of @NCE {
    IE.MemoryResource 2097152 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    IE.ExecutorResource 1 of @SHAVE_UPA
    IE.ExecutorResource 1 of @SHAVE_ACT
    IE.ExecutorResource 1 of @DPU
}
IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "input" : tensor<1x1000xf16>
    }
    outputsInfo : {
        IE.DataInfo "sigmoid" : tensor<1x1000xf16>
    }

// Sub-module, which holds SW kernel declarations and optional implementations.
// Used to group those declarations for faster access.
module @VPU.SW {
    // The declaration should match C++ params structure in decomposed form.
    // `memref` will be translated to `MemRefData`, while raw scalars will be translated as is.
    func.func private @builtin_sigmoid(%input : memref<*xf16>, %output : memref<*xf16>)
        attributes {
            VPU.kernel_code = "activation_sigmoid.cpp",
            VPU.kernel_entry = "activation_sigmoid"
        }

    func.func private @builtin_softmax(%input : memref<*xf16>, %output : memref<*xf16>, %axis : i64)
        attributes {
            VPU.kernel_code = "softmax.cpp",
            VPU.kernel_entry = "softmax"
        }
}

func.func @main(%1: memref<1x1x1x1000xf16>, %2: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {

    %in_tile0_cmx  = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %buf0_tile0_cmx = VPURT.DeclareBuffer <CMX_NN> [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %buf1_tile0_cmx = VPURT.DeclareBuffer <CMX_NN> [0] <4000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %out_tile0_cmx = VPURT.DeclareBuffer <CMX_NN> [0] <6000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>

    %b0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    %b1 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
    %b2 = VPURT.ConfigureBarrier<2> -> !VPURT.Barrier
    %b3 = VPURT.ConfigureBarrier<3> -> !VPURT.Barrier

    VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%1 : memref<1x1x1x1000xf16>) outputs(%in_tile0_cmx : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    }

    // Genetic Kernel information for the scheduler.
    VPURT.Task waits(%b0 : !VPURT.Barrier) updates(%b1 : !VPURT.Barrier) {
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
                    @VPU.SW::@builtin_sigmoid            // The reference to the Kernel function.
                    inputs(%in_tile0_cmx as %arg0: memref<1x1x1x1000xf16, [@CMX_NN, 0]>)     // Inputs/outputs buffers for generic operation interface
                    outputs(%buf0_tile0_cmx as %arg1: memref<1x1x1x1000xf16, [@CMX_NN, 0]>)   // and their mapping to inner region.
                    on tile 0                           // The tile index to execute on.
        -> memref<1x1x1x1000xf16, [@CMX_NN, 0]> {

                VPUIP.SW.Kernel.run(%arg0, %arg1)
                    : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
                    , memref<1x1x1x1000xf16, [@CMX_NN, 0]>
        }
    }

    // Genetic Kernel information for the scheduler.
    VPURT.Task waits(%b1 : !VPURT.Barrier) updates(%b2 : !VPURT.Barrier) {
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
                    @VPU.SW::@builtin_softmax            // The reference to the Kernel function.
                    inputs(%buf0_tile0_cmx as %arg0: memref<1x1x1x1000xf16, [@CMX_NN, 0]>)     // Inputs/outputs buffers for generic operation interface
                    outputs(%buf1_tile0_cmx as %arg1: memref<1x1x1x1000xf16, [@CMX_NN, 0]>)   // and their mapping to inner region.
                    on tile 0                           // The tile index to execute on.
        -> memref<1x1x1x1000xf16, [@CMX_NN, 0]> {

                // The arguments mapping, the order must match the kernel parameter structure.
                VPUIP.SW.Kernel.run {attrs = [0]}(%arg0, %arg1)
                    : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
                    , memref<1x1x1x1000xf16, [@CMX_NN, 0]>
        }
    }

    // Genetic Kernel information for the scheduler.
    VPURT.Task waits(%b2 : !VPURT.Barrier) updates(%b3 : !VPURT.Barrier) {
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
                    @VPU.SW::@builtin_sigmoid            // The reference to the Kernel function.
                    inputs(%buf1_tile0_cmx as %arg0: memref<1x1x1x1000xf16, [@CMX_NN, 0]>)     // Inputs/outputs buffers for generic operation interface
                    outputs(%out_tile0_cmx as %arg1: memref<1x1x1x1000xf16, [@CMX_NN, 0]>)   // and their mapping to inner region.
                    on tile 0                           // The tile index to execute on.

        -> memref<1x1x1x1000xf16, [@CMX_NN, 0]> {

                // The arguments mapping, the order must match the kernel parameter structure.
                VPUIP.SW.Kernel.run(%arg0, %arg1)
                    : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
                    , memref<1x1x1x1000xf16, [@CMX_NN, 0]>
        }
    }

    VPURT.Task waits(%b3 : !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%out_tile0_cmx : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%2 : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>
    }

    return %2: memref<1x1x1x1000xf16>
}

}


// CHECK:    device: "VPUX37XX",
// CHECK:    act_kernel_runtime: {
// CHECK:      shaveStacks: [
// CHECK:        {
// CHECK:          name: "actSHAVE0_stack",
// CHECK:          locale: "GFEmbeddedKernel",
// CHECK:          referenced_data_size: 4096
// CHECK:        },
// CHECK:        {
// CHECK:          name: "actSHAVE1_stack",
// CHECK:          locale: "GFEmbeddedKernel",
// CHECK:          referenced_data_size: 4096
// CHECK:        },
// CHECK:        {
// CHECK:          name: "actSHAVE2_stack",
// CHECK:          locale: "GFEmbeddedKernel",
// CHECK:          referenced_data_size: 4096
// CHECK:        },
// CHECK:        {
// CHECK:          name: "actSHAVE3_stack",
// CHECK:          locale: "GFEmbeddedKernel",
// CHECK:          referenced_data_size: 4096
// CHECK:        }
// CHECK:      ],
// CHECK:      codeScratchBuffer: {
// CHECK:        name: "scratch_buffer",
// CHECK:        locale: "GFEmbeddedKernel",
// CHECK:        referenced_data_size: 65536
// CHECK:      }
// CHECK:  task_lists: [
// CHECK:          task_type: "NNDMATask",
// CHECK:          task: {
// CHECK:            src: {
// CHECK:              name: "input",
// CHECK:              dimensions: [
// CHECK:                1,
// CHECK:                1,
// CHECK:                1,
// CHECK:                1000
// CHECK:              ],
// CHECK:              data: {
// CHECK:                data_index: 0
// CHECK:              },
// CHECK:              locale: "ProgrammableInput",
// CHECK:              locale_index: [
// CHECK:                0
// CHECK:              ],
// CHECK:              data_dtype: "FP16",
// CHECK:              quant_zero: [
// CHECK:                0
// CHECK:              ],
// CHECK:              quant_mult: [
// CHECK:                1
// CHECK:              ],
// CHECK:              quant_shift: [
// CHECK:                0
// CHECK:              ],
// CHECK:              order: 4660,
// CHECK:              base_ptrs: [
// CHECK:              ],
// CHECK:              bit_strides: [
// CHECK:                16,
// CHECK:                16000,
// CHECK:                16000,
// CHECK:                16000,
// CHECK:                16
// CHECK:              ]
// CHECK:            },
// CHECK:            dst: {
// CHECK:              name: "temp-0",
// CHECK:              dimensions: [
// CHECK:                1,
// CHECK:                1,
// CHECK:                1,
// CHECK:                1000
// CHECK:              ],
// CHECK:              data: {
// CHECK:                data_index: 0
// CHECK:              },
// CHECK:              locale: "VPU_CMX_NN",
// CHECK:              locale_index: [
// CHECK:                0
// CHECK:              ],
// CHECK:              data_dtype: "FP16",
// CHECK:              quant_zero: [
// CHECK:                0
// CHECK:              ],
// CHECK:              quant_mult: [
// CHECK:                1
// CHECK:              ],
// CHECK:              quant_shift: [
// CHECK:                0
// CHECK:              ],
// CHECK:              order: 4660,
// CHECK:              base_ptrs: [
// CHECK:              ],
// CHECK:              bit_strides: [
// CHECK:                16,
// CHECK:                16000,
// CHECK:                16000,
// CHECK:                16000,
// CHECK:                16
// CHECK:              ]
// CHECK:          nodeID: 8,
// CHECK:          associated_barriers: {
// CHECK:            wait_barriers: [
// CHECK:              3
// CHECK:            ],
// CHECK:            virtual_wait_barriers: [
// CHECK:              3
// CHECK:            ]
// CHECK:          },
// CHECK:          task_type: "NNDMATask",
// CHECK:          task: {
// CHECK:            src: {
// CHECK:              name: "temp-3",
// CHECK:              dimensions: [
// CHECK:                1,
// CHECK:                1,
// CHECK:                1,
// CHECK:                1000
// CHECK:              ],
// CHECK:              data: {
// CHECK:                data_index: 6000
// CHECK:              },
// CHECK:              locale: "VPU_CMX_NN",
// CHECK:              locale_index: [
// CHECK:                0
// CHECK:              ],
// CHECK:              data_dtype: "FP16",
// CHECK:              quant_zero: [
// CHECK:                0
// CHECK:              ],
// CHECK:              quant_mult: [
// CHECK:                1
// CHECK:              ],
// CHECK:              quant_shift: [
// CHECK:                0
// CHECK:              ],
// CHECK:              order: 4660,
// CHECK:              base_ptrs: [
// CHECK:              ],
// CHECK:              bit_strides: [
// CHECK:                16,
// CHECK:                16000,
// CHECK:                16000,
// CHECK:                16000,
// CHECK:                16
// CHECK:              ]
// CHECK:            },
// CHECK:            dst: {
// CHECK:              name: "sigmoid",
// CHECK:              dimensions: [
// CHECK:                1,
// CHECK:                1,
// CHECK:                1,
// CHECK:                1000
// CHECK:              ],
// CHECK:              data: {
// CHECK:                data_index: 0
// CHECK:              },
// CHECK:              locale: "ProgrammableOutput",
// CHECK:              locale_index: [
// CHECK:                0
// CHECK:              ],
// CHECK:              data_dtype: "FP16",
// CHECK:              quant_zero: [
// CHECK:                0
// CHECK:              ],
// CHECK:              quant_mult: [
// CHECK:                1
// CHECK:              ],
// CHECK:              quant_shift: [
// CHECK:                0
// CHECK:              ],
// CHECK:              order: 4660,
// CHECK:              base_ptrs: [
// CHECK:              ],
// CHECK:              bit_strides: [
// CHECK:                16,
// CHECK:                16000,
// CHECK:                16000,
// CHECK:                16000,
// CHECK:                16
// CHECK:              ]
// CHECK:    {
// CHECK:      content: [
// CHECK:          task_type: "ActKernelTask",
// CHECK:          task: {
// CHECK:            kernel: {
// CHECK:              kernelText: {
// CHECK:                name: "builtin_sigmoid",
// CHECK:                locale: "GFEmbeddedKernel",
// CHECK:                referenced_data_size: {{[1-9][0-9]+}}
// CHECK:              }
// CHECK:            },
// CHECK:            invocations: [
// CHECK:              {
// CHECK:                associatedBarriers: {
// CHECK:                  wait_barriers: [
// CHECK:                    0
// CHECK:                  ],
// CHECK:                  update_barriers: [
// CHECK:                    1
// CHECK:                  ],
// CHECK:                  virtual_wait_barriers: [
// CHECK:                    0
// CHECK:                  ],
// CHECK:                  virtual_update_barriers: [
// CHECK:                    1
// CHECK:                  ]
// CHECK:                },
// CHECK:                dataSection: {
// CHECK:                  name: "builtin_sigmoid_invo",
// CHECK:                  locale: "GFEmbeddedKernel",
// CHECK:                },
// CHECK:                invocationArgs: {
// CHECK:                  name: "builtin_sigmoid_invo",
// CHECK:                  locale: "GFEmbeddedKernel",
// CHECK:                  referenced_data_size: 168
// CHECK:                }

// CHECK:          name: "",
// CHECK:          nodeID: 6,
// CHECK:          associated_barriers: {
// CHECK:            wait_barriers: [
// CHECK:              1
// CHECK:            ],
// CHECK:            update_barriers: [
// CHECK:              2
// CHECK:            ],
// CHECK:            virtual_wait_barriers: [
// CHECK:              1
// CHECK:            ],
// CHECK:            virtual_update_barriers: [
// CHECK:              2
// CHECK:            ]
// CHECK:          },
// CHECK:          task_type: "ActKernelTask",
// CHECK:          task: {
// CHECK:            kernel: {
// CHECK:              kernelText: {
// CHECK:                name: "builtin_softmax",
// CHECK:                locale: "GFEmbeddedKernel",
// CHECK:                referenced_data_size: {{[1-9][0-9]+}}
// CHECK:              }
// CHECK:            },
// CHECK:            invocations: [
// CHECK:              {
// CHECK:                associatedBarriers: {
// CHECK:                  wait_barriers: [
// CHECK:                    1
// CHECK:                  ],
// CHECK:                  update_barriers: [
// CHECK:                    2
// CHECK:                  ],
// CHECK:                  virtual_wait_barriers: [
// CHECK:                    1
// CHECK:                  ],
// CHECK:                  virtual_update_barriers: [
// CHECK:                    2
// CHECK:                  ]
// CHECK:                },
// CHECK:                dataSection: {
// CHECK:                  name: "builtin_softmax_invo",
// CHECK:                  locale: "GFEmbeddedKernel",
// CHECK:                },
// CHECK:                invocationArgs: {
// CHECK:                  name: "builtin_softmax_invo",
// CHECK:                  locale: "GFEmbeddedKernel",
// CHECK:                  referenced_data_size: {{[1-9][0-9]+}}
// CHECK:                }

// CHECK:          nodeID: 7,
// CHECK:          associated_barriers: {
// CHECK:            wait_barriers: [
// CHECK:              2
// CHECK:            ],
// CHECK:            update_barriers: [
// CHECK:              3
// CHECK:            ],
// CHECK:            virtual_wait_barriers: [
// CHECK:              2
// CHECK:            ],
// CHECK:            virtual_update_barriers: [
// CHECK:              3
// CHECK:            ]
// CHECK:          },
// CHECK:          task_type: "ActKernelTask",
// CHECK:          task: {
// CHECK:            kernel: {
// CHECK:              kernelText: {
// CHECK:                name: "builtin_sigmoid",
// CHECK:                locale: "GFEmbeddedKernel",
// CHECK:                referenced_data_size: {{[1-9][0-9]+}}
// CHECK:              }
// CHECK:            },
// CHECK:            invocations: [
// CHECK:              {
// CHECK:                associatedBarriers: {
// CHECK:                  wait_barriers: [
// CHECK:                    2
// CHECK:                  ],
// CHECK:                  update_barriers: [
// CHECK:                    3
// CHECK:                  ],
// CHECK:                  virtual_wait_barriers: [
// CHECK:                    2
// CHECK:                  ],
// CHECK:                  virtual_update_barriers: [
// CHECK:                    3
// CHECK:                  ]
// CHECK:                },
// CHECK:                dataSection: {
// CHECK:                  name: "builtin_sigmoid_invo_1",
// CHECK:                  locale: "GFEmbeddedKernel",
// CHECK:                },
// CHECK:                invocationArgs: {
// CHECK:                  name: "builtin_sigmoid_invo_1",
// CHECK:                  locale: "GFEmbeddedKernel",
// CHECK:                  referenced_data_size: 168
// CHECK:                }
