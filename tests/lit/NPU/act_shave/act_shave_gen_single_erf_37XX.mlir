//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" %s | vpux-translate --vpu-arch=%arch% --export-VPUIP -o %t
// RUN: flatc --raw-binary --json %vpuip_schema_file% -- %t
// RUN: FileCheck %s --input-file %basename_t.json
// RUN: rm %basename_t.json
// REQUIRES: arch-VPUX37XX
//
// This file generates a blob with erf activation shave
// demonstrate that the runtime cannot handle this.  It's also a lit test to help
// check for regressions in the VPUIP dialect.
//

module @Test {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "input" : tensor<1x1000xf16>
    }
    outputsInfo : {
        IE.DataInfo "erf" : tensor<1x1000xf16>
    }

VPURT.SW.Runtime
    entryPoint: @VPU.SW::@runtime
    stack_configuration: [
        4096,  // Size in bytes for the actSHAVE0 in the first tile.
        4096,  // Size in bytes for the actSHAVE1 in the first tile.
        4096,  // Size in bytes for the actSHAVE2 in the second tile.
        4096   // Size in bytes for the actSHAVE3 in the second tile.
    ]


// Sub-module, which holds SW kernel declarations and optional implementations.
// Used to group those declarations for faster access.
module @VPU.SW {
    // The declaration should match C++ params structure in decomposed form.
    // `memref` will be translated to `MemRefData`, while raw scalars will be translated as is.
    func.func private @builtin_erf(%input : memref<*xf16>, %output : memref<*xf16>)
        attributes {
            VPU.kernel_code = "activation_erf.cpp",
            VPU.kernel_entry = "activation_erf"
        }

    // management kernel definition
    func.func private @runtime()
        attributes {
            VPU.kernel_code = "nnActEntry"
        }
}



func.func @main(%1: memref<1x1x1x1000xf16>, %2: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {

    %in_tile0_cmx  = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %out_tile0_cmx = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>

    %b0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    %b1 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier

    VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%1 : memref<1x1x1x1000xf16>) outputs(%in_tile0_cmx : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    }

    // Genetic Kernel information for the scheduler.
    VPURT.Task waits(%b0  : !VPURT.Barrier) updates(%b1  : !VPURT.Barrier) {
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
                    @VPU.SW::@builtin_erf            // The reference to the Kernel function.
                    inputs(%in_tile0_cmx as %arg0: memref<1x1x1x1000xf16, [@CMX_NN, 0]>)     // Inputs/outputs buffers for generic operation interface
                    outputs(%out_tile0_cmx as %arg1: memref<1x1x1x1000xf16, [@CMX_NN, 0]>)   // and their mapping to inner region.
                    on tile 0                           // The tile index to execute on.

        -> memref<1x1x1x1000xf16, [@CMX_NN, 0]> {

                // The arguments mapping, the order must match the kernel parameter structure.
                VPUIP.SW.Kernel.run(%arg0, %arg1)
                    : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
                    , memref<1x1x1x1000xf16, [@CMX_NN, 0]>
        }
    }

    VPURT.Task waits(%b1 : !VPURT.Barrier) {
        %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%out_tile0_cmx : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%2 : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>
    }
    return %2: memref<1x1x1x1000xf16>

}


}

// CHECK:   identifier: "Test"

// CHECK:   net_input: [
// CHECK:     {
// CHECK:       name: "input",
// CHECK:       dimensions: [
// CHECK:           1,
// CHECK:           1,
// CHECK:           1,
// CHECK:           1000
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableInput",
// CHECK:       data_dtype: "FP16",
// CHECK:       bit_strides: [
// CHECK:           16,
// CHECK:           16000,
// CHECK:           16000,
// CHECK:           16000,
// CHECK:           16
// CHECK:       ]
// CHECK:     }
// CHECK:   ],

// CHECK:   net_output: [
// CHECK:     {
// CHECK:       name: "erf",
// CHECK:       dimensions: [
// CHECK:         1,
// CHECK:         1,
// CHECK:         1,
// CHECK:         1000
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableOutput",
// CHECK:       data_dtype: "FP16",
// CHECK:       bit_strides: [
// CHECK:         16,
// CHECK:         16000,
// CHECK:         16000,
// CHECK:         16000,
// CHECK:         16
// CHECK:       ]
// CHECK:     }
// CHECK:   ],

// CHECK:   task_count: 5,

// CHECK:   options: [
// CHECK:   ],

// CHECK:   in_tensor_desc: [
// CHECK:     {
// CHECK:       name: "input",
// CHECK:       dimensions: [
// CHECK:         1,
// CHECK:         1000
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableInput",
// CHECK:       data_dtype: "FP16",
// CHECK:       bit_strides: [
// CHECK:         16,
// CHECK:         16000,
// CHECK:         16
// CHECK:       ]
// CHECK:     }
// CHECK:   ],

// CHECK:   out_tensor_desc: [
// CHECK:     {
// CHECK:       name: "erf",
// CHECK:       dimensions: [
// CHECK:         1,
// CHECK:         1000
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableOutput",
// CHECK:       data_dtype: "FP16",
// CHECK:       bit_strides: [
// CHECK:         16,
// CHECK:         16000,
// CHECK:         16
// CHECK:       ]
// CHECK:     }
// CHECK:   ]


// CHECK:    device: "VPUX37XX",
// CHECK:    act_kernel_runtime: {
// CHECK:        shaveStacks: [
// CHECK:          {
// CHECK:            name: "actSHAVE0_stack",
// CHECK:            locale: "GFEmbeddedKernel",
// CHECK:            referenced_data_size: 4096
// CHECK:          },
// CHECK:          {
// CHECK:            name: "actSHAVE1_stack",
// CHECK:            locale: "GFEmbeddedKernel",
// CHECK:            referenced_data_size: 4096
// CHECK:          }
// CHECK:        ]
// CHECK:      kernel: {
// CHECK:        kernelText: {
// CHECK:          name: "nnActEntry",
// CHECK:          locale: "GFEmbeddedKernel",
// CHECK:          referenced_data_size: {{[1-9][0-9]+}}
// CHECK:        },
// CHECK:        globalArgs: {
// CHECK:          name: "nnActEntry.data",
// CHECK:          locale: "GFEmbeddedKernel",
// CHECK:        }
// CHECK:      }
// CHECK:    }

// CHECK:   task_lists: [
// CHECK:      {
// CHECK:        content: [
// CHECK:          {
// CHECK:            name: "",
// CHECK:            nodeID: 3,
// CHECK:            associated_barriers: {
// CHECK:              wait_barriers: [
// CHECK:                0
// CHECK:              ],
// CHECK:              update_barriers: [
// CHECK:                1
// CHECK:              ],
// CHECK:              virtual_wait_barriers: [
// CHECK:                0
// CHECK:              ],
// CHECK:              virtual_update_barriers: [
// CHECK:                1
// CHECK:              ]
// CHECK:            },
// CHECK:            task_type: "ActKernelTask",
// CHECK:            task: {
// CHECK:              kernel: {
// CHECK:                kernelText: {
// CHECK:                  name: "builtin_erf",
// CHECK:                  locale: "GFEmbeddedKernel",
// CHECK:                  referenced_data_size: {{[1-9][0-9]+}}
// CHECK:                }
// CHECK:              },
// CHECK:              invocations: [
// CHECK:                {
// CHECK:                  associatedBarriers: {
// CHECK:                    wait_barriers: [
// CHECK:                      0
// CHECK:                    ],
// CHECK:                    update_barriers: [
// CHECK:                      1
// CHECK:                    ]
// CHECK:                  },
// CHECK:                  dataSection: {
// CHECK:                    name: "builtin_erf_invo",
// CHECK:                    locale: "GFEmbeddedKernel",
// CHECK:                  },
// CHECK:                  invocationArgs: {
// CHECK:                    name: "builtin_erf_invo",
// CHECK:                    locale: "GFEmbeddedKernel",
// CHECK:                    referenced_data_size: 168
// CHECK:                  }
// CHECK:                }
// CHECK:              ]
// CHECK:            }
// CHECK:          }
// CHECK:        ]
// CHECK:      }
// CHECK:   ],


// CHECK:   kernel_data: [
// CHECK:      ]
