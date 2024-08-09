//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" %s | vpux-translate --vpu-arch=%arch% --export-VPUIP -o %t
// RUN: flatc --raw-binary --json %vpuip_schema_file% -- %t
// RUN: FileCheck %s --input-file %basename_t.json
// RUN: rm %basename_t.json
// REQUIRES: arch-NPU37XX
//
// This file generates a blob with gather_elements activation shave
// demonstrate that the runtime cannot handle this.  It's also a lit test to help
// check for regressions in the VPUIP dialect.
//

module @gatherEl {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "Parameter_143" : tensor<2x2xf32>
    }
    outputsInfo : {
        IE.DataInfo "GatherElements" : tensor<2x2xf32>
    }

VPURT.SW.Runtime
    entryPoint: @VPU.SW::@runtime
    stack_configuration: [
        4096, // Size in bytes for the SHAVEs in the first tile.
        4096, // Size in bytes for the SHAVEs in the second tile.
        4096, // Size in bytes for the SHAVEs in the third tile.
        4096  // Size in bytes for the SHAVEs in the fourth tile.
    ]

// Sub-module, which holds SW kernel declarations and optional implementations.
// Used to group those declarations for faster access.
module @VPU.SW {
    // The declaration should match C++ params structure in decomposed form.
    // `memref` will be translated to `MemRefData`, while raw scalars will be translated as is.
    func.func private @builtin_GatherElements(%input : memref<*xf16>, %input2 : memref<*xsi32>, %output : memref<*xf16>)
        attributes {
            VPU.kernel_code = "gather_elements.cpp",
            VPU.kernel_entry = "gather_elements"
        }

    func.func private @builtin_Convert(%input3: memref<*xf32>, %output2: memref<*xf16>)
        attributes {
            VPU.kernel_code = "convert.cpp",
            VPU.kernel_entry = "convert"
        }

    // management kernel definition
    func.func private @runtime()
        attributes {
            VPU.kernel_code = "nnActEntry"
        }
}

func.func @main(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>) -> memref<2x2xf32> {
    %cst = const.Declare memref<2x2xsi32> = dense<[[0, 0], [1, 1]]> : tensor<2x2xsi32>
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<2x2xf32, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <128> -> memref<2x2xf16, [@CMX_NN, 0]>
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <64> -> memref<2x2xsi32, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<2x2xf16, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <64> -> memref<2x2xf32, [@CMX_NN, 0]>

    %5 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    VPURT.Task updates(%5 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %11 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<2x2xf32>) outputs(%0 : memref<2x2xf32, [@CMX_NN, 0]>) -> memref<2x2xf32, [@CMX_NN, 0]>
    }

    %6 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    VPURT.Task waits(%5 : !VPURT.Barrier) updates(%6 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convert inputs(%0 as %arg2: memref<2x2xf32, [@CMX_NN, 0]>) outputs(%1 as %arg3: memref<2x2xf16, [@CMX_NN, 0]>) on tile 0 -> memref<2x2xf16, [@CMX_NN, 0]>{
            VPUIP.SW.Kernel.run(%arg2, %arg3) : memref<2x2xf32, [@CMX_NN, 0]>, memref<2x2xf16, [@CMX_NN, 0]>
        }
    }

    %7 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    VPURT.Task waits(%5 : !VPURT.Barrier) updates(%7 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %11 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst : memref<2x2xsi32>) outputs(%2 : memref<2x2xsi32, [@CMX_NN, 0]>) -> memref<2x2xsi32, [@CMX_NN, 0]>
    }

    %8 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    VPURT.Task waits(%6, %7 : !VPURT.Barrier, !VPURT.Barrier) updates(%8 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_GatherElements inputs(%1 as %arg2: memref<2x2xf16, [@CMX_NN, 0]>, %2 as %arg3: memref<2x2xsi32, [@CMX_NN, 0]>) outputs(%3 as %arg4: memref<2x2xf16, [@CMX_NN, 0]>) on tile 0 -> memref<2x2xf16, [@CMX_NN, 0]>{
            VPUIP.SW.Kernel.run {attrs = [0]}(%arg2, %arg3, %arg4) : memref<2x2xf16, [@CMX_NN, 0]>, memref<2x2xsi32, [@CMX_NN, 0]>, memref<2x2xf16, [@CMX_NN, 0]>
        }
    }

    %9 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    VPURT.Task waits(%8 : !VPURT.Barrier) updates(%9 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convert inputs(%3 as %arg2: memref<2x2xf16, [@CMX_NN, 0]>) outputs(%4 as %arg3: memref<2x2xf32, [@CMX_NN, 0]>) on tile 0 -> memref<2x2xf32, [@CMX_NN, 0]>{
            VPUIP.SW.Kernel.run(%arg2, %arg3) : memref<2x2xf16, [@CMX_NN, 0]>, memref<2x2xf32, [@CMX_NN, 0]>
        }
    }

    %10 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        VPURT.Task waits(%9 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
            %11 = VPUIP.NNDMA {port = 0 : i64} inputs(%4 : memref<2x2xf32, [@CMX_NN, 0]>) outputs(%arg1 : memref<2x2xf32>) -> memref<2x2xf32>
    }
  return %arg1 : memref<2x2xf32>

}

}

// CHECK:   identifier: "gatherEl"

// CHECK:   net_input: [
// CHECK:   {
// CHECK:       name: "Parameter_143",
// CHECK:       dimensions: [
// CHECK:           2,
// CHECK:           2
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableInput",
// CHECK:       data_dtype: "FP32",
// CHECK:       bit_strides: [
// CHECK:           32,
// CHECK:           64,
// CHECK:           32
// CHECK:       ]

// CHECK:     }
// CHECK:   ],

// CHECK:   net_output: [
// CHECK:     {
// CHECK:       name: "GatherElements",
// CHECK:       dimensions: [
// CHECK:         2,
// CHECK:         2
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableOutput",
// CHECK:       data_dtype: "FP32",
// CHECK:       bit_strides: [
// CHECK:         32,
// CHECK:         64,
// CHECK:         32
// CHECK:       ]
// CHECK:     }
// CHECK:   ],

// CHECK:   task_count: 6,

// CHECK:   options: [
// CHECK:   ],

// CHECK:   in_tensor_desc: [
// CHECK:     {
// CHECK:       name: "Parameter_143",
// CHECK:       dimensions: [
// CHECK:         2,
// CHECK:         2
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableInput",
// CHECK:       data_dtype: "FP32",
// CHECK:       bit_strides: [
// CHECK:         32,
// CHECK:         64,
// CHECK:         32
// CHECK:       ]
// CHECK:     }
// CHECK:   ],

// CHECK:   out_tensor_desc: [
// CHECK:     {
// CHECK:       name: "GatherElements",
// CHECK:       dimensions: [
// CHECK:         2,
// CHECK:         2
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableOutput",
// CHECK:       data_dtype: "FP32",
// CHECK:       bit_strides: [
// CHECK:         32,
// CHECK:         64,
// CHECK:         32
// CHECK:       ]
// CHECK:     }
// CHECK:   ]

// CHECK:    device: "VPUX37XX",
// CHECK:    act_kernel_runtime: {
// CHECK:        shaveStacks: [
// CHECK:          {
// CHECK:            name: "actSHAVE0_stack",
// CHECK:            locale: "GFEmbeddedKernel",
// CHECK:            referenced_data_size: {{[1-9][0-9]+}}
// CHECK:          },
// CHECK:          {
// CHECK:            name: "actSHAVE1_stack",
// CHECK:            locale: "GFEmbeddedKernel",
// CHECK:            referenced_data_size: {{[1-9][0-9]+}}
// CHECK:          },
// CHECK:          {
// CHECK:            name: "actSHAVE2_stack",
// CHECK:            locale: "GFEmbeddedKernel",
// CHECK:            referenced_data_size: {{[1-9][0-9]+}}
// CHECK:          },
// CHECK:          {
// CHECK:            name: "actSHAVE3_stack",
// CHECK:            locale: "GFEmbeddedKernel",
// CHECK:            referenced_data_size: {{[1-9][0-9]+}}
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
// CHECK:            nodeID: 1,
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
// CHECK:                  name: "builtin_Convert",
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
// CHECK:                    ],
// CHECK:                    virtual_wait_barriers: [
// CHECK:                      0
// CHECK:                    ],
// CHECK:                    virtual_update_barriers: [
// CHECK:                      1
// CHECK:                    ]
// CHECK:                  },
// CHECK:                  dataSection: {
// CHECK:                    name: "builtin_GatherElements_invo",
// CHECK:                    locale: "GFEmbeddedKernel",
// CHECK:                  },
// CHECK:                  invocationArgs: {
// CHECK:                    name: "builtin_GatherElements_invo",
// CHECK:                    locale: "GFEmbeddedKernel",
// CHECK:                    referenced_data_size: {{[1-9][0-9]+}}
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
