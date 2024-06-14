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
// This file generates a blob with select shave
// demonstrate that the runtime cannot handle this.  It's also a lit test to help
// check for regressions in the VPUIP dialect.
//

module @Test {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "input0" : tensor<1x1x1x1000xf16>
        IE.DataInfo "input1" : tensor<1x1x1x1000xf16>
        IE.DataInfo "input2" : tensor<1x1x1x1000xf16>
    }
    outputsInfo : {
        IE.DataInfo "select" : tensor<1x1x1x1000xf16>
    }

VPURT.SW.Runtime
    entryPoint: @VPU.SW::@runtime
    stack_configuration: [
        4096, // Size in bytes for the SHAVEs in the first tile.
        4096,  // Size in bytes for the SHAVEs in the second tile.
        4096,
        4096
    ]


// Sub-module, which holds SW kernel declarations and optional implementations.
// Used to group those declarations for faster access.
module @VPU.SW {
    // The declaration should match C++ params structure in decomposed form.
    // `memref` will be translated to `MemRefData`, while raw scalars will be translated as is.
    func.func private @builtin_select(%input0 : memref<*xf16>, %input1 : memref<*xf16>, %input2 : memref<*xf16>, %output : memref<*xf16>)
        attributes {
            VPU.kernel_code = "eltwise_select.cpp",
            VPU.kernel_entry = "eltwise_select"
        }

    // management kernel definition
    func.func private @runtime()
        attributes {
            VPU.kernel_code = "nnActEntry"
        }
}

func.func @main(%1: memref<1x1x1x1000xf16>, %2: memref<1x1x1x1000xf16>, %3: memref<1x1x1x1000xf16>, %4: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
    %in0_tile0_cmx  = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %in1_tile0_cmx  = VPURT.DeclareBuffer <CMX_NN> [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %in2_tile0_cmx  = VPURT.DeclareBuffer <CMX_NN> [0] <4000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %out_tile0_cmx = VPURT.DeclareBuffer <CMX_NN> [0] <6000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>

    %b0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    %b1 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier

    VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%1 : memref<1x1x1x1000xf16>) outputs(%in0_tile0_cmx : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    }
    VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%2 : memref<1x1x1x1000xf16>) outputs(%in1_tile0_cmx : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    }
    VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%3 : memref<1x1x1x1000xf16>) outputs(%in2_tile0_cmx : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%b0 : !VPURT.Barrier) updates(%b1 : !VPURT.Barrier) {
        VPUIP.SW.Kernel
                    {resultSegmentSizes = array<i32: 1, 0, 0>}
                    @VPU.SW::@builtin_select
                    inputs(%in0_tile0_cmx as %arg0: memref<1x1x1x1000xf16, [@CMX_NN, 0]>, %in1_tile0_cmx as %arg1: memref<1x1x1x1000xf16, [@CMX_NN, 0]>, %in2_tile0_cmx as %arg2: memref<1x1x1x1000xf16, [@CMX_NN, 0]>)
                    outputs(%out_tile0_cmx as %arg3: memref<1x1x1x1000xf16, [@CMX_NN, 0]>)
                    on tile 0
        -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>{
                VPUIP.SW.Kernel.run(%arg0, %arg1, %arg2, %arg3)
                    : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
                    , memref<1x1x1x1000xf16, [@CMX_NN, 0]>
                    , memref<1x1x1x1000xf16, [@CMX_NN, 0]>
                    , memref<1x1x1x1000xf16, [@CMX_NN, 0]>
        }
    }

    VPURT.Task waits(%b1 : !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%out_tile0_cmx : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%4 : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>
    }

    return %4: memref<1x1x1x1000xf16>
}

}


// CHECK:    identifier: "Test",
// CHECK:    net_input: [
// CHECK:      {
// CHECK:        name: "input0",
// CHECK:        dimensions: [
// CHECK:          1,
// CHECK:          1,
// CHECK:          1,
// CHECK:          1000
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableInput",
// CHECK:        data_dtype: "FP16",
// CHECK:        bit_strides: [
// CHECK:          16,
// CHECK:          16000,
// CHECK:          16000,
// CHECK:          16000,
// CHECK:          16
// CHECK:        ]
// CHECK:      },
// CHECK:      {
// CHECK:        name: "input1",
// CHECK:        dimensions: [
// CHECK:          1,
// CHECK:          1,
// CHECK:          1,
// CHECK:          1000
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableInput",
// CHECK:        data_dtype: "FP16",
// CHECK:        bit_strides: [
// CHECK:          16,
// CHECK:          16000,
// CHECK:          16000,
// CHECK:          16000,
// CHECK:          16
// CHECK:        ]
// CHECK:      },
// CHECK:      {
// CHECK:        name: "input2",
// CHECK:        dimensions: [
// CHECK:          1,
// CHECK:          1,
// CHECK:          1,
// CHECK:          1000
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableInput",
// CHECK:        data_dtype: "FP16",
// CHECK:        bit_strides: [
// CHECK:          16,
// CHECK:          16000,
// CHECK:          16000,
// CHECK:          16000,
// CHECK:          16
// CHECK:        ]
// CHECK:      }
// CHECK:    ],

// CHECK:    net_output: [
// CHECK:      {
// CHECK:        name: "select",
// CHECK:        dimensions: [
// CHECK:          1,
// CHECK:          1,
// CHECK:          1,
// CHECK:          1000
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableOutput",
// CHECK:        data_dtype: "FP16",
// CHECK:        bit_strides: [
// CHECK:          16,
// CHECK:          16000,
// CHECK:          16000,
// CHECK:          16000,
// CHECK:          16
// CHECK:        ]
// CHECK:      }
// CHECK:    ],

// CHECK:    task_count: 7,

// CHECK:    options: [
// CHECK:    ],

// CHECK:    in_tensor_desc: [
// CHECK:      {
// CHECK:        name: "input0",
// CHECK:        dimensions: [
// CHECK:          1,
// CHECK:          1,
// CHECK:          1,
// CHECK:          1000
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableInput",
// CHECK:        data_dtype: "FP16",
// CHECK:        bit_strides: [
// CHECK:          16,
// CHECK:          16000,
// CHECK:          16000,
// CHECK:          16000,
// CHECK:          16
// CHECK:        ]
// CHECK:      },
// CHECK:      {
// CHECK:        name: "input1",
// CHECK:        dimensions: [
// CHECK:          1,
// CHECK:          1,
// CHECK:          1,
// CHECK:          1000
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableInput",
// CHECK:        data_dtype: "FP16",
// CHECK:        bit_strides: [
// CHECK:          16,
// CHECK:          16000,
// CHECK:          16000,
// CHECK:          16000,
// CHECK:          16
// CHECK:        ]
// CHECK:      },
// CHECK:      {
// CHECK:        name: "input2",
// CHECK:        dimensions: [
// CHECK:          1,
// CHECK:          1,
// CHECK:          1,
// CHECK:          1000
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableInput",
// CHECK:        data_dtype: "FP16",
// CHECK:        bit_strides: [
// CHECK:          16,
// CHECK:          16000,
// CHECK:          16000,
// CHECK:          16000,
// CHECK:          16
// CHECK:        ]
// CHECK:      }
// CHECK:    ],

// CHECK:    out_tensor_desc: [
// CHECK:      {
// CHECK:        name: "select",
// CHECK:        dimensions: [
// CHECK:          1,
// CHECK:          1,
// CHECK:          1,
// CHECK:          1000
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableOutput",
// CHECK:        data_dtype: "FP16",
// CHECK:        bit_strides: [
// CHECK:          16,
// CHECK:          16000,
// CHECK:          16000,
// CHECK:          16000,
// CHECK:          16
// CHECK:        ]
// CHECK:      }
// CHECK:    ]

// CHECK:    device: "VPUX37XX",
// CHECK:    act_kernel_runtime: {
// CHECK:      shaveStacks: [
// CHECK:        {
// CHECK:          name: "actSHAVE0_stack",
// CHECK:          locale: "GFEmbeddedKernel",
// CHECK:          locale_offset: 2,
// CHECK:          referenced_data_size: 4096
// CHECK:        },
// CHECK:        {
// CHECK:          name: "actSHAVE1_stack",
// CHECK:          locale: "GFEmbeddedKernel",
// CHECK:          locale_offset: 3,
// CHECK:          referenced_data_size: 4096
// CHECK:        },
// CHECK:        {
// CHECK:          name: "actSHAVE2_stack",
// CHECK:          locale: "GFEmbeddedKernel",
// CHECK:          locale_offset: 4,
// CHECK:          referenced_data_size: 4096
// CHECK:        },
// CHECK:        {
// CHECK:          name: "actSHAVE3_stack",
// CHECK:          locale: "GFEmbeddedKernel",
// CHECK:          locale_offset: 5,
// CHECK:          referenced_data_size: 4096
// CHECK:        }
// CHECK:      ],
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

// CHECK:  task_lists: [
// CHECK:    {
// CHECK:      content: [
// CHECK:        {
// CHECK:          name: "",
// CHECK:          task_type: "ControllerTask",
// CHECK:          task: {
// CHECK:            task_type: "BarrierConfigurationTask",
// CHECK:            task: {
// CHECK:              target: {
// CHECK:                consumer_count: 1,
// CHECK:                producer_count: 3
// CHECK:              }
// CHECK:            }
// CHECK:          }
// CHECK:        },
// CHECK:        {
// CHECK:          name: "",
// CHECK:          nodeID: 1,
// CHECK:          task_type: "ControllerTask",
// CHECK:          task: {
// CHECK:            task_type: "BarrierConfigurationTask",
// CHECK:            task: {
// CHECK:              target: {
// CHECK:                barrier_id: 1,
// CHECK:                consumer_count: 1,
// CHECK:                producer_count: 1
// CHECK:              }
// CHECK:            }
// CHECK:          }
// CHECK:        }
// CHECK:      ]
// CHECK:    },
// CHECK:    {
// CHECK:      content: [
// CHECK:        {
// CHECK:          name: "",
// CHECK:          nodeID: 2,
// CHECK:          associated_barriers: {
// CHECK:            wait_barriers: [
// CHECK:            ],
// CHECK:            update_barriers: [
// CHECK:              0
// CHECK:            ],
// CHECK:            virtual_wait_barriers: [
// CHECK:            ],
// CHECK:            virtual_update_barriers: [
// CHECK:              0
// CHECK:            ]
// CHECK:          },
// CHECK:          task_type: "NNDMATask",
// CHECK:          task: {
// CHECK:            src: {
// CHECK:              name: "input0",
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
// CHECK:            },
// CHECK:            set_ord: true
// CHECK:          }
// CHECK:        },
// CHECK:        {
// CHECK:          name: "",
// CHECK:          nodeID: 3,
// CHECK:          associated_barriers: {
// CHECK:            wait_barriers: [
// CHECK:            ],
// CHECK:            update_barriers: [
// CHECK:              0
// CHECK:            ],
// CHECK:            virtual_wait_barriers: [
// CHECK:            ],
// CHECK:            virtual_update_barriers: [
// CHECK:              0
// CHECK:            ]
// CHECK:          },
// CHECK:          task_type: "NNDMATask",
// CHECK:          task: {
// CHECK:            src: {
// CHECK:              name: "input1",
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
// CHECK:                1
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
// CHECK:              name: "temp-1",
// CHECK:              dimensions: [
// CHECK:                1,
// CHECK:                1,
// CHECK:                1,
// CHECK:                1000
// CHECK:              ],
// CHECK:              data: {
// CHECK:                data_index: 2000
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
// CHECK:            set_ord: true
// CHECK:          }
// CHECK:        },
// CHECK:        {
// CHECK:          name: "",
// CHECK:          nodeID: 4,
// CHECK:          associated_barriers: {
// CHECK:            wait_barriers: [
// CHECK:            ],
// CHECK:            update_barriers: [
// CHECK:              0
// CHECK:            ],
// CHECK:            virtual_wait_barriers: [
// CHECK:            ],
// CHECK:            virtual_update_barriers: [
// CHECK:              0
// CHECK:            ]
// CHECK:          },
// CHECK:          task_type: "NNDMATask",
// CHECK:          task: {
// CHECK:            src: {
// CHECK:              name: "input2",
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
// CHECK:                2
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
// CHECK:              name: "temp-2",
// CHECK:              dimensions: [
// CHECK:                1,
// CHECK:                1,
// CHECK:                1,
// CHECK:                1000
// CHECK:              ],
// CHECK:              data: {
// CHECK:                data_index: 4000
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
// CHECK:            set_ord: true
// CHECK:          }
// CHECK:        },
// CHECK:        {
// CHECK:          name: "",
// CHECK:          nodeID: 6,
// CHECK:          associated_barriers: {
// CHECK:            wait_barriers: [
// CHECK:              1
// CHECK:            ],
// CHECK:            update_barriers: [
// CHECK:            ],
// CHECK:            virtual_wait_barriers: [
// CHECK:              1
// CHECK:            ],
// CHECK:            virtual_update_barriers: [
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
// CHECK:              name: "select",
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
// CHECK:            },
// CHECK:            set_ord: true
// CHECK:          }
// CHECK:        }
// CHECK:      ]
// CHECK:    },
// CHECK:    {
// CHECK:      content: [
// CHECK:        {
// CHECK:          name: "",
// CHECK:          nodeID: 5,
// CHECK:          associated_barriers: {
// CHECK:            wait_barriers: [
// CHECK:              0
// CHECK:            ],
// CHECK:            update_barriers: [
// CHECK:              1
// CHECK:            ],
// CHECK:            virtual_wait_barriers: [
// CHECK:              0
// CHECK:            ],
// CHECK:            virtual_update_barriers: [
// CHECK:              1
// CHECK:            ]
// CHECK:          },
// CHECK:          task_type: "ActKernelTask",
// CHECK:          task: {
// CHECK:            kernel: {
// CHECK:              kernelText: {
// CHECK:                name: "builtin_select",
// CHECK:                locale: "GFEmbeddedKernel",
// CHECK:                locale_offset: {{[1-9]}},
// CHECK:                data_offset: {{[1-9][0-9]+}},
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
// CHECK:                  name: "builtin_select_invo",
// CHECK:                  locale: "GFEmbeddedKernel",
// CHECK:                  locale_offset: {{[1-9]}},
// CHECK:                  data_offset: {{[1-9][0-9]+}}
// CHECK:                },
// CHECK:                invocationArgs: {
// CHECK:                  name: "builtin_select_invo",
// CHECK:                  locale: "GFEmbeddedKernel",
// CHECK:                  locale_offset: {{[1-9]}},
// CHECK:                  data_offset: {{[1-9][0-9]+}},
// CHECK:                  referenced_data_size: {{[1-9][0-9]+}}
// CHECK:                }
// CHECK:              }
// CHECK:            ]
// CHECK:          }
// CHECK:        }
// CHECK:      ]
// CHECK:    }
// CHECK:  ],

// CHECK:  kernel_data: [
// CHECK:    ]
