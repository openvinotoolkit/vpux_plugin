// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" %s | vpux-translate --vpu-arch=%arch% --export-VPUIP -o %t
// RUN: flatc --raw-binary --json %vpuip_schema_file% -- %t
// RUN: FileCheck %s --input-file %basename_t.json
// RUN: rm %basename_t.json
// REQUIRES: arch-VPUX37XX
//
// This file generates a blob with GRUCell activation shave
// demonstrate that the runtime cannot handle this. It's also a lit test to help
// check for regressions in the VPUIP dialect.
//

module @Test {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
            DataInfo "Parameter_187" : tensor<1x1x2x3xf32>
            DataInfo "Parameter_188" : tensor<1x1x3x2xf32>
        }
        outputsInfo : {
            DataInfo "GridSample_189" : tensor<1x1x1x3xf32>
        }

// Sub-module, which holds SW kernel declarations and optional implementations.
// Used to group those declarations for faster access.
module @VPU.SW {
    // The declaration should match C++ params structure in decomposed form.
    // `memref` will be translated to `MemRefData`, while raw scalars will be translated as is.
    func.func private @builtin_GridSample(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, i64, i64, i64) attributes {VPU.kernel_code = "grid_sample.cpp", VPU.kernel_entry = "grid_sample"}
    func.func private @builtin_Convert(memref<*xf32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>) attributes {VPU.kernel_code = "convert.cpp", VPU.kernel_entry = "convert"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

  func.func @main(%arg0: memref<1x1x2x3xf32>, %arg1: memref<1x1x3x2xf32>, %arg2: memref<1x1x1x3xf32>) -> memref<1x1x1x3xf32> {
    %0 = builtin.unrealized_conversion_cast %arg1 : memref<1x1x3x2xf32> to tensor<1x1x3x2xf32>
    %1 = builtin.unrealized_conversion_cast %arg0 : memref<1x1x2x3xf32> to tensor<1x1x2x3xf32>
    %2 = memref.alloc() : memref<1x1x2x3xf32, [@CMX_NN, 0]>
    %3 = VPUIP.Copy inputs(%arg0 : memref<1x1x2x3xf32>) outputs(%2 : memref<1x1x2x3xf32, [@CMX_NN, 0]>) -> memref<1x1x2x3xf32, [@CMX_NN, 0]>
    %4 = memref.alloc() : memref<1x1x2x3xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convert inputs(%3 as %arg3: memref<1x1x2x3xf32, [@CMX_NN, 0]>) outputs(%4 as %arg4: memref<1x1x2x3xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x2x3xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg3, %arg4) : memref<1x1x2x3xf32, [@CMX_NN, 0]>, memref<1x1x2x3xf16, [@CMX_NN, 0]>
    }
    %5 = memref.alloc() : memref<1x1x2x3xf16>
    %6 = VPUIP.Copy inputs(%results : memref<1x1x2x3xf16, [@CMX_NN, 0]>) outputs(%5 : memref<1x1x2x3xf16>) -> memref<1x1x2x3xf16>
    %7 = memref.alloc() : memref<1x1x3x2xf32, [@CMX_NN, 0]>
    %8 = VPUIP.Copy inputs(%arg1 : memref<1x1x3x2xf32>) outputs(%7 : memref<1x1x3x2xf32, [@CMX_NN, 0]>) -> memref<1x1x3x2xf32, [@CMX_NN, 0]>
    %9 = memref.alloc() : memref<1x1x3x2xf16, [@CMX_NN, 0]>
    %results_0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convert inputs(%8 as %arg3: memref<1x1x3x2xf32, [@CMX_NN, 0]>) outputs(%9 as %arg4: memref<1x1x3x2xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x3x2xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg3, %arg4) : memref<1x1x3x2xf32, [@CMX_NN, 0]>, memref<1x1x3x2xf16, [@CMX_NN, 0]>
    }
    %10 = memref.alloc() : memref<1x1x3x2xf16>
    %11 = VPUIP.Copy inputs(%results_0 : memref<1x1x3x2xf16, [@CMX_NN, 0]>) outputs(%10 : memref<1x1x3x2xf16>) -> memref<1x1x3x2xf16>
    %12 = memref.alloc() : memref<1x1x2x3xf16, [@CMX_NN, 0]>
    %13 = VPUIP.Copy inputs(%6 : memref<1x1x2x3xf16>) outputs(%12 : memref<1x1x2x3xf16, [@CMX_NN, 0]>) -> memref<1x1x2x3xf16, [@CMX_NN, 0]>
    %14 = memref.alloc() : memref<1x1x3x2xf16, [@CMX_NN, 0]>
    %15 = VPUIP.Copy inputs(%11 : memref<1x1x3x2xf16>) outputs(%14 : memref<1x1x3x2xf16, [@CMX_NN, 0]>) -> memref<1x1x3x2xf16, [@CMX_NN, 0]>
    %16 = memref.alloc() : memref<1x1x1x3xf16, [@CMX_NN, 0]>
    %results_1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_GridSample inputs(%13 as %arg3: memref<1x1x2x3xf16, [@CMX_NN, 0]>, %15 as %arg4: memref<1x1x3x2xf16, [@CMX_NN, 0]>) outputs(%16 as %arg5: memref<1x1x1x3xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x3xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [1, 0, 1]}(%arg3, %arg4, %arg5) : memref<1x1x2x3xf16, [@CMX_NN, 0]>, memref<1x1x3x2xf16, [@CMX_NN, 0]>, memref<1x1x1x3xf16, [@CMX_NN, 0]>
    }
    %17 = memref.alloc() : memref<1x1x1x3xf16>
    %18 = VPUIP.Copy inputs(%results_1 : memref<1x1x1x3xf16, [@CMX_NN, 0]>) outputs(%17 : memref<1x1x1x3xf16>) -> memref<1x1x1x3xf16>
    %19 = memref.alloc() : memref<1x1x1x3xf16, [@CMX_NN, 0]>
    %20 = VPUIP.Copy inputs(%18 : memref<1x1x1x3xf16>) outputs(%19 : memref<1x1x1x3xf16, [@CMX_NN, 0]>) -> memref<1x1x1x3xf16, [@CMX_NN, 0]>
    %21 = memref.alloc() : memref<1x1x1x3xf32, [@CMX_NN, 0]>
    %results_2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convert inputs(%20 as %arg3: memref<1x1x1x3xf16, [@CMX_NN, 0]>) outputs(%21 as %arg4: memref<1x1x1x3xf32, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x3xf32, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg3, %arg4) : memref<1x1x1x3xf16, [@CMX_NN, 0]>, memref<1x1x1x3xf32, [@CMX_NN, 0]>
    }
    %22 = memref.alloc() : memref<1x1x1x3xf32>
    %23 = VPUIP.Copy inputs(%results_2 : memref<1x1x1x3xf32, [@CMX_NN, 0]>) outputs(%22 : memref<1x1x1x3xf32>) -> memref<1x1x1x3xf32>
    %24 = builtin.unrealized_conversion_cast %23 : memref<1x1x1x3xf32> to tensor<1x1x1x3xf32>
    %25 = builtin.unrealized_conversion_cast %24 : tensor<1x1x1x3xf32> to memref<1x1x1x3xf32>
    %26 = VPUIP.Copy inputs(%25 : memref<1x1x1x3xf32>) outputs(%arg2 : memref<1x1x1x3xf32>) -> memref<1x1x1x3xf32>
    return %26 : memref<1x1x1x3xf32>
  }

}

// CHECK:   identifier: "Test"

// CHECK:    net_input: [
// CHECK:      {
// CHECK:        name: "Parameter_187",
// CHECK:        dimensions: [
// CHECK:           1,
// CHECK:           1,
// CHECK:           2,
// CHECK:           3
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableInput",
// CHECK:        bit_strides: [
// CHECK:           32,
// CHECK:           192,
// CHECK:           192,
// CHECK:           96,
// CHECK:           32
// CHECK:        ]
// CHECK:      },
// CHECK:      {
// CHECK:        name: "Parameter_188",
// CHECK:        dimensions: [
// CHECK          1,
// CHECK          1,
// CHECK          3,
// CHECK          2
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        bit_strides: [
// CHECK:          32,
// CHECK:          192,
// CHECK:          192,
// CHECK:          64,
// CHECK:          32
// CHECK:        ]
// CHECK:      }
// CHECK:    ],

// CHECK:    net_output: [
// CHECK:      {
// CHECK:        name: "GridSample_189",
// CHECK:        dimensions: [
// CHECK:           1,
// CHECK:           1,
// CHECK:           1,
// CHECK:           3
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        bit_strides: [
// CHECK:           32,
// CHECK:           96,
// CHECK:           96,
// CHECK:           96,
// CHECK:           32
// CHECK:        ]
// CHECK:      }
// CHECK:    ],

// CHECK:    out_tensor_desc: [
// CHECK:      {
// CHECK:        name: "GridSample_189",
// CHECK:        dimensions: [
// CHECK:          1,
// CHECK:          1,
// CHECK:          1,
// CHECK:          3
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        strides: [
// CHECK:          32,
// CHECK:          96,
// CHECK:          96,
// CHECK:          96,
// CHECK:          32
// CHECK:        ]
// CHECK:      }
// CHECK:    ]
