//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" %s | vpux-translate --vpu-arch=%arch% --export-VPUIP -o %t
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// RUN: flatc --raw-binary --json %vpuip_schema_file% -- %t
// RUN: FileCheck %s --input-file %basename_t.json
// RUN: rm %basename_t.json

module @Test {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
            DataInfo "Parameter_187" : tensor<5x6xf32>
        }
        outputsInfo : {
            DataInfo "EmbeddingBagOffsetsSum_192" : tensor<2x6xf32>
        }

// Sub-module, which holds SW kernel declarations and optional implementations.
// Used to group those declarations for faster access.
module @VPU.SW {
    // The declaration should match C++ params structure in decomposed form.
    // `memref` will be translated to `MemRefData`, while raw scalars will be translated as is.
    func.func private @builtin_EmbeddingBagOffsetsSum(memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, i32) attributes {VPU.kernel_code = "embedding_bag_offsets_sum.cpp", VPU.kernel_entry = "embedding_bag_offsets_sum"}
    func.func private @builtin_Convert(memref<*xf32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>) attributes {VPU.kernel_code = "convert.cpp", VPU.kernel_entry = "convert"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @main(%arg0: memref<5x6xf32>, %arg1: memref<2x6xf32>) -> memref<2x6xf32> {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<5x6xf32> to tensor<5x6xf32>
    %cst = const.Declare tensor<5xf16> = dense<[1.000000e+00, 4.753910e+00, 9.976560e+00, 7.484380e+00, 1.000000e+01]> : tensor<5xf16>
    %1 = builtin.unrealized_conversion_cast %cst : tensor<5xf16> to memref<5xf16>
    %cst_0 = const.Declare tensor<2xsi32> = dense<[0, 2]> : tensor<2xsi32>
    %2 = builtin.unrealized_conversion_cast %cst_0 : tensor<2xsi32> to memref<2xsi32>
    %cst_1 = const.Declare tensor<5xsi32> = dense<[0, 1, 2, 2, 3]> : tensor<5xsi32>
    %3 = builtin.unrealized_conversion_cast %cst_1 : tensor<5xsi32> to memref<5xsi32>
    %4 = VPU.AffineReshape(%0) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 5, 1, 6]} : tensor<5x6xf32> -> tensor<1x5x1x6xf32>
    %5 = builtin.unrealized_conversion_cast %4 : tensor<1x5x1x6xf32> to memref<1x5x1x6xf32>
    %6 = memref.alloc() : memref<1x5x1x6xf32, [@CMX_NN, 0]>
    %7 = VPUIP.Copy inputs(%5 : memref<1x5x1x6xf32>) outputs(%6 : memref<1x5x1x6xf32, [@CMX_NN, 0]>) -> memref<1x5x1x6xf32, [@CMX_NN, 0]>
    %8 = memref.alloc() : memref<1x5x1x6xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convert inputs(%7 as %arg2: memref<1x5x1x6xf32, [@CMX_NN, 0]>) outputs(%8 as %arg3: memref<1x5x1x6xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x5x1x6xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg2, %arg3) : memref<1x5x1x6xf32, [@CMX_NN, 0]>, memref<1x5x1x6xf16, [@CMX_NN, 0]>
    }
    %9 = memref.alloc() : memref<1x5x1x6xf16>
    %10 = VPUIP.Copy inputs(%results : memref<1x5x1x6xf16, [@CMX_NN, 0]>) outputs(%9 : memref<1x5x1x6xf16>) -> memref<1x5x1x6xf16>
    %11 = builtin.unrealized_conversion_cast %10 : memref<1x5x1x6xf16> to tensor<1x5x1x6xf16>
    %12 = VPU.AffineReshape(%11) {dim_mapping = [[0], [0], [0], [1]], shape_value = [5, 6]} : tensor<1x5x1x6xf16> -> tensor<5x6xf16>
    %13 = builtin.unrealized_conversion_cast %12 : tensor<5x6xf16> to memref<5x6xf16>
    %14 = memref.alloc() : memref<5x6xf16, [@CMX_NN, 0]>
    %15 = VPUIP.Copy inputs(%13 : memref<5x6xf16>) outputs(%14 : memref<5x6xf16, [@CMX_NN, 0]>) -> memref<5x6xf16, [@CMX_NN, 0]>
    %16 = memref.alloc() : memref<5xsi32, [@CMX_NN, 0]>
    %17 = VPUIP.Copy inputs(%3 : memref<5xsi32>) outputs(%16 : memref<5xsi32, [@CMX_NN, 0]>) -> memref<5xsi32, [@CMX_NN, 0]>
    %18 = memref.alloc() : memref<2xsi32, [@CMX_NN, 0]>
    %19 = VPUIP.Copy inputs(%2 : memref<2xsi32>) outputs(%18 : memref<2xsi32, [@CMX_NN, 0]>) -> memref<2xsi32, [@CMX_NN, 0]>
    %20 = memref.alloc() : memref<5xf16, [@CMX_NN, 0]>
    %21 = VPUIP.Copy inputs(%1 : memref<5xf16>) outputs(%20 : memref<5xf16, [@CMX_NN, 0]>) -> memref<5xf16, [@CMX_NN, 0]>
    %22 = memref.alloc() : memref<2x6xf16, [@CMX_NN, 0]>
    %results_2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_EmbeddingBagOffsetsSum inputs(%15 as %arg2: memref<5x6xf16, [@CMX_NN, 0]>, %17 as %arg3: memref<5xsi32, [@CMX_NN, 0]>, %19 as %arg4: memref<2xsi32, [@CMX_NN, 0]>, %21 as %arg5: memref<5xf16, [@CMX_NN, 0]>) outputs(%22 as %arg6: memref<2x6xf16, [@CMX_NN, 0]>) on tile 0 -> memref<2x6xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [0 : i32]}(%arg2, %arg3, %arg4, %arg5, %arg6) : memref<5x6xf16, [@CMX_NN, 0]>, memref<5xsi32, [@CMX_NN, 0]>, memref<2xsi32, [@CMX_NN, 0]>, memref<5xf16, [@CMX_NN, 0]>, memref<2x6xf16, [@CMX_NN, 0]>
    }
    %23 = memref.alloc() : memref<2x6xf16>
    %24 = VPUIP.Copy inputs(%results_2 : memref<2x6xf16, [@CMX_NN, 0]>) outputs(%23 : memref<2x6xf16>) -> memref<2x6xf16>
    %25 = builtin.unrealized_conversion_cast %24 : memref<2x6xf16> to tensor<2x6xf16>
    %26 = VPU.AffineReshape(%25) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 2, 1, 6]} : tensor<2x6xf16> -> tensor<1x2x1x6xf16>
    %27 = builtin.unrealized_conversion_cast %26 : tensor<1x2x1x6xf16> to memref<1x2x1x6xf16>
    %28 = memref.alloc() : memref<1x2x1x6xf16, [@CMX_NN, 0]>
    %29 = VPUIP.Copy inputs(%27 : memref<1x2x1x6xf16>) outputs(%28 : memref<1x2x1x6xf16, [@CMX_NN, 0]>) -> memref<1x2x1x6xf16, [@CMX_NN, 0]>
    %30 = memref.alloc() : memref<1x2x1x6xf32, [@CMX_NN, 0]>
    %results_3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convert inputs(%29 as %arg2: memref<1x2x1x6xf16, [@CMX_NN, 0]>) outputs(%30 as %arg3: memref<1x2x1x6xf32, [@CMX_NN, 0]>) on tile 0 -> memref<1x2x1x6xf32, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg2, %arg3) : memref<1x2x1x6xf16, [@CMX_NN, 0]>, memref<1x2x1x6xf32, [@CMX_NN, 0]>
    }
    %31 = memref.alloc() : memref<1x2x1x6xf32>
    %32 = VPUIP.Copy inputs(%results_3 : memref<1x2x1x6xf32, [@CMX_NN, 0]>) outputs(%31 : memref<1x2x1x6xf32>) -> memref<1x2x1x6xf32>
    %33 = builtin.unrealized_conversion_cast %32 : memref<1x2x1x6xf32> to tensor<1x2x1x6xf32>
    %34 = VPU.AffineReshape(%33) {dim_mapping = [[0], [0], [0], [1]], shape_value = [2, 6]} : tensor<1x2x1x6xf32> -> tensor<2x6xf32>
    %35 = builtin.unrealized_conversion_cast %34 : tensor<2x6xf32> to memref<2x6xf32>
    %36 = VPUIP.Copy inputs(%35 : memref<2x6xf32>) outputs(%arg1 : memref<2x6xf32>) -> memref<2x6xf32>
    return %36 : memref<2x6xf32>
  }

}

// CHECK:   identifier: "Test"

// CHECK:    net_input: [
// CHECK:      {
// CHECK:        name: "Parameter_187",
// CHECK:        dimensions: [
// CHECK:           5,
// CHECK:           6
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        locale: "ProgrammableInput",
// CHECK:        bit_strides: [
// CHECK:          32,
// CHECK:          192,
// CHECK:          32
// CHECK:        ]
// CHECK:      }
// CHECK:    ],

// CHECK:    net_output: [
// CHECK:      {
// CHECK:        name: "EmbeddingBagOffsetsSum_192",
// CHECK:        dimensions: [
// CHECK:           2,
// CHECK:           6
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        bit_strides: [
// CHECK:          32,
// CHECK:          192,
// CHECK:          32
// CHECK:        ]
// CHECK:      }
// CHECK:    ],

// CHECK:    out_tensor_desc: [
// CHECK:      {
// CHECK:        name: "EmbeddingBagOffsetsSum_192",
// CHECK:        dimensions: [
// CHECK:          2,
// CHECK:          6
// CHECK:        ],
// CHECK:        data: {
// CHECK:          data_index: 0
// CHECK:        },
// CHECK:        bit_strides: [
// CHECK:          32,
// CHECK:          192,
// CHECK:          32
// CHECK:        ]
// CHECK:      }
// CHECK:    ]
