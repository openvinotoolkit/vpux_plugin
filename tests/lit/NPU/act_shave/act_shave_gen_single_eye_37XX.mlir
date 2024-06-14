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
// This file generates a blob with Eye activation shave
// demonstrate that the runtime cannot handle this. It's also a lit test to help
// check for regressions in the VPUIP dialect.
//

module @Test {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
            DataInfo "Parameter_201" : tensor<1xsi32>
        }
        outputsInfo : {
            DataInfo "Eye_202" : tensor<128x128xf32>
        }

// Sub-module, which holds SW kernel declarations and optional implementations.
// Used to group those declarations for faster access.
module @VPU.SW {
    // The declaration should match C++ params structure in decomposed form.
    // `memref` will be translated to `MemRefData`, while raw scalars will be translated as is.
    func.func private @builtin_Convert(memref<*xf16, @CMX_NN>, memref<*xf32, @CMX_NN>)
        attributes {
            VPU.kernel_code = "convert.cpp",
            VPU.kernel_entry = "convert"
        }
    func.func private @builtin_Eye(memref<*xsi32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>)
        attributes {
            VPU.kernel_code = "eye.cpp",
            VPU.kernel_entry = "eye"
        }
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @main(%arg0: memref<1xsi32, @DDR>, %arg1: memref<128x128xf32, @DDR>) -> memref<128x128xf32, @DDR> {
    %0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    %1 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
    %2 = VPURT.ConfigureBarrier<2> -> !VPURT.Barrier
    %3 = VPURT.ConfigureBarrier<3> -> !VPURT.Barrier
    %4 = VPURT.ConfigureBarrier<4> -> !VPURT.Barrier
    %5 = VPURT.ConfigureBarrier<5> -> !VPURT.Barrier
    %6 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1xsi32, @DDR>
    %7 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<128x128xf32, @DDR>
    %8 = VPURT.DeclareBuffer <CMX_NN> [0] <32768> -> memref<1xsi32, [@CMX_NN, 0]>
    %9 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<128x128xf16, [@CMX_NN, 0]>
    %10 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x128x128xf16, @DDR>
    %11 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x64x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [16384, 16384, 128, 1]}, @DDR>
    %12 = VPURT.DeclareBuffer <DDR> <16384> -> memref<1x1x64x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [16384, 16384, 128, 1]}, @DDR>
    %13 = VPURT.DeclareBuffer <CMX_NN> [0] <32768> -> memref<1x1x64x128xf16, [@CMX_NN, 0]>
    %14 = VPURT.DeclareBuffer <CMX_NN> [1] <32768> -> memref<1x1x64x128xf16, [@CMX_NN, 1]>
    %15 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x64x128xf32, [@CMX_NN, 0]>
    %16 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x1x64x128xf32, [@CMX_NN, 1]>
    %17 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x64x128xf32, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [16384, 16384, 128, 1]}, @DDR>
    %18 = VPURT.DeclareBuffer <DDR> <32768> -> memref<1x1x64x128xf32, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [16384, 16384, 128, 1]}, @DDR>
    %19 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x128x128xf16, [@CMX_NN, 0]>
    %20 = VPURT.DeclareBuffer <CMX_NN> [0] <40960> -> memref<1x1x32x128xf16, [@CMX_NN, 0]>
    %21 = VPURT.DeclareBuffer <CMX_NN> [1] <40960> -> memref<1x1x32x128xf16, [@CMX_NN, 1]>
    %22 = VPURT.DeclareBuffer <CMX_NN> [0] <32768> -> memref<1x1x32x128xf16, [@CMX_NN, 0]>
    %23 = VPURT.DeclareBuffer <CMX_NN> [1] <32768> -> memref<1x1x32x128xf16, [@CMX_NN, 1]>
    %24 = VPURT.DeclareBuffer <CMX_NN> [0] <16384> -> memref<1x1x32x128xf32, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [8192, 8192, 128, 1]}, [@CMX_NN, 0]>
    %25 = VPURT.DeclareBuffer <CMX_NN> [1] <16384> -> memref<1x1x32x128xf32, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [8192, 8192, 128, 1]}, [@CMX_NN, 1]>
    %26 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x32x128xf32, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [8192, 8192, 128, 1]}, [@CMX_NN, 0]>
    %27 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x1x32x128xf32, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [8192, 8192, 128, 1]}, [@CMX_NN, 1]>
    %28 = VPURT.DeclareBuffer <DDR> <0> -> memref<128x128xf32, @DDR>
    VPURT.Task updates(%0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %29 = VPUIP.NNDMA {port = 0 : i64} inputs(%6 : memref<1xsi32, @DDR>) outputs(%8 : memref<1xsi32, [@CMX_NN, 0]>) -> memref<1xsi32, [@CMX_NN, 0]>
    }
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Eye inputs(%8 as %arg2: memref<1xsi32, [@CMX_NN, 0]>) outputs(%9 as %arg3: memref<128x128xf16, [@CMX_NN, 0]>) on tile 0 -> memref<128x128xf16, [@CMX_NN, 0]>{
            VPUIP.SW.Kernel.run(%arg2, %arg3) : memref<1xsi32, [@CMX_NN, 0]>, memref<128x128xf16, [@CMX_NN, 0]>
        }
    }
    VPURT.Task waits(%1 : !VPURT.Barrier) updates(%2 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %29 = VPUIP.NNDMA {port = 0 : i64} inputs(%19 : memref<1x1x128x128xf16, [@CMX_NN, 0]>) outputs(%10 : memref<1x1x128x128xf16, @DDR>) -> memref<1x1x128x128xf16, @DDR>
    }
    VPURT.Task updates(%3 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %29 = VPUIP.NNDMA {port = 0 : i64} inputs(%11 : memref<1x1x64x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [16384, 16384, 128, 1]}, @DDR>) outputs(%13 : memref<1x1x64x128xf16, [@CMX_NN, 0]>) -> memref<1x1x64x128xf16, [@CMX_NN, 0]>
    }
    VPURT.Task waits(%2 : !VPURT.Barrier) updates(%3 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %29 = VPUIP.NNDMA {port = 1 : i64} inputs(%12 : memref<1x1x64x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [16384, 16384, 128, 1]}, @DDR>) outputs(%14 : memref<1x1x64x128xf16, [@CMX_NN, 1]>) -> memref<1x1x64x128xf16, [@CMX_NN, 1]>
    }
    VPURT.Task waits(%3 : !VPURT.Barrier) updates(%4 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convert inputs(%22 as %arg2: memref<1x1x32x128xf16, [@CMX_NN, 0]>) outputs(%26 as %arg3: memref<1x1x32x128xf32, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [8192, 8192, 128, 1]}, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x32x128xf32, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [8192, 8192, 128, 1]}, [@CMX_NN, 0]>{
            VPUIP.SW.Kernel.run(%arg2, %arg3) : memref<1x1x32x128xf16, [@CMX_NN, 0]>, memref<1x1x32x128xf32, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [8192, 8192, 128, 1]}, [@CMX_NN, 0]>
        }
    }
    VPURT.Task waits(%3 : !VPURT.Barrier) updates(%4 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convert inputs(%23 as %arg2: memref<1x1x32x128xf16, [@CMX_NN, 1]>) outputs(%27 as %arg3: memref<1x1x32x128xf32, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [8192, 8192, 128, 1]}, [@CMX_NN, 1]>) on tile 1 -> memref<1x1x32x128xf32, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [8192, 8192, 128, 1]}, [@CMX_NN, 1]>{
            VPUIP.SW.Kernel.run(%arg2, %arg3) : memref<1x1x32x128xf16, [@CMX_NN, 1]>, memref<1x1x32x128xf32, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [8192, 8192, 128, 1]}, [@CMX_NN, 1]>
        }
    }
    VPURT.Task waits(%3 : !VPURT.Barrier) updates(%4 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convert inputs(%20 as %arg2: memref<1x1x32x128xf16, [@CMX_NN, 0]>) outputs(%24 as %arg3: memref<1x1x32x128xf32, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [8192, 8192, 128, 1]}, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x32x128xf32, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [8192, 8192, 128, 1]}, [@CMX_NN, 0]>{
            VPUIP.SW.Kernel.run(%arg2, %arg3) : memref<1x1x32x128xf16, [@CMX_NN, 0]>, memref<1x1x32x128xf32, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [8192, 8192, 128, 1]}, [@CMX_NN, 0]>
        }
    }
    VPURT.Task waits(%3 : !VPURT.Barrier) updates(%4 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convert inputs(%21 as %arg2: memref<1x1x32x128xf16, [@CMX_NN, 1]>) outputs(%25 as %arg3: memref<1x1x32x128xf32, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [8192, 8192, 128, 1]}, [@CMX_NN, 1]>) on tile 1 -> memref<1x1x32x128xf32, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [8192, 8192, 128, 1]}, [@CMX_NN, 1]>{
            VPUIP.SW.Kernel.run(%arg2, %arg3) : memref<1x1x32x128xf16, [@CMX_NN, 1]>, memref<1x1x32x128xf32, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [8192, 8192, 128, 1]}, [@CMX_NN, 1]>
        }
    }
    VPURT.Task waits(%4 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %29 = VPUIP.NNDMA {port = 0 : i64} inputs(%15 : memref<1x1x64x128xf32, [@CMX_NN, 0]>) outputs(%17 : memref<1x1x64x128xf32, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [16384, 16384, 128, 1]}, @DDR>) -> memref<1x1x64x128xf32, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [16384, 16384, 128, 1]}, @DDR>
    }
    VPURT.Task waits(%4 : !VPURT.Barrier) updates(%5 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %29 = VPUIP.NNDMA {port = 1 : i64} inputs(%16 : memref<1x1x64x128xf32, [@CMX_NN, 1]>) outputs(%18 : memref<1x1x64x128xf32, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [16384, 16384, 128, 1]}, @DDR>) -> memref<1x1x64x128xf32, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [16384, 16384, 128, 1]}, @DDR>
    }
    VPURT.Task waits(%5 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %29 = VPUIP.NNDMA {port = 0 : i64} inputs(%28 : memref<128x128xf32, @DDR>) outputs(%7 : memref<128x128xf32, @DDR>) -> memref<128x128xf32, @DDR>
      }
    return %arg1 : memref<128x128xf32, @DDR>
}

}

// CHECK:   identifier: "Test"

// CHECK:   net_input: [
// CHECK:     {
// CHECK:       name: "Parameter_201",
// CHECK:       dimensions: [
// CHECK:           1
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableInput",
// CHECK:       data_dtype: "I32",
// CHECK:       bit_strides: [
// CHECK:           32,
// CHECK:           32
// CHECK:       ]
// CHECK:     }
// CHECK:   ],

// CHECK:   net_output: [
// CHECK:     {
// CHECK:       name: "Eye_202",
// CHECK:       dimensions: [
// CHECK:         128,
// CHECK:         128
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableOutput",
// CHECK:       data_dtype: "FP32",
// CHECK:       bit_strides: [
// CHECK:         32,
// CHECK:         4096,
// CHECK:         32
// CHECK:       ]
// CHECK:     }
// CHECK:   ],

// CHECK:   task_count: 18,

// CHECK:   options: [
// CHECK:   ],

// CHECK:   in_tensor_desc: [
// CHECK:     {
// CHECK:       name: "Parameter_201",
// CHECK:       dimensions: [
// CHECK:           1
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableInput",
// CHECK:       data_dtype: "I32",
// CHECK:       bit_strides: [
// CHECK:           32,
// CHECK:           32
// CHECK:       ]
// CHECK:     }
// CHECK:   ],

// CHECK:   out_tensor_desc: [
// CHECK:     {
// CHECK:       name: "Eye_202",
// CHECK:       dimensions: [
// CHECK:         128,
// CHECK:         128
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableOutput",
// CHECK:       data_dtype: "FP32",
// CHECK:       bit_strides: [
// CHECK:         32,
// CHECK:         4096,
// CHECK:         32
// CHECK:       ]
// CHECK:     }
// CHECK:   ]
