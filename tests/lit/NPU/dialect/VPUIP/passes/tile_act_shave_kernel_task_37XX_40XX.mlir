//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" --tile-act-shave-kernel-task %s | FileCheck %s
// REQUIRES: arch-VPUX37XX || arch-VPUX40XX

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

IE.TileResource 1 of @NCE at 1.300000e+03 MHz {
    IE.ExecutorResource 1 of @DPU
}

module @VPU.SW  {
    func.func private @builtin_TopK(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xsi32, @CMX_NN>, i64, i64, i64, i64) attributes {VPU.kernel_code = "topk.cpp", VPU.kernel_entry = "topk"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileTopKSubViewWithOneDPUGroup(%arg0: memref<1x1x10x1xf16, #NCWH>)
        -> (memref<1x1x10x1xf16, #NCWH>, memref<1x1x10x1xsi32, #NCWH>) {
    %0 = memref.alloc() : memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x1x10x1xf16, #NCWH>) outputs(%0 : memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>) -> memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>
    %3 = memref.alloc() : memref<1x1x10x1xsi32, #NCWH, [@CMX_NN, 0]>
    %4:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_TopK inputs(%1 as %arg10: memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>) outputs(%2 as %arg11: memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>, %3 as %arg12: memref<1x1x10x1xsi32, #NCWH, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>, memref<1x1x10x1xsi32, #NCWH, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [1, 0, 0, 1]}(%arg10, %arg11, %arg12) : memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>, memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>, memref<1x1x10x1xsi32, #NCWH, [@CMX_NN, 0]>
    }
    %5 = memref.alloc() : memref<1x1x10x1xf16, #NCWH>
    %6 = VPUIP.Copy inputs(%4#0 : memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>) outputs(%5 : memref<1x1x10x1xf16, #NCWH>) -> memref<1x1x10x1xf16, #NCWH>
    %7 = memref.alloc() : memref<1x1x10x1xsi32, #NCWH>
    %8 = VPUIP.Copy inputs(%4#1 : memref<1x1x10x1xsi32, #NCWH, [@CMX_NN, 0]>) outputs(%7 : memref<1x1x10x1xsi32, #NCWH>) -> memref<1x1x10x1xsi32, #NCWH>
    return %6, %8 : memref<1x1x10x1xf16, #NCWH>, memref<1x1x10x1xsi32, #NCWH>


    // CHECK: [[IN_BUF:%.*]] = memref.alloc() : memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>
    // CHECK: [[IN_COPY:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x1x10x1xf16, #NCWH>) outputs([[IN_BUF]] : memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>) -> memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>
    // CHECK: [[OUT_BUF_0:%.*]] = memref.alloc() : memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>
    // CHECK: [[OUT_BUF_1:%.*]] = memref.alloc() : memref<1x1x10x1xsi32, #NCWH, [@CMX_NN, 0]>

    // CHECK: [[IN_SUBVIEW_0:%.*]] = VPUIP.SubView [[IN_COPY]] [0, 0, 0, 0] [1, 1, 5, 1] : memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]> to memref<1x1x5x1xf16, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>
    // CHECK: [[OUT_SUBVIEW_0:%.*]] = VPUIP.SubView [[OUT_BUF_0]] [0, 0, 0, 0] [1, 1, 5, 1] : memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]> to memref<1x1x5x1xf16, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>
    // CHECK: [[OUT_SUBVIEW_1:%.*]] = VPUIP.SubView [[OUT_BUF_1]] [0, 0, 0, 0] [1, 1, 5, 1] : memref<1x1x10x1xsi32, #NCWH, [@CMX_NN, 0]> to memref<1x1x5x1xsi32, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>

    // CHECK: [[IN_SUBVIEW_1:%.*]] = VPUIP.SubView [[IN_COPY]] [0, 0, 5, 0] [1, 1, 5, 1] : memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]> to memref<1x1x5x1xf16, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>
    // CHECK: [[OUT_SUBVIEW_2:%.*]] = VPUIP.SubView [[OUT_BUF_0]] [0, 0, 5, 0] [1, 1, 5, 1] : memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]> to memref<1x1x5x1xf16, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>
    // CHECK: [[OUT_SUBVIEW_3:%.*]] = VPUIP.SubView [[OUT_BUF_1]] [0, 0, 5, 0] [1, 1, 5, 1] : memref<1x1x10x1xsi32, #NCWH, [@CMX_NN, 0]> to memref<1x1x5x1xsi32, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>

    // CHECK: [[TOPK:%.*]]:4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 4, 0, 0>} @VPU.SW::@builtin_TopK inputs([[IN_SUBVIEW_0]] as %arg1: memref<1x1x5x1xf16, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>, [[IN_SUBVIEW_1]] as %arg2: memref<1x1x5x1xf16, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>) outputs([[OUT_SUBVIEW_0]] as %arg3: memref<1x1x5x1xf16, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>, [[OUT_SUBVIEW_1]] as %arg4: memref<1x1x5x1xsi32, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>, [[OUT_SUBVIEW_2]] as %arg5: memref<1x1x5x1xf16, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>, [[OUT_SUBVIEW_3]] as %arg6: memref<1x1x5x1xsi32, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x5x1xf16, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>, memref<1x1x5x1xsi32, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>, memref<1x1x5x1xf16, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>, memref<1x1x5x1xsi32, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>){
    // CHECK:    VPUIP.SW.Kernel.run {attrs = [1, 0, 0, 1]}(%arg1, %arg3, %arg4) : memref<1x1x5x1xf16, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>, memref<1x1x5x1xf16, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>, memref<1x1x5x1xsi32, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>
    // CHECK:    VPUIP.SW.Kernel.run {attrs = [1, 0, 0, 1]}(%arg2, %arg5, %arg6) : memref<1x1x5x1xf16, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>, memref<1x1x5x1xf16, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>, memref<1x1x5x1xsi32, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>

    // CHECK: [[CONCAT_0:%.*]] = VPUIP.ConcatView inputs([[TOPK]]#0, [[TOPK]]#2 : memref<1x1x5x1xf16, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>, memref<1x1x5x1xf16, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>) outputs([[OUT_BUF_0]] : memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>) -> memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>
    // CHECK: [[CONCAT_1:%.*]] = VPUIP.ConcatView inputs([[TOPK]]#1, [[TOPK]]#3 : memref<1x1x5x1xsi32, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>, memref<1x1x5x1xsi32, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>) outputs([[OUT_BUF_1]] : memref<1x1x10x1xsi32, #NCWH, [@CMX_NN, 0]>) -> memref<1x1x10x1xsi32, #NCWH, [@CMX_NN, 0]>

    // CHECK: [[OUT_DDR_BUF_0:%.*]] = memref.alloc() : memref<1x1x10x1xf16, #NCWH>
    // CHECK: [[OUT_DDR_COPY_0:%.*]] = VPUIP.Copy inputs([[CONCAT_0]] : memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>) outputs([[OUT_DDR_BUF_0]] : memref<1x1x10x1xf16, #NCWH>) -> memref<1x1x10x1xf16, #NCWH>
    // CHECK: [[OUT_DDR_BUF_1:%.*]] = memref.alloc() : memref<1x1x10x1xsi32, #NCWH>
    // CHECK: [[OUT_DDR_COPY_1:%.*]] = VPUIP.Copy inputs([[CONCAT_1]] : memref<1x1x10x1xsi32, #NCWH, [@CMX_NN, 0]>) outputs([[OUT_DDR_BUF_1]] : memref<1x1x10x1xsi32, #NCWH>) -> memref<1x1x10x1xsi32, #NCWH>
    // CHECK: return [[OUT_DDR_COPY_0]], [[OUT_DDR_COPY_1]] : memref<1x1x10x1xf16, #NCWH>, memref<1x1x10x1xsi32, #NCWH>
}

// -----

IE.TileResource 1 of @NCE at 1.300000e+03 MHz {
    IE.ExecutorResource 1 of @DPU
}

module @VPU.SW  {
    func.func private @builtin_TopK(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xsi32, @CMX_NN>, i64, i64, i64, i64) attributes {VPU.kernel_code = "topk.cpp", VPU.kernel_entry = "topk"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileTopKCopyWithOneDPUGroup(%arg0: memref<1x8x16x16xf16>)
        -> (memref<1x1x16x16xf16>, memref<1x1x16x16xsi32>) {
    %0 = memref.alloc() : memref<1x8x16x16xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x8x16x16xf16>) outputs(%0 : memref<1x8x16x16xf16, [@CMX_NN, 0]>) -> memref<1x8x16x16xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x1x16x16xf16, [@CMX_NN, 0]>
    %3 = memref.alloc() : memref<1x1x16x16xsi32, [@CMX_NN, 0]>
    %4:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_TopK inputs(%1 as %arg10: memref<1x8x16x16xf16, [@CMX_NN, 0]>) outputs(%2 as %arg11: memref<1x1x16x16xf16, [@CMX_NN, 0]>, %3 as %arg12: memref<1x1x16x16xsi32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x16x16xf16, [@CMX_NN, 0]>, memref<1x1x16x16xsi32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [0, 0, 2, 1]}(%arg10, %arg11, %arg12) : memref<1x8x16x16xf16, [@CMX_NN, 0]>, memref<1x1x16x16xf16, [@CMX_NN, 0]>, memref<1x1x16x16xsi32, [@CMX_NN, 0]>
    }
    %5 = memref.alloc() : memref<1x1x16x16xf16>
    %6 = VPUIP.Copy inputs(%4#0 : memref<1x1x16x16xf16, [@CMX_NN, 0]>) outputs(%5 : memref<1x1x16x16xf16>) -> memref<1x1x16x16xf16>
    %7 = memref.alloc() : memref<1x1x16x16xsi32>
    %8 = VPUIP.Copy inputs(%4#1 : memref<1x1x16x16xsi32, [@CMX_NN, 0]>) outputs(%7 : memref<1x1x16x16xsi32>) -> memref<1x1x16x16xsi32>
    return %6, %8 : memref<1x1x16x16xf16>, memref<1x1x16x16xsi32>

    // CHECK: [[IN_BUF:%.*]] = memref.alloc() : memref<1x8x16x16xf16, [@CMX_NN, 0]>
    // CHECK: [[IN_COPY:%.*]] = VPUIP.Copy inputs({{[^:]+}} : memref<1x8x16x16xf16>) outputs([[IN_BUF]] : memref<1x8x16x16xf16, [@CMX_NN, 0]>) -> memref<1x8x16x16xf16, [@CMX_NN, 0]>
    // CHECK: [[SUBVIEW_1:%.*]] = VPUIP.SubView {{[^:]+}} [0, 0, 0, 0] [1, 1, 8, 16] : memref<1x8x16x16xf16> to memref<1x1x8x16xf16, {order = #NCHW, strides = [2048, 256, 16, 1]}>
    // CHECK: [[SUBVIEW_1_OUT:%.*]] = memref.alloc() : memref<1x1x8x16xf16, [@CMX_NN, 0]>
    // CHECK: [[SUBVIEW_1_COPY:%.*]] = VPUIP.Copy inputs([[SUBVIEW_1]] : memref<1x1x8x16xf16, {order = #NCHW, strides = [2048, 256, 16, 1]}>) outputs([[SUBVIEW_1_OUT]] : memref<1x1x8x16xf16, [@CMX_NN, 0]>) -> memref<1x1x8x16xf16, [@CMX_NN, 0]>

    // CHECK: [[RUN_1_OUTPUT_1:%.*]] = memref.alloc() : memref<1x1x8x16xf16, [@CMX_NN, 0]>
    // CHECK: [[RUN_1_OUTPUT_2:%.*]] = memref.alloc() : memref<1x1x8x16xsi32, [@CMX_NN, 0]>

    // CHECK: [[SUBVIEW_2:%.*]] = VPUIP.SubView {{[^:]+}} [0, 0, 8, 0] [1, 1, 8, 16] : memref<1x8x16x16xf16> to memref<1x1x8x16xf16, {order = #NCHW, strides = [2048, 256, 16, 1]}>
    // CHECK: [[SUBVIEW_2_OUT:%.*]] = memref.alloc() : memref<1x1x8x16xf16, [@CMX_NN, 0]>
    // CHECK: [[SUBVIEW_2_COPY:%.*]]  = VPUIP.Copy inputs([[SUBVIEW_2]] : memref<1x1x8x16xf16, {order = #NCHW, strides = [2048, 256, 16, 1]}>) outputs([[SUBVIEW_2_OUT]] : memref<1x1x8x16xf16, [@CMX_NN, 0]>) -> memref<1x1x8x16xf16, [@CMX_NN, 0]>

    // CHECK: [[RUN_2_OUTPUT_1:%.*]] = memref.alloc() : memref<1x1x8x16xf16, [@CMX_NN, 0]>
    // CHECK: [[RUN_2_OUTPUT_2:%.*]] = memref.alloc() : memref<1x1x8x16xsi32, [@CMX_NN, 0]>

    // CHECK: [[TOPK:%.*]]:4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 4, 0, 0>} @VPU.SW::@builtin_TopK inputs([[SUBVIEW_1_COPY]] as %arg1: memref<1x1x8x16xf16, [@CMX_NN, 0]>, [[SUBVIEW_2_COPY]] as %arg2: memref<1x1x8x16xf16, [@CMX_NN, 0]>) outputs([[RUN_1_OUTPUT_1]] as %arg3: memref<1x1x8x16xf16, [@CMX_NN, 0]>, [[RUN_1_OUTPUT_2]] as %arg4: memref<1x1x8x16xsi32, [@CMX_NN, 0]>, [[RUN_2_OUTPUT_1]] as %arg5: memref<1x1x8x16xf16, [@CMX_NN, 0]>, [[RUN_2_OUTPUT_2]] as %arg6: memref<1x1x8x16xsi32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x8x16xf16, [@CMX_NN, 0]>, memref<1x1x8x16xsi32, [@CMX_NN, 0]>, memref<1x1x8x16xf16, [@CMX_NN, 0]>, memref<1x1x8x16xsi32, [@CMX_NN, 0]>){
    // CHECK:   VPUIP.SW.Kernel.run {attrs = [0, 0, 2, 1]}(%arg1, %arg3, %arg4) : memref<1x1x8x16xf16, [@CMX_NN, 0]>, memref<1x1x8x16xf16, [@CMX_NN, 0]>, memref<1x1x8x16xsi32, [@CMX_NN, 0]>
    // CHECK:   VPUIP.SW.Kernel.run {attrs = [0, 0, 2, 1]}(%arg2, %arg5, %arg6) : memref<1x1x8x16xf16, [@CMX_NN, 0]>, memref<1x1x8x16xf16, [@CMX_NN, 0]>, memref<1x1x8x16xsi32, [@CMX_NN, 0]>
    // CHECK:  }

    // CHECK: [[CONCAT_1_OUTPUT:%.*]] = memref.alloc() : memref<1x1x16x16xsi32, [@CMX_NN, 0]>
    // CHECK: [[OUTPUT_SUBVIEW_1_1:%.*]] = VPUIP.SubView [[CONCAT_1_OUTPUT]] [0, 0, 0, 0] [1, 1, 8, 16] : memref<1x1x16x16xsi32, [@CMX_NN, 0]> to memref<1x1x8x16xsi32, {order = #NCHW, strides = [256, 256, 16, 1]}, [@CMX_NN, 0]>
    // CHECK: [[OUTPUT_SUBVIEW_1_1_CP:%.*]] = VPUIP.Copy inputs([[TOPK]]#1 : memref<1x1x8x16xsi32, [@CMX_NN, 0]>) outputs([[OUTPUT_SUBVIEW_1_1]] : memref<1x1x8x16xsi32, {order = #NCHW, strides = [256, 256, 16, 1]}, [@CMX_NN, 0]>) -> memref<1x1x8x16xsi32, {order = #NCHW, strides = [256, 256, 16, 1]}, [@CMX_NN, 0]>

    // CHECK: [[OUTPUT_SUBVIEW_1_2:%.*]] = VPUIP.SubView [[CONCAT_1_OUTPUT]] [0, 0, 8, 0] [1, 1, 8, 16] : memref<1x1x16x16xsi32, [@CMX_NN, 0]> to memref<1x1x8x16xsi32, {order = #NCHW, strides = [256, 256, 16, 1]}, [@CMX_NN, 0]>
    // CHECK: [[OUTPUT_SUBVIEW_1_2_CP:%.*]] = VPUIP.Copy inputs([[TOPK]]#3 : memref<1x1x8x16xsi32, [@CMX_NN, 0]>) outputs([[OUTPUT_SUBVIEW_1_2]] : memref<1x1x8x16xsi32, {order = #NCHW, strides = [256, 256, 16, 1]}, [@CMX_NN, 0]>) -> memref<1x1x8x16xsi32, {order = #NCHW, strides = [256, 256, 16, 1]}, [@CMX_NN, 0]>
    // CHECK: [[CONCAT_1:%.*]] = VPUIP.ConcatView inputs([[OUTPUT_SUBVIEW_1_1_CP]], [[OUTPUT_SUBVIEW_1_2_CP]] : memref<1x1x8x16xsi32, {order = #NCHW, strides = [256, 256, 16, 1]}, [@CMX_NN, 0]>, memref<1x1x8x16xsi32, {order = #NCHW, strides = [256, 256, 16, 1]}, [@CMX_NN, 0]>) outputs([[CONCAT_1_OUTPUT]] : memref<1x1x16x16xsi32, [@CMX_NN, 0]>) -> memref<1x1x16x16xsi32, [@CMX_NN, 0]>

    // CHECK: [[CONCAT_2_OUTPUT:%.*]]  = memref.alloc() : memref<1x1x16x16xf16, [@CMX_NN, 0]>
    // CHECK: [[OUTPUT_SUBVIEW_2_1:%.*]] = VPUIP.SubView [[CONCAT_2_OUTPUT]] [0, 0, 0, 0] [1, 1, 8, 16] : memref<1x1x16x16xf16, [@CMX_NN, 0]> to memref<1x1x8x16xf16, {order = #NCHW, strides = [256, 256, 16, 1]}, [@CMX_NN, 0]>
    // CHECK: [[OUTPUT_SUBVIEW_2_1_CP:%.*]] = VPUIP.Copy inputs([[TOPK]]#0 : memref<1x1x8x16xf16, [@CMX_NN, 0]>) outputs([[OUTPUT_SUBVIEW_2_1]] : memref<1x1x8x16xf16, {order = #NCHW, strides = [256, 256, 16, 1]}, [@CMX_NN, 0]>) -> memref<1x1x8x16xf16, {order = #NCHW, strides = [256, 256, 16, 1]}, [@CMX_NN, 0]>
    // CHECK: [[OUTPUT_SUBVIEW_2_2:%.*]] = VPUIP.SubView [[CONCAT_2_OUTPUT]] [0, 0, 8, 0] [1, 1, 8, 16] : memref<1x1x16x16xf16, [@CMX_NN, 0]> to memref<1x1x8x16xf16, {order = #NCHW, strides = [256, 256, 16, 1]}, [@CMX_NN, 0]>
    // CHECK: [[OUTPUT_SUBVIEW_2_2_CP:%.*]] = VPUIP.Copy inputs([[TOPK]]#2 : memref<1x1x8x16xf16, [@CMX_NN, 0]>) outputs([[OUTPUT_SUBVIEW_2_2]] : memref<1x1x8x16xf16, {order = #NCHW, strides = [256, 256, 16, 1]}, [@CMX_NN, 0]>) -> memref<1x1x8x16xf16, {order = #NCHW, strides = [256, 256, 16, 1]}, [@CMX_NN, 0]>
    // CHECK: [[CONCAT_2:%.*]] = VPUIP.ConcatView inputs([[OUTPUT_SUBVIEW_2_1_CP]], [[OUTPUT_SUBVIEW_2_2_CP]] : memref<1x1x8x16xf16, {order = #NCHW, strides = [256, 256, 16, 1]}, [@CMX_NN, 0]>, memref<1x1x8x16xf16, {order = #NCHW, strides = [256, 256, 16, 1]}, [@CMX_NN, 0]>) outputs([[CONCAT_2_OUTPUT]] : memref<1x1x16x16xf16, [@CMX_NN, 0]>) -> memref<1x1x16x16xf16, [@CMX_NN, 0]>

    // CHECK: [[CONCAT_2_DDR:%.*]] = memref.alloc() : memref<1x1x16x16xf16>
    // CHECK: [[CONCAT_2_DDR_COPY:%.*]] = VPUIP.Copy inputs([[CONCAT_2]] : memref<1x1x16x16xf16, [@CMX_NN, 0]>) outputs([[CONCAT_2_DDR]] : memref<1x1x16x16xf16>) -> memref<1x1x16x16xf16>
    // CHECK: [[CONCAT_1_DDR:%.*]]  = memref.alloc() : memref<1x1x16x16xsi32>
    // CHECK: [[CONCAT_1_DDR_COPY:%.*]] = VPUIP.Copy inputs([[CONCAT_1]] : memref<1x1x16x16xsi32, [@CMX_NN, 0]>) outputs([[CONCAT_1_DDR]] : memref<1x1x16x16xsi32>) -> memref<1x1x16x16xsi32>
    // CHECK: return [[CONCAT_2_DDR_COPY]], [[CONCAT_1_DDR_COPY]] : memref<1x1x16x16xf16>, memref<1x1x16x16xsi32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
    func.func private @builtin_MVN6(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i64, f64, i64, none) attributes {VPU.kernel_code = "mvn6.cpp", VPU.kernel_entry = "mvn6", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileMvn6OverC(%arg0: memref<1x32x15x64xf16, [@CMX_NN, 0]>, %arg1: memref<1x32x15x64xf16, [@CMX_NN, 0]>) -> memref<1x32x15x64xf16, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x32x15x64xf16, [@CMX_NN, 0]>

    // note: at VPUIP dialect, MVN6 axes are numbered in memory order, thus [1] means [H] for NCHW
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN6
                    inputs(%arg0 as %arg2 : memref<1x32x15x64xf16, [@CMX_NN, 0]>)
                    outputs(%0 as %arg3: memref<1x32x15x64xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x32x15x64xf16, [@CMX_NN, 0]>{
              VPUIP.SW.Kernel.run {attrs = [true, 0, 2.0E-7, 1, [1]]} (%arg2, %arg3) : memref<1x32x15x64xf16, [@CMX_NN, 0]>, memref<1x32x15x64xf16, [@CMX_NN, 0]>
    }

    return %results : memref<1x32x15x64xf16, [@CMX_NN, 0]>

    // CHECK: [[OUTPUT_BUF:%.*]] = memref.alloc() : memref<1x32x15x64xf16, [@CMX_NN, 0]>

    // CHECK: [[SUBVIEW_INP_0:%.*]] = VPUIP.SubView {{[^:]+}}      [0, 0, 0, 0] [1, 16, 15, 64] : memref<1x32x15x64xf16, [@CMX_NN, 0]> to memref<1x16x15x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>
    // CHECK: [[SUBVIEW_OUT_0:%.*]] = VPUIP.SubView [[OUTPUT_BUF]] [0, 0, 0, 0] [1, 16, 15, 64] : memref<1x32x15x64xf16, [@CMX_NN, 0]> to memref<1x16x15x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>
    // CHECK: [[SUBVIEW_INP_1:%.*]] = VPUIP.SubView {{[^:]+}}      [0, 16, 0, 0] [1, 16, 15, 64] : memref<1x32x15x64xf16, [@CMX_NN, 0]> to memref<1x16x15x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>
    // CHECK: [[SUBVIEW_OUT_1:%.*]] = VPUIP.SubView [[OUTPUT_BUF]] [0, 16, 0, 0] [1, 16, 15, 64] : memref<1x32x15x64xf16, [@CMX_NN, 0]> to memref<1x16x15x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>

    // CHECK: [[MVN6:%.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN6
    // CHECK_SAME(LITEAL):   inputs([[SUBVIEW_INP_0]] as {{[^:]+}}]: memref<1x16x15x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>,
    // CHECK_SAME(LITEAL):          [[SUBVIEW_INP_1]] as {{[^:]+}}]: memref<1x16x15x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>)
    // CHECK_SAME(LITEAL):  outputs([[SUBVIEW_OUT_0]] as {{[^:]+}}]: memref<1x16x15x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>,
    // CHECK_SAME(LITEAL):          [[SUBVIEW_OUT_1]] as {{[^:]+}}]: memref<1x16x15x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x16x15x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>, memref<1x16x15x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>)

    // CHECK: VPUIP.SW.Kernel.run {attrs = [true, 0, 2.000000e-07, 1, [1]]}({{[^:]+}}, {{[^:]+}}) : memref<1x16x15x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>, memref<1x16x15x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>
    // CHECK: VPUIP.SW.Kernel.run {attrs = [true, 0, 2.000000e-07, 1, [1]]}({{[^:]+}}, {{[^:]+}}) : memref<1x16x15x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>, memref<1x16x15x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>

    // CHECK:  [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[MVN6]]#0, [[MVN6]]#1 : memref<1x16x15x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>, memref<1x16x15x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>) outputs(%alloc : memref<1x32x15x64xf16, [@CMX_NN, 0]>) -> memref<1x32x15x64xf16, [@CMX_NN, 0]>
    // CHECK:  return [[CONCAT]] : memref<1x32x15x64xf16, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
    func.func private @builtin_MVN6(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i64, f64, i64, none) attributes {VPU.kernel_code = "mvn6.cpp", VPU.kernel_entry = "mvn6", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}


func.func @TileMvn6OverH(%arg0: memref<1x32x15x64xf16, [@CMX_NN, 0]>, %arg1: memref<1x32x15x64xf16, [@CMX_NN, 0]>) -> memref<1x32x15x64xf16, [@CMX_NN, 0]> {

    %0 = memref.alloc() : memref<1x32x15x64xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x32x15x64xf16, [@CMX_NN, 0]>) outputs(%0 : memref<1x32x15x64xf16, [@CMX_NN, 0]>) -> memref<1x32x15x64xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x32x15x64xf16, [@CMX_NN, 0]>

    // note: at VPUIP dialect, MVN6 axes are numbered in memory order, thus [0, 2] mean [W, C] for NCHW
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN6
                    inputs(%1 as %arg2 : memref<1x32x15x64xf16, [@CMX_NN, 0]>)
                    outputs(%2 as %arg3: memref<1x32x15x64xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x32x15x64xf16, [@CMX_NN, 0]>{
              VPUIP.SW.Kernel.run {attrs = [true, 0, 2.0E-7, 1, [0, 2]]} (%arg2, %arg3) : memref<1x32x15x64xf16, [@CMX_NN, 0]>, memref<1x32x15x64xf16, [@CMX_NN, 0]>
    }

    return %results : memref<1x32x15x64xf16, [@CMX_NN, 0]>

    // CHECK:  %[[VAL_2:.*]] = memref.alloc() : memref<1x32x15x64xf16, [@CMX_NN, 0]>
    // CHECK:  %[[VAL_3:.*]] = VPUIP.Copy inputs({{[^:]+}} : memref<1x32x15x64xf16, [@CMX_NN, 0]>) outputs(%[[VAL_2]] : memref<1x32x15x64xf16, [@CMX_NN, 0]>) -> memref<1x32x15x64xf16, [@CMX_NN, 0]>
    // CHECK:  %[[VAL_4:.*]] = VPUIP.SubView {{[^:]+}} [0, 0, 0, 0] [1, 32, 8, 64] : memref<1x32x15x64xf16, [@CMX_NN, 0]> to memref<1x32x8x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>
    // CHECK:  %[[VAL_5:.*]] = memref.alloc() : memref<1x32x8x64xf16, [@CMX_NN, 0]>
    // CHECK:  %[[VAL_6:.*]] = VPUIP.Copy inputs(%[[VAL_4]] : memref<1x32x8x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>) outputs(%[[VAL_5]] : memref<1x32x8x64xf16, [@CMX_NN, 0]>) -> memref<1x32x8x64xf16, [@CMX_NN, 0]>
    // CHECK:  %[[VAL_7:.*]] = memref.alloc() : memref<1x32x8x64xf16, [@CMX_NN, 0]>
    // CHECK:  %[[VAL_8:.*]] = VPUIP.SubView {{[^:]+}} [0, 0, 8, 0] [1, 32, 7, 64] : memref<1x32x15x64xf16, [@CMX_NN, 0]> to memref<1x32x7x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>
    // CHECK:  %[[VAL_9:.*]] = memref.alloc() : memref<1x32x7x64xf16, [@CMX_NN, 0]>
    // CHECK:  %[[VAL_10:.*]] = VPUIP.Copy inputs(%[[VAL_8]] : memref<1x32x7x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>) outputs(%[[VAL_9]] : memref<1x32x7x64xf16, [@CMX_NN, 0]>) -> memref<1x32x7x64xf16, [@CMX_NN, 0]>
    // CHECK:  %[[VAL_11:.*]] = memref.alloc() : memref<1x32x7x64xf16, [@CMX_NN, 0]>
    // CHECK:  %[[VAL_12:.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN6 inputs(%[[VAL_6]] as %[[VAL_13:.*]]: memref<1x32x8x64xf16, [@CMX_NN, 0]>, %[[VAL_10]] as %[[VAL_14:.*]]: memref<1x32x7x64xf16, [@CMX_NN, 0]>) outputs(%[[VAL_7]] as %[[VAL_15:.*]]: memref<1x32x8x64xf16, [@CMX_NN, 0]>, %[[VAL_11]] as %[[VAL_16:.*]]: memref<1x32x7x64xf16, [@CMX_NN, 0]>) on tile 0 -> (memref<1x32x8x64xf16, [@CMX_NN, 0]>, memref<1x32x7x64xf16, [@CMX_NN, 0]>){
    // CHECK:    VPUIP.SW.Kernel.run {attrs = [true, 0, 2.000000e-07, 1, [0, 2]]}(%[[VAL_13]], %[[VAL_15]]) : memref<1x32x8x64xf16, [@CMX_NN, 0]>, memref<1x32x8x64xf16, [@CMX_NN, 0]>
    // CHECK:    VPUIP.SW.Kernel.run {attrs = [true, 0, 2.000000e-07, 1, [0, 2]]}(%[[VAL_14]], %[[VAL_16]]) : memref<1x32x7x64xf16, [@CMX_NN, 0]>, memref<1x32x7x64xf16, [@CMX_NN, 0]>
    // CHECK:  }
    // CHECK:  %[[VAL_17:.*]] = memref.alloc() : memref<1x32x15x64xf16, [@CMX_NN, 0]>
    // CHECK:  %[[VAL_18:.*]] = VPUIP.SubView %[[VAL_17]] [0, 0, 0, 0] [1, 32, 8, 64] : memref<1x32x15x64xf16, [@CMX_NN, 0]> to memref<1x32x8x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>
    // CHECK:  %[[VAL_19:.*]] = VPUIP.Copy inputs(%[[VAL_20:.*]]#0 : memref<1x32x8x64xf16, [@CMX_NN, 0]>) outputs(%[[VAL_18]] : memref<1x32x8x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>) -> memref<1x32x8x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>
    // CHECK:  %[[VAL_21:.*]] = VPUIP.SubView %[[VAL_17]] [0, 0, 8, 0] [1, 32, 7, 64] : memref<1x32x15x64xf16, [@CMX_NN, 0]> to memref<1x32x7x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>
    // CHECK:  %[[VAL_22:.*]] = VPUIP.Copy inputs(%[[VAL_20]]#1 : memref<1x32x7x64xf16, [@CMX_NN, 0]>) outputs(%[[VAL_21]] : memref<1x32x7x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>) -> memref<1x32x7x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>
    // CHECK:  %[[VAL_23:.*]] = VPUIP.ConcatView inputs(%[[VAL_19]], %[[VAL_22]] : memref<1x32x8x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>, memref<1x32x7x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>) outputs(%[[VAL_17]] : memref<1x32x15x64xf16, [@CMX_NN, 0]>) -> memref<1x32x15x64xf16, [@CMX_NN, 0]>
    // CHECK:  return %[[VAL_23]] : memref<1x32x15x64xf16, [@CMX_NN, 0]>

}

// -----

module @VPU.SW {
    func.func private @builtin_Minimum(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "eltwise_min.cpp", VPU.kernel_entry = "eltwise_min"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileMinimum(%arg0: memref<1x4x96x160xf16, [@CMX_NN, 0]>, %arg1: memref<1x4x96x160xf16, [@CMX_NN, 0]>) -> memref<1x4x96x160xf16, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x4x96x160xf16, [@CMX_NN, 0]>

    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Minimum inputs(%arg0 as %arg2 : memref<1x4x96x160xf16, [@CMX_NN, 0]>,%arg1 as %arg3: memref<1x4x96x160xf16, [@CMX_NN, 0]>) outputs(%0 as %4: memref<1x4x96x160xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x4x96x160xf16, [@CMX_NN, 0]>{

      VPUIP.SW.Kernel.run {attrs = []}(%arg2, %arg3,%4) : memref<1x4x96x160xf16, [@CMX_NN, 0]>, memref<1x4x96x160xf16, [@CMX_NN, 0]>,  memref<1x4x96x160xf16, [@CMX_NN, 0]>
    }

    return %results : memref<1x4x96x160xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUTPUT_BUF_0:%.*]] = memref.alloc() : memref<1x4x96x160xf16, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_0:%.*]] = VPUIP.SubView {{[^:]+}} [0, 0, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_1:%.*]] = VPUIP.SubView {{[^:]+}} [0, 0, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_2:%.*]] = VPUIP.SubView [[OUTPUT_BUF_0]] [0, 0, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_3:%.*]] = VPUIP.SubView {{[^:]+}} [0, 2, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_4:%.*]] = VPUIP.SubView {{[^:]+}} [0, 2, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_5:%.*]] = VPUIP.SubView [[OUTPUT_BUF_0]] [0, 2, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[MINIMUM:%.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Minimum inputs([[SUBVIEW_0]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_1]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_3]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_4]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>) outputs([[SUBVIEW_2]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_5]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>){
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = []}({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = []}({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[MINIMUM]]#0, [[MINIMUM]]#1 : memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>) outputs([[OUTPUT_BUF_0]] : memref<1x4x96x160xf16, [@CMX_NN, 0]>) -> memref<1x4x96x160xf16, [@CMX_NN, 0]>
    // CHECK:    return [[CONCAT]] : memref<1x4x96x160xf16, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
    func.func private @builtin_Interpolate(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64, i64, i64, none, none, none, none, none) attributes {VPU.kernel_code = "interpolate.cpp", VPU.kernel_entry = "interpolate"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileCubicInterpolate(%arg0: memref<1x1x460x620xf16>, %arg1: memref<1x1x800x1000xf16>) -> memref<1x1x800x1000xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x460x620xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 232, 620], [1, 1, 232, 620]], compute_offsets = [[0, 0, 0, 0], [0, 0, 228, 0]], memory_shapes = [[1, 1, 232, 620], [1, 1, 232, 620]], memory_offsets = [[0, 0, 0, 0], [0, 0, 228, 0]]}>
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg2: memref<1x1x460x620xf16>) outputs(%0 as %arg3: memref<1x1x460x620xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x1x460x620xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 232, 620], [1, 1, 232, 620]], compute_offsets = [[0, 0, 0, 0], [0, 0, 228, 0]], memory_shapes = [[1, 1, 232, 620], [1, 1, 232, 620]], memory_offsets = [[0, 0, 0, 0], [0, 0, 228, 0]]}> {
      %6 = VPUIP.Copy inputs(%arg2 : memref<1x1x460x620xf16>) outputs(%arg3 : memref<1x1x460x620xf16, @CMX_NN>) -> memref<1x1x460x620xf16, @CMX_NN>
    }
    %2 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x800x1000xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPUIP.NCEClusterTiling inputs(%1 as %arg2: memref<1x1x460x620xf16, @CMX_NN>) outputs(%2 as %arg3: memref<1x1x800x1000xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x1x800x1000xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Interpolate inputs(%arg2 as %arg4: memref<1x1x460x620xf16, @CMX_NN>) outputs(%arg3 as %arg5: memref<1x1x800x1000xf16, @CMX_NN>) on tile 0 -> memref<1x1x800x1000xf16, @CMX_NN>{
        VPUIP.SW.Kernel.run {attrs = [3, 1, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [620, 460, 1, 1], [1000, 800, 1, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}(%arg4, %arg5) : memref<1x1x460x620xf16, @CMX_NN>, memref<1x1x800x1000xf16, @CMX_NN>
      }
    }
    %alloc = memref.alloc() : memref<1x1x800x1000xf16>
    %4 = VPUIP.NCEClusterTiling inputs(%3 as %arg2: memref<1x1x800x1000xf16, @CMX_NN>) outputs(%alloc as %arg3: memref<1x1x800x1000xf16>) -> memref<1x1x800x1000xf16> {
      %6 = VPUIP.Copy inputs(%arg2 : memref<1x1x800x1000xf16, @CMX_NN>) outputs(%arg3 : memref<1x1x800x1000xf16>) -> memref<1x1x800x1000xf16>
    }
    %5 = VPUIP.Copy inputs(%4 : memref<1x1x800x1000xf16>) outputs(%arg1 : memref<1x1x800x1000xf16>) -> memref<1x1x800x1000xf16>
    return %5 : memref<1x1x800x1000xf16>

    //CHECK:    [[IN_SUBVIEW0:%.*]] = VPUIP.SubView %arg0 [0, 0, 228, 0] [1, 1, 232, 620] : memref<1x1x460x620xf16> to memref<1x1x232x620xf16, {order = #NCHW, strides = [285200, 285200, 620, 1]}>
    //CHECK:    [[ALLOCDISTRIBUTED0:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x232x620xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 1, 119, 620], [1, 1, 117, 620]], compute_offsets = [[0, 0, 0, 0], [0, 0, 115, 0]], memory_shapes = [[1, 1, 119, 620], [1, 1, 117, 620]], memory_offsets = [[0, 0, 0, 0], [0, 0, 115, 0]]}>

    //CHECK:    [[NCECLUSTERTILING0:%.*]] = VPUIP.NCEClusterTiling inputs([[IN_SUBVIEW0]] as %arg2: memref<1x1x232x620xf16, {order = #NCHW, strides = [285200, 285200, 620, 1]}>) outputs([[ALLOCDISTRIBUTED0]] as %arg3: memref<1x1x232x620xf16, @CMX_NN>)
    //CHECK-SAME{LITERAL}:  -> !VPUIP.DistributedBuffer<1x1x232x620xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 119, 620], [1, 1, 117, 620]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 115, 0]], memory_shapes = [[1, 1, 119, 620], [1, 1, 117, 620]], memory_offsets = [[0, 0, 0, 0], [0, 0, 115, 0]]}> {
    //CHECK:        VPUIP.Copy inputs(%arg2 : memref<1x1x232x620xf16, {order = #NCHW, strides = [285200, 285200, 620, 1]}>) outputs(%arg3 : memref<1x1x232x620xf16, @CMX_NN>) -> memref<1x1x232x620xf16, @CMX_NN>
    //CHECK:    }

    //CHECK:    [[IN_SUBVIEW1:%.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 1, 232, 620] : memref<1x1x460x620xf16> to memref<1x1x232x620xf16, {order = #NCHW, strides = [285200, 285200, 620, 1]}>
    //CHECK:    [[ALLOCDISTRIBUTED1:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x232x620xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 117, 620], [1, 1, 119, 620]], compute_offsets = [[0, 0, 0, 0], [0, 0, 113, 0]], memory_shapes = [[1, 1, 117, 620], [1, 1, 119, 620]], memory_offsets = [[0, 0, 0, 0], [0, 0, 113, 0]]}>

    //CHECK:    [[ALLOCDISTRIBUTED2:%.*]] = VPUIP.NCEClusterTiling inputs([[IN_SUBVIEW1]] as %arg2: memref<1x1x232x620xf16, {order = #NCHW, strides = [285200, 285200, 620, 1]}>) outputs([[ALLOCDISTRIBUTED1]] as %arg3: memref<1x1x232x620xf16, @CMX_NN>)
    //CHECK-SAME{LITERAL}: -> !VPUIP.DistributedBuffer<1x1x232x620xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 117, 620], [1, 1, 119, 620]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 113, 0]], memory_shapes = [[1, 1, 117, 620], [1, 1, 119, 620]], memory_offsets = [[0, 0, 0, 0], [0, 0, 113, 0]]}> {
    //CHECK:        VPUIP.Copy inputs(%arg2 : memref<1x1x232x620xf16, {order = #NCHW, strides = [285200, 285200, 620, 1]}>) outputs(%arg3 : memref<1x1x232x620xf16, @CMX_NN>) -> memref<1x1x232x620xf16, @CMX_NN>
    //CHECK:    }

    //CHECK:    [[ALLOCDISTRIBUTED3:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x400x1000xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:    [[ALLOCDISTRIBUTED4:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x400x1000xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[NCECLUSTERTILING1:%.*]]:2 = VPUIP.NCEClusterTiling inputs([[ALLOCDISTRIBUTED2]] as %arg2: memref<1x1x232x620xf16, @CMX_NN>, [[NCECLUSTERTILING0]] as %arg3: memref<1x1x232x620xf16, @CMX_NN>)
    //CHECK-SAME:   outputs([[ALLOCDISTRIBUTED4]] as %arg4: memref<1x1x400x1000xf16, @CMX_NN>, [[ALLOCDISTRIBUTED3]] as %arg5: memref<1x1x400x1000xf16, @CMX_NN>)
    //CHECK-SAME:   -> (!VPUIP.DistributedBuffer<1x1x400x1000xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    //CHECK-SAME:   !VPUIP.DistributedBuffer<1x1x400x1000xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) {
    //CHECK:            VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Interpolate inputs(%arg2 as %arg6: memref<1x1x232x620xf16, @CMX_NN>, %arg3 as %arg7: memref<1x1x232x620xf16, @CMX_NN>)
    //CHECK-SAME:       outputs(%arg4 as %arg8: memref<1x1x400x1000xf16, @CMX_NN>, %arg5 as %arg9: memref<1x1x400x1000xf16, @CMX_NN>) on tile 0 -> (memref<1x1x400x1000xf16, @CMX_NN>, memref<1x1x400x1000xf16, @CMX_NN>){
    //CHECK:                VPUIP.SW.Kernel.run {attrs = [3, 1, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [620, 460, 1, 1], [1000, 800, 1, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}(%arg6, %arg8) :
    //CHECK-SAME:           memref<1x1x232x620xf16, @CMX_NN>, memref<1x1x400x1000xf16, @CMX_NN>
    //CHECK:                VPUIP.SW.Kernel.run {attrs = [3, 1, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [620, 460, 1, 1], [1000, 800, 1, 1], [2, 3], -7.500000e-01, [0, 228, 0, 0], [0, 400, 0, 0]]}(%arg7, %arg9) :
    //CHECK-SAME:           memref<1x1x232x620xf16, @CMX_NN>, memref<1x1x400x1000xf16, @CMX_NN>
    //CHECK:            }
    //CHECK:        }

    //CHECK:    [[ALLOC:%.*]] = memref.alloc() : memref<1x1x800x1000xf16>
    //CHECK:    [[OUT_SUBVIEW0:%.*]] = VPUIP.SubView [[ALLOC]] [0, 0, 0, 0] [1, 1, 400, 1000] : memref<1x1x800x1000xf16> to memref<1x1x400x1000xf16, {order = #NCHW, strides = [800000, 800000, 1000, 1]}>

    //CHECK:        [[NCECLUSTERTILING2:%.*]]  = VPUIP.NCEClusterTiling inputs([[NCECLUSTERTILING1]]#0 as %arg2: memref<1x1x400x1000xf16, @CMX_NN>) outputs([[OUT_SUBVIEW0]] as %arg3: memref<1x1x400x1000xf16,
    //CHECK-SAME:   {order = #NCHW, strides = [800000, 800000, 1000, 1]}>) -> memref<1x1x400x1000xf16, {order = #NCHW, strides = [800000, 800000, 1000, 1]}> {
    //CHECK:            VPUIP.Copy inputs(%arg2 : memref<1x1x400x1000xf16, @CMX_NN>) outputs(%arg3 : memref<1x1x400x1000xf16, {order = #NCHW, strides = [800000, 800000, 1000, 1]}>)
    //CHECK-SAME:       -> memref<1x1x400x1000xf16, {order = #NCHW, strides = [800000, 800000, 1000, 1]}>
    //CHECK:        }

    //CHECK:    [[OUT_SUBVIEW1:%.*]] = VPUIP.SubView [[ALLOC]] [0, 0, 400, 0] [1, 1, 400, 1000] : memref<1x1x800x1000xf16> to memref<1x1x400x1000xf16, {order = #NCHW, strides = [800000, 800000, 1000, 1]}>

    //CHECK:        [[NCECLUSTERTILING3:%.*]] = VPUIP.NCEClusterTiling inputs([[NCECLUSTERTILING1]]#1 as %arg2: memref<1x1x400x1000xf16, @CMX_NN>) outputs(%11 as %arg3: memref<1x1x400x1000xf16,
    //CHECK-SAME:   {order = #NCHW, strides = [800000, 800000, 1000, 1]}>) -> memref<1x1x400x1000xf16, {order = #NCHW, strides = [800000, 800000, 1000, 1]}> {
    //CHECK:            VPUIP.Copy inputs(%arg2 : memref<1x1x400x1000xf16, @CMX_NN>) outputs(%arg3 : memref<1x1x400x1000xf16, {order = #NCHW, strides = [800000, 800000, 1000, 1]}>)
    //CHECK-SAME:       -> memref<1x1x400x1000xf16, {order = #NCHW, strides = [800000, 800000, 1000, 1]}>
    //CHECK:        }

    //CHECK:        [[CONCATVIEW:%.*]] = VPUIP.ConcatView inputs([[NCECLUSTERTILING2]], [[NCECLUSTERTILING3]] : memref<1x1x400x1000xf16, {order = #NCHW, strides = [800000, 800000, 1000, 1]}>,
    //CHECK-SAME:   memref<1x1x400x1000xf16, {order = #NCHW, strides = [800000, 800000, 1000, 1]}>) outputs([[ALLOC]] : memref<1x1x800x1000xf16>) -> memref<1x1x800x1000xf16>

    //CHECK:    [[COPY:%.*]] = VPUIP.Copy inputs([[CONCATVIEW]] : memref<1x1x800x1000xf16>) outputs(%arg1 : memref<1x1x800x1000xf16>) -> memref<1x1x800x1000xf16>
    //CHECK:    return [[COPY]] : memref<1x1x800x1000xf16>
}

// -----


#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
func.func private @builtin_ReduceSum(memref<*xf32, @CMX_NN>, memref<*xf32, @CMX_NN>, i64, i64, none) attributes {VPU.kernel_code = "reduce_sum.cpp", VPU.kernel_entry = "reduce_sum", VPU.task_type = @COMPUTE}
func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL:   func.func @TileReduceSum1(
// CHECK-SAME:                    %[[VAL_0:.*]]: memref<1x1024x7x7xf32>,
// CHECK-SAME:                    %[[VAL_1:.*]]: memref<1x1024x1x1xf32>) -> memref<1x1024x1x1xf32> {
func.func @TileReduceSum1(%arg0: memref<1x1024x7x7xf32>, %arg1: memref<1x1024x1x1xf32>) -> memref<1x1024x1x1xf32> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1024x7x7xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg2: memref<1x1024x7x7xf32>) outputs(%0 as %arg3: memref<1x1024x7x7xf32, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x1024x7x7xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
      %6 = VPUIP.Copy inputs(%arg2 : memref<1x1024x7x7xf32>) outputs(%arg3 : memref<1x1024x7x7xf32, @CMX_NN>) -> memref<1x1024x7x7xf32, @CMX_NN>
    }
    %2 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1024x1x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    %3 = VPUIP.NCEClusterTiling inputs(%1 as %arg2: memref<1x1024x7x7xf32, @CMX_NN>) outputs(%2 as %arg3: memref<1x1024x1x1xf32, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x1024x1x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_ReduceSum inputs(%arg2 as %arg4: memref<1x1024x7x7xf32, @CMX_NN>) outputs(%arg3 as %arg5: memref<1x1024x1x1xf32, @CMX_NN>) on tile 0 -> memref<1x1024x1x1xf32, @CMX_NN>{
        VPUIP.SW.Kernel.run {attrs = [1, 2, [1, 0]]}(%arg4, %arg5) : memref<1x1024x7x7xf32, @CMX_NN>, memref<1x1024x1x1xf32, @CMX_NN>
      }
    }
    %alloc = memref.alloc() : memref<1x1024x1x1xf32>
    %4 = VPUIP.NCEClusterTiling inputs(%3 as %arg2: memref<1x1024x1x1xf32, @CMX_NN>) outputs(%alloc as %arg3: memref<1x1024x1x1xf32>) -> memref<1x1024x1x1xf32> {
      %6 = VPUIP.Copy inputs(%arg2 : memref<1x1024x1x1xf32, @CMX_NN>) outputs(%arg3 : memref<1x1024x1x1xf32>) -> memref<1x1024x1x1xf32>
    }
    %5 = VPUIP.Copy inputs(%4 : memref<1x1024x1x1xf32>) outputs(%arg1 : memref<1x1024x1x1xf32>) -> memref<1x1024x1x1xf32>
    return %5 : memref<1x1024x1x1xf32>

    // CHECK:   %[[VAL_2:.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1024x7x7xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:   %[[VAL_3:.*]] = VPUIP.NCEClusterTiling inputs(%[[VAL_0]] as %[[VAL_4:.*]]: memref<1x1024x7x7xf32>) outputs(%[[VAL_2]] as %[[VAL_5:.*]]: memref<1x1024x7x7xf32, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x1024x7x7xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> {
    // CHECK:    %[[VAL_6:.*]] = VPUIP.Copy inputs(%[[VAL_4]] : memref<1x1024x7x7xf32>) outputs(%[[VAL_5]] : memref<1x1024x7x7xf32, @CMX_NN>) -> memref<1x1024x7x7xf32, @CMX_NN>
    // CHECK:   }
    // CHECK:   %[[VAL_7:.*]] = VPUIP.SubView %[[VAL_3]] [0, 512, 0, 0] [1, 512, 7, 7] : !VPUIP.DistributedBuffer<1x1024x7x7xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:   %[[VAL_8:.*]] = VPUIP.SubView %[[VAL_3]] [0, 0, 0, 0] [1, 512, 7, 7] : !VPUIP.DistributedBuffer<1x1024x7x7xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:   %[[VAL_9:.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1024x1x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:   %[[VAL_10:.*]] = VPUIP.SubView %[[VAL_9]] [0, 512, 0, 0] [1, 512, 1, 1] : !VPUIP.DistributedBuffer<1x1024x1x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:   %[[VAL_11:.*]] = VPUIP.SubView %[[VAL_9]] [0, 0, 0, 0] [1, 512, 1, 1] : !VPUIP.DistributedBuffer<1x1024x1x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:   %[[VAL_12:.*]]:2 = VPUIP.NCEClusterTiling inputs(%[[VAL_8]] as %[[VAL_13:.*]]: memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, @CMX_NN>, %[[VAL_7]] as %[[VAL_14:.*]]: memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, @CMX_NN>) outputs(%[[VAL_11]] as %[[VAL_15:.*]]: memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, @CMX_NN>, %[[VAL_10]] as %[[VAL_16:.*]]: memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, @CMX_NN>) -> (!VPUIP.DistributedBuffer<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) {
    // CHECK:    %[[VAL_17:.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_ReduceSum inputs(%[[VAL_13]] as %[[VAL_18:.*]]: memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, @CMX_NN>, %[[VAL_14]] as %[[VAL_19:.*]]: memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, @CMX_NN>) outputs(%[[VAL_15]] as %[[VAL_20:.*]]: memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, @CMX_NN>, %[[VAL_16]] as %[[VAL_21:.*]]: memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, @CMX_NN>) strides({{\[\[}}512, 1, 1, 1], [512, 1, 1, 1]]) on tile 0 -> (memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, @CMX_NN>, memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, @CMX_NN>){
    // CHECK:      VPUIP.SW.Kernel.run {attrs = [1, 2, [1, 0]]}(%[[VAL_18]], %[[VAL_20]]) : memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, @CMX_NN>, memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, @CMX_NN>
    // CHECK:      VPUIP.SW.Kernel.run {attrs = [1, 2, [1, 0]]}(%[[VAL_19]], %[[VAL_21]]) : memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, @CMX_NN>, memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, @CMX_NN>
    // CHECK:    }
    // CHECK:   }
    // CHECK:   %[[VAL_22:.*]] = VPUIP.ConcatView inputs(%[[VAL_23:.*]]#0, %[[VAL_23]]#1 : !VPUIP.DistributedBuffer<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) outputs(%[[VAL_9]] : !VPUIP.DistributedBuffer<1x1024x1x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x1024x1x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:   %[[VAL_24:.*]] = memref.alloc() : memref<1x1024x1x1xf32>
    // CHECK:   %[[VAL_25:.*]] = VPUIP.NCEClusterTiling inputs(%[[VAL_22]] as %[[VAL_26:.*]]: memref<1x1024x1x1xf32, @CMX_NN>) outputs(%[[VAL_24]] as %[[VAL_27:.*]]: memref<1x1024x1x1xf32>) -> memref<1x1024x1x1xf32> {
    // CHECK:    %[[VAL_28:.*]] = VPUIP.Copy inputs(%[[VAL_26]] : memref<1x1024x1x1xf32, @CMX_NN>) outputs(%[[VAL_27]] : memref<1x1024x1x1xf32>) -> memref<1x1024x1x1xf32>
    // CHECK:   }
    // CHECK:   %[[VAL_29:.*]] = VPUIP.Copy inputs(%[[VAL_25]] : memref<1x1024x1x1xf32>) outputs(%[[VAL_1]] : memref<1x1024x1x1xf32>) -> memref<1x1024x1x1xf32>
    // CHECK:   return %[[VAL_29]] : memref<1x1024x1x1xf32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
func.func private @builtin_ReduceSum(memref<*xf32, @CMX_NN>, memref<*xf32, @CMX_NN>, i64, i64, none) attributes {VPU.kernel_code = "reduce_sum.cpp", VPU.kernel_entry = "reduce_sum", VPU.task_type = @COMPUTE}
func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL:   func.func @TileReduceSum2(
// CHECK-SAME:                    %[[VAL_0:.*]]: memref<1x16x32x64xf32>,
// CHECK-SAME:                    %[[VAL_1:.*]]: memref<1x1x32x1xf32>) -> memref<1x1x32x1xf32> {
func.func @TileReduceSum2(%arg0: memref<1x16x32x64xf32>, %arg1: memref<1x1x32x1xf32>) -> memref<1x1x32x1xf32> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x32x64xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg2: memref<1x16x32x64xf32>) outputs(%0 as %arg3: memref<1x16x32x64xf32, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x16x32x64xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %6 = VPUIP.Copy inputs(%arg2 : memref<1x16x32x64xf32>) outputs(%arg3 : memref<1x16x32x64xf32, @CMX_NN>) -> memref<1x16x32x64xf32, @CMX_NN>
    }
    %2 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x32x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPUIP.NCEClusterTiling inputs(%1 as %arg2: memref<1x16x32x64xf32, @CMX_NN>) outputs(%2 as %arg3: memref<1x1x32x1xf32, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x1x32x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_ReduceSum inputs(%arg2 as %arg4: memref<1x16x32x64xf32, @CMX_NN>) outputs(%arg3 as %arg5: memref<1x1x32x1xf32, @CMX_NN>) on tile 0 -> memref<1x1x32x1xf32, @CMX_NN>{
        VPUIP.SW.Kernel.run {attrs = [1, 2, [2, 0]]}(%arg4, %arg5) : memref<1x16x32x64xf32, @CMX_NN>, memref<1x1x32x1xf32, @CMX_NN>
      }
    }
    %alloc = memref.alloc() : memref<1x1x32x1xf32>
    %4 = VPUIP.NCEClusterTiling inputs(%3 as %arg2: memref<1x1x32x1xf32, @CMX_NN>) outputs(%alloc as %arg3: memref<1x1x32x1xf32>) -> memref<1x1x32x1xf32> {
      %6 = VPUIP.Copy inputs(%arg2 : memref<1x1x32x1xf32, @CMX_NN>) outputs(%arg3 : memref<1x1x32x1xf32>) -> memref<1x1x32x1xf32>
    }
    %5 = VPUIP.Copy inputs(%4 : memref<1x1x32x1xf32>) outputs(%arg1 : memref<1x1x32x1xf32>) -> memref<1x1x32x1xf32>
    return %5 : memref<1x1x32x1xf32>

    // CHECK:   %[[VAL_2:.*]] = VPUIP.SubView %[[VAL_0]] [0, 0, 16, 0] [1, 16, 16, 64] : memref<1x16x32x64xf32> to memref<1x16x16x64xf32, {order = #NCHW, strides = [32768, 2048, 64, 1]}>
    // CHECK:   %[[VAL_3:.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x16x64xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:   %[[VAL_4:.*]] = VPUIP.NCEClusterTiling inputs(%[[VAL_2]] as %[[VAL_5:.*]]: memref<1x16x16x64xf32, {order = #NCHW, strides = [32768, 2048, 64, 1]}>) outputs(%[[VAL_3]] as %[[VAL_6:.*]]: memref<1x16x16x64xf32, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x16x16x64xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:    %[[VAL_7:.*]] = VPUIP.Copy inputs(%[[VAL_5]] : memref<1x16x16x64xf32, {order = #NCHW, strides = [32768, 2048, 64, 1]}>) outputs(%[[VAL_6]] : memref<1x16x16x64xf32, @CMX_NN>) -> memref<1x16x16x64xf32, @CMX_NN>
    // CHECK:   }
    // CHECK:   %[[VAL_8:.*]] = VPUIP.SubView %[[VAL_0]] [0, 0, 0, 0] [1, 16, 16, 64] : memref<1x16x32x64xf32> to memref<1x16x16x64xf32, {order = #NCHW, strides = [32768, 2048, 64, 1]}>
    // CHECK:   %[[VAL_9:.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x16x64xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:   %[[VAL_10:.*]] = VPUIP.NCEClusterTiling inputs(%[[VAL_8]] as %[[VAL_11:.*]]: memref<1x16x16x64xf32, {order = #NCHW, strides = [32768, 2048, 64, 1]}>) outputs(%[[VAL_9]] as %[[VAL_12:.*]]: memref<1x16x16x64xf32, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x16x16x64xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:    %[[VAL_13:.*]] = VPUIP.Copy inputs(%[[VAL_11]] : memref<1x16x16x64xf32, {order = #NCHW, strides = [32768, 2048, 64, 1]}>) outputs(%[[VAL_12]] : memref<1x16x16x64xf32, @CMX_NN>) -> memref<1x16x16x64xf32, @CMX_NN>
    // CHECK:   }
    // CHECK:   %[[VAL_14:.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x16x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:   %[[VAL_15:.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x16x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:   %[[VAL_16:.*]]:2 = VPUIP.NCEClusterTiling inputs(%[[VAL_10]] as %[[VAL_17:.*]]: memref<1x16x16x64xf32, @CMX_NN>, %[[VAL_4]] as %[[VAL_18:.*]]: memref<1x16x16x64xf32, @CMX_NN>) outputs(%[[VAL_15]] as %[[VAL_19:.*]]: memref<1x1x16x1xf32, @CMX_NN>, %[[VAL_14]] as %[[VAL_20:.*]]: memref<1x1x16x1xf32, @CMX_NN>) -> (!VPUIP.DistributedBuffer<1x1x16x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x16x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) {
    // CHECK:    %[[VAL_21:.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_ReduceSum inputs(%[[VAL_17]] as %[[VAL_22:.*]]: memref<1x16x16x64xf32, @CMX_NN>, %[[VAL_18]] as %[[VAL_23:.*]]: memref<1x16x16x64xf32, @CMX_NN>) outputs(%[[VAL_19]] as %[[VAL_24:.*]]: memref<1x1x16x1xf32, @CMX_NN>, %[[VAL_20]] as %[[VAL_25:.*]]: memref<1x1x16x1xf32, @CMX_NN>) on tile 0 -> (memref<1x1x16x1xf32, @CMX_NN>, memref<1x1x16x1xf32, @CMX_NN>){
    // CHECK:      VPUIP.SW.Kernel.run {attrs = [1, 2, [2, 0]]}(%[[VAL_22]], %[[VAL_24]]) : memref<1x16x16x64xf32, @CMX_NN>, memref<1x1x16x1xf32, @CMX_NN>
    // CHECK:      VPUIP.SW.Kernel.run {attrs = [1, 2, [2, 0]]}(%[[VAL_23]], %[[VAL_25]]) : memref<1x16x16x64xf32, @CMX_NN>, memref<1x1x16x1xf32, @CMX_NN>
    // CHECK:    }
    // CHECK:   }
    // CHECK:   %[[VAL_26:.*]] = memref.alloc() : memref<1x1x32x1xf32>
    // CHECK:   %[[VAL_27:.*]] = VPUIP.SubView %[[VAL_26]] [0, 0, 0, 0] [1, 1, 16, 1] : memref<1x1x32x1xf32> to memref<1x1x16x1xf32, {order = #NCHW, strides = [32, 32, 1, 1]}>
    // CHECK:   %[[VAL_28:.*]] = VPUIP.NCEClusterTiling inputs(%[[VAL_29:.*]]#0 as %[[VAL_30:.*]]: memref<1x1x16x1xf32, @CMX_NN>) outputs(%[[VAL_27]] as %[[VAL_31:.*]]: memref<1x1x16x1xf32, {order = #NCHW, strides = [32, 32, 1, 1]}>) -> memref<1x1x16x1xf32, {order = #NCHW, strides = [32, 32, 1, 1]}> {
    // CHECK:    %[[VAL_32:.*]] = VPUIP.Copy inputs(%[[VAL_30]] : memref<1x1x16x1xf32, @CMX_NN>) outputs(%[[VAL_31]] : memref<1x1x16x1xf32, {order = #NCHW, strides = [32, 32, 1, 1]}>) -> memref<1x1x16x1xf32, {order = #NCHW, strides = [32, 32, 1, 1]}>
    // CHECK:   }
    // CHECK:   %[[VAL_33:.*]] = VPUIP.SubView %[[VAL_26]] [0, 0, 16, 0] [1, 1, 16, 1] : memref<1x1x32x1xf32> to memref<1x1x16x1xf32, {order = #NCHW, strides = [32, 32, 1, 1]}>
    // CHECK:   %[[VAL_34:.*]] = VPUIP.NCEClusterTiling inputs(%[[VAL_35:.*]]#1 as %[[VAL_36:.*]]: memref<1x1x16x1xf32, @CMX_NN>) outputs(%[[VAL_33]] as %[[VAL_37:.*]]: memref<1x1x16x1xf32, {order = #NCHW, strides = [32, 32, 1, 1]}>) -> memref<1x1x16x1xf32, {order = #NCHW, strides = [32, 32, 1, 1]}> {
    // CHECK:    %[[VAL_38:.*]] = VPUIP.Copy inputs(%[[VAL_36]] : memref<1x1x16x1xf32, @CMX_NN>) outputs(%[[VAL_37]] : memref<1x1x16x1xf32, {order = #NCHW, strides = [32, 32, 1, 1]}>) -> memref<1x1x16x1xf32, {order = #NCHW, strides = [32, 32, 1, 1]}>
    // CHECK:   }
    // CHECK:   %[[VAL_39:.*]] = VPUIP.ConcatView inputs(%[[VAL_28]], %[[VAL_34]] : memref<1x1x16x1xf32, {order = #NCHW, strides = [32, 32, 1, 1]}>, memref<1x1x16x1xf32, {order = #NCHW, strides = [32, 32, 1, 1]}>) outputs(%[[VAL_26]] : memref<1x1x32x1xf32>) -> memref<1x1x32x1xf32>
    // CHECK:   %[[VAL_40:.*]] = VPUIP.Copy inputs(%[[VAL_39]] : memref<1x1x32x1xf32>) outputs(%[[VAL_1]] : memref<1x1x32x1xf32>) -> memref<1x1x32x1xf32>
    // CHECK:   return %[[VAL_40]] : memref<1x1x32x1xf32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
func.func private @builtin_ReduceL1(memref<*xf32, [@CMX_NN, 0]>, memref<*xf32, [@CMX_NN, 0]>, i64, i64, none) attributes {VPU.kernel_code = "reduce_l1.cpp", VPU.kernel_entry = "reduce_l1", VPU.task_type = @COMPUTE}
func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: func.func @TileReduceL1CMX(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<1x1024x7x7xf32, [@CMX_NN, 0]>,
// CHECK-SAME:      %[[VAL_1:.*]]: memref<1x1024x1x1xf32, [@CMX_NN, 0]>) -> memref<1x1024x1x1xf32, [@CMX_NN, 0]> {
func.func @TileReduceL1CMX(%arg0: memref<1x1024x7x7xf32, [@CMX_NN, 0]>, %arg1: memref<1x1024x1x1xf32, [@CMX_NN, 0]>) -> memref<1x1024x1x1xf32, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x1024x1x1xf32, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_ReduceL1 inputs(%arg0 as %arg2: memref<1x1024x7x7xf32, [@CMX_NN, 0]>) outputs(%0 as %arg3: memref<1x1024x1x1xf32, [@CMX_NN, 0]>) on tile 0 -> memref<1x1024x1x1xf32, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [1, 2, [1, 0]]}(%arg2, %arg3) : memref<1x1024x7x7xf32, [@CMX_NN, 0]>, memref<1x1024x1x1xf32, [@CMX_NN, 0]>
    }
    return %results : memref<1x1024x1x1xf32, [@CMX_NN, 0]>

    // CHECK:   %[[VAL_2:.*]] = memref.alloc() : memref<1x1024x1x1xf32, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_3:.*]] = VPUIP.SubView %[[VAL_0]] [0, 0, 0, 0] [1, 512, 7, 7] : memref<1x1024x7x7xf32, [@CMX_NN, 0]> to memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_4:.*]] = VPUIP.SubView %[[VAL_2]] [0, 0, 0, 0] [1, 512, 1, 1] : memref<1x1024x1x1xf32, [@CMX_NN, 0]> to memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_5:.*]] = VPUIP.SubView %[[VAL_0]] [0, 512, 0, 0] [1, 512, 7, 7] : memref<1x1024x7x7xf32, [@CMX_NN, 0]> to memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_6:.*]] = VPUIP.SubView %[[VAL_2]] [0, 512, 0, 0] [1, 512, 1, 1] : memref<1x1024x1x1xf32, [@CMX_NN, 0]> to memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_7:.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_ReduceL1 inputs(%[[VAL_3]] as %[[VAL_8:.*]]: memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>, %[[VAL_5]] as %[[VAL_9:.*]]: memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>) outputs(%[[VAL_4]] as %[[VAL_10:.*]]: memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>, %[[VAL_6]] as %[[VAL_11:.*]]: memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>, memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>){
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [1, 2, [1, 0]]}(%[[VAL_8]], %[[VAL_10]]) : memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>, memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [1, 2, [1, 0]]}(%[[VAL_9]], %[[VAL_11]]) : memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>, memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:   }
    // CHECK:   %[[VAL_12:.*]] = VPUIP.ConcatView inputs(%[[VAL_13:.*]]#0, %[[VAL_13]]#1 : memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>, memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>) outputs(%[[VAL_2]] : memref<1x1024x1x1xf32, [@CMX_NN, 0]>) -> memref<1x1024x1x1xf32, [@CMX_NN, 0]>
    // CHECK:   return %[[VAL_12]] : memref<1x1024x1x1xf32, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
func.func private @builtin_ReduceL2(memref<*xf32, [@CMX_NN, 0]>, memref<*xf32, [@CMX_NN, 0]>, i64, i64, none) attributes {VPU.kernel_code = "reduce_l2.cpp", VPU.kernel_entry = "reduce_l2", VPU.task_type = @COMPUTE}
func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: func.func @TileReduceL2CMX(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<1x1024x7x7xf32, [@CMX_NN, 0]>,
// CHECK-SAME:      %[[VAL_1:.*]]: memref<1x1x7x1xf32, [@CMX_NN, 0]>) -> memref<1x1x7x1xf32, [@CMX_NN, 0]> {
func.func @TileReduceL2CMX(%arg0: memref<1x1024x7x7xf32, [@CMX_NN, 0]>, %arg1: memref<1x1x7x1xf32, [@CMX_NN, 0]>) -> memref<1x1x7x1xf32, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x1x7x1xf32, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_ReduceL2 inputs(%arg0 as %arg2: memref<1x1024x7x7xf32, [@CMX_NN, 0]>) outputs(%0 as %arg3: memref<1x1x7x1xf32, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x7x1xf32, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [1, 2, [2, 0]]}(%arg2, %arg3) : memref<1x1024x7x7xf32, [@CMX_NN, 0]>, memref<1x1x7x1xf32, [@CMX_NN, 0]>
    }
    return %results : memref<1x1x7x1xf32, [@CMX_NN, 0]>

    // CHECK:   %[[VAL_2:.*]] = VPUIP.SubView %[[VAL_0]] [0, 0, 0, 0] [1, 1024, 4, 7] : memref<1x1024x7x7xf32, [@CMX_NN, 0]> to memref<1x1024x4x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_3:.*]] = memref.alloc() : memref<1x1024x4x7xf32, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_4:.*]] = VPUIP.Copy inputs(%[[VAL_2]] : memref<1x1024x4x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>) outputs(%[[VAL_3]] : memref<1x1024x4x7xf32, [@CMX_NN, 0]>) -> memref<1x1024x4x7xf32, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_5:.*]] = memref.alloc() : memref<1x1x4x1xf32, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_6:.*]] = VPUIP.SubView %[[VAL_0]] [0, 0, 4, 0] [1, 1024, 3, 7] : memref<1x1024x7x7xf32, [@CMX_NN, 0]> to memref<1x1024x3x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_7:.*]] = memref.alloc() : memref<1x1024x3x7xf32, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_8:.*]] = VPUIP.Copy inputs(%[[VAL_6]] : memref<1x1024x3x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>) outputs(%[[VAL_7]] : memref<1x1024x3x7xf32, [@CMX_NN, 0]>) -> memref<1x1024x3x7xf32, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_9:.*]] = memref.alloc() : memref<1x1x3x1xf32, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_10:.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_ReduceL2 inputs(%[[VAL_4]] as %[[VAL_11:.*]]: memref<1x1024x4x7xf32, [@CMX_NN, 0]>, %[[VAL_8]] as %[[VAL_12:.*]]: memref<1x1024x3x7xf32, [@CMX_NN, 0]>) outputs(%[[VAL_5]] as %[[VAL_13:.*]]: memref<1x1x4x1xf32, [@CMX_NN, 0]>, %[[VAL_9]] as %[[VAL_14:.*]]: memref<1x1x3x1xf32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x4x1xf32, [@CMX_NN, 0]>, memref<1x1x3x1xf32, [@CMX_NN, 0]>){
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [1, 2, [2, 0]]}(%[[VAL_11]], %[[VAL_13]]) : memref<1x1024x4x7xf32, [@CMX_NN, 0]>, memref<1x1x4x1xf32, [@CMX_NN, 0]>
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [1, 2, [2, 0]]}(%[[VAL_12]], %[[VAL_14]]) : memref<1x1024x3x7xf32, [@CMX_NN, 0]>, memref<1x1x3x1xf32, [@CMX_NN, 0]>
    // CHECK:   }
    // CHECK:   %[[VAL_15:.*]] = memref.alloc() : memref<1x1x7x1xf32, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_16:.*]] = VPUIP.SubView %[[VAL_15]] [0, 0, 0, 0] [1, 1, 4, 1] : memref<1x1x7x1xf32, [@CMX_NN, 0]> to memref<1x1x4x1xf32, {order = #NCHW, strides = [7, 7, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_17:.*]] = VPUIP.Copy inputs(%[[VAL_18:.*]]#0 : memref<1x1x4x1xf32, [@CMX_NN, 0]>) outputs(%[[VAL_16]] : memref<1x1x4x1xf32, {order = #NCHW, strides = [7, 7, 1, 1]}, [@CMX_NN, 0]>) -> memref<1x1x4x1xf32, {order = #NCHW, strides = [7, 7, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_19:.*]] = VPUIP.SubView %[[VAL_15]] [0, 0, 4, 0] [1, 1, 3, 1] : memref<1x1x7x1xf32, [@CMX_NN, 0]> to memref<1x1x3x1xf32, {order = #NCHW, strides = [7, 7, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_20:.*]] = VPUIP.Copy inputs(%[[VAL_18]]#1 : memref<1x1x3x1xf32, [@CMX_NN, 0]>) outputs(%[[VAL_19]] : memref<1x1x3x1xf32, {order = #NCHW, strides = [7, 7, 1, 1]}, [@CMX_NN, 0]>) -> memref<1x1x3x1xf32, {order = #NCHW, strides = [7, 7, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_21:.*]] = VPUIP.ConcatView inputs(%[[VAL_17]], %[[VAL_20]] : memref<1x1x4x1xf32, {order = #NCHW, strides = [7, 7, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x3x1xf32, {order = #NCHW, strides = [7, 7, 1, 1]}, [@CMX_NN, 0]>) outputs(%[[VAL_15]] : memref<1x1x7x1xf32, [@CMX_NN, 0]>) -> memref<1x1x7x1xf32, [@CMX_NN, 0]>
    // CHECK:   return %[[VAL_21]] : memref<1x1x7x1xf32, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
func.func private @builtin_ReduceLogicalAnd(memref<*xf32, [@CMX_NN, 0]>, memref<*xf32, [@CMX_NN, 0]>, i64, i64, none) attributes {VPU.kernel_code = "reduce_logical_and.cpp", VPU.kernel_entry = "reduce_logical_and", VPU.task_type = @COMPUTE}
func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: func.func @TileReduceLogicalAndCMX(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<1x1024x7x7xf32, [@CMX_NN, 0]>,
// CHECK-SAME:      %[[VAL_1:.*]]: memref<1x1x1x7xf32, [@CMX_NN, 0]>) -> memref<1x1x1x7xf32, [@CMX_NN, 0]> {
func.func @TileReduceLogicalAndCMX(%arg0: memref<1x1024x7x7xf32, [@CMX_NN, 0]>, %arg1: memref<1x1x1x7xf32, [@CMX_NN, 0]>) -> memref<1x1x1x7xf32, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x1x1x7xf32, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_ReduceLogicalAnd inputs(%arg0 as %arg2: memref<1x1024x7x7xf32, [@CMX_NN, 0]>) outputs(%0 as %arg3: memref<1x1x1x7xf32, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x7xf32, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [1, 2, [2, 1]]}(%arg2, %arg3) : memref<1x1024x7x7xf32, [@CMX_NN, 0]>, memref<1x1x1x7xf32, [@CMX_NN, 0]>
    }
    return %results : memref<1x1x1x7xf32, [@CMX_NN, 0]>

    // CHECK:   %[[VAL_2:.*]] = VPUIP.SubView %[[VAL_0]] [0, 0, 0, 0] [1, 1024, 7, 4] : memref<1x1024x7x7xf32, [@CMX_NN, 0]> to memref<1x1024x7x4xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_3:.*]] = memref.alloc() : memref<1x1024x7x4xf32, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_4:.*]] = VPUIP.Copy inputs(%[[VAL_2]] : memref<1x1024x7x4xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>) outputs(%[[VAL_3]] : memref<1x1024x7x4xf32, [@CMX_NN, 0]>) -> memref<1x1024x7x4xf32, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_5:.*]] = memref.alloc() : memref<1x1x1x4xf32, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_6:.*]] = VPUIP.SubView %[[VAL_0]] [0, 0, 0, 4] [1, 1024, 7, 3] : memref<1x1024x7x7xf32, [@CMX_NN, 0]> to memref<1x1024x7x3xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_7:.*]] = memref.alloc() : memref<1x1024x7x3xf32, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_8:.*]] = VPUIP.Copy inputs(%[[VAL_6]] : memref<1x1024x7x3xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>) outputs(%[[VAL_7]] : memref<1x1024x7x3xf32, [@CMX_NN, 0]>) -> memref<1x1024x7x3xf32, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_9:.*]] = memref.alloc() : memref<1x1x1x3xf32, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_10:.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_ReduceLogicalAnd inputs(%[[VAL_4]] as %[[VAL_11:.*]]: memref<1x1024x7x4xf32, [@CMX_NN, 0]>, %[[VAL_8]] as %[[VAL_12:.*]]: memref<1x1024x7x3xf32, [@CMX_NN, 0]>) outputs(%[[VAL_5]] as %[[VAL_13:.*]]: memref<1x1x1x4xf32, [@CMX_NN, 0]>, %[[VAL_9]] as %[[VAL_14:.*]]: memref<1x1x1x3xf32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x1x4xf32, [@CMX_NN, 0]>, memref<1x1x1x3xf32, [@CMX_NN, 0]>){
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [1, 2, [2, 1]]}(%[[VAL_11]], %[[VAL_13]]) : memref<1x1024x7x4xf32, [@CMX_NN, 0]>, memref<1x1x1x4xf32, [@CMX_NN, 0]>
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [1, 2, [2, 1]]}(%[[VAL_12]], %[[VAL_14]]) : memref<1x1024x7x3xf32, [@CMX_NN, 0]>, memref<1x1x1x3xf32, [@CMX_NN, 0]>
    // CHECK:   }
    // CHECK:   %[[VAL_15:.*]] = memref.alloc() : memref<1x1x1x7xf32, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_16:.*]] = VPUIP.SubView %[[VAL_15]] [0, 0, 0, 0] [1, 1, 1, 4] : memref<1x1x1x7xf32, [@CMX_NN, 0]> to memref<1x1x1x4xf32, {order = #NCHW, strides = [7, 7, 7, 1]}, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_17:.*]] = VPUIP.Copy inputs(%[[VAL_18:.*]]#0 : memref<1x1x1x4xf32, [@CMX_NN, 0]>) outputs(%[[VAL_16]] : memref<1x1x1x4xf32, {order = #NCHW, strides = [7, 7, 7, 1]}, [@CMX_NN, 0]>) -> memref<1x1x1x4xf32, {order = #NCHW, strides = [7, 7, 7, 1]}, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_19:.*]] = VPUIP.SubView %[[VAL_15]] [0, 0, 0, 4] [1, 1, 1, 3] : memref<1x1x1x7xf32, [@CMX_NN, 0]> to memref<1x1x1x3xf32, {order = #NCHW, strides = [7, 7, 7, 1]}, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_20:.*]] = VPUIP.Copy inputs(%[[VAL_18]]#1 : memref<1x1x1x3xf32, [@CMX_NN, 0]>) outputs(%[[VAL_19]] : memref<1x1x1x3xf32, {order = #NCHW, strides = [7, 7, 7, 1]}, [@CMX_NN, 0]>) -> memref<1x1x1x3xf32, {order = #NCHW, strides = [7, 7, 7, 1]}, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_21:.*]] = VPUIP.ConcatView inputs(%[[VAL_17]], %[[VAL_20]] : memref<1x1x1x4xf32, {order = #NCHW, strides = [7, 7, 7, 1]}, [@CMX_NN, 0]>, memref<1x1x1x3xf32, {order = #NCHW, strides = [7, 7, 7, 1]}, [@CMX_NN, 0]>) outputs(%[[VAL_15]] : memref<1x1x1x7xf32, [@CMX_NN, 0]>) -> memref<1x1x1x7xf32, [@CMX_NN, 0]>
    // CHECK:   return %[[VAL_21]] : memref<1x1x1x7xf32, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
func.func private @builtin_ReduceLogicalOr(memref<*xf32, [@CMX_NN, 0]>, memref<*xf32, [@CMX_NN, 0]>, i64, i64, none) attributes {VPU.kernel_code = "reduce_logical_or.cpp", VPU.kernel_entry = "reduce_logical_or", VPU.task_type = @COMPUTE}
func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: func.func @TileReduceLogicalOrCMXShort(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<1x1024x7x7xf32, [@CMX_NN, 0]>,
// CHECK-SAME:      %[[VAL_1:.*]]: memref<1x1024x1x1xf32, [@CMX_NN, 0]>) -> memref<1x1024x1x1xf32, [@CMX_NN, 0]> {
func.func @TileReduceLogicalOrCMXShort(%arg0: memref<1x1024x7x7xf32, [@CMX_NN, 0]>, %arg1: memref<1x1024x1x1xf32, [@CMX_NN, 0]>) -> memref<1x1024x1x1xf32, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x1024x1x1xf32, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_ReduceLogicalOr inputs(%arg0 as %arg2: memref<1x1024x7x7xf32, [@CMX_NN, 0]>) outputs(%0 as %arg3: memref<1x1024x1x1xf32, [@CMX_NN, 0]>) on tile 0 -> memref<1x1024x1x1xf32, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [1, 2, [1, 0]]}(%arg2, %arg3) : memref<1x1024x7x7xf32, [@CMX_NN, 0]>, memref<1x1024x1x1xf32, [@CMX_NN, 0]>
    }
    return %results : memref<1x1024x1x1xf32, [@CMX_NN, 0]>

    // CHECK:   %[[VAL_1:.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_ReduceLogicalOr inputs({{[^:]+}} as %[[VAL_2:.*]]: memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>, {{[^:]+}} as %[[VAL_3:.*]]: memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>) outputs({{[^:]+}} as %[[VAL_4:.*]]: memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>, {{[^:]+}} as %[[VAL_5:.*]]: memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>, memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>){
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [1, 2, [1, 0]]}(%[[VAL_2]], %[[VAL_4]]) : memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>, memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [1, 2, [1, 0]]}(%[[VAL_3]], %[[VAL_5]]) : memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>, memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
func.func private @builtin_ReduceMax(memref<*xf32, [@CMX_NN, 0]>, memref<*xf32, [@CMX_NN, 0]>, i64, i64, none) attributes {VPU.kernel_code = "reduce_max.cpp", VPU.kernel_entry = "reduce_max", VPU.task_type = @COMPUTE}
func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: func.func @TileReduceMaxCMXShort(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<1x1024x7x7xf32, [@CMX_NN, 0]>,
// CHECK-SAME:      %[[VAL_1:.*]]: memref<1x1024x1x1xf32, [@CMX_NN, 0]>) -> memref<1x1024x1x1xf32, [@CMX_NN, 0]> {
func.func @TileReduceMaxCMXShort(%arg0: memref<1x1024x7x7xf32, [@CMX_NN, 0]>, %arg1: memref<1x1024x1x1xf32, [@CMX_NN, 0]>) -> memref<1x1024x1x1xf32, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x1024x1x1xf32, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_ReduceMax inputs(%arg0 as %arg2: memref<1x1024x7x7xf32, [@CMX_NN, 0]>) outputs(%0 as %arg3: memref<1x1024x1x1xf32, [@CMX_NN, 0]>) on tile 0 -> memref<1x1024x1x1xf32, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [1, 2, [1, 0]]}(%arg2, %arg3) : memref<1x1024x7x7xf32, [@CMX_NN, 0]>, memref<1x1024x1x1xf32, [@CMX_NN, 0]>
    }
    return %results : memref<1x1024x1x1xf32, [@CMX_NN, 0]>

    // CHECK:   %[[VAL_1:.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_ReduceMax inputs({{[^:]+}} as %[[VAL_2:.*]]: memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>, {{[^:]+}} as %[[VAL_3:.*]]: memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>) outputs({{[^:]+}} as %[[VAL_4:.*]]: memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>, {{[^:]+}} as %[[VAL_5:.*]]: memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>, memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>){
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [1, 2, [1, 0]]}(%[[VAL_2]], %[[VAL_4]]) : memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>, memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [1, 2, [1, 0]]}(%[[VAL_3]], %[[VAL_5]]) : memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>, memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
func.func private @builtin_ReduceMean(memref<*xf32, [@CMX_NN, 0]>, memref<*xf32, [@CMX_NN, 0]>, i64, i64, none) attributes {VPU.kernel_code = "reduce_mean.cpp", VPU.kernel_entry = "reduce_mean", VPU.task_type = @COMPUTE}
func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: func.func @TileReduceMeanCMXShort(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<1x1024x7x7xf32, [@CMX_NN, 0]>,
// CHECK-SAME:      %[[VAL_1:.*]]: memref<1x1024x1x1xf32, [@CMX_NN, 0]>) -> memref<1x1024x1x1xf32, [@CMX_NN, 0]> {
func.func @TileReduceMeanCMXShort(%arg0: memref<1x1024x7x7xf32, [@CMX_NN, 0]>, %arg1: memref<1x1024x1x1xf32, [@CMX_NN, 0]>) -> memref<1x1024x1x1xf32, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x1024x1x1xf32, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_ReduceMean inputs(%arg0 as %arg2: memref<1x1024x7x7xf32, [@CMX_NN, 0]>) outputs(%0 as %arg3: memref<1x1024x1x1xf32, [@CMX_NN, 0]>) on tile 0 -> memref<1x1024x1x1xf32, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [1, 2, [1, 0]]}(%arg2, %arg3) : memref<1x1024x7x7xf32, [@CMX_NN, 0]>, memref<1x1024x1x1xf32, [@CMX_NN, 0]>
    }
    return %results : memref<1x1024x1x1xf32, [@CMX_NN, 0]>

    // CHECK:   %[[VAL_1:.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_ReduceMean inputs({{[^:]+}} as %[[VAL_2:.*]]: memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>, {{[^:]+}} as %[[VAL_3:.*]]: memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>) outputs({{[^:]+}} as %[[VAL_4:.*]]: memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>, {{[^:]+}} as %[[VAL_5:.*]]: memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>, memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>){
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [1, 2, [1, 0]]}(%[[VAL_2]], %[[VAL_4]]) : memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>, memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [1, 2, [1, 0]]}(%[[VAL_3]], %[[VAL_5]]) : memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>, memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
func.func private @builtin_ReduceMin(memref<*xf32, [@CMX_NN, 0]>, memref<*xf32, [@CMX_NN, 0]>, i64, i64, none) attributes {VPU.kernel_code = "reduce_min.cpp", VPU.kernel_entry = "reduce_min", VPU.task_type = @COMPUTE}
func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: func.func @TileReduceMinCMXShort(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<1x1024x7x7xf32, [@CMX_NN, 0]>,
// CHECK-SAME:      %[[VAL_1:.*]]: memref<1x1024x1x1xf32, [@CMX_NN, 0]>) -> memref<1x1024x1x1xf32, [@CMX_NN, 0]> {
func.func @TileReduceMinCMXShort(%arg0: memref<1x1024x7x7xf32, [@CMX_NN, 0]>, %arg1: memref<1x1024x1x1xf32, [@CMX_NN, 0]>) -> memref<1x1024x1x1xf32, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x1024x1x1xf32, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_ReduceMin inputs(%arg0 as %arg2: memref<1x1024x7x7xf32, [@CMX_NN, 0]>) outputs(%0 as %arg3: memref<1x1024x1x1xf32, [@CMX_NN, 0]>) on tile 0 -> memref<1x1024x1x1xf32, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [1, 2, [1, 0]]}(%arg2, %arg3) : memref<1x1024x7x7xf32, [@CMX_NN, 0]>, memref<1x1024x1x1xf32, [@CMX_NN, 0]>
    }
    return %results : memref<1x1024x1x1xf32, [@CMX_NN, 0]>

    // CHECK:   %[[VAL_1:.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_ReduceMin inputs({{[^:]+}} as %[[VAL_2:.*]]: memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>, {{[^:]+}} as %[[VAL_3:.*]]: memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>) outputs({{[^:]+}} as %[[VAL_4:.*]]: memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>, {{[^:]+}} as %[[VAL_5:.*]]: memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>, memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>){
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [1, 2, [1, 0]]}(%[[VAL_2]], %[[VAL_4]]) : memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>, memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [1, 2, [1, 0]]}(%[[VAL_3]], %[[VAL_5]]) : memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>, memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
func.func private @builtin_ReduceProd(memref<*xf32, [@CMX_NN, 0]>, memref<*xf32, [@CMX_NN, 0]>, i64, i64, none) attributes {VPU.kernel_code = "reduce_prod.cpp", VPU.kernel_entry = "reduce_prod", VPU.task_type = @COMPUTE}
func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: func.func @TileReduceProdCMXShort(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<1x1024x7x7xf32, [@CMX_NN, 0]>,
// CHECK-SAME:      %[[VAL_1:.*]]: memref<1x1024x1x1xf32, [@CMX_NN, 0]>) -> memref<1x1024x1x1xf32, [@CMX_NN, 0]> {
func.func @TileReduceProdCMXShort(%arg0: memref<1x1024x7x7xf32, [@CMX_NN, 0]>, %arg1: memref<1x1024x1x1xf32, [@CMX_NN, 0]>) -> memref<1x1024x1x1xf32, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x1024x1x1xf32, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_ReduceProd inputs(%arg0 as %arg2: memref<1x1024x7x7xf32, [@CMX_NN, 0]>) outputs(%0 as %arg3: memref<1x1024x1x1xf32, [@CMX_NN, 0]>) on tile 0 -> memref<1x1024x1x1xf32, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [1, 2, [1, 0]]}(%arg2, %arg3) : memref<1x1024x7x7xf32, [@CMX_NN, 0]>, memref<1x1024x1x1xf32, [@CMX_NN, 0]>
    }
    return %results : memref<1x1024x1x1xf32, [@CMX_NN, 0]>

    // CHECK:   %[[VAL_1:.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_ReduceProd inputs({{[^:]+}} as %[[VAL_2:.*]]: memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>, {{[^:]+}} as %[[VAL_3:.*]]: memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>) outputs({{[^:]+}} as %[[VAL_4:.*]]: memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>, {{[^:]+}} as %[[VAL_5:.*]]: memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>, memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>){
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [1, 2, [1, 0]]}(%[[VAL_2]], %[[VAL_4]]) : memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>, memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [1, 2, [1, 0]]}(%[[VAL_3]], %[[VAL_5]]) : memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>, memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
    func.func private @builtin_Floor(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "activation_floor.cpp", VPU.kernel_entry = "activation_floor"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileFloor(%arg0: memref<1x16x16x512xf16, [@CMX_NN, 0]>) -> memref<1x16x16x512xf16, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x16x16x512xf16, [@CMX_NN, 0]>

    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Floor
                  inputs(%arg0 as %arg3: memref<1x16x16x512xf16, #NCHW, [@CMX_NN, 0]>)
                  outputs(%0 as %arg4: memref<1x16x16x512xf16, #NCHW, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x16x512xf16, #NCHW, [@CMX_NN, 0]> {
      VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}(%arg3, %arg4) : memref<1x16x16x512xf16, [@CMX_NN, 0]>, memref<1x16x16x512xf16, [@CMX_NN, 0]>
    }

    return %results : memref<1x16x16x512xf16, [@CMX_NN, 0]>

    // CHECK:    [[OUTPUT_BUF_0:%.*]] = memref.alloc() : memref<1x16x16x512xf16, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_0:%.*]] = VPUIP.SubView {{[^:]+}} [0, 0, 0, 0] [1, 8, 16, 512] : memref<1x16x16x512xf16, [@CMX_NN, 0]> to memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_1:%.*]] = VPUIP.SubView [[OUTPUT_BUF_0]] [0, 0, 0, 0] [1, 8, 16, 512] : memref<1x16x16x512xf16, [@CMX_NN, 0]> to memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_2:%.*]] = VPUIP.SubView {{[^:]+}} [0, 8, 0, 0] [1, 8, 16, 512] : memref<1x16x16x512xf16, [@CMX_NN, 0]> to memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_3:%.*]] = VPUIP.SubView [[OUTPUT_BUF_0]] [0, 8, 0, 0] [1, 8, 16, 512] : memref<1x16x16x512xf16, [@CMX_NN, 0]> to memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[FLOOR:%.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Floor inputs([[SUBVIEW_0]] as [[SUBVIEW_ARG1:%.+]]: memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_2]] as [[SUBVIEW_ARG2:%.+]]: memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>) outputs([[SUBVIEW_1]] as [[SUBVIEW_ARG3:%.+]]: memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_3]] as [[SUBVIEW_ARG4:%.+]]: memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>, memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>){
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}([[SUBVIEW_ARG1]], [[SUBVIEW_ARG3]]) : memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>, memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}([[SUBVIEW_ARG2]], [[SUBVIEW_ARG4]]) : memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>, memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[FLOOR]]#0, [[FLOOR]]#1 : memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>, memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>) outputs(%alloc : memref<1x16x16x512xf16, [@CMX_NN, 0]>) -> memref<1x16x16x512xf16, [@CMX_NN, 0]>
    // CHECK:    return [[CONCAT]] : memref<1x16x16x512xf16, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
  func.func private @builtin_FakeQuantize(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "fake_quantize.cpp", VPU.kernel_entry = "fake_quantize", VPU.task_type = @COMPUTE}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @TileFakeQuantize
// CHECK-SAME:  ([[INPUT_DATA:%.+]]: memref<1x16x32x32xf16>)
func.func @TileFakeQuantize(%arg0: memref<1x16x32x32xf16>) -> memref<1x16x32x32xf16, [@CMX_NN, 0]> {
    %input_buff = memref.alloc() : memref<1x16x32x32xf16, [@CMX_NN, 0]>
    %input = VPUIP.Copy inputs(%arg0 : memref<1x16x32x32xf16>) outputs(%input_buff : memref<1x16x32x32xf16, [@CMX_NN, 0]>) -> memref<1x16x32x32xf16, [@CMX_NN, 0]>

    %out_high_data = const.Declare memref<1x16x1x1xf16> = dense<3.0> : tensor<1x16x1x1xf16>
    %out_low_data = const.Declare memref<1x16x1x1xf16> = dense<-3.0> : tensor<1x16x1x1xf16>
    %in_high_data = const.Declare memref<1x16x1x1xf16> = dense<2.0> : tensor<1x16x1x1xf16>
    %in_low_data = const.Declare memref<1x16x1x1xf16> = dense<-2.0> : tensor<1x16x1x1xf16>

    %in_low_buffer = memref.alloc() : memref<1x16x1x1xf16, [@CMX_NN, 0]>
    %in_low = VPUIP.Copy inputs(%in_low_data : memref<1x16x1x1xf16>) outputs(%in_low_buffer : memref<1x16x1x1xf16, [@CMX_NN, 0]>) -> memref<1x16x1x1xf16, [@CMX_NN, 0]>
    %in_high_buffer = memref.alloc() : memref<1x16x1x1xf16, [@CMX_NN, 0]>
    %in_high = VPUIP.Copy inputs(%in_high_data : memref<1x16x1x1xf16>) outputs(%in_high_buffer : memref<1x16x1x1xf16, [@CMX_NN, 0]>) -> memref<1x16x1x1xf16, [@CMX_NN, 0]>

    %out_low_buffer = memref.alloc() : memref<1x16x1x1xf16, [@CMX_NN, 0]>
    %out_low = VPUIP.Copy inputs(%out_low_data : memref<1x16x1x1xf16>) outputs(%out_low_buffer : memref<1x16x1x1xf16, [@CMX_NN, 0]>) -> memref<1x16x1x1xf16, [@CMX_NN, 0]>
    %out_high_buffer = memref.alloc() : memref<1x16x1x1xf16, [@CMX_NN, 0]>
    %out_high = VPUIP.Copy inputs(%out_high_data : memref<1x16x1x1xf16>) outputs(%out_high_buffer : memref<1x16x1x1xf16, [@CMX_NN, 0]>) -> memref<1x16x1x1xf16, [@CMX_NN, 0]>

    %out_buffer = memref.alloc() : memref<1x16x32x32xf16, [@CMX_NN, 0]>
    %sw_fq = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_FakeQuantize
                  inputs(%input as %arg1: memref<1x16x32x32xf16, [@CMX_NN, 0]>, %in_low as %arg2: memref<1x16x1x1xf16, [@CMX_NN, 0]>, %in_high as %arg3: memref<1x16x1x1xf16, [@CMX_NN, 0]>,
                         %out_low as %arg4: memref<1x16x1x1xf16, [@CMX_NN, 0]>, %out_high as %arg5: memref<1x16x1x1xf16, [@CMX_NN, 0]>)
                  outputs(%out_buffer as %arg6: memref<1x16x32x32xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x32x32xf16, [@CMX_NN, 0]>{
        VPUIP.SW.Kernel.run {attrs = [256]}(%arg1, %arg2, %arg3, %arg4, %arg5, %arg6) : memref<1x16x32x32xf16, [@CMX_NN, 0]>, memref<1x16x1x1xf16, [@CMX_NN, 0]>, memref<1x16x1x1xf16, [@CMX_NN, 0]>, memref<1x16x1x1xf16, [@CMX_NN, 0]>, memref<1x16x1x1xf16, [@CMX_NN, 0]>, memref<1x16x32x32xf16, [@CMX_NN, 0]>
    }
    return %sw_fq : memref<1x16x32x32xf16, [@CMX_NN, 0]>

    // CHECK-DAG: [[IN_LOW_DATA:%.+]] = const.Declare memref<1x16x1x1xf16> = dense<-2.000000e+00> : tensor<1x16x1x1xf16>
    // CHECK-DAG: [[IN_HIGH_DATA:%.+]] = const.Declare memref<1x16x1x1xf16> = dense<2.000000e+00> : tensor<1x16x1x1xf16>
    // CHECK-DAG: [[OUT_LOW_DATA:%.+]] = const.Declare memref<1x16x1x1xf16> = dense<-3.000000e+00> : tensor<1x16x1x1xf16>
    // CHECK-DAG: [[OUT_HIGH_DATA:%.+]] = const.Declare memref<1x16x1x1xf16> = dense<3.000000e+00> : tensor<1x16x1x1xf16>

    // CHECK: [[INPUT_BUFFER:%.+]] = memref.alloc() : memref<1x16x32x32xf16, [@CMX_NN, 0]>
    // CHECK: [[INPUT:%.+]] = VPUIP.Copy inputs([[INPUT_DATA]] : memref<1x16x32x32xf16>) outputs(%alloc : memref<1x16x32x32xf16, [@CMX_NN, 0]>) -> memref<1x16x32x32xf16, [@CMX_NN, 0]>

    // CHECK: [[IN_LOW_BUFFER:%.+]] = memref.alloc() : memref<1x16x1x1xf16, [@CMX_NN, 0]>
    // CHECK: [[IN_LOW_BUFFER_COPY:%.+]] = VPUIP.Copy inputs([[IN_LOW_DATA]] : memref<1x16x1x1xf16>) outputs([[IN_LOW_BUFFER]] : memref<1x16x1x1xf16, [@CMX_NN, 0]>) -> memref<1x16x1x1xf16, [@CMX_NN, 0]>
    // CHECK: [[IN_HIGH_BUFFER:%.+]] = memref.alloc() : memref<1x16x1x1xf16, [@CMX_NN, 0]>
    // CHECK: [[IN_HIGH_BUFFER_COPY:%.+]] = VPUIP.Copy inputs([[IN_HIGH_DATA]] : memref<1x16x1x1xf16>) outputs([[IN_HIGH_BUFFER]] : memref<1x16x1x1xf16, [@CMX_NN, 0]>) -> memref<1x16x1x1xf16, [@CMX_NN, 0]>

    // CHECK: [[OUT_LOW_BUFFER:%.+]] = memref.alloc() : memref<1x16x1x1xf16, [@CMX_NN, 0]>
    // CHECK: [[OUT_LOW_BUFFER_COPY:%.+]] = VPUIP.Copy inputs([[OUT_LOW_DATA]] : memref<1x16x1x1xf16>) outputs([[OUT_LOW_BUFFER]] : memref<1x16x1x1xf16, [@CMX_NN, 0]>) -> memref<1x16x1x1xf16, [@CMX_NN, 0]>
    // CHECK: [[OUT_HIGH_BUFFER:%.+]] = memref.alloc() : memref<1x16x1x1xf16, [@CMX_NN, 0]>
    // CHECK: [[OUT_HIGH_BUFFER_COPY:%.+]] = VPUIP.Copy inputs([[OUT_HIGH_DATA]] : memref<1x16x1x1xf16>) outputs([[OUT_HIGH_BUFFER]] : memref<1x16x1x1xf16, [@CMX_NN, 0]>) -> memref<1x16x1x1xf16, [@CMX_NN, 0]>

    // CHECK: [[OUT_BUFFER:%.+]] = memref.alloc() : memref<1x16x32x32xf16, [@CMX_NN, 0]>
    // CHECK: [[INPUT_SUBVIEW_SHAVE_0:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 0, 0] [1, 8, 32, 32] : memref<1x16x32x32xf16, [@CMX_NN, 0]> to memref<1x8x32x32xf16, {order = #NCHW, strides = [16384, 1024, 32, 1]}, [@CMX_NN, 0]>
    // CHECK: [[IN_LOW_SUBVIEW_SHAVE_0:%.+]] = VPUIP.SubView [[IN_LOW_BUFFER_COPY]] [0, 0, 0, 0] [1, 8, 1, 1] : memref<1x16x1x1xf16, [@CMX_NN, 0]> to memref<1x8x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, [@CMX_NN, 0]>
    // CHECK: [[IN_HIGH_SUBVIEW_SHAVE_0:%.+]] = VPUIP.SubView [[IN_HIGH_BUFFER_COPY]] [0, 0, 0, 0] [1, 8, 1, 1] : memref<1x16x1x1xf16, [@CMX_NN, 0]> to memref<1x8x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, [@CMX_NN, 0]>
    // CHECK: [[OUT_LOW_SUBVIEW_SHAVE_0:%.+]] = VPUIP.SubView [[OUT_LOW_BUFFER_COPY]] [0, 0, 0, 0] [1, 8, 1, 1] : memref<1x16x1x1xf16, [@CMX_NN, 0]> to memref<1x8x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, [@CMX_NN, 0]>
    // CHECK: [[OUT_HIGH_SUBVIEW_SHAVE_0:%.+]] = VPUIP.SubView [[OUT_HIGH_BUFFER_COPY]] [0, 0, 0, 0] [1, 8, 1, 1] : memref<1x16x1x1xf16, [@CMX_NN, 0]> to memref<1x8x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, [@CMX_NN, 0]>
    // CHECK: [[OUT_SUBVIEW_SHAVE_0:%.+]] = VPUIP.SubView [[OUT_BUFFER]] [0, 0, 0, 0] [1, 8, 32, 32] : memref<1x16x32x32xf16, [@CMX_NN, 0]> to memref<1x8x32x32xf16, {order = #NCHW, strides = [16384, 1024, 32, 1]}, [@CMX_NN, 0]>

    // CHECK: [[INPUT_SUBVIEW_SHAVE_1:%.+]] = VPUIP.SubView [[INPUT]] [0, 8, 0, 0] [1, 8, 32, 32] : memref<1x16x32x32xf16, [@CMX_NN, 0]> to memref<1x8x32x32xf16, {order = #NCHW, strides = [16384, 1024, 32, 1]}, [@CMX_NN, 0]>
    // CHECK: [[IN_LOW_SUBVIEW_SHAVE_1:%.+]] = VPUIP.SubView [[IN_LOW_BUFFER_COPY]] [0, 8, 0, 0] [1, 8, 1, 1] : memref<1x16x1x1xf16, [@CMX_NN, 0]> to memref<1x8x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, [@CMX_NN, 0]>
    // CHECK: [[IN_HIGH_SUBVIEW_SHAVE_1:%.+]] = VPUIP.SubView [[IN_HIGH_BUFFER_COPY]] [0, 8, 0, 0] [1, 8, 1, 1] : memref<1x16x1x1xf16, [@CMX_NN, 0]> to memref<1x8x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, [@CMX_NN, 0]>
    // CHECK: [[OUT_LOW_SUBVIEW_SHAVE_1:%.+]] = VPUIP.SubView [[OUT_LOW_BUFFER_COPY]] [0, 8, 0, 0] [1, 8, 1, 1] : memref<1x16x1x1xf16, [@CMX_NN, 0]> to memref<1x8x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, [@CMX_NN, 0]>
    // CHECK: [[OUT_HIGH_SUBVIEW_SHAVE_1:%.+]] = VPUIP.SubView [[OUT_HIGH_BUFFER_COPY]] [0, 8, 0, 0] [1, 8, 1, 1] : memref<1x16x1x1xf16, [@CMX_NN, 0]> to memref<1x8x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, [@CMX_NN, 0]>
    // CHECK: [[OUT_SUBVIEW_SHAVE_1:%.+]] = VPUIP.SubView [[OUT_BUFFER]] [0, 8, 0, 0] [1, 8, 32, 32] : memref<1x16x32x32xf16, [@CMX_NN, 0]> to memref<1x8x32x32xf16, {order = #NCHW, strides = [16384, 1024, 32, 1]}, [@CMX_NN, 0]>

    // CHECK: [[FQ:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_FakeQuantize inputs([[INPUT_SUBVIEW_SHAVE_0]] as [[ARG1:[^:]+]]: memref<1x8x32x32xf16, {order = #NCHW, strides = [16384, 1024, 32, 1]}, [@CMX_NN, 0]>, [[IN_LOW_SUBVIEW_SHAVE_0]] as [[ARG2:[^:]+]]: memref<1x8x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, [@CMX_NN, 0]>, [[IN_HIGH_SUBVIEW_SHAVE_0]] as [[ARG3:[^:]+]]: memref<1x8x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, [@CMX_NN, 0]>, [[OUT_LOW_SUBVIEW_SHAVE_0]] as [[ARG4:[^:]+]]: memref<1x8x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, [@CMX_NN, 0]>, [[OUT_HIGH_SUBVIEW_SHAVE_0]] as [[ARG5:[^:]+]]: memref<1x8x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, [@CMX_NN, 0]>, [[INPUT_SUBVIEW_SHAVE_1]] as [[ARG6:[^:]+]]: memref<1x8x32x32xf16, {order = #NCHW, strides = [16384, 1024, 32, 1]}, [@CMX_NN, 0]>, [[IN_LOW_SUBVIEW_SHAVE_1]] as [[ARG7:[^:]+]]: memref<1x8x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, [@CMX_NN, 0]>, [[IN_HIGH_SUBVIEW_SHAVE_1]] as [[ARG8:[^:]+]]: memref<1x8x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, [@CMX_NN, 0]>, [[OUT_LOW_SUBVIEW_SHAVE_1]] as [[ARG9:[^:]+]]: memref<1x8x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, [@CMX_NN, 0]>, [[OUT_HIGH_SUBVIEW_SHAVE_1]] as [[ARG10:[^:]+]]: memref<1x8x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, [@CMX_NN, 0]>) outputs([[OUT_SUBVIEW_SHAVE_0:%.+]] as [[ARG11:[^:]+]]: memref<1x8x32x32xf16, {order = #NCHW, strides = [16384, 1024, 32, 1]}, [@CMX_NN, 0]>, [[OUT_SUBVIEW_SHAVE_1:%.+]] as [[ARG12:[^:]+]]: memref<1x8x32x32xf16, {order = #NCHW, strides = [16384, 1024, 32, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x8x32x32xf16, {order = #NCHW, strides = [16384, 1024, 32, 1]}, [@CMX_NN, 0]>, memref<1x8x32x32xf16, {order = #NCHW, strides = [16384, 1024, 32, 1]}, [@CMX_NN, 0]>){
    // CHECK:   VPUIP.SW.Kernel.run {attrs = [256]}([[ARG1]], [[ARG2]], [[ARG3]], [[ARG4]], [[ARG5]], [[ARG11]]) : memref<1x8x32x32xf16, {order = #NCHW, strides = [16384, 1024, 32, 1]}, [@CMX_NN, 0]>, memref<1x8x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, [@CMX_NN, 0]>, memref<1x8x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, [@CMX_NN, 0]>, memref<1x8x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, [@CMX_NN, 0]>, memref<1x8x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, [@CMX_NN, 0]>, memref<1x8x32x32xf16, {order = #NCHW, strides = [16384, 1024, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:   VPUIP.SW.Kernel.run {attrs = [256]}([[ARG6]], [[ARG7]], [[ARG8]], [[ARG9]], [[ARG10]], [[ARG12]]) : memref<1x8x32x32xf16, {order = #NCHW, strides = [16384, 1024, 32, 1]}, [@CMX_NN, 0]>, memref<1x8x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, [@CMX_NN, 0]>, memref<1x8x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, [@CMX_NN, 0]>, memref<1x8x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, [@CMX_NN, 0]>, memref<1x8x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, [@CMX_NN, 0]>, memref<1x8x32x32xf16, {order = #NCHW, strides = [16384, 1024, 32, 1]}, [@CMX_NN, 0]>

    // CHECK: [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[FQ]]#0, [[FQ]]#1 : memref<1x8x32x32xf16, {order = #NCHW, strides = [16384, 1024, 32, 1]}, [@CMX_NN, 0]>, memref<1x8x32x32xf16, {order = #NCHW, strides = [16384, 1024, 32, 1]}, [@CMX_NN, 0]>) outputs([[OUT_BUFFER]] : memref<1x16x32x32xf16, [@CMX_NN, 0]>) -> memref<1x16x32x32xf16, [@CMX_NN, 0]>
    // CHECK: return [[CONCAT]] : memref<1x16x32x32xf16, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
  func.func private @builtin_FakeQuantize(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "fake_quantize.cpp", VPU.kernel_entry = "fake_quantize", VPU.task_type = @COMPUTE}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @TileClusterFakeQuantize
// CHECK-SAME:  ([[INPUT_DATA:%.+]]: memref<1x3x384x320xf16>)
func.func @TileClusterFakeQuantize(%arg0: memref<1x3x384x320xf16>) -> memref<1x3x384x320xf16> {
    %input_buff = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x3x384x320xf16, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %input = VPUIP.NCEClusterTiling inputs(%arg0 as %arg2: memref<1x3x384x320xf16>) outputs(%input_buff as %arg3: memref<1x3x384x320xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x3x384x320xf16, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %inner_copy = VPUIP.Copy inputs(%arg2 : memref<1x3x384x320xf16>) outputs(%arg3 : memref<1x3x384x320xf16, @CMX_NN>) -> memref<1x3x384x320xf16, @CMX_NN>
    }

    %out_high_data = const.Declare memref<1x1x1x1xf16> = dense<2.1171875> : tensor<1x1x1x1xf32>, [#const.ConvertElemType<f16>]
    %out_low_data = const.Declare memref<1x1x1x1xf16> = dense<-2.13476563> : tensor<1x1x1x1xf32>, [#const.ConvertElemType<f16>]
    %in_high_data = const.Declare memref<1x3x1x1xf16> = dense<[[[[247.329773]], [[237.251968]], [[225.064804]]]]> : tensor<1x3x1x1xf32>, [#const.ConvertElemType<f16>]
    %in_low_data = const.Declare memref<1x3x1x1xf16> = dense<[[[[-0.965240657]], [[-5.63905859]], [[-18.9422073]]]]> : tensor<1x3x1x1xf32>, [#const.ConvertElemType<f16>]

    %in_low_buffer = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x3x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %in_low = VPUIP.NCEClusterTiling inputs(%in_low_data as %arg2: memref<1x3x1x1xf16>) outputs(%in_low_buffer as %arg3: memref<1x3x1x1xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x3x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
      %inner_copy = VPUIP.Copy inputs(%arg2 : memref<1x3x1x1xf16>) outputs(%arg3 : memref<1x3x1x1xf16, @CMX_NN>) -> memref<1x3x1x1xf16, @CMX_NN>
    }

    %in_high_buffer = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x3x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %in_high = VPUIP.NCEClusterTiling inputs(%in_high_data as %arg2: memref<1x3x1x1xf16>) outputs(%in_high_buffer as %arg3: memref<1x3x1x1xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x3x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
      %inner_copy = VPUIP.Copy inputs(%arg2 : memref<1x3x1x1xf16>) outputs(%arg3 : memref<1x3x1x1xf16, @CMX_NN>) -> memref<1x3x1x1xf16, @CMX_NN>
    }

    %out_low_buffer = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %out_low = VPUIP.NCEClusterTiling inputs(%out_low_data as %arg2: memref<1x1x1x1xf16>) outputs(%out_low_buffer as %arg3: memref<1x1x1x1xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x1x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
      %inner_copy = VPUIP.Copy inputs(%arg2 : memref<1x1x1x1xf16>) outputs(%arg3 : memref<1x1x1x1xf16, @CMX_NN>) -> memref<1x1x1x1xf16, @CMX_NN>
    }

    %out_high_buffer = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %out_high = VPUIP.NCEClusterTiling inputs(%out_high_data as %arg2: memref<1x1x1x1xf16>) outputs(%out_high_buffer as %arg3: memref<1x1x1x1xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x1x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
      %inner_copy = VPUIP.Copy inputs(%arg2 : memref<1x1x1x1xf16>) outputs(%arg3 : memref<1x1x1x1xf16, @CMX_NN>) -> memref<1x1x1x1xf16, @CMX_NN>
    }

    %out = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x3x384x320xf16, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %sw_cluster = VPUIP.NCEClusterTiling inputs(%input as %arg2: memref<1x3x384x320xf16, @CMX_NN>, %in_low as %arg3: memref<1x3x1x1xf16, @CMX_NN>, %in_high as %arg4: memref<1x3x1x1xf16, @CMX_NN>, %out_low as %arg5: memref<1x1x1x1xf16, @CMX_NN>, %out_high as %arg6: memref<1x1x1x1xf16, @CMX_NN>) outputs(%out as %arg7: memref<1x3x384x320xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x3x384x320xf16, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %results_5251 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_FakeQuantize inputs(%arg2 as %arg8: memref<1x3x384x320xf16, @CMX_NN>, %arg3 as %arg9: memref<1x3x1x1xf16, @CMX_NN>, %arg4 as %arg10: memref<1x3x1x1xf16, @CMX_NN>, %arg5 as %arg11: memref<1x1x1x1xf16, @CMX_NN>, %arg6 as %arg12: memref<1x1x1x1xf16, @CMX_NN>) outputs(%arg7 as %arg13: memref<1x3x384x320xf16, @CMX_NN>) on tile 0 -> memref<1x3x384x320xf16, @CMX_NN>{
        VPUIP.SW.Kernel.run {attrs = [256]}(%arg8, %arg9, %arg10, %arg11, %arg12, %arg13) : memref<1x3x384x320xf16, @CMX_NN>, memref<1x3x1x1xf16, @CMX_NN>, memref<1x3x1x1xf16, @CMX_NN>, memref<1x1x1x1xf16, @CMX_NN>, memref<1x1x1x1xf16, @CMX_NN>, memref<1x3x384x320xf16, @CMX_NN>
      }
    }

    %alloc = memref.alloc() : memref<1x3x384x320xf16>
    %out_ddr = VPUIP.NCEClusterTiling inputs(%sw_cluster as %arg2: memref<1x3x384x320xf16, @CMX_NN>) outputs(%alloc as %arg3: memref<1x3x384x320xf16>) -> memref<1x3x384x320xf16> {
      %inner_copy = VPUIP.Copy inputs(%arg2 : memref<1x3x384x320xf16, @CMX_NN>) outputs(%arg3 : memref<1x3x384x320xf16>) -> memref<1x3x384x320xf16>
    }
    return %out_ddr: memref<1x3x384x320xf16>

    // CHECK-DAG: [[IN_LOW_DATA:%.+]] = const.Declare memref<1x3x1x1xf16>
    // CHECK-DAG: [[IN_HIGH_DATA:%.+]] = const.Declare memref<1x3x1x1xf16>
    // CHECK-DAG: [[OUT_LOW_DATA:%.+]] = const.Declare memref<1x1x1x1xf16>
    // CHECK-DAG: [[OUT_HIGH_DATA:%.+]] = const.Declare memref<1x1x1x1xf16>

    // CHECK: [[SUBVIEW_DDR_0:%.+]] = VPUIP.SubView [[INPUT_DATA]] [0, 0, 192, 0] [1, 3, 192, 320] : memref<1x3x384x320xf16> to memref<1x3x192x320xf16, {order = #NCHW, strides = [368640, 122880, 320, 1]}>
    // CHECK: [[SUBVIEW_CMX_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x3x192x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[SUBVIEW_CMX_0_COPY:%.+]] = VPUIP.NCEClusterTiling inputs([[SUBVIEW_DDR_0]] as [[ARG1:[^:]+]]: memref<1x3x192x320xf16, {order = #NCHW, strides = [368640, 122880, 320, 1]}>) outputs([[SUBVIEW_CMX_0]] as [[ARG2:[^:]+]]: memref<1x3x192x320xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x3x192x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:   VPUIP.Copy inputs([[ARG1]] : memref<1x3x192x320xf16, {order = #NCHW, strides = [368640, 122880, 320, 1]}>) outputs([[ARG2]] : memref<1x3x192x320xf16, @CMX_NN>) -> memref<1x3x192x320xf16, @CMX_NN>
    // CHECK: }

    // CHECK: [[SUBVIEW_DDR_1:%.+]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 3, 192, 320] : memref<1x3x384x320xf16> to memref<1x3x192x320xf16, {order = #NCHW, strides = [368640, 122880, 320, 1]}>
    // CHECK: [[SUBVIEW_CMX_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x3x192x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[SUBVIEW_CMX_1_COPY:%.+]] = VPUIP.NCEClusterTiling inputs([[SUBVIEW_DDR_1]] as [[ARG1:[^:]+]]: memref<1x3x192x320xf16, {order = #NCHW, strides = [368640, 122880, 320, 1]}>) outputs([[SUBVIEW_CMX_1]] as [[ARG2:[^:]+]]: memref<1x3x192x320xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x3x192x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK: VPUIP.Copy inputs(%arg1 : memref<1x3x192x320xf16, {order = #NCHW, strides = [368640, 122880, 320, 1]}>) outputs(%arg2 : memref<1x3x192x320xf16, @CMX_NN>) -> memref<1x3x192x320xf16, @CMX_NN>

    // CHECK: [[IN_LOW_BUFFER_SHAVE_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x3x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK: [[IN_LOW_BUFFER_SHAVE_0_COPY:%.+]] = VPUIP.NCEClusterTiling inputs([[IN_LOW_DATA]] as [[ARG1:[^:]+]]: memref<1x3x1x1xf16>) outputs([[IN_LOW_BUFFER_SHAVE_0]] as [[ARG2:[^:]+]]: memref<1x3x1x1xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x3x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    // CHECK: VPUIP.Copy inputs([[ARG1]] : memref<1x3x1x1xf16>) outputs([[ARG2]] : memref<1x3x1x1xf16, @CMX_NN>) -> memref<1x3x1x1xf16, @CMX_NN>

    // CHECK: [[IN_LOW_BUFFER_SHAVE_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x3x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK: [[IN_LOW_BUFFER_SHAVE_1_COPY:%.+]] = VPUIP.NCEClusterTiling inputs([[IN_LOW_DATA]] as [[ARG1:[^:]+]]: memref<1x3x1x1xf16>) outputs([[IN_LOW_BUFFER_SHAVE_1]] as [[ARG2:[^:]+]]: memref<1x3x1x1xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x3x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    // CHECK: VPUIP.Copy inputs([[ARG1]] : memref<1x3x1x1xf16>) outputs([[ARG2]] : memref<1x3x1x1xf16, @CMX_NN>) -> memref<1x3x1x1xf16, @CMX_NN>

    // CHECK: [[IN_HIGH_BUFFER_SHAVE_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x3x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK: [[IN_HIGH_BUFFER_SHAVE_0_COPY:%.+]] = VPUIP.NCEClusterTiling inputs([[IN_HIGH_DATA]] as [[ARG1:[^:]+]]: memref<1x3x1x1xf16>) outputs([[IN_HIGH_BUFFER_SHAVE_0]] as [[ARG2:[^:]+]]: memref<1x3x1x1xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x3x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    // CHECK: VPUIP.Copy inputs([[ARG1]] : memref<1x3x1x1xf16>) outputs([[ARG2]] : memref<1x3x1x1xf16, @CMX_NN>) -> memref<1x3x1x1xf16, @CMX_NN>

    // CHECK: [[IN_HIGH_BUFFER_SHAVE_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x3x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK: [[IN_HIGH_BUFFER_SHAVE_1_COPY:%.+]] = VPUIP.NCEClusterTiling inputs([[IN_HIGH_DATA]] as [[ARG1:[^:]+]]: memref<1x3x1x1xf16>) outputs([[IN_HIGH_BUFFER_SHAVE_1]] as [[ARG2:[^:]+]]: memref<1x3x1x1xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x3x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    // CHECK: VPUIP.Copy inputs([[ARG1]] : memref<1x3x1x1xf16>) outputs([[ARG2]] : memref<1x3x1x1xf16, @CMX_NN>) -> memref<1x3x1x1xf16, @CMX_NN>

    // CHECK: [[OUT_LOW_BUFFER_SHAVE_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK: [[OUT_LOW_BUFFER_SHAVE_0_COPY:%.+]] = VPUIP.NCEClusterTiling inputs([[OUT_LOW_DATA]] as [[ARG1:[^:]+]]: memref<1x1x1x1xf16>) outputs([[OUT_LOW_BUFFER_SHAVE_0]] as [[ARG2:[^:]+]]: memref<1x1x1x1xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    // CHECK:    VPUIP.Copy inputs([[ARG1]] : memref<1x1x1x1xf16>) outputs([[ARG2]] : memref<1x1x1x1xf16, @CMX_NN>) -> memref<1x1x1x1xf16, @CMX_NN>

    // CHECK: [[OUT_LOW_BUFFER_SHAVE_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK: [[OUT_LOW_BUFFER_SHAVE_1_COPY:%.+]] = VPUIP.NCEClusterTiling inputs([[OUT_LOW_DATA]] as [[ARG1:[^:]+]]: memref<1x1x1x1xf16>) outputs([[OUT_LOW_BUFFER_SHAVE_1]] as [[ARG2:[^:]+]]: memref<1x1x1x1xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    // CHECK:    VPUIP.Copy inputs([[ARG1]] : memref<1x1x1x1xf16>) outputs([[ARG2]] : memref<1x1x1x1xf16, @CMX_NN>) -> memref<1x1x1x1xf16, @CMX_NN>

    // CHECK: [[OUT_HIGH_BUFFER_SHAVE_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK: [[OUT_HIGH_BUFFER_SHAVE_0_COPY:%.+]] = VPUIP.NCEClusterTiling inputs([[OUT_HIGH_DATA]] as [[ARG1:[^:]+]]: memref<1x1x1x1xf16>) outputs([[OUT_HIGH_BUFFER_SHAVE_0]] as [[ARG2:[^:]+]]: memref<1x1x1x1xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    // CHECK:    VPUIP.Copy inputs([[ARG1]] : memref<1x1x1x1xf16>) outputs([[ARG2]] : memref<1x1x1x1xf16, @CMX_NN>) -> memref<1x1x1x1xf16, @CMX_NN>

    // CHECK: [[OUT_HIGH_BUFFER_SHAVE_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK: [[OUT_HIGH_BUFFER_SHAVE_1_COPY:%.+]] = VPUIP.NCEClusterTiling inputs([[OUT_HIGH_DATA]] as [[ARG1:[^:]+]]: memref<1x1x1x1xf16>) outputs([[OUT_HIGH_BUFFER_SHAVE_1]] as [[ARG2:[^:]+]]: memref<1x1x1x1xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    // CHECK:    VPUIP.Copy inputs([[ARG1]] : memref<1x1x1x1xf16>) outputs([[ARG2]] : memref<1x1x1x1xf16, @CMX_NN>) -> memref<1x1x1x1xf16, @CMX_NN>

    // CHECK: [[OUT_BUFFER_SHAVE_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x3x192x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[OUT_BUFFER_SHAVE_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x3x192x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[SW_FQ:%.+]]:2 = VPUIP.NCEClusterTiling inputs([[SUBVIEW_CMX_1_COPY]] as [[ARG1:[^:]+]]: memref<1x3x192x320xf16, @CMX_NN>, [[IN_LOW_BUFFER_SHAVE_1_COPY]] as [[ARG2:[^:]+]]: memref<1x3x1x1xf16, @CMX_NN>, [[IN_HIGH_BUFFER_SHAVE_1_COPY]] as [[ARG3:[^:]+]]: memref<1x3x1x1xf16, @CMX_NN>, [[OUT_LOW_BUFFER_SHAVE_1_COPY]] as [[ARG4:[^:]+]]: memref<1x1x1x1xf16, @CMX_NN>, [[OUT_HIGH_BUFFER_SHAVE_1_COPY]] as [[ARG5:[^:]+]]: memref<1x1x1x1xf16, @CMX_NN>, [[SUBVIEW_CMX_0_COPY]] as [[ARG6:[^:]+]]: memref<1x3x192x320xf16, @CMX_NN>, [[IN_LOW_BUFFER_SHAVE_0_COPY]] as [[ARG7:[^:]+]]: memref<1x3x1x1xf16, @CMX_NN>, [[IN_HIGH_BUFFER_SHAVE_0_COPY]] as [[ARG8:[^:]+]]: memref<1x3x1x1xf16, @CMX_NN>, [[OUT_LOW_BUFFER_SHAVE_0_COPY]] as [[ARG9:[^:]+]]: memref<1x1x1x1xf16, @CMX_NN>, [[OUT_HIGH_BUFFER_SHAVE_0_COPY]] as [[ARG10:[^:]+]]: memref<1x1x1x1xf16, @CMX_NN>) outputs([[OUT_BUFFER_SHAVE_1]] as [[ARG11:[^:]+]]: memref<1x3x192x320xf16, @CMX_NN>, [[OUT_BUFFER_SHAVE_0]] as [[ARG12:[^:]+]]: memref<1x3x192x320xf16, @CMX_NN>) -> (!VPUIP.DistributedBuffer<1x3x192x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x3x192x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) {
    // CHECK:   VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_FakeQuantize inputs([[ARG1]] as [[ARG13:[^:]+]]: memref<1x3x192x320xf16, @CMX_NN>, [[ARG2]] as [[ARG14:[^:]+]]: memref<1x3x1x1xf16, @CMX_NN>, [[ARG3]] as [[ARG15:[^:]+]]: memref<1x3x1x1xf16, @CMX_NN>, [[ARG4]] as [[ARG16:[^:]+]]: memref<1x1x1x1xf16, @CMX_NN>, [[ARG5]] as [[ARG17:[^:]+]]: memref<1x1x1x1xf16, @CMX_NN>, [[ARG6]] as [[ARG18:[^:]+]]: memref<1x3x192x320xf16, @CMX_NN>, [[ARG7]] as [[ARG19:[^:]+]]: memref<1x3x1x1xf16, @CMX_NN>, [[ARG8]] as [[ARG20:[^:]+]]: memref<1x3x1x1xf16, @CMX_NN>, [[ARG9]] as [[ARG21:[^:]+]]: memref<1x1x1x1xf16, @CMX_NN>, [[ARG10]] as [[ARG22:[^:]+]]: memref<1x1x1x1xf16, @CMX_NN>) outputs([[ARG11]] as [[ARG23:[^:]+]]: memref<1x3x192x320xf16, @CMX_NN>, [[ARG12]] as [[ARG24:[^:]+]]: memref<1x3x192x320xf16, @CMX_NN>) on tile 0 -> (memref<1x3x192x320xf16, @CMX_NN>, memref<1x3x192x320xf16, @CMX_NN>){
    // CHECK:    VPUIP.SW.Kernel.run {attrs = [256]}([[ARG13]], [[ARG14]], [[ARG15]], [[ARG16]], [[ARG17]], [[ARG23]]) : memref<1x3x192x320xf16, @CMX_NN>, memref<1x3x1x1xf16, @CMX_NN>, memref<1x3x1x1xf16, @CMX_NN>, memref<1x1x1x1xf16, @CMX_NN>, memref<1x1x1x1xf16, @CMX_NN>, memref<1x3x192x320xf16, @CMX_NN>
    // CHECK:    VPUIP.SW.Kernel.run {attrs = [256]}([[ARG18]], [[ARG19]], [[ARG20]], [[ARG21]], [[ARG22]], [[ARG24]]) : memref<1x3x192x320xf16, @CMX_NN>, memref<1x3x1x1xf16, @CMX_NN>, memref<1x3x1x1xf16, @CMX_NN>, memref<1x1x1x1xf16, @CMX_NN>, memref<1x1x1x1xf16, @CMX_NN>, memref<1x3x192x320xf16, @CMX_NN>
    // CHECK:  }

    // CHECK: [[OUT_DDR:%.+]] = memref.alloc() : memref<1x3x384x320xf16>
    // CHECK: [[OUT_SUBVIEW_0:%.+]] = VPUIP.SubView [[OUT_DDR]] [0, 0, 0, 0] [1, 3, 192, 320] : memref<1x3x384x320xf16> to memref<1x3x192x320xf16, {order = #NCHW, strides = [368640, 122880, 320, 1]}>
    // CHECK: [[OUT_SUBVIEW_0_COPY:%.+]] = VPUIP.NCEClusterTiling inputs([[SW_FQ]]#0 as [[ARG1:[^:]+]]: memref<1x3x192x320xf16, @CMX_NN>) outputs([[OUT_SUBVIEW_0]] as [[ARG2:[^:]+]]: memref<1x3x192x320xf16, {order = #NCHW, strides = [368640, 122880, 320, 1]}>) -> memref<1x3x192x320xf16, {order = #NCHW, strides = [368640, 122880, 320, 1]}> {
    // CHECK:       VPUIP.Copy inputs([[ARG1]] : memref<1x3x192x320xf16, @CMX_NN>) outputs([[ARG2]] : memref<1x3x192x320xf16, {order = #NCHW, strides = [368640, 122880, 320, 1]}>) -> memref<1x3x192x320xf16, {order = #NCHW, strides = [368640, 122880, 320, 1]}>

    // CHECK: [[OUT_SUBVIEW_1:%.+]] = VPUIP.SubView [[OUT_DDR]] [0, 0, 192, 0] [1, 3, 192, 320] : memref<1x3x384x320xf16> to memref<1x3x192x320xf16, {order = #NCHW, strides = [368640, 122880, 320, 1]}>
    // CHECK: [[OUT_SUBVIEW_1_COPY:%.+]] = VPUIP.NCEClusterTiling inputs([[SW_FQ]]#1 as [[ARG1:[^:]+]]: memref<1x3x192x320xf16, @CMX_NN>) outputs([[OUT_SUBVIEW_1]] as [[ARG2:[^:]+]]: memref<1x3x192x320xf16, {order = #NCHW, strides = [368640, 122880, 320, 1]}>) -> memref<1x3x192x320xf16, {order = #NCHW, strides = [368640, 122880, 320, 1]}> {
    // CHECK:     VPUIP.Copy inputs([[ARG1]] : memref<1x3x192x320xf16, @CMX_NN>) outputs([[ARG2]] : memref<1x3x192x320xf16, {order = #NCHW, strides = [368640, 122880, 320, 1]}>) -> memref<1x3x192x320xf16, {order = #NCHW, strides = [368640, 122880, 320, 1]}>

    // CHECK: [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[OUT_SUBVIEW_0_COPY]], [[OUT_SUBVIEW_1_COPY]] : memref<1x3x192x320xf16, {order = #NCHW, strides = [368640, 122880, 320, 1]}>, memref<1x3x192x320xf16, {order = #NCHW, strides = [368640, 122880, 320, 1]}>) outputs([[OUT_DDR]] : memref<1x3x384x320xf16>) -> memref<1x3x384x320xf16>
    // CHECK: return [[CONCAT]] : memref<1x3x384x320xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!DistributedBuffer = !VPUIP.DistributedBuffer<
    1x1x16x255xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!DistributedBuffer1 = !VPUIP.DistributedBuffer<
    1x1x1x255xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

module @VPU.SW {
    func.func private @builtin_Select(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "eltwise_select.cpp", VPU.kernel_entry = "eltwise_select"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileSelect() -> !DistributedBuffer {
    %input0 = VPURT.AllocDistributed -> !DistributedBuffer
    %input1 = VPURT.AllocDistributed -> !DistributedBuffer1
    %input2 = VPURT.AllocDistributed -> !DistributedBuffer
    %alloc_cmx = VPURT.AllocDistributed -> !DistributedBuffer
    %select = VPUIP.NCEClusterTiling inputs(%input0 as %arg3: memref<1x1x16x255xf16, @CMX_NN>,
                                            %input1 as %arg4: memref<1x1x1x255xf16, @CMX_NN>,
                                            %input2 as %arg5: memref<1x1x16x255xf16, @CMX_NN>)
                                     outputs(%alloc_cmx as %arg6: memref<1x1x16x255xf16, @CMX_NN>) -> !DistributedBuffer {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Select
                  inputs(%arg3 as %arg7: memref<1x1x16x255xf16, @CMX_NN>,
                         %arg4 as %arg8: memref<1x1x1x255xf16, @CMX_NN>,
                         %arg5 as %arg9: memref<1x1x16x255xf16, @CMX_NN>)
                  outputs(%arg6 as %arg10: memref<1x1x16x255xf16, @CMX_NN>) on tile 0 -> memref<1x1x16x255xf16, @CMX_NN>{
        VPUIP.SW.Kernel.run(%arg7, %arg8, %arg9, %arg10) : memref<1x1x16x255xf16, @CMX_NN>, memref<1x1x1x255xf16, @CMX_NN>, memref<1x1x16x255xf16, @CMX_NN>, memref<1x1x16x255xf16, @CMX_NN>
      }
    }
    return %select : !DistributedBuffer

    // CHECK: [[IN_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x16x255xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[IN_0_DDR_BUFFER_0:%.+]] = memref.alloc() : memref<1x1x16x255xf16, @DDR>
    // CHECK: [[IN_0_COPY_TO_DDR_SHAVE_0:%.+]] = VPUIP.NCEClusterTiling inputs([[IN_0]] as [[ARG0:[^:]+]]: memref<1x1x16x255xf16, @CMX_NN>) outputs([[IN_0_DDR_BUFFER_0]] as [[ARG1:[^:]+]]: memref<1x1x16x255xf16, @DDR>) -> memref<1x1x16x255xf16, @DDR> {
    // CHECK:     VPUIP.Copy inputs([[ARG0]] : memref<1x1x16x255xf16, @CMX_NN>) outputs([[ARG1]] : memref<1x1x16x255xf16, @DDR>) -> memref<1x1x16x255xf16, @DDR>

    // CHECK: [[IN_0_DDR_BUFFER_SHAVE_1:%.+]] = VPUIP.SubView [[IN_0_COPY_TO_DDR_SHAVE_0]] [0, 0, 8, 0] [1, 1, 8, 255] : memref<1x1x16x255xf16, @DDR> to memref<1x1x8x255xf16, {order = #NCHW, strides = [4080, 4080, 255, 1]}, @DDR>
    // CHECK: [[IN_0_CMX_BUFFER_SHAVE_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x8x255xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[IN_0_COPY_SHAVE_1:%.+]] = VPUIP.NCEClusterTiling inputs([[IN_0_DDR_BUFFER_SHAVE_1]] as [[ARG2:[^:]+]]: memref<1x1x8x255xf16, {order = #NCHW, strides = [4080, 4080, 255, 1]}, @DDR>)
    // CHECK-SAME:                                               outputs([[IN_0_CMX_BUFFER_SHAVE_1]] as [[ARG3:[^:]+]]: memref<1x1x8x255xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x1x8x255xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:    VPUIP.Copy inputs([[ARG2]] : memref<1x1x8x255xf16, {order = #NCHW, strides = [4080, 4080, 255, 1]}, @DDR>) outputs([[ARG3]] : memref<1x1x8x255xf16, @CMX_NN>) -> memref<1x1x8x255xf16, @CMX_NN>

    // CHECK: [[IN_0_DDR_BUFFER_1:%.+]] = memref.alloc() : memref<1x1x16x255xf16, @DDR>
    // CHECK: [[IN_0_COPY_TO_DDR_SHAVE_1:%.+]] = VPUIP.NCEClusterTiling inputs([[IN_0]] as [[ARG4:[^:]+]]: memref<1x1x16x255xf16, @CMX_NN>) outputs([[IN_0_DDR_BUFFER_1]] as [[ARG5:[^:]+]]: memref<1x1x16x255xf16, @DDR>) -> memref<1x1x16x255xf16, @DDR> {
    // CHECK:     VPUIP.Copy inputs([[ARG4]] : memref<1x1x16x255xf16, @CMX_NN>) outputs([[ARG5]] : memref<1x1x16x255xf16, @DDR>) -> memref<1x1x16x255xf16, @DDR>

    // CHECK: [[IN_0_DDR_BUFFER_SHAVE_0:%.+]] = VPUIP.SubView [[IN_0_COPY_TO_DDR_SHAVE_1]] [0, 0, 0, 0] [1, 1, 8, 255] : memref<1x1x16x255xf16, @DDR> to memref<1x1x8x255xf16, {order = #NCHW, strides = [4080, 4080, 255, 1]}, @DDR>
    // CHECK: [[IN_0_CMX_BUFFER_SHAVE_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x8x255xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[IN_0_COPY_SHAVE_0:%.+]] = VPUIP.NCEClusterTiling inputs([[IN_0_DDR_BUFFER_SHAVE_0]] as [[ARG6:[^:]+]]: memref<1x1x8x255xf16, {order = #NCHW, strides = [4080, 4080, 255, 1]}, @DDR>)
    // CHECK-SAME:                                               outputs([[IN_0_CMX_BUFFER_SHAVE_0]] as [[ARG7:[^:]+]]: memref<1x1x8x255xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x1x8x255xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:     VPUIP.Copy inputs([[ARG6]] : memref<1x1x8x255xf16, {order = #NCHW, strides = [4080, 4080, 255, 1]}, @DDR>) outputs([[ARG7]] : memref<1x1x8x255xf16, @CMX_NN>) -> memref<1x1x8x255xf16, @CMX_NN>

    // CHECK: [[IN_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x255xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK: [[IN_1_DDR_BUFFER_0:%.+]] = memref.alloc() : memref<1x1x1x255xf16, @DDR>
    // CHECK: [[IN_1_COPY_TO_DDR_SHAVE_0:%.+]] = VPUIP.NCEClusterTiling inputs([[IN_1]] as [[ARG8:[^:]+]]: memref<1x1x1x255xf16, @CMX_NN>) outputs([[IN_1_DDR_BUFFER_0]] as [[ARG9:[^:]+]]: memref<1x1x1x255xf16, @DDR>) -> memref<1x1x1x255xf16, @DDR> {
    // CHECK:     VPUIP.Copy inputs([[ARG8]] : memref<1x1x1x255xf16, @CMX_NN>) outputs([[ARG9]] : memref<1x1x1x255xf16, @DDR>) -> memref<1x1x1x255xf16, @DDR>

    // CHECK: [[IN_1_CMX_BUFFER_SHAVE_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x255xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK: [[IN_1_COPY_SHAVE_1:%.+]] = VPUIP.NCEClusterTiling inputs([[IN_1_COPY_TO_DDR_SHAVE_0]] as [[ARG10:[^:]+]]: memref<1x1x1x255xf16, @DDR>)
    // CHECK-SAME:                                               outputs([[IN_1_CMX_BUFFER_SHAVE_1]] as [[ARG11:[^:]+]]: memref<1x1x1x255xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x1x1x255xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    // CHECK:     VPUIP.Copy inputs([[ARG10]] : memref<1x1x1x255xf16, @DDR>) outputs([[ARG11]] : memref<1x1x1x255xf16, @CMX_NN>) -> memref<1x1x1x255xf16, @CMX_NN>

    // CHECK: [[IN_1_DDR_BUFFER_1:%.+]] = memref.alloc() : memref<1x1x1x255xf16, @DDR>
    // CHECK: [[IN_1_COPY_TO_DDR_SHAVE_1:%.+]] = VPUIP.NCEClusterTiling inputs([[IN_1]] as [[ARG12:[^:]+]]: memref<1x1x1x255xf16, @CMX_NN>) outputs([[IN_1_DDR_BUFFER_1]] as [[ARG13:[^:]+]]: memref<1x1x1x255xf16, @DDR>) -> memref<1x1x1x255xf16, @DDR> {
    // CHECK:     VPUIP.Copy inputs([[ARG12]] : memref<1x1x1x255xf16, @CMX_NN>) outputs([[ARG13]] : memref<1x1x1x255xf16, @DDR>) -> memref<1x1x1x255xf16, @DDR>

    // CHECK: [[IN_1_CMX_BUFFER_SHAVE_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x255xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK: [[IN_1_COPY_SHAVE_0:%.+]] = VPUIP.NCEClusterTiling inputs([[IN_1_COPY_TO_DDR_SHAVE_1]] as [[ARG14:[^:]+]]: memref<1x1x1x255xf16, @DDR>)
    // CHECK-SAME:                                               outputs([[IN_1_CMX_BUFFER_SHAVE_0]] as [[ARG15:[^:]+]]: memref<1x1x1x255xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x1x1x255xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    // CHECK:     VPUIP.Copy inputs([[ARG14]] : memref<1x1x1x255xf16, @DDR>) outputs([[ARG15]] : memref<1x1x1x255xf16, @CMX_NN>) -> memref<1x1x1x255xf16, @CMX_NN>

    // CHECK: [[IN_2:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x16x255xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[IN_2_DDR_BUFFER_0:%.+]] = memref.alloc() : memref<1x1x16x255xf16, @DDR>
    // CHECK: [[IN_2_COPY_TO_DDR_SHAVE_0:%.+]] = VPUIP.NCEClusterTiling inputs([[IN_2]] as [[ARG16:[^:]+]]: memref<1x1x16x255xf16, @CMX_NN>) outputs([[IN_2_DDR_BUFFER_0]] as [[ARG17:[^:]+]]: memref<1x1x16x255xf16, @DDR>) -> memref<1x1x16x255xf16, @DDR> {
    // CHECK:     VPUIP.Copy inputs([[ARG16]] : memref<1x1x16x255xf16, @CMX_NN>) outputs([[ARG17]] : memref<1x1x16x255xf16, @DDR>) -> memref<1x1x16x255xf16, @DDR>

    // CHECK: [[IN_2_DDR_BUFFER_SHAVE_1:%.+]] = VPUIP.SubView [[IN_2_COPY_TO_DDR_SHAVE_0]] [0, 0, 8, 0] [1, 1, 8, 255] : memref<1x1x16x255xf16, @DDR> to memref<1x1x8x255xf16, {order = #NCHW, strides = [4080, 4080, 255, 1]}, @DDR>
    // CHECK: [[IN_2_CMX_BUFFER_SHAVE_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x8x255xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[IN_2_COPY_SHAVE_1:%.+]] = VPUIP.NCEClusterTiling inputs([[IN_2_DDR_BUFFER_SHAVE_1]] as [[ARG18:[^:]+]]: memref<1x1x8x255xf16, {order = #NCHW, strides = [4080, 4080, 255, 1]}, @DDR>)
    // CHECK-SAME:                                               outputs([[IN_2_CMX_BUFFER_SHAVE_1]] as [[ARG19:[^:]+]]: memref<1x1x8x255xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x1x8x255xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:     VPUIP.Copy inputs([[ARG18]] : memref<1x1x8x255xf16, {order = #NCHW, strides = [4080, 4080, 255, 1]}, @DDR>) outputs([[ARG19]] : memref<1x1x8x255xf16, @CMX_NN>) -> memref<1x1x8x255xf16, @CMX_NN>

    // CHECK: [[IN_2_DDR_BUFFER_1:%.+]] = memref.alloc() : memref<1x1x16x255xf16, @DDR>
    // CHECK: [[IN_2_COPY_TO_DDR_SHAVE_1:%.+]] = VPUIP.NCEClusterTiling inputs([[IN_2]] as [[ARG20:[^:]+]]: memref<1x1x16x255xf16, @CMX_NN>) outputs([[IN_2_DDR_BUFFER_1]] as [[ARG21:[^:]+]]: memref<1x1x16x255xf16, @DDR>) -> memref<1x1x16x255xf16, @DDR> {
    // CHECK:     VPUIP.Copy inputs([[ARG20]] : memref<1x1x16x255xf16, @CMX_NN>) outputs([[ARG21]] : memref<1x1x16x255xf16, @DDR>) -> memref<1x1x16x255xf16, @DDR>

    // CHECK: [[IN_2_DDR_BUFFER_SHAVE_0:%.+]] = VPUIP.SubView [[IN_2_COPY_TO_DDR_SHAVE_1]] [0, 0, 0, 0] [1, 1, 8, 255] : memref<1x1x16x255xf16, @DDR> to memref<1x1x8x255xf16, {order = #NCHW, strides = [4080, 4080, 255, 1]}, @DDR>
    // CHECK: [[IN_2_CMX_BUFFER_SHAVE_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x8x255xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[IN_2_COPY_SHAVE_0:%.+]] = VPUIP.NCEClusterTiling inputs([[IN_2_DDR_BUFFER_SHAVE_0]] as [[ARG22:[^:]+]]: memref<1x1x8x255xf16, {order = #NCHW, strides = [4080, 4080, 255, 1]}, @DDR>)
    // CHECK-SAME:                                               outputs([[IN_2_CMX_BUFFER_SHAVE_0]] as [[ARG23:[^:]+]]: memref<1x1x8x255xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x1x8x255xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:     VPUIP.Copy inputs([[ARG22]] : memref<1x1x8x255xf16, {order = #NCHW, strides = [4080, 4080, 255, 1]}, @DDR>) outputs([[ARG23]] : memref<1x1x8x255xf16, @CMX_NN>) -> memref<1x1x8x255xf16, @CMX_NN>

    // CHECK: [[OUT_CMX_BUFFER_SHAVE_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x8x255xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[OUT_CMX_BUFFER_SHAVE_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x8x255xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[SW_FQ:%.+]]:2 = VPUIP.NCEClusterTiling inputs([[IN_0_COPY_SHAVE_0]] as [[ARG24:[^:]+]]: memref<1x1x8x255xf16, @CMX_NN>, [[IN_1_COPY_SHAVE_0]] as [[ARG25:[^:]+]]: memref<1x1x1x255xf16, @CMX_NN>, [[IN_2_COPY_SHAVE_0]] as [[ARG26:[^:]+]]: memref<1x1x8x255xf16, @CMX_NN>,
    // CHECK-SAME:                                            [[IN_0_COPY_SHAVE_1]] as [[ARG27:[^:]+]]: memref<1x1x8x255xf16, @CMX_NN>, [[IN_1_COPY_SHAVE_1]] as [[ARG28:[^:]+]]: memref<1x1x1x255xf16, @CMX_NN>, [[IN_2_COPY_SHAVE_1]] as [[ARG29:[^:]+]]: memref<1x1x8x255xf16, @CMX_NN>)
    // CHECK-SAME:                                     outputs([[OUT_CMX_BUFFER_SHAVE_1]] as [[ARG30:[^:]+]]: memref<1x1x8x255xf16, @CMX_NN>, [[OUT_CMX_BUFFER_SHAVE_0]] as [[ARG31:[^:]+]]: memref<1x1x8x255xf16, @CMX_NN>) -> (!VPUIP.DistributedBuffer<1x1x8x255xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x8x255xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) {
    // CHECK:      VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Select inputs([[ARG24]] as [[ARG32:[^:]+]]: memref<1x1x8x255xf16, @CMX_NN>, [[ARG25]] as [[ARG33:[^:]+]]: memref<1x1x1x255xf16, @CMX_NN>, [[ARG26]] as [[ARG34:[^:]+]]: memref<1x1x8x255xf16, @CMX_NN>, [[ARG27]] as [[ARG35:[^:]+]]: memref<1x1x8x255xf16, @CMX_NN>, [[ARG28]] as [[ARG36:[^:]+]]: memref<1x1x1x255xf16, @CMX_NN>, [[ARG29]] as [[ARG37:[^:]+]]: memref<1x1x8x255xf16, @CMX_NN>) outputs([[ARG30]] as [[ARG38:[^:]+]]: memref<1x1x8x255xf16, @CMX_NN>, [[ARG31]] as [[ARG39:[^:]+]]: memref<1x1x8x255xf16, @CMX_NN>) on tile 0 -> (memref<1x1x8x255xf16, @CMX_NN>, memref<1x1x8x255xf16, @CMX_NN>){
    // CHECK:      VPUIP.SW.Kernel.run {attrs = []}([[ARG32]], [[ARG33]], [[ARG34]], [[ARG38]]) : memref<1x1x8x255xf16, @CMX_NN>, memref<1x1x1x255xf16, @CMX_NN>, memref<1x1x8x255xf16, @CMX_NN>, memref<1x1x8x255xf16, @CMX_NN>
    // CHECK:      VPUIP.SW.Kernel.run {attrs = []}([[ARG35]], [[ARG36]], [[ARG37]], [[ARG39]]) : memref<1x1x8x255xf16, @CMX_NN>, memref<1x1x1x255xf16, @CMX_NN>, memref<1x1x8x255xf16, @CMX_NN>, memref<1x1x8x255xf16, @CMX_NN>

    // CHECK: [[OUT:%.+]] = memref.alloc() : memref<1x1x16x255xf16, @DDR>
    // CHECK: [[OUT_SLICE_0:%.+]] = VPUIP.SubView [[OUT]] [0, 0, 0, 0] [1, 1, 8, 255] : memref<1x1x16x255xf16, @DDR> to memref<1x1x8x255xf16, {order = #NCHW, strides = [4080, 4080, 255, 1]}, @DDR>
    // CHECK: [[OUT_SLICE_0_COPY:%.+]] = VPUIP.NCEClusterTiling inputs([[SW_FQ]]#0 as [[ARG40:[^:]+]]: memref<1x1x8x255xf16, @CMX_NN>)
    // CHECK-SAME:                                              outputs([[OUT_SLICE_0]] as [[ARG41:[^:]+]]: memref<1x1x8x255xf16, {order = #NCHW, strides = [4080, 4080, 255, 1]}, @DDR>) -> memref<1x1x8x255xf16, {order = #NCHW, strides = [4080, 4080, 255, 1]}, @DDR> {
    // CHECK:     VPUIP.Copy inputs([[ARG40]] : memref<1x1x8x255xf16, @CMX_NN>) outputs([[ARG41]] : memref<1x1x8x255xf16, {order = #NCHW, strides = [4080, 4080, 255, 1]}, @DDR>) -> memref<1x1x8x255xf16, {order = #NCHW, strides = [4080, 4080, 255, 1]}, @DDR>
    // CHECK: [[OUT_SLICE_1:%.+]] = VPUIP.SubView [[OUT]] [0, 0, 8, 0] [1, 1, 8, 255] : memref<1x1x16x255xf16, @DDR> to memref<1x1x8x255xf16, {order = #NCHW, strides = [4080, 4080, 255, 1]}, @DDR>
    // CHECK: [[OUT_SLICE_1_COPY:%.+]] = VPUIP.NCEClusterTiling inputs([[SW_FQ]]#1 as [[ARG42:[^:]+]]: memref<1x1x8x255xf16, @CMX_NN>)
    // CHECK-SAME:                                              outputs([[OUT_SLICE_1]] as [[ARG43:[^:]+]]: memref<1x1x8x255xf16, {order = #NCHW, strides = [4080, 4080, 255, 1]}, @DDR>) -> memref<1x1x8x255xf16, {order = #NCHW, strides = [4080, 4080, 255, 1]}, @DDR> {
    // CHECK:     VPUIP.Copy inputs([[ARG42]] : memref<1x1x8x255xf16, @CMX_NN>) outputs([[ARG43]] : memref<1x1x8x255xf16, {order = #NCHW, strides = [4080, 4080, 255, 1]}, @DDR>) -> memref<1x1x8x255xf16, {order = #NCHW, strides = [4080, 4080, 255, 1]}, @DDR>
    // CHECK: [[OUT_CONCAT:%.+]] = VPUIP.ConcatView inputs([[OUT_SLICE_0_COPY]], [[OUT_SLICE_1_COPY]] : memref<1x1x8x255xf16, {order = #NCHW, strides = [4080, 4080, 255, 1]}, @DDR>, memref<1x1x8x255xf16, {order = #NCHW, strides = [4080, 4080, 255, 1]}, @DDR>) outputs(%alloc_5 : memref<1x1x16x255xf16, @DDR>) -> memref<1x1x16x255xf16, @DDR>
}
