//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% ppe-version=IntPPE" --one-shot-bufferize-VPU-to-VPUIP %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK:  module @VPU.SW {
// CHECK-NEXT:    func.func private @builtin_Convert(memref<*xi8, [@CMX_NN, 0]>, memref<*xf32, [@CMX_NN, 0]>) attributes {VPU.kernel_code = "convert.cpp", VPU.kernel_entry = "convert", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @builtin_Equal(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xi8, [@CMX_NN, 0]>) attributes {VPU.kernel_code = "eltwise_equal.cpp", VPU.kernel_entry = "eltwise_equal", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func @EqualOpSWLayer
// CHECK-SAME:     ([[INPUT1:%.+]]: memref<1x1x1x5xf16>, [[INPUT2:%.+]]: memref<1x1x1x1xf16>)
func.func @EqualOpSWLayer(%input1: tensor<1x1x1x5xf16>, %input2: tensor<1x1x1x1xf16>) -> tensor<1x1x1x5xf32> {
    %equalop = VPU.Equal(%input1, %input2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x5xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x5xi8>
    %output = VPU.Convert(%equalop) {dstElemType = f32} : tensor<1x1x1x5xi8> -> tensor<1x1x1x5xf32>
    return %output : tensor<1x1x1x5xf32>

    // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1x1x1x5xf16, [@CMX_NN, 0]>
    // CHECK: [[COPY0:%.+]] = VPUIP.Copy inputs([[INPUT1]] : memref<1x1x1x5xf16>) outputs([[ALLOC]] : memref<1x1x1x5xf16, [@CMX_NN, 0]>) -> memref<1x1x1x5xf16, [@CMX_NN, 0]>

    // CHECK: [[ALLOC0:%.+]] = memref.alloc() : memref<1x1x1x1xf16, [@CMX_NN, 0]>
    // CHECK: [[COPY1:%.+]] = VPUIP.Copy inputs([[INPUT2]] : memref<1x1x1x1xf16>) outputs([[ALLOC0]] : memref<1x1x1x1xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1xf16, [@CMX_NN, 0]>

    // CHECK: [[ALLOC1:%.+]] = memref.alloc() : memref<1x1x1x5xi8, [@CMX_NN, 0]>
    // CHECK: [[RES:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Equal inputs([[COPY0]] as {{[^:]+}}: memref<1x1x1x5xf16, [@CMX_NN, 0]>, [[COPY1]] as {{[^:]+}}: memref<1x1x1x1xf16, [@CMX_NN, 0]>) outputs([[ALLOC1]] as {{[^:]+}}: memref<1x1x1x5xi8, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x5xi8, [@CMX_NN, 0]>{
    // CHECK:   VPUIP.SW.Kernel.run({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<1x1x1x5xf16, [@CMX_NN, 0]>, memref<1x1x1x1xf16, [@CMX_NN, 0]>, memref<1x1x1x5xi8, [@CMX_NN, 0]>
    // CHECK: }

    // CHECK: [[ALLOC2:%.+]] = memref.alloc() : memref<1x1x1x5xi8>
    // CHECK: [[COPY2:%.+]] = VPUIP.Copy inputs([[RES]] : memref<1x1x1x5xi8, [@CMX_NN, 0]>) outputs([[ALLOC2]] : memref<1x1x1x5xi8>) -> memref<1x1x1x5xi8>

    // CHECK: [[ALLOC3:%.+]] = memref.alloc() : memref<1x1x1x5xi8, [@CMX_NN, 0]>
    // CHECK: [[COPY3:%.+]] = VPUIP.Copy inputs([[COPY2]] : memref<1x1x1x5xi8>) outputs([[ALLOC3]] : memref<1x1x1x5xi8, [@CMX_NN, 0]>) -> memref<1x1x1x5xi8, [@CMX_NN, 0]>

    // CHECK: [[ALLOC4:%.+]] = memref.alloc() : memref<1x1x1x5xf32, [@CMX_NN, 0]>
    // CHECK: [[OUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convert inputs([[COPY3]] as {{[^:]+}}: memref<1x1x1x5xi8, [@CMX_NN, 0]>) outputs([[ALLOC4]] as {{[^:]+}}: memref<1x1x1x5xf32, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x5xf32, [@CMX_NN, 0]>{
    // CHECK:   VPUIP.SW.Kernel.run({{[^:]+}}, {{[^:]+}}) : memref<1x1x1x5xi8, [@CMX_NN, 0]>, memref<1x1x1x5xf32, [@CMX_NN, 0]>
    // CHECK: }

    // CHECK: [[ALLOC6:%.+]] = memref.alloc() : memref<1x1x1x5xf32>
    // CHECK: [[COPY4:%.+]] = VPUIP.Copy inputs([[OUT]] : memref<1x1x1x5xf32, [@CMX_NN, 0]>) outputs([[ALLOC6]] : memref<1x1x1x5xf32>) -> memref<1x1x1x5xf32>

    // CHECK: return [[COPY4]] : memref<1x1x1x5xf32>

}

// -----

// CHECK: module @VPU.SW {
// CHECK-NEXT:   func.func private @builtin_ConditionalCopyOp(memref<*xsi8, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>) attributes {VPU.kernel_code = "conditional_copy.cpp", VPU.kernel_entry = "conditional_copy", VPU.task_type = @COMPUTE}
// CHECK-NEXT:   func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT: }

// CHECK-LABEL:  func.func @ConditionalCopySWLayer
// CHECK-SAME:     ([[COND:%.+]]: memref<1xsi8>, [[INPUT1:%.+]]: memref<1x1x4x4xf16>, [[INPUT2:%.+]]: memref<1x1x4x4xf16>)
func.func @ConditionalCopySWLayer(%cond: tensor<1xsi8>, %input1: tensor<1x1x4x4xf16>, %input2: tensor<1x1x4x4xf16>) -> (tensor<1x1x4x4xf16>) {
    %output = VPU.ConditionalCopyOp(%cond, %input1, %input2) : tensor<1xsi8>, tensor<1x1x4x4xf16>, tensor<1x1x4x4xf16> -> tensor<1x1x4x4xf16>
    return %output : tensor<1x1x4x4xf16>

    // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1xsi8, [@CMX_NN, 0]>
    // CHECK: [[COPY0:%.+]] = VPUIP.Copy inputs([[COND]] : memref<1xsi8>) outputs([[ALLOC]] : memref<1xsi8, [@CMX_NN, 0]>) -> memref<1xsi8, [@CMX_NN, 0]>

    // CHECK: [[ALLOC0:%.+]] = memref.alloc() : memref<1x1x4x4xf16, [@CMX_NN, 0]>
    // CHECK: [[COPY1:%.+]] = VPUIP.Copy inputs([[INPUT1]] : memref<1x1x4x4xf16>) outputs([[ALLOC0]] : memref<1x1x4x4xf16, [@CMX_NN, 0]>) -> memref<1x1x4x4xf16, [@CMX_NN, 0]>

    // CHECK: [[ALLOC1:%.+]] = memref.alloc() : memref<1x1x4x4xf16, [@CMX_NN, 0]>
    // CHECK: [[COPY2:%.+]] = VPUIP.Copy inputs([[INPUT2]] : memref<1x1x4x4xf16>) outputs([[ALLOC1]] : memref<1x1x4x4xf16, [@CMX_NN, 0]>) -> memref<1x1x4x4xf16, [@CMX_NN, 0]>

    // CHECK: [[ALLOC2:%.+]] = memref.alloc() : memref<1x1x4x4xf16, [@CMX_NN, 0]>
    // CHECK: [[RES:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_ConditionalCopyOp inputs([[COPY0]] as {{[^:]+}}: memref<1xsi8, [@CMX_NN, 0]>, [[COPY1]] as {{[^:]+}}: memref<1x1x4x4xf16, [@CMX_NN, 0]>, [[COPY2]] as {{[^:]+}}: memref<1x1x4x4xf16, [@CMX_NN, 0]>) outputs([[ALLOC2]] as {{[^:]+}}: memref<1x1x4x4xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x4x4xf16, [@CMX_NN, 0]>{
    // CHECK:   VPUIP.SW.Kernel.run({{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<1xsi8, [@CMX_NN, 0]>, memref<1x1x4x4xf16, [@CMX_NN, 0]>, memref<1x1x4x4xf16, [@CMX_NN, 0]>, memref<1x1x4x4xf16, [@CMX_NN, 0]>
    // CHECK: }

    // CHECK: [[ALLOC3:%.+]] = memref.alloc() : memref<1x1x4x4xf16>
    // CHECK: [[COPY3:%.+]] = VPUIP.Copy inputs([[RES]] : memref<1x1x4x4xf16, [@CMX_NN, 0]>) outputs([[ALLOC3]] : memref<1x1x4x4xf16>) -> memref<1x1x4x4xf16>
    // CHECK: return [[COPY3]] : memref<1x1x4x4xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK:  module @VPU.SW {
// CHECK-NEXT:      func.func private @builtin_ReduceSum(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, i64, i64, none) attributes {VPU.kernel_code = "reduce_sum.cpp", VPU.kernel_entry = "reduce_sum", VPU.task_type = @COMPUTE}
// CHECK-NEXT:      func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func @ReduceSumSWLayer
// CHECK-SAME:     ([[INPUT:%.+]]: memref<1x7x2x3xf16, #NHWC>)
func.func @ReduceSumSWLayer(%input: tensor<1x7x2x3xf16, {order = #NHWC}>) -> tensor<1x1x2x3xf16, {order = #NHWC}> {
    %output = VPU.ReduceSum(%input) {axes_value = [1], keep_dims} : tensor<1x7x2x3xf16, {order = #NHWC}> -> tensor<1x1x2x3xf16, {order = #NHWC}>
    return %output : tensor<1x1x2x3xf16, {order = #NHWC}>

    // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1x7x2x3xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK: [[COPY0:%.+]] = VPUIP.Copy inputs([[INPUT]] : memref<1x7x2x3xf16, #NHWC>) outputs([[ALLOC]] : memref<1x7x2x3xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x7x2x3xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK: [[ALLOC0:%.+]] = memref.alloc() : memref<1x1x2x3xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK: [[RES:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_ReduceSum inputs([[COPY0]] as {{[^:]+}}: memref<1x7x2x3xf16, #NHWC, [@CMX_NN, 0]>) outputs([[ALLOC0]] as {{[^:]+}}: memref<1x1x2x3xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x2x3xf16, #NHWC, [@CMX_NN, 0]>{
    // CHECK:   VPUIP.SW.Kernel.run {attrs = [1, 1, [0]]}({{[^:]+}}, {{[^:]+}}) : memref<1x7x2x3xf16, #NHWC, [@CMX_NN, 0]>, memref<1x1x2x3xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK: }

    // CHECK: [[ALLOC1:%.+]] = memref.alloc() : memref<1x1x2x3xf16, #NHWC>
    // CHECK: [[COPY1:%.+]] = VPUIP.Copy inputs([[RES]] : memref<1x1x2x3xf16, #NHWC, [@CMX_NN, 0]>) outputs([[ALLOC1]] : memref<1x1x2x3xf16, #NHWC>) -> memref<1x1x2x3xf16, #NHWC>
    // CHECK: return [[COPY1]] : memref<1x1x2x3xf16, #NHWC>
}

// -----

// CHECK:  module @VPU.SW {
// CHECK-NEXT:    func.func private @builtin_Log(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>) attributes {VPU.kernel_code = "activation_log.cpp", VPU.kernel_entry = "activation_log", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func @ActivationLog
// CHECK-SAME:     ([[INPUT:%.+]]: memref<1x50x1x1xf16>)
func.func @ActivationLog(%input: tensor<1x50x1x1xf16>) -> tensor<1x50x1x1xf16> {
    %output = VPU.Log(%input) : tensor<1x50x1x1xf16> -> tensor<1x50x1x1xf16>
    return %output : tensor<1x50x1x1xf16>

    // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1x50x1x1xf16, [@CMX_NN, 0]>
    // CHECK: [[COPY0:%.+]] = VPUIP.Copy inputs([[INPUT]] : memref<1x50x1x1xf16>) outputs([[ALLOC]] : memref<1x50x1x1xf16, [@CMX_NN, 0]>) -> memref<1x50x1x1xf16, [@CMX_NN, 0]>

    // CHECK: [[ALLOC0:%.+]] = memref.alloc() : memref<1x50x1x1xf16, [@CMX_NN, 0]>
    // CHECK: [[RES:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Log inputs([[COPY0]] as {{[^:]+}}: memref<1x50x1x1xf16, [@CMX_NN, 0]>) outputs([[ALLOC0]] as {{[^:]+}}: memref<1x50x1x1xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x50x1x1xf16, [@CMX_NN, 0]>{
    // CHECK:   VPUIP.SW.Kernel.run({{[^:]+}}, {{[^:]+}}) : memref<1x50x1x1xf16, [@CMX_NN, 0]>, memref<1x50x1x1xf16, [@CMX_NN, 0]>
    // CHECK: }

    // CHECK: [[ALLOC1:%.+]] = memref.alloc() : memref<1x50x1x1xf16>
    // CHECK: [[COPY1:%.+]] = VPUIP.Copy inputs([[RES]] : memref<1x50x1x1xf16, [@CMX_NN, 0]>) outputs([[ALLOC1]] : memref<1x50x1x1xf16>) -> memref<1x50x1x1xf16>
    // CHECK: return [[COPY1]] : memref<1x50x1x1xf16>

}

// -----

// CHECK:  module @VPU.SW {
// CHECK-NEXT:    func.func private @builtin_Convert(memref<*xf16, [@CMX_NN, 0]>, memref<*xf32, [@CMX_NN, 0]>) attributes {VPU.kernel_code = "convert.cpp", VPU.kernel_entry = "convert", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func @ConvertFP16ToFP32UsingSW
// CHECK-SAME:       ([[ARG:%.+]]: memref<1x3x4x4xf16>)
func.func @ConvertFP16ToFP32UsingSW(%input: tensor<1x3x4x4xf16>) -> tensor<1x3x4x4xf32> {
    %output = VPU.Convert(%input) {dstElemType = f32} : tensor<1x3x4x4xf16> -> tensor<1x3x4x4xf32>
    return %output : tensor<1x3x4x4xf32>

    // CHECK-NOT: VPU.Convert
    // CHECK: [[INPUT_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x3x4x4xf16, [@CMX_NN, 0]>
    // CHECK: [[INPUT_CMX:%.+]] = VPUIP.Copy inputs([[ARG]] : memref<1x3x4x4xf16>) outputs([[INPUT_BUFFER_CMX]] : memref<1x3x4x4xf16, [@CMX_NN, 0]>) -> memref<1x3x4x4xf16, [@CMX_NN, 0]>
    // CHECK: [[CONVERT_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x3x4x4xf32, [@CMX_NN, 0]>

    // CHECK: [[OUTPUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convert inputs([[INPUT_CMX]] as {{[^:]+}}: memref<1x3x4x4xf16, [@CMX_NN, 0]>) outputs([[CONVERT_BUFFER_CMX]] as {{[^:]+}}: memref<1x3x4x4xf32, [@CMX_NN, 0]>) on tile 0 -> memref<1x3x4x4xf32, [@CMX_NN, 0]>{
    // CHECK:   VPUIP.SW.Kernel.run({{[^:]+}}, {{[^:]+}}) : memref<1x3x4x4xf16, [@CMX_NN, 0]>, memref<1x3x4x4xf32, [@CMX_NN, 0]>
    // CHECK: }

    // CHECK: [[OUTPUT_BUFFER:%.+]] = memref.alloc() : memref<1x3x4x4xf32>
    // CHECK: [[OUTPUT_DDR:%.+]] = VPUIP.Copy inputs([[OUTPUT]] : memref<1x3x4x4xf32, [@CMX_NN, 0]>) outputs([[OUTPUT_BUFFER]] : memref<1x3x4x4xf32>) -> memref<1x3x4x4xf32>
    // CHECK: return [[OUTPUT_DDR]] : memref<1x3x4x4xf32>
}

// -----

// CHECK:  module @VPU.SW {
// CHECK-NEXT:    func.func private @builtin_Convert(memref<*xf16>, memref<*xf32>) attributes {VPU.kernel_code = "convert.cpp", VPU.kernel_entry = "convert", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func @NCEClusterTilingConvertFP16ToFP32
// CHECK-SAME:      ([[ARG:%.+]]: memref<1x3x4x4xf16>)
func.func @NCEClusterTilingConvertFP16ToFP32(%input: tensor<1x3x4x4xf16>) -> tensor<1x3x4x4xf32> {
    %output = VPU.NCE.ClusterTiling (%input as %arg0: tensor<1x3x4x4xf16>) -> tensor<1x3x4x4xf32> {
        %cvt = VPU.Convert(%arg0) {dstElemType = f32} : tensor<1x3x4x4xf16> -> tensor<1x3x4x4xf32>
        VPU.Yield %cvt
    }
    return %output : tensor<1x3x4x4xf32>

    // CHECK: [[OUTBUF:%.+]] = memref.alloc() : memref<1x3x4x4xf32>
    // CHECK: [[OUTPUT:%.+]] = VPUIP.NCEClusterTiling inputs([[ARG]] as {{[^:]+}}: memref<1x3x4x4xf16>) outputs([[OUTBUF]] as {{[^:]+}}: memref<1x3x4x4xf32>) -> memref<1x3x4x4xf32> {
    // CHECK:   [[SWKERNEL_OUTPUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convert inputs({{[^:]+}} as {{[^:]+}}: memref<1x3x4x4xf16>) outputs({{[^:]+}} as {{[^:]+}}: memref<1x3x4x4xf32>) on tile 0 -> memref<1x3x4x4xf32>{
    // CHECK:     VPUIP.SW.Kernel.run({{[^:]+}}, {{[^:]+}}) : memref<1x3x4x4xf16>, memref<1x3x4x4xf32>
    // CHECK:   }
    // CHECK: }

    // CHECK: return [[OUTPUT]] : memref<1x3x4x4xf32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK:  module @VPU.SW {
// CHECK-NEXT:    func.func private @builtin_SoftMax(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, i64, i64) attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func @SingleSWLayer
// CHECK-SAME:      ([[ARG:%.+]]: memref<1x1x1x1000xf16>)
func.func @SingleSWLayer(%input: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %output = VPU.SoftMax(%input) {axisInd = 3, padSize = 3} : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    return %output: tensor<1x1x1x1000xf16>

    // CHECK: [[INPUT_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    // CHECK: [[INPUT_CMX:%.+]] = VPUIP.Copy inputs([[ARG]] : memref<1x1x1x1000xf16>) outputs([[INPUT_BUFFER_CMX]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    // CHECK: [[SOFTMAX_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x1x1x1000xf16, [@CMX_NN, 0]>

    // CHECK: [[OUTPUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs([[INPUT_CMX]] as {{[^:]+}}: memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs([[SOFTMAX_BUFFER_CMX]] as {{[^:]+}}: memref<1x1x1x1000xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>{
    // CHECK:   VPUIP.SW.Kernel.run {attrs = [0, 3]}({{[^:]+}}, {{[^:]+}}) : memref<1x1x1x1000xf16, [@CMX_NN, 0]>, memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    // CHECK:  }

    // CHECK: [[OUTPUT_BUFFER:%.+]] = memref.alloc() : memref<1x1x1x1000xf16>
    // CHECK: [[OUTPUT_DDR:%.+]] = VPUIP.Copy inputs([[OUTPUT]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs([[OUTPUT_BUFFER]] : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>
    // CHECK: return [[OUTPUT_DDR]] : memref<1x1x1x1000xf16>
}

// -----
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK: module @VPU.SW  {
// CHECK-NEXT: func.func private @builtin_Sigmoid(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>) attributes {VPU.kernel_code = "activation_sigmoid.cpp", VPU.kernel_entry = "activation_sigmoid", VPU.task_type = @COMPUTE}
// CHECK-NEXT: func.func private @builtin_SoftMax(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, i64, i64) attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax", VPU.task_type = @COMPUTE}
// CHECK-NEXT: func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT: }

// CHECK-LABEL:  func.func @ThreeSWLayers
// CHECK-SAME:      ([[ARG:%.+]]: memref<1x1x1x2000xf16>)
func.func @ThreeSWLayers(%input: tensor<1x1x1x2000xf16>) -> tensor<1x1x1x2000xf16> {
    %sftmax = VPU.SoftMax(%input) {axisInd = 3} : tensor<1x1x1x2000xf16> -> tensor<1x1x1x2000xf16>
    %sigmoid = VPU.Sigmoid(%sftmax) {axisInd = 3} : tensor<1x1x1x2000xf16> -> tensor<1x1x1x2000xf16>
    %output = VPU.SoftMax(%sigmoid) {axisInd = 3} : tensor<1x1x1x2000xf16> -> tensor<1x1x1x2000xf16>

    return %output : tensor<1x1x1x2000xf16>

    // CHECK: [[INPUT_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x1x1x2000xf16, [@CMX_NN, 0]>
    // CHECK: [[INPUT_CMX:%.+]] = VPUIP.Copy inputs([[ARG]] : memref<1x1x1x2000xf16>) outputs([[INPUT_BUFFER_CMX]] : memref<1x1x1x2000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x2000xf16, [@CMX_NN, 0]>

    // CHECK: [[SOFTMAX1_SW_OUTPUT_BUFFER:%.+]] = memref.alloc() : memref<1x1x1x2000xf16, [@CMX_NN, 0]>
    // CHECK: [[SOFTMAX1_SW_OUTPUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs([[INPUT_CMX]] as {{[^:]+}}: memref<1x1x1x2000xf16, [@CMX_NN, 0]>) outputs([[SOFTMAX1_SW_OUTPUT_BUFFER]] as {{[^:]+}}: memref<1x1x1x2000xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x2000xf16, [@CMX_NN, 0]>{
    // CHECK:   VPUIP.SW.Kernel.run {attrs = [0, 0]}({{[^:]+}}, {{[^:]+}}) : memref<1x1x1x2000xf16, [@CMX_NN, 0]>, memref<1x1x1x2000xf16, [@CMX_NN, 0]>
    // CHECK: }

    // CHECK: [[SOFTMAX1_SW_OUTPUT_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x1x1x2000xf16>
    // CHECK: [[SOFTMAX1_SW_OUTPUT_CMX:%.+]] = VPUIP.Copy inputs([[SOFTMAX1_SW_OUTPUT]] : memref<1x1x1x2000xf16, [@CMX_NN, 0]>) outputs([[SOFTMAX1_SW_OUTPUT_BUFFER_CMX]] : memref<1x1x1x2000xf16>) -> memref<1x1x1x2000xf16>

    // CHECK: [[SIGMOID_SW_INPUT_BUFFER:%.+]] = memref.alloc() : memref<1x1x1x2000xf16, [@CMX_NN, 0]>
    // CHECK: [[SIGMOID_SW_INPUT_CMX:%.+]] = VPUIP.Copy inputs([[SOFTMAX1_SW_OUTPUT_CMX]] : memref<1x1x1x2000xf16>) outputs([[SIGMOID_SW_INPUT_BUFFER]] : memref<1x1x1x2000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x2000xf16, [@CMX_NN, 0]>
    // CHECK: [[SIGMOID_SW_OUTPUT_BUFFER:%.+]] = memref.alloc() : memref<1x1x1x2000xf16, [@CMX_NN, 0]>
    // CHECK: [[SIGMOID_SW_OUTPUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Sigmoid inputs([[SIGMOID_SW_INPUT_CMX]] as {{[^:]+}}: memref<1x1x1x2000xf16, [@CMX_NN, 0]>) outputs([[SIGMOID_SW_OUTPUT_BUFFER]] as {{[^:]+}}: memref<1x1x1x2000xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x2000xf16, [@CMX_NN, 0]>{
    // CHECK:   VPUIP.SW.Kernel.run({{[^:]+}}, {{[^:]+}}) : memref<1x1x1x2000xf16, [@CMX_NN, 0]>, memref<1x1x1x2000xf16, [@CMX_NN, 0]>
    // CHECK: }

    // CHECK: [[SIGMOID_SW_OUTPUT_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x1x1x2000xf16>
    // CHECK: [[SIGMOID_SW_OUTPUT_CMX:%.+]] = VPUIP.Copy inputs([[SIGMOID_SW_OUTPUT]] : memref<1x1x1x2000xf16, [@CMX_NN, 0]>) outputs([[SIGMOID_SW_OUTPUT_BUFFER_CMX]] : memref<1x1x1x2000xf16>) -> memref<1x1x1x2000xf16>

    // CHECK: [[SOFTMAX2_SW_INPUT_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x1x1x2000xf16, [@CMX_NN, 0]>
    // CHECK: [[SOFTMAX2_SW_INPUT_CMX:%.+]] = VPUIP.Copy inputs([[SIGMOID_SW_OUTPUT_CMX]] : memref<1x1x1x2000xf16>) outputs([[SOFTMAX2_SW_INPUT_BUFFER_CMX]] : memref<1x1x1x2000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x2000xf16, [@CMX_NN, 0]>
    // CHECK: [[SOFTMAX2_SW_OUTPUT_BUFFER:%.+]] = memref.alloc() : memref<1x1x1x2000xf16, [@CMX_NN, 0]>
    // CHECK: [[SOFTMAX2_SW_OUTPUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs([[SOFTMAX2_SW_INPUT_CMX]] as {{[^:]+}}: memref<1x1x1x2000xf16, [@CMX_NN, 0]>) outputs([[SOFTMAX2_SW_OUTPUT_BUFFER]] as {{[^:]+}}: memref<1x1x1x2000xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x2000xf16, [@CMX_NN, 0]>{
    // CHECK:   VPUIP.SW.Kernel.run {attrs = [0, 0]}({{[^:]+}}, {{[^:]+}}) : memref<1x1x1x2000xf16, [@CMX_NN, 0]>, memref<1x1x1x2000xf16, [@CMX_NN, 0]>
    // CHECK: }

    // CHECK: [[SOFTMAX2_SW_OUTPUT_BUFFER_DDR:%.+]] = memref.alloc() : memref<1x1x1x2000xf16>
    // CHECK: [[SOFTMAX2_SW_OUTPUT_DDR:%.+]] = VPUIP.Copy inputs([[SOFTMAX2_SW_OUTPUT]] : memref<1x1x1x2000xf16, [@CMX_NN, 0]>) outputs([[SOFTMAX2_SW_OUTPUT_BUFFER_DDR]] : memref<1x1x1x2000xf16>) -> memref<1x1x1x2000xf16>
    // CHECK: return [[SOFTMAX2_SW_OUTPUT_DDR]] : memref<1x1x1x2000xf16>
}

// -----

// CHECK:  module @VPU.SW {
// CHECK-NEXT:    func.func private @builtin_ReduceMean(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, i64, i64, none) attributes {VPU.kernel_code = "reduce_mean.cpp", VPU.kernel_entry = "reduce_mean", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func @ReduceMean
// CHECK-SAME:      ([[ARG0:%.+]]: memref<1x512x7x7xf16>, [[ARG1:%.+]]: memref<1x512x7xf16>)
func.func @ReduceMean(%input0: tensor<1x512x7x7xf16>, %input1: tensor<1x512x7xf16>) -> tensor<1x512x7xf16> {
    %output = VPU.ReduceMean(%input0) {axes_value = [2]} : tensor<1x512x7x7xf16> -> tensor<1x512x7xf16>
    return %output : tensor<1x512x7xf16>

    // CHECK: [[INPUT_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x512x7x7xf16, [@CMX_NN, 0]>
    // CHECK: [[INPUT_CMX:%.+]] = VPUIP.Copy inputs([[ARG0]] : memref<1x512x7x7xf16>) outputs([[INPUT_BUFFER_CMX]] : memref<1x512x7x7xf16, [@CMX_NN, 0]>) -> memref<1x512x7x7xf16, [@CMX_NN, 0]>
    // CHECK: [[REDUCEMEAN_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x512x7xf16, [@CMX_NN, 0]>

    // CHECK: [[OUTPUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_ReduceMean inputs([[INPUT_CMX]] as {{[^:]+}}: memref<1x512x7x7xf16, [@CMX_NN, 0]>) outputs([[REDUCEMEAN_BUFFER_CMX]] as {{[^:]+}}: memref<1x512x7xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x512x7xf16, [@CMX_NN, 0]>{
    // CHECK:   VPUIP.SW.Kernel.run {attrs = [0, 1, [1]]}({{[^:]+}}, {{[^:]+}}) : memref<1x512x7x7xf16, [@CMX_NN, 0]>, memref<1x512x7xf16, [@CMX_NN, 0]>
    // CHECK: }

    // CHECK: [[OUTPUT_BUFFER:%.+]] = memref.alloc() : memref<1x512x7xf16>
    // CHECK: [[OUTPUT_DDR:%.+]] = VPUIP.Copy inputs([[OUTPUT]] : memref<1x512x7xf16, [@CMX_NN, 0]>) outputs([[OUTPUT_BUFFER]] : memref<1x512x7xf16>) -> memref<1x512x7xf16>
    // CHECK: return [[OUTPUT_DDR]] : memref<1x512x7xf16>
}

// -----

// CHECK:  module @VPU.SW {
// CHECK-NEXT:    func.func private @builtin_Interpolate(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, i64, i64, i64, i64, none, none, none, none, f64, none, none) attributes {VPU.kernel_code = "interpolate.cpp", VPU.kernel_entry = "interpolate", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func @InterpolateSWLayerWithUnnecessaryScalingAxes
// CHECK-SAME:      ([[ARG:%.+]]: memref<1x128x1x1xf16>)
func.func @InterpolateSWLayerWithUnnecessaryScalingAxes(%input: tensor<1x128x1x1xf16>) -> tensor<1x128x32x32xf16> {
    %output = VPU.Interpolate(%input) {attr = #IE.Interpolate<antialias = false, coord_mode = <ALIGN_CORNERS>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [0, 1, 2, 3], initial_input_dims_attr = [1, 128, 1, 1], initial_input_offset_attr = [0, 0, 0, 0], initial_output_dims_attr = [1, 128, 32, 32], initial_output_offset_attr = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00, 3.200000e+00, 3.200000e+00], sizes_attr = [1, 128, 32, 32], tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]} : tensor<1x128x1x1xf16> -> tensor<1x128x32x32xf16>

    return %output : tensor<1x128x32x32xf16>

    // CHECK: [[INPUT_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x128x1x1xf16, [@CMX_NN, 0]>
    // CHECK: [[INPUT_CMX:%.+]] = VPUIP.Copy inputs([[ARG]] : memref<1x128x1x1xf16>) outputs([[INPUT_BUFFER_CMX]] : memref<1x128x1x1xf16, [@CMX_NN, 0]>) -> memref<1x128x1x1xf16, [@CMX_NN, 0]>
    // CHECK: [[INTERPOLATE_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x128x32x32xf16, [@CMX_NN, 0]>

    // CHECK: [[OUTPUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Interpolate inputs([[INPUT_CMX]] as {{[^:]+}}: memref<1x128x1x1xf16, [@CMX_NN, 0]>) outputs([[INTERPOLATE_BUFFER_CMX]] as {{[^:]+}}: memref<1x128x32x32xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x128x32x32xf16, [@CMX_NN, 0]>{
    // CHECK:   VPUIP.SW.Kernel.run {attrs = [2, 4, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [1, 1, 128, 1], [32, 32, 128, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}({{[^:]+}}, {{[^:]+}}) : memref<1x128x1x1xf16, [@CMX_NN, 0]>, memref<1x128x32x32xf16, [@CMX_NN, 0]>
    // CHECK: }

    // CHECK: [[OUTPUT_BUFFER:%.+]] = memref.alloc() : memref<1x128x32x32xf16>
    // CHECK: [[OUTPUT_DDR:%.+]] = VPUIP.Copy inputs([[OUTPUT]] : memref<1x128x32x32xf16, [@CMX_NN, 0]>) outputs([[OUTPUT_BUFFER]] : memref<1x128x32x32xf16>) -> memref<1x128x32x32xf16>
    // CHECK: return [[OUTPUT_DDR]] : memref<1x128x32x32xf16>
}

// -----
// Case A: SW Kernel's input and output buffers don't all fit in NNCMX. Try to work as much as possible with NNCMX.
// Input buffer is smaller so input buffer will be placed in DDR and output buffer will be placed in NNCMX.

// CHECK:  module @VPU.SW {
// CHECK-NEXT:    func.func private @builtin_Interpolate(memref<*xf16>, memref<*xf16, [@CMX_NN, 0]>, i64, i64, i64, i64, none, none, none, none, f64, none, none) attributes {VPU.kernel_code = "interpolate.cpp", VPU.kernel_entry = "interpolate", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func @SingleSWLayerTooLargeForCMXCaseA
// CHECK-SAME:      ({{[^:]+}}: memref<1x128x50x50xf16>)
func.func @SingleSWLayerTooLargeForCMXCaseA(%input: tensor<1x128x50x50xf16>) -> tensor<1x128x75x75xf16> {
    %output = VPU.Interpolate(%input) {attr = #IE.Interpolate<antialias = false, coord_mode = <ALIGN_CORNERS>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [0, 1, 2, 3], initial_input_dims_attr = [1, 128, 50, 50], initial_input_offset_attr = [0, 0, 0, 0], initial_output_dims_attr = [1, 128, 75, 75], initial_output_offset_attr = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00, 1.50000e+00, 1.50000e+00], sizes_attr = [1, 128, 75, 75], tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]} : tensor<1x128x50x50xf16> -> tensor<1x128x75x75xf16>
    return %output : tensor<1x128x75x75xf16>

    // CHECK: [[OUTPUT_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x128x75x75xf16, [@CMX_NN, 0]>

    // CHECK-NOT: VPUIP.Copy
    // CHECK-NOT: memref.alloc()

    // CHECK: [[OUTPUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Interpolate inputs({{[^:]+}} as {{[^:]+}}: memref<1x128x50x50xf16>) outputs([[OUTPUT_BUFFER_CMX]] as {{[^:]+}}: memref<1x128x75x75xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x128x75x75xf16, [@CMX_NN, 0]>{
    // CHECK:   VPUIP.SW.Kernel.run {attrs = [2, 4, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [50, 50, 128, 1], [75, 75, 128, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}({{[^:]+}}, {{[^:]+}}) : memref<1x128x50x50xf16>, memref<1x128x75x75xf16, [@CMX_NN, 0]>
    // CHECK: }

    // CHECK: [[OUTPUT_BUFFER:%.+]] = memref.alloc() : memref<1x128x75x75xf16>
    // CHECK: [[OUTPUT_DDR:%.+]] = VPUIP.Copy inputs([[OUTPUT]] : memref<1x128x75x75xf16, [@CMX_NN, 0]>) outputs([[OUTPUT_BUFFER]] : memref<1x128x75x75xf16>) -> memref<1x128x75x75xf16>
    // CHECK: return [[OUTPUT_DDR]] : memref<1x128x75x75xf16>
}

// -----
// Case B: SW Kernel's input and output buffers don't all fit in NNCMX. Try to work as much as possible with NNCMX.
// Input buffer is larger so input buffer will be placed in NNCMX and output buffer will be placed in DDR.

// CHECK:  module @VPU.SW {
// CHECK-NEXT:    func.func private @builtin_Interpolate(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16>, i64, i64, i64, i64, none, none, none, none, f64, none, none) attributes {VPU.kernel_code = "interpolate.cpp", VPU.kernel_entry = "interpolate", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func @SingleSWLayerTooLargeForCMXCaseB
// CHECK-SAME:      ([[ARG:%.+]]: memref<1x128x75x75xf16>)
func.func @SingleSWLayerTooLargeForCMXCaseB(%input: tensor<1x128x75x75xf16>) -> tensor<1x128x50x50xf16> {
    %output = VPU.Interpolate(%input) {attr = #IE.Interpolate<antialias = false, coord_mode = <ALIGN_CORNERS>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [0, 1, 2, 3], initial_input_dims_attr = [1, 128, 75, 75], initial_input_offset_attr = [0, 0, 0, 0], initial_output_dims_attr = [1, 128, 50, 50], initial_output_offset_attr = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00, 0.666666e+00, 0.666666e+00], sizes_attr = [1, 128, 50, 50], tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]} : tensor<1x128x75x75xf16> -> tensor<1x128x50x50xf16>
    return %output : tensor<1x128x50x50xf16>

    // CHECK: [[INPUT_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x128x75x75xf16, [@CMX_NN, 0]>
    // CHECK: [[INPUT_CMX:%.+]] = VPUIP.Copy inputs([[ARG]] : memref<1x128x75x75xf16>) outputs([[INPUT_BUFFER_CMX]] : memref<1x128x75x75xf16, [@CMX_NN, 0]>) -> memref<1x128x75x75xf16, [@CMX_NN, 0]>
    // CHECK: [[INTERPOLATE_BUFFER_DDR:%.+]] = memref.alloc() : memref<1x128x50x50xf16>

    // CHECK: [[OUTPUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Interpolate inputs({{[^:]+}} as {{[^:]+}}: memref<1x128x75x75xf16, [@CMX_NN, 0]>) outputs([[INTERPOLATE_BUFFER_DDR]] as {{[^:]+}}: memref<1x128x50x50xf16>) on tile 0 -> memref<1x128x50x50xf16>{
    // CHECK:   VPUIP.SW.Kernel.run {attrs = [2, 4, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [75, 75, 128, 1], [50, 50, 128, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}({{[^:]+}}, {{[^:]+}}) : memref<1x128x75x75xf16, [@CMX_NN, 0]>, memref<1x128x50x50xf16>
    // CHECK: }

    // CHECK-NOT: memref.alloc()
    // CHECK-NOT: VPUIP.Copy

    // CHECK: return [[OUTPUT]] : memref<1x128x50x50xf16>
}

// -----
// Case C: Neither of SW Kernel's input and output buffers fit in NNCMX.
// Both buffers will be placed in DDR.

// CHECK:  module @VPU.SW {
// CHECK-NEXT:    func.func private @builtin_SoftMax(memref<*xf16>, memref<*xf16>, i64, i64) attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func @SingleSWLayerTooLargeForCMXCaseC
// CHECK-SAME:      ({{[^:]+}}: memref<1x1x1x1000000xf16>)
func.func @SingleSWLayerTooLargeForCMXCaseC(%input: tensor<1x1x1x1000000xf16>) -> tensor<1x1x1x1000000xf16> {
    %output = VPU.SoftMax(%input) {axisInd = 3} : tensor<1x1x1x1000000xf16> -> tensor<1x1x1x1000000xf16>
    return %output: tensor<1x1x1x1000000xf16>

    // CHECK: [[INPUT_BUFFER_DDR:%.+]] = memref.alloc() : memref<1x1x1x1000000xf16>

    // CHECK-NOT: memref.alloc()
    // CHECK-NOT: VPUIP.Copy

    // CHECK: [[OUTPUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs({{[^:]+}} as {{[^:]+}}: memref<1x1x1x1000000xf16>) outputs([[INPUT_BUFFER_DDR]] as {{[^:]+}}: memref<1x1x1x1000000xf16>) on tile 0 -> memref<1x1x1x1000000xf16>{
    // CHECK:   VPUIP.SW.Kernel.run {attrs = [0, 0]}({{[^:]+}}, {{[^:]+}}) : memref<1x1x1x1000000xf16>, memref<1x1x1x1000000xf16>
    // CHECK: }

    // CHECK-NOT: memref.alloc()
    // CHECK-NOT: VPUIP.Copy

    // CHECK: return [[OUTPUT]] : memref<1x1x1x1000000xf16>
}

// -----
// Case D: SW Kernel's input and output buffers don't all fit in NNCMX. Try to work as much as possible with NNCMX.
// Both input buffers can fit together in NNCMX and together are larger than the output buffer.
// Both input buffers will be placed in NNCMX and the output buffer will be placed in DDR.

// CHECK:  module @VPU.SW {
// CHECK-NEXT:    func.func private @builtin_GroupConvolution(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16>, none, none, none, none, i64) attributes {VPU.kernel_code = "convolution.cpp", VPU.kernel_entry = "convolution", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func @SingleSWLayerTooLargeForCMXCaseD
// CHECK-SAME:      ([[ARG:%.+]]: memref<1x16x210x210xf16>)
func.func @SingleSWLayerTooLargeForCMXCaseD(%input: tensor<1x16x210x210xf16>) -> tensor<1x8x208x208xf16> {
    %cst = const.Declare tensor<8x8x3x3xf16> = dense<2.0> : tensor<2x4x8x3x3xf16>, [#const.Reshape<[8, 8, 3, 3]>]
    %output = VPU.GroupConvolution(%input, %cst) {dilations = [1, 1], groups = 2 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x210x210xf16>, tensor<8x8x3x3xf16> -> tensor<1x8x208x208xf16>
    return %output : tensor<1x8x208x208xf16>

    // CHECK: [[CST_DECLARE:%.+]] = const.Declare memref<8x8x3x3xf16> = dense<2.000000e+00> : tensor<2x4x8x3x3xf16>, [#const.Reshape<[8, 8, 3, 3]>]

    // CHECK: [[INPUT_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x16x210x210xf16, [@CMX_NN, 0]>
    // CHECK: [[INPUT_CMX:%.+]] = VPUIP.Copy inputs([[ARG]] : memref<1x16x210x210xf16>) outputs([[INPUT_BUFFER_CMX]] : memref<1x16x210x210xf16, [@CMX_NN, 0]>) -> memref<1x16x210x210xf16, [@CMX_NN, 0]>

    // CHECK: [[CST_DECLARE_BUFFER_CMX:%.+]] = memref.alloc() : memref<8x8x3x3xf16, [@CMX_NN, 0]>
    // CHECK: [[CST_DECLARE_CMX:%.+]] = VPUIP.Copy inputs([[CST_DECLARE]] : memref<8x8x3x3xf16>) outputs([[CST_DECLARE_BUFFER_CMX]] : memref<8x8x3x3xf16, [@CMX_NN, 0]>) -> memref<8x8x3x3xf16, [@CMX_NN, 0]>

    // CHECK: [[GROUPCONV_BUFFER:%.+]] = memref.alloc() : memref<1x8x208x208xf16>
    // CHECK: [[OUTPUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_GroupConvolution inputs([[INPUT_CMX]] as {{[^:]+}}: memref<1x16x210x210xf16, [@CMX_NN, 0]>, [[CST_DECLARE_CMX]] as {{[^:]+}}: memref<8x8x3x3xf16, [@CMX_NN, 0]>) outputs([[GROUPCONV_BUFFER]] as {{[^:]+}}: memref<1x8x208x208xf16>) on tile 0 -> memref<1x8x208x208xf16>{
    // CHECK:   VPUIP.SW.Kernel.run
    // CHECK-SAME{LITERAL}: {attrs = [[1, 1], [0, 0], [0, 0], [1, 1], 2]}
    // CHECK:   ({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<1x16x210x210xf16, [@CMX_NN, 0]>, memref<8x8x3x3xf16, [@CMX_NN, 0]>, memref<1x8x208x208xf16>
    // CHECK: }

    // CHECK-NOT: memref.alloc()
    // CHECK-NOT: VPUIP.Copy

    // CHECK: return [[OUTPUT]] : memref<1x8x208x208xf16>
}

// -----

// CHECK:  module @VPU.SW {
// CHECK-NEXT:    func.func private @builtin_StridedSlice(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, i64, none, none, none, i64, i64, i64) attributes {VPU.kernel_code = "strided_slice.cpp", VPU.kernel_entry = "strided_slice", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func @StridedSlice2Dim
// CHECK-SAME:      ([[ARG:%.+]]: memref<3x40x40x15xf16>)
func.func @StridedSlice2Dim(%input: tensor<3x40x40x15xf16>) -> tensor<3x40x20x5xf16> {
    %output = VPU.StridedSlice(%input) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [3, 40, 40, 15], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 3]} : tensor<3x40x40x15xf16> -> tensor<3x40x20x5xf16>
    return %output : tensor<3x40x20x5xf16>

    // CHECK: [[INPUT_BUFFER_CMX:%.+]] = memref.alloc() : memref<3x40x40x15xf16, [@CMX_NN, 0]>
    // CHECK: [[INPUT_CMX:%.+]] = VPUIP.Copy inputs([[ARG]] : memref<3x40x40x15xf16>) outputs([[INPUT_BUFFER_CMX]] : memref<3x40x40x15xf16, [@CMX_NN, 0]>) -> memref<3x40x40x15xf16, [@CMX_NN, 0]>

    // CHECK: [[STRIDESLICE_BUFFER_CMX:%.+]] = memref.alloc() : memref<3x40x20x5xf16, [@CMX_NN, 0]>
    // CHECK: [[OUTPUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_StridedSlice inputs([[INPUT_CMX]] as {{[^:]+}}: memref<3x40x40x15xf16, [@CMX_NN, 0]>) outputs([[STRIDESLICE_BUFFER_CMX]] as {{[^:]+}}: memref<3x40x20x5xf16, [@CMX_NN, 0]>) on tile 0 -> memref<3x40x20x5xf16, [@CMX_NN, 0]>{
    // CHECK:   VPUIP.SW.Kernel.run
    // CHECK-SAME{LITERAL}: {attrs = [9223372036854775807, [0, 0, 0, 0], [3, 40, 40, 15], [1, 1, 2, 3], 1, 1, 1]}
    // CHECK:  ({{[^:]+}}, {{[^:]+}}) : memref<3x40x40x15xf16, [@CMX_NN, 0]>, memref<3x40x20x5xf16, [@CMX_NN, 0]>
    // CHECK: }

    // CHECK: [[OUTPUT_BUFFER:%.+]] = memref.alloc() : memref<3x40x20x5xf16>
    // CHECK: [[OUTPUT_DDR:%.+]] = VPUIP.Copy inputs([[OUTPUT]] : memref<3x40x20x5xf16, [@CMX_NN, 0]>) outputs([[OUTPUT_BUFFER:%.+]] : memref<3x40x20x5xf16>) -> memref<3x40x20x5xf16>
    // CHECK: return [[OUTPUT_DDR]] : memref<3x40x20x5xf16>
}

// -----

// CHECK:  module @VPU.SW {
// CHECK-NEXT:    func.func private @builtin_Convolution(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none, none, none, none, i64) attributes {VPU.kernel_code = "convolution.cpp", VPU.kernel_entry = "convolution", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func @Convolution
// CHECK-SAME:      ([[ARG0:%.+]]: memref<1x32x64x64xf16>, [[ARG1:%.+]]: memref<64x32x3x3xf16>)
func.func @Convolution(
        %input: tensor<1x32x64x64xf16>,
        %filter: tensor<64x32x3x3xf16>)
        -> tensor<1x64x62x62xf16> {
    %output = VPU.Convolution(%input, %filter) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x32x64x64xf16>, tensor<64x32x3x3xf16> -> tensor<1x64x62x62xf16>
    return %output : tensor<1x64x62x62xf16>

    // CHECK: [[INPUT_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x32x64x64xf16, [@CMX_NN, 0]>
    // CHECK: [[INPUT_CMX:%.+]] = VPUIP.Copy inputs([[ARG0]] : memref<1x32x64x64xf16>) outputs([[INPUT_BUFFER_CMX]] : memref<1x32x64x64xf16, [@CMX_NN, 0]>) -> memref<1x32x64x64xf16, [@CMX_NN, 0]>
    // CHECK: [[FILTER_BUFFER_CMX:%.+]] = memref.alloc() : memref<64x32x3x3xf16, [@CMX_NN, 0]>
    // CHECK: [[FILTER_CMX:%.+]] = VPUIP.Copy inputs([[ARG1]] : memref<64x32x3x3xf16>) outputs([[FILTER_BUFFER_CMX]] : memref<64x32x3x3xf16, [@CMX_NN, 0]>) -> memref<64x32x3x3xf16, [@CMX_NN, 0]>
    // CHECK: [[CONV_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x64x62x62xf16, [@CMX_NN, 0]>

    // CHECK: [[OUTPUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convolution inputs([[INPUT_CMX]] as {{[^:]+}}: memref<1x32x64x64xf16, [@CMX_NN, 0]>, [[FILTER_CMX]] as {{[^:]+}}: memref<64x32x3x3xf16, [@CMX_NN, 0]>) outputs([[CONV_BUFFER_CMX]] as {{[^:]+}}: memref<1x64x62x62xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x64x62x62xf16, [@CMX_NN, 0]>{
    // CHECK:   VPUIP.SW.Kernel.run
    // CHECK-SAME{LITERAL}: {attrs = [[1, 1], [0, 0], [0, 0], [1, 1], 1]}
    // CHECK:   ({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<1x32x64x64xf16, [@CMX_NN, 0]>, memref<64x32x3x3xf16, [@CMX_NN, 0]>, memref<1x64x62x62xf16, [@CMX_NN, 0]>
    // CHECK: }

    // CHECK: [[OUTPUT_BUFFER:%.+]] = memref.alloc() : memref<1x64x62x62xf16>
    // CHECK: [[OUTPUT_DDR:%.+]] = VPUIP.Copy inputs([[OUTPUT]] : memref<1x64x62x62xf16, [@CMX_NN, 0]>) outputs([[OUTPUT_BUFFER]] : memref<1x64x62x62xf16>) -> memref<1x64x62x62xf16>
    // CHECK: return [[OUTPUT_DDR]] : memref<1x64x62x62xf16>
}

// -----
// Neither of SW Kernel's input and output buffers fit in NNCMX, so both of them should be placed in DDR
// but they will later be converted from SW Kernel to VPUIP.PermuteDMA operations.
// Leave input and output buffers in NNCMX to not add a performance hit for DMA for working with DDR.

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK:  module @VPU.SW {
// CHECK-NEXT:    func.func private @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func @MemPermuteSWLayer
// CHECK-SAME:      ([[ARG:%.+]]: memref<1x3x1024x1024xf16>)
func.func @MemPermuteSWLayer(%input: tensor<1x3x1024x1024xf16, {order = #NCHW}>) -> tensor<1x1024x3x1024xf16, {order = #NHWC}> {
    %memPermute = VPU.MemPermute(%input) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x3x1024x1024xf16, {order = #NCHW}> -> tensor<1x1024x3x1024xf16, {order = #NHWC}>
    return %memPermute: tensor<1x1024x3x1024xf16, {order = #NHWC}>

    // CHECK: [[MEMPERMUTE_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x1024x3x1024xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK: [[INPUT_BUFFER_CMX:%.+]] = memref.alloc() : memref<1x3x1024x1024xf16, [@CMX_NN, 0]>
    // CHECK: [[INPUT_CMX:%.+]] = VPUIP.Copy inputs([[ARG]] : memref<1x3x1024x1024xf16>) outputs([[INPUT_BUFFER_CMX]] : memref<1x3x1024x1024xf16, [@CMX_NN, 0]>) -> memref<1x3x1024x1024xf16, [@CMX_NN, 0]>

    // CHECK: [[OUTPUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MemPermute inputs([[INPUT_CMX]] as {{[^:]+}}: memref<1x3x1024x1024xf16, [@CMX_NN, 0]>) outputs([[MEMPERMUTE_BUFFER_CMX]] as {{[^:]+}}: memref<1x1024x3x1024xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x1024x3x1024xf16, #NHWC, [@CMX_NN, 0]>{
    // CHECK:   VPUIP.SW.Kernel.run
    // CHECK-SAME{LITERAL}: {attrs = [[0, 1, 2, 3]]}
    // CHECK:   ({{[^:]+}}, {{[^:]+}}) : memref<1x3x1024x1024xf16, [@CMX_NN, 0]>, memref<1x1024x3x1024xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK: }

    // CHECK: [[OUTPUT_BUFFER:%.+]] = memref.alloc() : memref<1x1024x3x1024xf16, #NHWC>
    // CHECK: [[OUTPUT_DDR:%.+]] = VPUIP.Copy inputs([[OUTPUT]] : memref<1x1024x3x1024xf16, #NHWC, [@CMX_NN, 0]>) outputs([[OUTPUT_BUFFER]] : memref<1x1024x3x1024xf16, #NHWC>) -> memref<1x1024x3x1024xf16, #NHWC>
    // CHECK: return [[OUTPUT_DDR]] : memref<1x1024x3x1024xf16, #NHWC>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK:  module @VPU.SW {
// CHECK-NEXT:    func.func private @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func @MemPermuteSWLayerTooLargeForCMXButDMAConvertible
// CHECK-SAME:      ([[ARG:%.+]]: memref<1x3x1024x1024xf16>)
func.func @MemPermuteSWLayerTooLargeForCMXButDMAConvertible(%arg0: tensor<1x3x1024x1024xf16>) -> tensor<1x3x1024x1024xf16, {order = #NHWC}> {
    %output = VPU.MemPermute(%arg0) {mem_perm = #NHWC, dst_order = #NHWC} : tensor<1x3x1024x1024xf16> -> tensor<1x3x1024x1024xf16, {order = #NHWC}>
    return %output: tensor<1x3x1024x1024xf16, {order = #NHWC}>

    // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1x3x1024x1024xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK: [[ALLOC0:%.+]] = memref.alloc() : memref<1x3x1024x1024xf16, [@CMX_NN, 0]>
    // CHECK: [[COPY0:%.+]] = VPUIP.Copy inputs([[ARG]] : memref<1x3x1024x1024xf16>) outputs([[ALLOC0]] : memref<1x3x1024x1024xf16, [@CMX_NN, 0]>) -> memref<1x3x1024x1024xf16, [@CMX_NN, 0]>

    // CHECK: [[RES:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MemPermute inputs([[COPY0]] as {{[^:]+}}: memref<1x3x1024x1024xf16, [@CMX_NN, 0]>) outputs([[ALLOC]] as {{[^:]+}}: memref<1x3x1024x1024xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x3x1024x1024xf16, #NHWC, [@CMX_NN, 0]>{
    // CHECK:   VPUIP.SW.Kernel.run
    // CHECK-SAME{LITERAL}: {attrs = [[2, 0, 1, 3]]}
    // CHECK: ({{[^:]+}}, {{[^:]+}}) : memref<1x3x1024x1024xf16, [@CMX_NN, 0]>, memref<1x3x1024x1024xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK: }

    // CHECK: [[ALLOC1:%.+]] = memref.alloc() : memref<1x3x1024x1024xf16, #NHWC>
    // CHECK: [[COPY1:%.+]] = VPUIP.Copy inputs([[RES]] : memref<1x3x1024x1024xf16, #NHWC, [@CMX_NN, 0]>) outputs([[ALLOC1]] : memref<1x3x1024x1024xf16, #NHWC>) -> memref<1x3x1024x1024xf16, #NHWC>
    // CHECK: return [[COPY1]] : memref<1x3x1024x1024xf16, #NHWC>
}

// -----

// CHECK:  module @VPU.SW {
// CHECK-NEXT:    func.func private @builtin_ROIAlign(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, i64, i64, i64, f64, i64, i64) attributes {VPU.kernel_code = "roi_align.cpp", VPU.kernel_entry = "roi_align", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func @ROIAlignSWLayer
// CHECK-SAME:      ([[ARG:%.+]]: memref<2x22x20x20xf16>)
func.func @ROIAlignSWLayer(%input0: tensor<2x22x20x20xf16>) -> tensor<2x22x8x8xf16> {
    %cst = const.Declare tensor<2x4xf16> = dense<[[0.000000e+00, 0.000000e+00, 0.000000e+00, 3.500000e+00], [0.000000e+00, 3.781250e+00, 0.000000e+00, 3.906250e+00]]> : tensor<2x4xf16>
    %cst_0 = const.Declare tensor<2xsi32> = dense<[0, 1]> : tensor<2xsi32>
    %output = VPU.ROIAlign(%input0, %cst, %cst_0) {alignedMode = #IE.roi_align_aligned_method<ASYMMETRIC>, pooled_h = 8 : i64, pooled_w = 8 : i64, poolingMode = #IE.roi_align_method<AVG>, sampling_ratio = 2 : i64, spatial_scale = 3.125000e-02 : f64} : tensor<2x22x20x20xf16>, tensor<2x4xf16>, tensor<2xsi32> -> tensor<2x22x8x8xf16>
    return %output : tensor<2x22x8x8xf16>

    // CHECK:  VPUIP.SW.Kernel
    // CHECK-SAME:  {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_ROIAlign
    // CHECK-SAME:  inputs({{[^:]+}} as {{[^:]+}}: memref<2x22x20x20xf16, [@CMX_NN, 0]>, {{[^:]+}} as {{[^:]+}}: memref<2x4xf16, [@CMX_NN, 0]>, {{[^:]+}} as {{[^:]+}}: memref<2xsi32, [@CMX_NN, 0]>)
    // CHECK-SAME:  outputs({{[^:]+}} as {{[^:]+}}: memref<2x22x8x8xf16, [@CMX_NN, 0]>) on tile 0
    // CHECK-SAME:  -> memref<2x22x8x8xf16, [@CMX_NN, 0]>{

    // CHECK:  VPUIP.SW.Kernel.run
    // CHECK-SAME{LITERAL}:  {attrs = [8, 8, 2, 3.125000e-02, 0, 0]}
    // CHECK:  ({{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<2x22x20x20xf16, [@CMX_NN, 0]>, memref<2x4xf16, [@CMX_NN, 0]>, memref<2xsi32, [@CMX_NN, 0]>, memref<2x22x8x8xf16, [@CMX_NN, 0]>
    // CHECK:  }
}

// -----
// CHECK-LABEL:  func.func @SpaceToBatchSWLayer
// CHECK-SAME:      ([[ARG:%.+]]: memref<2x8x8x3x3xf16>)
func.func @SpaceToBatchSWLayer(%input0: tensor<2x8x8x3x3xf16>) -> tensor<48x2x2x3x3xf16> {
    %output = VPU.SpaceToBatch(%input0) {block_shape_value = [1, 6, 4, 1, 1], pads_begin_value = [0, 1, 0, 0, 0], pads_end_value = [0, 3, 0, 0, 0]} : tensor<2x8x8x3x3xf16> -> tensor<48x2x2x3x3xf16>
    return %output : tensor<48x2x2x3x3xf16>

    // CHECK:  VPUIP.SW.Kernel
    // CHECK-SAME:  {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SpaceToBatch
    // CHECK-SAME:  inputs({{[^:]+}} as {{[^:]+}}: memref<2x8x8x3x3xf16, [@CMX_NN, 0]>)
    // CHECK-SAME:  outputs({{[^:]+}} as {{[^:]+}}: memref<48x2x2x3x3xf16, [@CMX_NN, 0]>) on tile 0
    // CHECK-SAME:  -> memref<48x2x2x3x3xf16, [@CMX_NN, 0]>{

    // CHECK:  VPUIP.SW.Kernel.run
    // CHECK-SAME{LITERAL}:  {attrs = [[1, 6, 4, 1, 1], [0, 1, 0, 0, 0], [0, 3, 0, 0, 0]]}
    // CHECK:  ({{[^:]+}}, {{[^:]+}}) : memref<2x8x8x3x3xf16, [@CMX_NN, 0]>, memref<48x2x2x3x3xf16, [@CMX_NN, 0]>
    // CHECK:  }
}

// -----

// CHECK-LABEL: func.func @GroupNormalization
// CHECK-SAME:      ([[ARG0:%.+]]: memref<1x4x16x16xf16>, [[ARG1:%.+]]: memref<4xf16>, [[ARG2:%.+]]: memref<4xf16>)
func.func @GroupNormalization(%arg0: tensor<1x4x16x16xf16>, %arg1: tensor<4xf16>, %arg2: tensor<4xf16>) -> tensor<1x4x16x16xf16> {
    %0 = VPU.GroupNormalization(%arg0, %arg1, %arg2) {epsilon = 9.9999997473787516E-5 : f32, num_groups = 2 : i32} : tensor<1x4x16x16xf16>, tensor<4xf16>, tensor<4xf16> -> tensor<1x4x16x16xf16>
    return %0 : tensor<1x4x16x16xf16>

    // CHECK:  VPUIP.SW.Kernel
    // CHECK-SAME:  {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_GroupNormalization
    // CHECK-SAME:  inputs({{[^:]+}} as {{[^:]+}}: memref<1x4x16x16xf16, [@CMX_NN, 0]>, {{[^:]+}} as {{[^:]+}}: memref<4xf16, [@CMX_NN, 0]>, {{[^:]+}} as {{[^:]+}}: memref<4xf16, [@CMX_NN, 0]>)
    // CHECK-SAME:  outputs({{[^:]+}} as {{[^:]+}}: memref<1x4x16x16xf16, [@CMX_NN, 0]>) on tile 0
    // CHECK-SAME:  -> memref<1x4x16x16xf16, [@CMX_NN, 0]>{

    // CHECK:       VPUIP.SW.Kernel.run
    // CHECK-SAME: {attrs = [9.9999997473787516E-5, 2]}
    // CHECK:  ({{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<1x4x16x16xf16, [@CMX_NN, 0]>, memref<4xf16, [@CMX_NN, 0]>, memref<4xf16, [@CMX_NN, 0]>, memref<1x4x16x16xf16, [@CMX_NN, 0]>
    // CHECK:  }
}

// -----
// CHECK-LABEL:  func.func @AdaptiveMaxPoolSWLayer
// CHECK-SAME:      ([[ARG:%.+]]: memref<2x3x7xf16>)
func.func @AdaptiveMaxPoolSWLayer(%arg0: tensor<2x3x7xf16>) -> (tensor<2x3x1xf16>, tensor<2x3x1xsi32>) {
    %cst = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi32>
    %output, %output_index = VPU.AdaptiveMaxPool(%arg0, %cst) {index_element_type = si32} : tensor<2x3x7xf16>, tensor<1xsi32> -> tensor<2x3x1xf16>, tensor<2x3x1xsi32>
    return %output, %output_index : tensor<2x3x1xf16>, tensor<2x3x1xsi32>

    // CHECK:  VPUIP.SW.Kernel
    // CHECK-SAME:  {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_AdaptiveMaxPool
    // CHECK-SAME:  inputs({{[^:]+}} as {{[^:]+}}: memref<2x3x7xf16, [@CMX_NN, 0]>, {{[^:]+}} as {{[^:]+}}: memref<1xsi32, [@CMX_NN, 0]>)
    // CHECK-SAME:  outputs({{[^:]+}} as {{[^:]+}}: memref<2x3x1xf16, [@CMX_NN, 0]>, {{[^:]+}} as {{[^:]+}}: memref<2x3x1xsi32, [@CMX_NN, 0]>) on tile 0
    // CHECK-SAME:  -> (memref<2x3x1xf16, [@CMX_NN, 0]>, memref<2x3x1xsi32, [@CMX_NN, 0]>){

    // CHECK:  VPUIP.SW.Kernel.run({{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<2x3x7xf16, [@CMX_NN, 0]>, memref<1xsi32, [@CMX_NN, 0]>, memref<2x3x1xf16, [@CMX_NN, 0]>, memref<2x3x1xsi32, [@CMX_NN, 0]>
    // CHECK:  }
}


// -----
// CHECK-LABEL:  func.func @BucketizeSWLayer
// CHECK-SAME:      ([[ARG0:%.+]]: memref<1x20x20xf16>, [[ARG1:%.+]]: memref<100xf16>)
func.func @BucketizeSWLayer(%input0: tensor<1x20x20xf16>, %input1: tensor<100xf16>) -> tensor<1x20x20xsi32> {
    %output = VPU.Bucketize(%input0, %input1) {output_type = si32, with_right_bound} : tensor<1x20x20xf16>, tensor<100xf16> -> tensor<1x20x20xsi32>
    return %output : tensor<1x20x20xsi32>

    // CHECK:  VPUIP.SW.Kernel
    // CHECK-SAME:  {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Bucketize
    // CHECK-SAME:  inputs({{[^:]+}} as {{[^:]+}}: memref<1x20x20xf16, [@CMX_NN, 0]>, {{[^:]+}} as {{[^:]+}}: memref<100xf16, [@CMX_NN, 0]>)
    // CHECK-SAME:  outputs({{[^:]+}} as {{[^:]+}}: memref<1x20x20xsi32, [@CMX_NN, 0]>) on tile 0
    // CHECK-SAME:  -> memref<1x20x20xsi32, [@CMX_NN, 0]>{

    // CHECK:       VPUIP.SW.Kernel.run
    // CHECK-SAME{LITERAL}: {attrs = [1]}
    // CHECK:  ({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<1x20x20xf16, [@CMX_NN, 0]>, memref<100xf16, [@CMX_NN, 0]>, memref<1x20x20xsi32, [@CMX_NN, 0]>
    // CHECK:  }
}

// -----
// CHECK-LABEL: func.func @MaxPool8SWLayer
// CHECK-SAME:  ([[ARG0:%.+]]: memref<1x3x30x30xf16>) -> (memref<1x3x13x26xf16>, memref<1x3x13x26xsi32>)
func.func @MaxPool8SWLayer(%arg0: tensor<1x3x30x30xf16>) -> (tensor<1x3x13x26xf16>, tensor<1x3x13x26xsi32>) {
    %output, %output_index = VPU.MaxPool8(%arg0) {axis = 0 : i64, dilations = [2, 2], index_element_type = si32, kernel_size = [3, 5], pads_begin = [0, 2], pads_end = [0, 2], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 1]} : tensor<1x3x30x30xf16> -> tensor<1x3x13x26xf16>, tensor<1x3x13x26xsi32>
    return %output, %output_index : tensor<1x3x13x26xf16>, tensor<1x3x13x26xsi32>


    // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1x3x30x30xf16, [@CMX_NN, 0]>
    // CHECK: [[COPY0:%.+]] = VPUIP.Copy inputs([[ARG0]] : memref<1x3x30x30xf16>) outputs([[ALLOC]] : memref<1x3x30x30xf16, [@CMX_NN, 0]>) -> memref<1x3x30x30xf16, [@CMX_NN, 0]>
    // CHECK: [[ALLOC0:%.+]] = memref.alloc() : memref<1x3x13x26xf16, [@CMX_NN, 0]>
    // CHECK: [[ALLOC1:%.+]] = memref.alloc() : memref<1x3x13x26xsi32, [@CMX_NN, 0]>

    // CHECK: VPUIP.SW.Kernel
    // CHECK-SAME: {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MaxPool8
    // CHECK-SAME: inputs([[COPY0:%.+]] as {{[^:]+}}: memref<1x3x30x30xf16, [@CMX_NN, 0]>)
    // CHECK-SAME: outputs([[ALLOC0]] as {{[^:]+}}: memref<1x3x13x26xf16, [@CMX_NN, 0]>, [[ALLOC1]] as {{[^:]+}}: memref<1x3x13x26xsi32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x3x13x26xf16, [@CMX_NN, 0]>, memref<1x3x13x26xsi32, [@CMX_NN, 0]>){

    // CHECK: VPUIP.SW.Kernel.run
    // CHECK-SAME{LITERAL}: {attrs = [[1, 3, 5], [1, 2, 1], [1, 2, 2], [0, 0, 2], [0, 0, 2], 3]}
    // CHECK-SAME: memref<1x3x30x30xf16, [@CMX_NN, 0]>, memref<1x3x13x26xf16, [@CMX_NN, 0]>, memref<1x3x13x26xsi32, [@CMX_NN, 0]>
    // CHECK: }
}

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TensorsWithBounds
// CHECK-SAME:          ([[ARG:%.+]]: !VPUIP.BoundedBuffer<data=memref<1x18x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>) ->
// CHECK-SAME:          !VPUIP.BoundedBuffer<data=memref<1x18x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>
func.func @TensorsWithBounds(%arg0: tensor<1x18x3x3xf32, {bounds = [1, 18, 3, 3], order = #NHWC}>) -> tensor<1x18x3x3xf32, {bounds = [1, 18, 3, 3], order = #NHWC}> {
    %0 = VPU.ReLU(%arg0) : tensor<1x18x3x3xf32, {bounds = [1, 18, 3, 3], order = #NHWC}> -> tensor<1x18x3x3xf32, {bounds = [1, 18, 3, 3], order = #NHWC}>

// CHECK:       [[INPUT_DATA:%.+]] = memref.alloc() : memref<1x18x3x3xf32, #NHWC, [@CMX_NN, 0]>
// CHECK:       [[INPUT_SHAPE:%.+]] = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>
// CHECK:       [[INPUT_BOUNDED_BUFFER:%.+]] = VPUIP.GroupBoundedBuffer([[INPUT_DATA]], [[INPUT_SHAPE]]) : memref<1x18x3x3xf32, #NHWC, [@CMX_NN, 0]>, memref<4xsi32, [@CMX_NN, 0]>
// CHECK-SAME:    -> !VPUIP.BoundedBuffer<data=memref<1x18x3x3xf32, #NHWC, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>
// CHECK:       [[COPY_INPUT_BOUNDED_BUFFER:%.+]] = VPUIP.Copy
// CHECK-SAME:    inputs([[ARG]] : !VPUIP.BoundedBuffer<data=memref<1x18x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>)
// CHECK-SAME:    outputs([[INPUT_BOUNDED_BUFFER]] : !VPUIP.BoundedBuffer<data=memref<1x18x3x3xf32, #NHWC, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>)
// CHECK-SAME:    -> !VPUIP.BoundedBuffer<data=memref<1x18x3x3xf32, #NHWC, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>
// CHECK:       [[SW_OP_RESULT_DATA:%.+]] = memref.alloc() : memref<1x18x3x3xf32, #NHWC, [@CMX_NN, 0]>
// CHECK:       [[SW_OP_RESULT_SHAPE:%.+]] = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>
// CHECK:       [[SW_OP_RESULT_BOUNDED_BUFFER:%.+]] = VPUIP.GroupBoundedBuffer([[SW_OP_RESULT_DATA]], [[SW_OP_RESULT_SHAPE]]) : memref<1x18x3x3xf32, #NHWC, [@CMX_NN, 0]>, memref<4xsi32, [@CMX_NN, 0]>
// CHECK-SAME:    -> !VPUIP.BoundedBuffer<data=memref<1x18x3x3xf32, #NHWC, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>
// CHECK:       [[SW_OP_RESULT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_ReLU
// CHECK-SAME:    inputs([[COPY_INPUT_BOUNDED_BUFFER]] as [[ARG_0:%.+]]: !VPUIP.BoundedBuffer<data=memref<1x18x3x3xf32, #NHWC, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>) outputs([[SW_OP_RESULT_BOUNDED_BUFFER]] as [[ARG_1:%.+]]: !VPUIP.BoundedBuffer<data=memref<1x18x3x3xf32, #NHWC, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>) on tile 0
// CHECK-SAME:    -> !VPUIP.BoundedBuffer<data=memref<1x18x3x3xf32, #NHWC, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>{
// CHECK:         VPUIP.SW.Kernel.run([[ARG_0]], [[ARG_1]]) : !VPUIP.BoundedBuffer<data=memref<1x18x3x3xf32, #NHWC, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>, !VPUIP.BoundedBuffer<data=memref<1x18x3x3xf32, #NHWC, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>
// CHECK:       }
// CHECK:       [[OUTPUT_DATA:%.+]] = memref.alloc() : memref<1x18x3x3xf32, #NHWC>
// CHECK:       [[OUTPUT_SHAPE:%.+]] = memref.alloc() : memref<4xsi32>
// CHECK:       [[OUTPUT_BOUNDED_BUFFER:%.+]] = VPUIP.GroupBoundedBuffer([[OUTPUT_DATA]], [[OUTPUT_SHAPE]]) : memref<1x18x3x3xf32, #NHWC>, memref<4xsi32>
// CHECK-SAME:    -> !VPUIP.BoundedBuffer<data=memref<1x18x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>
// CHECK:       [[COPY_OUTPUT_BOUNDED_BUFFER:%.+]] = VPUIP.Copy
// CHECK-SAME:    inputs([[SW_OP_RESULT]] : !VPUIP.BoundedBuffer<data=memref<1x18x3x3xf32, #NHWC, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>)
// CHECK-SAME:    outputs([[OUTPUT_BOUNDED_BUFFER]] : !VPUIP.BoundedBuffer<data=memref<1x18x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>)
// CHECK-SAME:    -> !VPUIP.BoundedBuffer<data=memref<1x18x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>

    return %0 : tensor<1x18x3x3xf32, {bounds = [1, 18, 3, 3], order = #NHWC}>
    // CHECK: return
    // CHECK-SAME: !VPUIP.BoundedBuffer<
    // CHECK-SAME:  data=memref<1x18x3x3xf32, #NHWC>,
    // CHECK-SAME:  dynamic_shape=memref<4xsi32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ShapeOf
// CHECK-SAME:          ([[ARG:%.+]]: !VPUIP.BoundedBuffer<data=memref<1x8x48x48xf16>, dynamic_shape=memref<4xsi32>>)
func.func @ShapeOf(%DATA: tensor<1x8x?x?xf16, {bounds = [1, 8, 48, 48], order = #NCHW}>) -> tensor<4xsi32> {

    %SHAPE_OF = VPU.ShapeOf(%DATA) :
        tensor<1x8x?x?xf16, {bounds = [1, 8, 48, 48], order = #NCHW}> -> tensor<4xsi32>

    // CHECK: [[DATA:%.+]]  = memref.alloc() : memref<1x8x48x48xf16, [@CMX_NN, 0]>
    // CHECK: [[SHAPE:%.+]] = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>
    // CHECK: [[BOUNDED_BUFFER:%.*]] = VPUIP.GroupBoundedBuffer([[DATA]], [[SHAPE]]) : memref<1x8x48x48xf16, [@CMX_NN, 0]>, memref<4xsi32, [@CMX_NN, 0]>
    // CHECK-SAME: -> !VPUIP.BoundedBuffer<data=memref<1x8x48x48xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>

    // CHECK: [[COPY:%.*]] = VPUIP.Copy
    // CHECK-SAME: inputs([[ARG]]
    // CHECK-SAME: outputs([[BOUNDED_BUFFER]]
    // CHECK-SAME: -> !VPUIP.BoundedBuffer<data=memref<1x8x48x48xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>

    // CHECK: [[OUT_SHAPE:%.*]] = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>
    // CHECK: [[OUT:%.*]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_ShapeOf
    // CHECK-SAME: inputs([[COPY]]
    // CHECK-SAME: outputs([[OUT_SHAPE]]

    // CHECK: [[RES_SHAPE:%.*]] = memref.alloc() : memref<4xsi32>

    // CHECK: [[COPY_OUT:%.*]] = VPUIP.Copy
    // CHECK-SAME: inputs([[OUT]]
    // CHECK-SAME: outputs([[RES_SHAPE]]
    // CHECK-SAME: -> memref<4xsi32>

    return %SHAPE_OF: tensor<4xsi32>
    // CHECK:   return [[COPY_OUT]] : memref<4xsi32>
}

// -----
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SkipStaticPermuteCast
// CHECK-SAME:         ([[ARG0:%.+]]: memref<1x32x32x16xf16>)
func.func @SkipStaticPermuteCast(%arg0: tensor<1x32x32x16xf16, {order = #NCHW}>)
    -> tensor<1x16x32x32xf16, {order = #NHWC}> {

    %PERMUTE_CAST = VPU.PermuteCast(%arg0) {
        dst_order = #NHWC,
        mem_perm = #NCHW
    } : tensor<1x32x32x16xf16, {order = #NCHW}> -> tensor<1x16x32x32xf16, {order = #NHWC}>

    return %PERMUTE_CAST : tensor<1x16x32x32xf16, {order = #NHWC}>

    // CHECK-NOT:   VPUIP.SW.Kernel
    // CHECK:   [[PERMUTE_CAST:%.*]] = VPUIP.PermuteCast {
    // CHECK-SAME:      dst_order = #NHWC,
    // CHECK-SAME:      mem_perm = #NCHW
    // CHECK-SAME: }
    // CHECK-SAME: inputs([[ARG0]]
    // CHECK-SAME:      -> memref<1x16x32x32xf16, #NHWC>

    // CHECK:       return [[PERMUTE_CAST]] : memref<1x16x32x32xf16, #NHWC>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @DynamicPermuteCast
// CHECK-SAME:         ([[ARG:%.+]]: !VPUIP.BoundedBuffer<data=memref<1x32x64x16xf16>, dynamic_shape=memref<4xsi32>>)
func.func @DynamicPermuteCast(%arg: tensor<1x?x?x16xf16, {bounds = [1, 32, 64, 16], order = #NCHW}>)
   -> (tensor<1x16x?x?xf16, {bounds = [1, 16, 32, 64], order = #NHWC}>) {

    %permute_cast = VPU.PermuteCast(%arg) {
        dst_order = #NHWC,
        mem_perm = #NCHW
    } : tensor<1x?x?x16xf16, {bounds = [1, 32, 64, 16], order = #NCHW}>
        -> tensor<1x16x?x?xf16, {bounds = [1, 16, 32, 64], order = #NHWC}>
    // CHECK: [[IN_DATA:%.+]] = memref.alloc() : memref<1x32x64x16xf16, [@CMX_NN, 0]>
    // CHECK: [[IN_SHAPE:%.+]] = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>
    // CHECK: [[IN_BOUNDED_BUFFER:%.+]] = VPUIP.GroupBoundedBuffer([[IN_DATA]], [[IN_SHAPE]])

    // CHECK: [[COPY_IN:%.+]] = VPUIP.Copy
    // CHECK-SAME: inputs([[ARG]]
    // CHECK-SAME: outputs([[IN_BOUNDED_BUFFER]]

    // CHECK: [[OUT_DATA:%.+]] = memref.alloc() : memref<1x16x32x64xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK: [[OUT_SHAPE:%.+]] = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>
    // CHECK: [[OUT_BOUNDED_BUFFER:%.+]] = VPUIP.GroupBoundedBuffer([[OUT_DATA]], [[OUT_SHAPE]])

    // CHECK: [[SW_OP_RESULT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_PermuteCast
    // CHECK-SAME: inputs([[COPY_IN]]
    // CHECK-SAME: outputs([[OUT_BOUNDED_BUFFER]]

    // CHECK: [[RES_DATA:%.+]] = memref.alloc() : memref<1x16x32x64xf16, #NHWC>
    // CHECK: [[RES_SHAPE:%.+]] = memref.alloc() : memref<4xsi32>
    // CHECK: [[RES_BOUNDED_BUFFER:%.+]] = VPUIP.GroupBoundedBuffer([[RES_DATA]], [[RES_SHAPE]])

    // CHECK: [[COPY_OUT:%.+]] = VPUIP.Copy
    // CHECK-SAME: inputs([[SW_OP_RESULT]]
    // CHECK-SAME: outputs([[RES_BOUNDED_BUFFER]]

    return %permute_cast : tensor<1x16x?x?xf16, {bounds = [1, 16, 32, 64], order = #NHWC}>
    // CHECK: return [[COPY_OUT]] : !VPUIP.BoundedBuffer<data=memref<1x16x32x64xf16, #NHWC>, dynamic_shape=memref<4xsi32>
}

// -----

// CHECK:  module @VPU.SW {
// CHECK-NEXT:    func.func private @builtin_Gather(memref<*xf16>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, i64, i64, i64) attributes {VPU.kernel_code = "gather.cpp", VPU.kernel_entry = "gather", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func @GatherWithDDRAccessOutputAtCMX
// CHECK-SAME:      ([[INPUT:%.+]]: memref<51865x512xf16>)
func.func @GatherWithDDRAccessOutputAtCMX(%arg0: tensor<51865x512xf16>) -> tensor<1x16x512xf16> {
    %cst = const.Declare tensor<1x16xsi32> = dense<1> : tensor<1x16xsi64>, [#const.CastElemType<si32>]
    %output = VPU.Gather(%arg0, %cst) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<51865x512xf16>, tensor<1x16xsi32> -> tensor<1x16x512xf16>
    return %output: tensor<1x16x512xf16>

    // CHECK-DAG: [[INDICES:%.+]] = const.Declare memref<1x16xsi32> = dense<1> : tensor<1x16xsi64>, [#const.CastElemType<si32>]
    // CHECK: [[INDICES_ALLOC:%.+]] = memref.alloc() : memref<1x16xsi32, [@CMX_NN, 0]>
    // CHECK: [[INDICES_COPY:%.+]] = VPUIP.Copy inputs([[INDICES]] : memref<1x16xsi32>) outputs([[INDICES_ALLOC]] : memref<1x16xsi32, [@CMX_NN, 0]>) -> memref<1x16xsi32, [@CMX_NN, 0]>

    // CHECK: [[OUTPUT_CMX:%.+]] = memref.alloc() : memref<1x16x512xf16, [@CMX_NN, 0]>
    // CHECK: [[GATHER:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Gather inputs([[INPUT]] as {{[^:]+}}: memref<51865x512xf16>, [[INDICES_COPY]] as {{[^:]+}}: memref<1x16xsi32, [@CMX_NN, 0]>) outputs([[OUTPUT_CMX]] as {{[^:]+}}: memref<1x16x512xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x512xf16, [@CMX_NN, 0]>{
    // CHECK:   VPUIP.SW.Kernel.run
    // CHECK-SAME{LITERAL}: {attrs = [1, 0, 2]}
    // CHECK: ({{[^:]+}}, {{[^:]+}}) : memref<51865x512xf16>, memref<1x16xsi32, [@CMX_NN, 0]>, memref<1x16x512xf16, [@CMX_NN, 0]>
    // CHECK: }

    // CHECK: [[OUTPUT_DDR:%.+]] = memref.alloc() : memref<1x16x512xf16>
    // CHECK: [[OUTPUT_COPY:%.+]] = VPUIP.Copy inputs([[GATHER]] : memref<1x16x512xf16, [@CMX_NN, 0]>) outputs([[OUTPUT_DDR]] : memref<1x16x512xf16>) -> memref<1x16x512xf16>
    // CHECK: return [[OUTPUT_COPY]] : memref<1x16x512xf16>
}

// -----
// Using DDR Access for GatherOp with the output buffer in DDR leads to suboptimal performance and should be avoided

// CHECK:  module @VPU.SW {
// CHECK-NEXT:    func.func private @builtin_Gather(memref<*xf16>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xf16>, i64, i64, i64) attributes {VPU.kernel_code = "gather.cpp", VPU.kernel_entry = "gather", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func @GatherWithDDRAccessOutputAtDDR
// CHECK-SAME:      ([[INPUT:%.+]]: memref<51865x512xf16>)
func.func @GatherWithDDRAccessOutputAtDDR(%arg0: tensor<51865x512xf16>) -> tensor<1x2000x512xf16> {
    %cst = const.Declare tensor<1x2000xsi32> = dense<1> : tensor<1x2000xsi64>, [#const.CastElemType<si32>]
    %output = VPU.Gather(%arg0, %cst) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<51865x512xf16>, tensor<1x2000xsi32> -> tensor<1x2000x512xf16>
    return %output: tensor<1x2000x512xf16>

    // CHECK-DAG: [[INDICES:%.+]] = const.Declare memref<1x2000xsi32> = dense<1> : tensor<1x2000xsi64>, [#const.CastElemType<si32>]
    // CHECK: [[INDICES_ALLOC:%.+]] = memref.alloc() : memref<1x2000xsi32, [@CMX_NN, 0]>
    // CHECK: [[INDICES_COPY:%.+]] = VPUIP.Copy inputs([[INDICES]] : memref<1x2000xsi32>) outputs([[INDICES_ALLOC]] : memref<1x2000xsi32, [@CMX_NN, 0]>) -> memref<1x2000xsi32, [@CMX_NN, 0]>

    // CHECK: [[OUTPUT_DDR:%.+]] = memref.alloc() : memref<1x2000x512xf16>
    // CHECK: [[GATHER:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Gather inputs([[INPUT]] as {{[^:]+}}: memref<51865x512xf16>, [[INDICES_COPY]] as {{[^:]+}}: memref<1x2000xsi32, [@CMX_NN, 0]>) outputs([[OUTPUT_DDR]] as {{[^:]+}}: memref<1x2000x512xf16>) on tile 0 -> memref<1x2000x512xf16>{
    // CHECK:   VPUIP.SW.Kernel.run
    // CHECK-SAME{LITERAL}: {attrs = [1, 0, 2]}
    // CHECK: ({{[^:]+}}, {{[^:]+}}) : memref<51865x512xf16>, memref<1x2000xsi32, [@CMX_NN, 0]>, memref<1x2000x512xf16>
    // CHECK: }

    // CHECK: return [[GATHER]] : memref<1x2000x512xf16>
}

// -----

// CHECK:  module @VPU.SW {
// CHECK-NEXT:    func.func private @builtin_GRUSequence(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, i64, i64, i64, i64, f64) attributes {VPU.kernel_code = "gru_sequence.cpp", VPU.kernel_entry = "gru_sequence", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func @GRUSequenceWithDDRAccess
// CHECK-SAME:      [[INPUT0:%.+]]: memref<1x1x200xf16>
// CHECK-SAME:      [[INPUT1:%.+]]: memref<1x1x1024xf16>
func.func @GRUSequenceWithDDRAccess(%arg0: tensor<1x1x200xf16>, %arg1: tensor<1x1x1024xf16>) -> (tensor<1x1x1x1024xf16>, tensor<1x1x1024xf16>) {
    %cst = const.Declare tensor<1x3072x200xf16> = dense<1.000000e+00> : tensor<1x3072x200xf16>
    %cst_0 = const.Declare tensor<1x3072x1024xf16> = dense<1.000000e+00> : tensor<1x3072x1024xf16>
    %cst_1 = const.Declare tensor<1x4096xf16> = dense<1.000000e+00> : tensor<1x4096xf16>
    %middle_hidden_state, %output_hidden_state = VPU.GRUSequence(%arg0, %arg1, %cst, %cst_0, %cst_1) {__inplace_operands_attr__ = ["true", "true", "true", "true", "true"], clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<FORWARD>, hidden_size = 1024 : i64, seq_length = 1 : i64, should_linear_before_reset} : tensor<1x1x200xf16>, tensor<1x1x1024xf16>, tensor<1x3072x200xf16>, tensor<1x3072x1024xf16>, tensor<1x4096xf16> -> tensor<1x1x1x1024xf16>, tensor<1x1x1024xf16>
    return {__inplace_operands_attr__ = ["true", "true"]} %middle_hidden_state, %output_hidden_state : tensor<1x1x1x1024xf16>, tensor<1x1x1024xf16>

    // CHECK: [[CST:%.+]] = const.Declare memref<1x3072x200xf16> = dense<1.000000e+00> : tensor<1x3072x200xf16>
    // CHECK: [[CST0:%.+]] = const.Declare memref<1x3072x1024xf16> = dense<1.000000e+00> : tensor<1x3072x1024xf16>
    // CHECK: [[CST1:%.+]] = const.Declare memref<1x4096xf16> = dense<1.000000e+00> : tensor<1x4096xf16>
    // CHECK: [[ALLOC0:%.+]] = memref.alloc() : memref<1x1x200xf16, [@CMX_NN, 0]>
    // CHECK: [[COPY0:%.+]] = VPUIP.Copy inputs([[INPUT0]] : memref<1x1x200xf16>) outputs([[ALLOC0]] : memref<1x1x200xf16, [@CMX_NN, 0]>) -> memref<1x1x200xf16, [@CMX_NN, 0]>
    // CHECK: [[ALLOC1:%.+]] = memref.alloc() : memref<1x1x1024xf16, [@CMX_NN, 0]>
    // CHECK: [[COPY1:%.+]] = VPUIP.Copy inputs([[INPUT1]] : memref<1x1x1024xf16>) outputs([[ALLOC1]] : memref<1x1x1024xf16, [@CMX_NN, 0]>) -> memref<1x1x1024xf16, [@CMX_NN, 0]>
    // CHECK: [[ALLOC2:%.+]] = memref.alloc() : memref<1x3072x200xf16, [@CMX_NN, 0]>
    // CHECK: [[COPY2:%.+]] = VPUIP.Copy inputs([[CST]] : memref<1x3072x200xf16>) outputs([[ALLOC2]] : memref<1x3072x200xf16, [@CMX_NN, 0]>) -> memref<1x3072x200xf16, [@CMX_NN, 0]>
    // CHECK: [[ALLOC3:%.+]] = memref.alloc() : memref<1x4096xf16, [@CMX_NN, 0]>
    // CHECK: [[COPY3:%.+]] = VPUIP.Copy inputs([[CST1]] : memref<1x4096xf16>) outputs([[ALLOC3]] : memref<1x4096xf16, [@CMX_NN, 0]>) -> memref<1x4096xf16, [@CMX_NN, 0]>
    // CHECK: [[ALLOC4:%.+]] = memref.alloc() : memref<1x1x1x1024xf16, [@CMX_NN, 0]>
    // CHECK: [[ALLOC5:%.+]] = memref.alloc() : memref<1x1x1024xf16, [@CMX_NN, 0]>
    // CHECK: [[OUT:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_GRUSequence inputs([[COPY0]] as %arg2: memref<1x1x200xf16, [@CMX_NN, 0]>, [[COPY1]] as %arg3: memref<1x1x1024xf16, [@CMX_NN, 0]>, [[COPY2]] as %arg4: memref<1x3072x200xf16, [@CMX_NN, 0]>, [[CST0]] as %arg5: memref<1x3072x1024xf16>, [[COPY3]] as %arg6: memref<1x4096xf16, [@CMX_NN, 0]>) outputs([[ALLOC4]] as %arg7: memref<1x1x1x1024xf16, [@CMX_NN, 0]>, [[ALLOC5:%.+]] as %arg8: memref<1x1x1024xf16, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x1x1024xf16, [@CMX_NN, 0]>, memref<1x1x1024xf16, [@CMX_NN, 0]>){
    // CHECK:  VPUIP.SW.Kernel.run {attrs = [1024, 0, 1, 1, 0.000000e+00]}(%arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8) : memref<1x1x200xf16, [@CMX_NN, 0]>, memref<1x1x1024xf16, [@CMX_NN, 0]>, memref<1x3072x200xf16, [@CMX_NN, 0]>, memref<1x3072x1024xf16>, memref<1x4096xf16, [@CMX_NN, 0]>, memref<1x1x1x1024xf16, [@CMX_NN, 0]>, memref<1x1x1024xf16, [@CMX_NN, 0]>
    // CHECK:}
    // CHECK: [[ALLOC6:%.+]] = memref.alloc() : memref<1x1x1x1024xf16>
    // CHECK: [[COPY4:%.+]] = VPUIP.Copy inputs([[OUT]]#0 : memref<1x1x1x1024xf16, [@CMX_NN, 0]>) outputs([[ALLOC6]] : memref<1x1x1x1024xf16>) -> memref<1x1x1x1024xf16>
    // CHECK: [[ALLOC7:%.+]] = memref.alloc() : memref<1x1x1024xf16>
    // CHECK: [[COPY5:%.+]] = VPUIP.Copy inputs([[OUT]]#1 : memref<1x1x1024xf16, [@CMX_NN, 0]>) outputs(%alloc_8 : memref<1x1x1024xf16>) -> memref<1x1x1024xf16>

    // CHECK: return [[COPY4]], [[COPY5]] : memref<1x1x1x1024xf16>, memref<1x1x1024xf16>
}

// -----

// CHECK:  module @VPU.SW {
// CHECK-NEXT:    func.func private @builtin_GRUSequenceLastPart(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, i64, i64, i64, i64, f64) attributes {VPU.kernel_code = "gru_sequence_last_part.cpp", VPU.kernel_entry = "gru_sequence_last_part", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func @GRUSequenceLastPartWithDDRAccess
// CHECK-SAME:      [[INPUT0:%.+]]: memref<1x1x1x3072xf16>
// CHECK-SAME:      [[INPUT1:%.+]]: memref<1x1x1024xf16>
func.func @GRUSequenceLastPartWithDDRAccess_(%arg0: tensor<1x1x1x3072xf16>, %arg1: tensor<1x1x1024xf16>) -> (tensor<1x1x1x1024xf16>, tensor<1x1x1024xf16>) {
    %cst = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %cst_0 = const.Declare tensor<1x3072x1024xf16> = dense<1.000000e+00> : tensor<1x3072x1024xf16>
    %cst_1 = const.Declare tensor<1x4096xf16> = dense<1.000000e+00> : tensor<1x4096xf16>
    %middle_hidden_state, %output_hidden_state = VPU.GRUSequenceLastPart(%arg0, %arg1, %cst_0, %cst_1) {__inplace_operands_attr__ = ["true", "true", "true", "true"], clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<FORWARD>, hidden_size = 1024 : i64, seq_length = 1 : i64, should_linear_before_reset} : tensor<1x1x1x3072xf16>, tensor<1x1x1024xf16>, tensor<1x3072x1024xf16>, tensor<1x4096xf16> -> tensor<1x1x1x1024xf16>, tensor<1x1x1024xf16>
    return {__inplace_operands_attr__ = ["true", "true"]} %middle_hidden_state, %output_hidden_state : tensor<1x1x1x1024xf16>, tensor<1x1x1024xf16>

    // CHECK: [[CST:%.+]] = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    // CHECK: [[CST0:%.+]] = const.Declare memref<1x3072x1024xf16> = dense<1.000000e+00> : tensor<1x3072x1024xf16>
    // CHECK: [[CST1:%.+]] = const.Declare memref<1x4096xf16> = dense<1.000000e+00> : tensor<1x4096xf16>
    // CHECK: [[ALLOC0:%.+]] = memref.alloc() : memref<1x1x1x3072xf16, [@CMX_NN, 0]>
    // CHECK: [[COPY0:%.+]] = VPUIP.Copy inputs([[INPUT0]] : memref<1x1x1x3072xf16>) outputs([[ALLOC0]] : memref<1x1x1x3072xf16, [@CMX_NN, 0]>) -> memref<1x1x1x3072xf16, [@CMX_NN, 0]>
    // CHECK: [[ALLOC1:%.+]] = memref.alloc() : memref<1x1x1024xf16, [@CMX_NN, 0]>
    // CHECK: [[COPY1:%.+]] = VPUIP.Copy inputs([[INPUT1]] : memref<1x1x1024xf16>) outputs([[ALLOC1]] : memref<1x1x1024xf16, [@CMX_NN, 0]>) -> memref<1x1x1024xf16, [@CMX_NN, 0]>
    // CHECK: [[ALLOC2:%.+]] = memref.alloc() : memref<1x4096xf16, [@CMX_NN, 0]>
    // CHECK: [[COPY2:%.+]] = VPUIP.Copy inputs([[CST1]] : memref<1x4096xf16>) outputs([[ALLOC2]] : memref<1x4096xf16, [@CMX_NN, 0]>) -> memref<1x4096xf16, [@CMX_NN, 0]>
    // CHECK: [[ALLOC3:%.+]] = memref.alloc() : memref<1x1x1x1024xf16, [@CMX_NN, 0]>
    // CHECK: [[ALLOC4:%.+]] = memref.alloc() : memref<1x1x1024xf16, [@CMX_NN, 0]>
    // CHECK: [[RESULT:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_GRUSequenceLastPart inputs([[COPY0]] as %arg2: memref<1x1x1x3072xf16, [@CMX_NN, 0]>, [[COPY1]] as %arg3: memref<1x1x1024xf16, [@CMX_NN, 0]>, [[CST0]] as %arg4: memref<1x3072x1024xf16>, [[COPY2]] as %arg5: memref<1x4096xf16, [@CMX_NN, 0]>) outputs([[ALLOC3]] as %arg6: memref<1x1x1x1024xf16, [@CMX_NN, 0]>, [[ALLOC4]] as %arg7: memref<1x1x1024xf16, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x1x1024xf16, [@CMX_NN, 0]>, memref<1x1x1024xf16, [@CMX_NN, 0]>){
    // CHECK:  VPUIP.SW.Kernel.run {attrs = [1024, 0, 1, 1, 0.000000e+00]}(%arg2, %arg3, %arg4, %arg5, %arg6, %arg7) : memref<1x1x1x3072xf16, [@CMX_NN, 0]>, memref<1x1x1024xf16, [@CMX_NN, 0]>, memref<1x3072x1024xf16>, memref<1x4096xf16, [@CMX_NN, 0]>, memref<1x1x1x1024xf16, [@CMX_NN, 0]>, memref<1x1x1024xf16, [@CMX_NN, 0]>
    // CHECK: }
    // CHECK: [[ALLOC5:%.+]] = memref.alloc() : memref<1x1x1x1024xf16>
    // CHECK: [[COPY3:%.+]] = VPUIP.Copy inputs([[RESULT]]#0 : memref<1x1x1x1024xf16, [@CMX_NN, 0]>) outputs([[ALLOC5]] : memref<1x1x1x1024xf16>) -> memref<1x1x1x1024xf16>
    // CHECK: [[ALLOC6:%.+]] = memref.alloc() : memref<1x1x1024xf16>
    // CHECK: [[COPY4:%.+]] = VPUIP.Copy inputs([[RESULT]]#1 : memref<1x1x1024xf16, [@CMX_NN, 0]>) outputs([[ALLOC6]] : memref<1x1x1024xf16>) -> memref<1x1x1024xf16>
    // CHECK: return [[COPY3]], [[COPY4]] : memref<1x1x1x1024xf16>, memref<1x1x1024xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK:  module @VPU.SW {
// CHECK-NEXT:    func.func private @builtin_Concat(memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, none, none) attributes {VPU.kernel_code = "concat.cpp", VPU.kernel_entry = "concat", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:  }

// CHECK-LABEL:   func.func @ConcatSWLayer

// CHECK-SAME:         ([[ARG0:%.+]]: !VPUIP.BoundedBuffer<data=memref<1x2x3x8xf16>, dynamic_shape=memref<4xsi32>>,
// CHECK-SAME:         [[ARG1:%.+]]: !VPUIP.BoundedBuffer<data=memref<1x2x3x8xf16>, dynamic_shape=memref<4xsi32>>)
// CHECK-SAME:         -> !VPUIP.BoundedBuffer<data=memref<1x4x3x8xf16>, dynamic_shape=memref<4xsi32>>
func.func @ConcatSWLayer(%arg0: tensor<1x2x3x?xf16, {bounds = [1, 2, 3, 8], order = #NCHW}>,
                                %arg1: tensor<1x2x3x?xf16, {bounds = [1, 2, 3, 8], order = #NCHW}>)
                                -> tensor<1x4x3x?xf16, {bounds = [1, 4, 3, 8], order = #NCHW}> {

    %0 = VPU.Concat(%arg0, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 2, 0, 0]]} : tensor<1x2x3x?xf16, {bounds = [1, 2, 3, 8], order = #NCHW}>, tensor<1x2x3x?xf16, {bounds = [1, 2, 3, 8], order = #NCHW}> -> tensor<1x4x3x?xf16, {bounds = [1, 4, 3, 8], order = #NCHW}>
    return %0 : tensor<1x4x3x?xf16, {bounds = [1, 4, 3, 8], order = #NCHW}>

    // CHECK:  [[ALLOC:%.+]] = memref.alloc() : memref<1x2x3x8xf16, [@CMX_NN, 0]>
    // CHECK:  [[ALLOC0:%.+]] = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>
    // CHECK:  [[BUFFER0:%.+]] = VPUIP.GroupBoundedBuffer([[ALLOC]], [[ALLOC0]])
    // CHECK:            : memref<1x2x3x8xf16, [@CMX_NN, 0]>, memref<4xsi32, [@CMX_NN, 0]>
    // CHECK:            -> !VPUIP.BoundedBuffer<data=memref<1x2x3x8xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>
    // CHECK:  [[COPY0:%.+]] = VPUIP.Copy inputs({{[^:]+}}  : !VPUIP.BoundedBuffer<data=memref<1x2x3x8xf16>, dynamic_shape=memref<4xsi32>>)
    // CHECK:  outputs([[BUFFER0]] : !VPUIP.BoundedBuffer<data=memref<1x2x3x8xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>)
    // CHECK:  -> !VPUIP.BoundedBuffer<data=memref<1x2x3x8xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>

    // CHECK:  [[ALLOC1:%.+]] = memref.alloc() : memref<1x2x3x8xf16, [@CMX_NN, 0]>
    // CHECK:  [[ALLOC2:%.+]] = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>
    // CHECK:  [[BUFFER1:%.+]] = VPUIP.GroupBoundedBuffer([[ALLOC1]], [[ALLOC2]])
    // CHECK:           : memref<1x2x3x8xf16, [@CMX_NN, 0]>, memref<4xsi32, [@CMX_NN, 0]>
    // CHECK:           -> !VPUIP.BoundedBuffer<data=memref<1x2x3x8xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>
    // CHECK:  [[COPY1:%.+]] = VPUIP.Copy inputs({{[^:]+}} : !VPUIP.BoundedBuffer<data=memref<1x2x3x8xf16>, dynamic_shape=memref<4xsi32>>)
    // CHECK:           outputs([[BUFFER1]] : !VPUIP.BoundedBuffer<data=memref<1x2x3x8xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>)
    // CHECK:           -> !VPUIP.BoundedBuffer<data=memref<1x2x3x8xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>

    // CHECK:  [[ALLOC3:%.+]] = memref.alloc() : memref<1x4x3x8xf16, [@CMX_NN, 0]>
    // CHECK:  [[ALLOC4:%.+]] = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>
    // CHECK:  [[BUFFER2:%.+]] = VPUIP.GroupBoundedBuffer([[ALLOC3]], [[ALLOC4]])
    // CHECK:           : memref<1x4x3x8xf16, [@CMX_NN, 0]>, memref<4xsi32, [@CMX_NN, 0]>
    // CHECK:           -> !VPUIP.BoundedBuffer<data=memref<1x4x3x8xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>

    // CHECK:  [[RES:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
    // CHECK:           @VPU.SW::@builtin_Concat inputs([[COPY0]] as {{[^:]+}}:
    // CHECK:               !VPUIP.BoundedBuffer<data=memref<1x2x3x8xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>,
    // CHECK:               [[COPY1]] as {{[^:]+}}: !VPUIP.BoundedBuffer<data=memref<1x2x3x8xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>)
    // CHECK:               outputs([[BUFFER2]] as {{[^:]+}}: !VPUIP.BoundedBuffer<data=memref<1x4x3x8xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>) on tile 0
    // CHECK:               -> !VPUIP.BoundedBuffer<data=memref<1x4x3x8xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>{
    // CHECK:                       VPUIP.SW.Kernel.run
    // CHECK-SAME{LITERAL}:             {attrs = [[0, 0, 0, 0], [0, 0, 2, 0]]}
    // CHECK:                           ({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : !VPUIP.BoundedBuffer<data=memref<1x2x3x8xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>,
    // CHECK:                                   !VPUIP.BoundedBuffer<data=memref<1x2x3x8xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>,
    // CHECK:                                   !VPUIP.BoundedBuffer<data=memref<1x4x3x8xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>
    // CHECK:  }

    // CHECK:  [[ALLOC5:%.+]] = memref.alloc() : memref<1x4x3x8xf16>
    // CHECK:  [[ALLOC6:%.+]] = memref.alloc() : memref<4xsi32>
    // CHECK:  [[BUFFER3:%.+]] = VPUIP.GroupBoundedBuffer([[ALLOC5]], [[ALLOC6]])
    // CHECK:           : memref<1x4x3x8xf16>, memref<4xsi32>
    // CHECK:           -> !VPUIP.BoundedBuffer<data=memref<1x4x3x8xf16>, dynamic_shape=memref<4xsi32>>
    // CHECK:  [[COPY2:%.+]] = VPUIP.Copy inputs([[RES]]
    // CHECK:           : !VPUIP.BoundedBuffer<data=memref<1x4x3x8xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>)
    // CHECK:           outputs([[BUFFER3]] : !VPUIP.BoundedBuffer<data=memref<1x4x3x8xf16>, dynamic_shape=memref<4xsi32>>) -> !VPUIP.BoundedBuffer<data=memref<1x4x3x8xf16>, dynamic_shape=memref<4xsi32>>

    // CHECK:  return [[COPY2]] : !VPUIP.BoundedBuffer<data=memref<1x4x3x8xf16>, dynamic_shape=memref<4xsi32>>
}

// -----

// CHECK:  module @VPU.SW {
// CHECK-NEXT:    func.func private @builtin_RMS(memref<*xf32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf32, [@CMX_NN, 0]>, f64) attributes {VPU.kernel_code = "rms_norm.cpp", VPU.kernel_entry = "rms_norm", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func @RMSNorm
// CHECK-SAME:      [[INPUT:%.+]]: memref<1x2x6xf32>
func.func @RMSNorm(%input: tensor<1x2x6xf32>) -> tensor<1x2x6xf32> {
    %cst = const.Declare tensor<6xf16> = dense<[2.900000e-02, 1.400000e-02, 3.000000e-03, 1.300000e-02, 1.500000e-02, 0.00899999961]> : tensor<6xf32>, [#const.CastElemType<f16>]
    %rmsop = VPU.RMS(%input, %cst) {epsilon = 9.9999997473787516E-6 : f64} : tensor<1x2x6xf32>, tensor<6xf16> -> tensor<1x2x6xf32>
    return %rmsop : tensor<1x2x6xf32>

// CHECK:    [[CST:%.+]] = const.Declare memref<6xf16> = dense<[2.900000e-02, 1.400000e-02, 3.000000e-03, 1.300000e-02, 1.500000e-02, 0.00899999961]> : tensor<6xf32>, [#const.CastElemType<f16>]
// CHECK:    [[ALLOC:%.+]] = memref.alloc() : memref<1x2x6xf32, [@CMX_NN, 0]>
// CHECK:    [[COPY0:%.+]] = VPUIP.Copy inputs([[INPUT]] : memref<1x2x6xf32>) outputs([[ALLOC]] : memref<1x2x6xf32, [@CMX_NN, 0]>) -> memref<1x2x6xf32, [@CMX_NN, 0]>

// CHECK:    [[ALLOC0:%.+]] = memref.alloc() : memref<6xf16, [@CMX_NN, 0]>
// CHECK:    [[COPY1:%.+]] = VPUIP.Copy inputs([[CST]] : memref<6xf16>) outputs([[ALLOC0]] : memref<6xf16, [@CMX_NN, 0]>) -> memref<6xf16, [@CMX_NN, 0]>

// CHECK:    [[ALLOC1:%.+]] = memref.alloc() : memref<1x2x6xf32, [@CMX_NN, 0]>
// CHECK:    [[RES:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_RMS inputs([[COPY0]] as {{[^:]+}}: memref<1x2x6xf32, [@CMX_NN, 0]>, [[COPY1]] as {{[^:]+}}: memref<6xf16, [@CMX_NN, 0]>) outputs([[ALLOC1]] as {{[^:]+}}: memref<1x2x6xf32, [@CMX_NN, 0]>) on tile 0 -> memref<1x2x6xf32, [@CMX_NN, 0]>{
// CHECK:      VPUIP.SW.Kernel.run {attrs = [9.9999997473787516E-6]}({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<1x2x6xf32, [@CMX_NN, 0]>, memref<6xf16, [@CMX_NN, 0]>, memref<1x2x6xf32, [@CMX_NN, 0]>
// CHECK:    }

// CHECK:    [[ALLOC2:%.+]] = memref.alloc() : memref<1x2x6xf32>
// CHECK:    [[COPY2:%.+]] = VPUIP.Copy inputs([[RES]] : memref<1x2x6xf32, [@CMX_NN, 0]>) outputs([[ALLOC2]] : memref<1x2x6xf32>) -> memref<1x2x6xf32>
// CHECK:    return [[COPY2]] : memref<1x2x6xf32>
}

// -----

// CHECK:  module @VPU.SW {
// CHECK-NEXT:    func.func private @builtin_Inverse(memref<*xf32, [@CMX_NN, 0]>, memref<*xf32, [@CMX_NN, 0]>, i64) attributes {VPU.kernel_code = "inverse.cpp", VPU.kernel_entry = "inverse", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func @Inverse
// CHECK-SAME:      [[INPUT:%.+]]: memref<1x10x2x2xf32>
func.func @Inverse(%input: tensor<1x10x2x2xf32>) -> tensor<1x10x2x2xf32> {
    %inverseop = VPU.Inverse(%input) {__inplace_operands_attr__ = ["true"], adjoint} : tensor<1x10x2x2xf32> -> tensor<1x10x2x2xf32>
    return %inverseop : tensor<1x10x2x2xf32>

// CHECK:    [[ALLOC:%.+]] = memref.alloc() : memref<1x10x2x2xf32, [@CMX_NN, 0]>
// CHECK:    [[COPY0:%.+]] = VPUIP.Copy inputs([[INPUT]] : memref<1x10x2x2xf32>) outputs([[ALLOC]] : memref<1x10x2x2xf32, [@CMX_NN, 0]>) -> memref<1x10x2x2xf32, [@CMX_NN, 0]>

// CHECK:    [[ALLOC0:%.+]] = memref.alloc() : memref<1x10x2x2xf32, [@CMX_NN, 0]>

// CHECK:    [[RES:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Inverse inputs([[COPY0]] as {{[^:]+}}: memref<1x10x2x2xf32, [@CMX_NN, 0]>) outputs([[ALLOC0]] as {{[^:]+}}: memref<1x10x2x2xf32, [@CMX_NN, 0]>) on tile 0 -> memref<1x10x2x2xf32, [@CMX_NN, 0]>{
// CHECK:      VPUIP.SW.Kernel.run {attrs = [1]}({{[^:]+}}, {{[^:]+}}) : memref<1x10x2x2xf32, [@CMX_NN, 0]>, memref<1x10x2x2xf32, [@CMX_NN, 0]>
// CHECK:    }

// CHECK:    [[ALLOC1:%.+]] = memref.alloc() : memref<1x10x2x2xf32>
// CHECK:    [[COPY1:%.+]] = VPUIP.Copy inputs([[RES]] : memref<1x10x2x2xf32, [@CMX_NN, 0]>) outputs([[ALLOC1]] : memref<1x10x2x2xf32>) -> memref<1x10x2x2xf32>
// CHECK:    return [[COPY1]] : memref<1x10x2x2xf32>
}

// -----

// CHECK-LABEL: func.func @DeformableConvolutionSWLayer
// CHECK-SAME:  ([[ARG0:%.+]]: memref<1x128x19x19xf16>, [[ARG1:%.+]]: memref<1x18x19x19xf16>, [[ARG2:%.+]]: memref<128x128x3x3xf16>, [[ARG3:%.+]]: memref<1x9x19x19xf16>) -> memref<1x128x19x19xf16>
func.func @DeformableConvolutionSWLayer(%arg0: tensor<1x128x19x19xf16>, %arg1: tensor<1x18x19x19xf16>, %arg2: tensor<128x128x3x3xf16>, %arg3: tensor<1x9x19x19xf16>) -> tensor<1x128x19x19xf16> {
    %output = VPU.DeformableConvolution(%arg0, %arg1, %arg2, %arg3) {biliniar_interpolate_pad, deformable_group = 1 : i64, dilations = [1, 1], group = 1 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x128x19x19xf16>, tensor<1x18x19x19xf16>, tensor<128x128x3x3xf16>, tensor<1x9x19x19xf16> -> tensor<1x128x19x19xf16>
    return %output : tensor<1x128x19x19xf16>

    // CHECK:   [[ALLOC:%.+]] = memref.alloc() : memref<1x128x19x19xf16, [@CMX_NN, 0]>
    // CHECK:   [[COPY0:%.+]] = VPUIP.Copy inputs([[ARG0]] : memref<1x128x19x19xf16>) outputs([[ALLOC]] : memref<1x128x19x19xf16, [@CMX_NN, 0]>) -> memref<1x128x19x19xf16, [@CMX_NN, 0]>
    // CHECK:   [[ALLOC0:%.+]] = memref.alloc() : memref<1x18x19x19xf16, [@CMX_NN, 0]>
    // CHECK:   [[COPY1:%.+]] = VPUIP.Copy inputs([[ARG1]] : memref<1x18x19x19xf16>) outputs([[ALLOC0]] : memref<1x18x19x19xf16, [@CMX_NN, 0]>) -> memref<1x18x19x19xf16, [@CMX_NN, 0]>
    // CHECK:   [[ALLOC1:%.+]] = memref.alloc() : memref<128x128x3x3xf16, [@CMX_NN, 0]>
    // CHECK:   [[COPY2:%.+]] = VPUIP.Copy inputs([[ARG2]] : memref<128x128x3x3xf16>) outputs([[ALLOC1]] : memref<128x128x3x3xf16, [@CMX_NN, 0]>) -> memref<128x128x3x3xf16, [@CMX_NN, 0]>
    // CHECK:   [[ALLOC2:%.+]] = memref.alloc() : memref<1x9x19x19xf16, [@CMX_NN, 0]>
    // CHECK:   [[COPY3:%.+]] = VPUIP.Copy inputs([[ARG3]] : memref<1x9x19x19xf16>) outputs([[ALLOC2]] : memref<1x9x19x19xf16, [@CMX_NN, 0]>) -> memref<1x9x19x19xf16, [@CMX_NN, 0]>
    // CHECK:   [[ALLOC3:%.+]] = memref.alloc() : memref<1x128x19x19xf16, [@CMX_NN, 0]>

    // CHECK:   VPUIP.SW.Kernel
    // CHECK-SAME:  {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_DeformableConvolution
    // CHECK-SAME:  inputs([[COPY0]] as {{[^:]+}}: memref<1x128x19x19xf16, [@CMX_NN, 0]>, [[COPY1]] as {{[^:]+}}: memref<1x18x19x19xf16, [@CMX_NN, 0]>,
    // CHECK-SAME:  [[COPY2]] as {{[^:]+}}: memref<128x128x3x3xf16, [@CMX_NN, 0]>, [[COPY3]] as {{[^:]+}}: memref<1x9x19x19xf16, [@CMX_NN, 0]>)
    // CHECK-SAME:  outputs([[ALLOC3]] as {{[^:]+}}: memref<1x128x19x19xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x128x19x19xf16, [@CMX_NN, 0]>{

    // CHECK:   VPUIP.SW.Kernel.run
    // CHECK-SAME{LITERAL}: {attrs = [[1, 1], [1, 1], [1, 1], [1, 1], 1, 1, 1]}
    // CHECK-SAME:  memref<1x128x19x19xf16, [@CMX_NN, 0]>, memref<1x18x19x19xf16, [@CMX_NN, 0]>, memref<128x128x3x3xf16, [@CMX_NN, 0]>, memref<1x9x19x19xf16, [@CMX_NN, 0]>, memref<1x128x19x19xf16, [@CMX_NN, 0]>
    // CHECK:   }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @DynamicBroadcastShapeSubgraph {
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "input_1" : tensor<4x1x1xf16>
    DataInfo "input_0" : tensor<1x4x5x5xf16>
  } outputsInfo : {
    DataInfo "Broadcast_63" friendlyName = "Result_67" : tensor<1x4x5x5xf16>
  }
  // CHECK: func.func @main([[ARG0:%.+]]: memref<4x1x1xf16>, [[ARG1:%.+]]: !VPUIP.BoundedBuffer<data=memref<1x4x5x5xf16>, dynamic_shape=memref<4xsi32>>) -> !VPUIP.BoundedBuffer<data=memref<1x4x5x5xf16>, dynamic_shape=memref<4xsi32>> {
  func.func @main(%arg0: tensor<4x1x1xf16>, %arg1: tensor<1x4x?x?xf16, {bounds = [1, 4, 5, 5], order = #NCHW}>) -> tensor<1x4x?x?xf16, {bounds = [1, 4, 5, 5], order = #NCHW}> {
    %0 = VPU.ShapeOf(%arg1) : tensor<1x4x?x?xf16, {bounds = [1, 4, 5, 5], order = #NCHW}> -> tensor<4xsi32>
    %1 = VPU.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 4, 1, 1]} : tensor<4x1x1xf16> -> tensor<1x4x1x1xf16>
    %2 = VPU.DynamicTile(%1, %0) {output_bounds = [1, 4, 5, 5], output_shape = [1, 4, -9223372036854775808, -9223372036854775808]} : tensor<1x4x1x1xf16>, tensor<4xsi32> -> tensor<1x4x?x?xf16, {bounds = [1, 4, 5, 5], order = #NCHW}>
    return %2 : tensor<1x4x?x?xf16, {bounds = [1, 4, 5, 5], order = #NCHW}>

    // CHECK:    [[ALLOC:%.+]] = memref.alloc() : memref<1x4x5x5xf16, [@CMX_NN, 0]>
    // CHECK:    [[ALLOC_0:%.+]] = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>
    // CHECK:    [[BUFF:%.+]] = VPUIP.GroupBoundedBuffer([[ALLOC]], [[ALLOC_0]]) : memref<1x4x5x5xf16, [@CMX_NN, 0]>, memref<4xsi32, [@CMX_NN, 0]> -> !VPUIP.BoundedBuffer<data=memref<1x4x5x5xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>
    // CHECK:    [[COPY:%.+]] = VPUIP.Copy inputs(%arg1 : !VPUIP.BoundedBuffer<data=memref<1x4x5x5xf16>, dynamic_shape=memref<4xsi32>>) outputs([[BUFF]] : !VPUIP.BoundedBuffer<data=memref<1x4x5x5xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>) -> !VPUIP.BoundedBuffer<data=memref<1x4x5x5xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>
    // CHECK:    [[ALLOC_1:%.+]] = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>
    // CHECK:    [[RESULT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_ShapeOf inputs([[COPY]] as %arg2: !VPUIP.BoundedBuffer<data=memref<1x4x5x5xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>) outputs([[ALLOC_1]] as %arg3: memref<4xsi32, [@CMX_NN, 0]>) on tile 0 -> memref<4xsi32, [@CMX_NN, 0]>{
    // CHECK:      VPUIP.SW.Kernel.run(%arg2, %arg3) : !VPUIP.BoundedBuffer<data=memref<1x4x5x5xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>, memref<4xsi32, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    [[ALLOC_2:%.+]] = memref.alloc() : memref<4xsi32>
    // CHECK:    [[COPY_0:%.+]] = VPUIP.Copy inputs([[RESULT]] : memref<4xsi32, [@CMX_NN, 0]>) outputs([[ALLOC_2]] : memref<4xsi32>) -> memref<4xsi32>
    // CHECK:    [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs(%arg0 : memref<4x1x1xf16>) -> memref<1x4x1x1xf16>
    // CHECK:    [[ALLOC_3:%.+]] = memref.alloc() : memref<1x4x1x1xf16, [@CMX_NN, 0]>
    // CHECK:    [[COPY_1:%.+]] = VPUIP.Copy inputs([[RESHAPE]] : memref<1x4x1x1xf16>) outputs([[ALLOC_3]] : memref<1x4x1x1xf16, [@CMX_NN, 0]>) -> memref<1x4x1x1xf16, [@CMX_NN, 0]>
    // CHECK:    [[ALLOC_4:%.+]] = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>
    // CHECK:    [[COPY_2:%.+]] = VPUIP.Copy inputs([[COPY_0]] : memref<4xsi32>) outputs([[ALLOC_4]] : memref<4xsi32, [@CMX_NN, 0]>) -> memref<4xsi32, [@CMX_NN, 0]>
    // CHECK:    [[ALLOC_5:%.+]] = memref.alloc() : memref<1x4x5x5xf16, [@CMX_NN, 0]>
    // CHECK:    [[ALLOC_6:%.+]] = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>
    // CHECK:    [[BUFF_0:%.+]] = VPUIP.GroupBoundedBuffer([[ALLOC_5]], [[ALLOC_6]]) : memref<1x4x5x5xf16, [@CMX_NN, 0]>, memref<4xsi32, [@CMX_NN, 0]> -> !VPUIP.BoundedBuffer<data=memref<1x4x5x5xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>
    // CHECK:    [[RESULT_0:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_DynamicTile inputs([[COPY_1]] as %arg2: memref<1x4x1x1xf16, [@CMX_NN, 0]>, [[COPY_2]] as %arg3: memref<4xsi32, [@CMX_NN, 0]>) outputs([[BUFF_0]] as %arg4: !VPUIP.BoundedBuffer<data=memref<1x4x5x5xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>) on tile 0 -> !VPUIP.BoundedBuffer<data=memref<1x4x5x5xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>{
    // CHECK:      VPUIP.SW.Kernel.run {attrs = [4, [1, 1, 1, 1]]}(%arg2, %arg3, %arg4) : memref<1x4x1x1xf16, [@CMX_NN, 0]>, memref<4xsi32, [@CMX_NN, 0]>, !VPUIP.BoundedBuffer<data=memref<1x4x5x5xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>
    // CHECK:    }
    // CHECK:    [[ALLOC_8:%.+]] = memref.alloc() : memref<1x4x5x5xf16>
    // CHECK:    [[ALLOC_9:%.+]] = memref.alloc() : memref<4xsi32>
    // CHECK:    [[BUFF_1:%.+]] = VPUIP.GroupBoundedBuffer([[ALLOC_8]], [[ALLOC_9]]) : memref<1x4x5x5xf16>, memref<4xsi32> -> !VPUIP.BoundedBuffer<data=memref<1x4x5x5xf16>, dynamic_shape=memref<4xsi32>>
    // CHECK:    [[COPY_3:%.+]] = VPUIP.Copy inputs([[RESULT_0]] : !VPUIP.BoundedBuffer<data=memref<1x4x5x5xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>) outputs([[BUFF_1]] : !VPUIP.BoundedBuffer<data=memref<1x4x5x5xf16>, dynamic_shape=memref<4xsi32>>) -> !VPUIP.BoundedBuffer<data=memref<1x4x5x5xf16>, dynamic_shape=memref<4xsi32>>
    // CHECK:    return [[COPY_3]] : !VPUIP.BoundedBuffer<data=memref<1x4x5x5xf16>, dynamic_shape=memref<4xsi32>>
  }
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK: func.func @DynamicTileFromBroadcast([[ARG0:%.+]]: !VPUIP.BoundedBuffer<data=memref<1x1x10xsi64>, dynamic_shape=memref<3xsi32>>, [[ARG1:%.+]]: memref<4xsi32>) -> !VPUIP.BoundedBuffer<data=memref<1x1x10x5xsi64>, dynamic_shape=memref<4xsi32>> {
func.func @DynamicTileFromBroadcast(%arg0: tensor<1x1x?xsi64, {bounds = [1, 1, 10], order = #CHW}>, %arg1: tensor<4xsi32>) -> tensor<1x1x?x?xsi64, {bounds = [1, 1, 10, 5], order = #NCHW}> {
    %0 = VPU.DynamicTile(%arg0, %arg1) {output_bounds = [1, 1, 10, 5], output_shape = [1, 1, -9223372036854775808, -9223372036854775808]} : tensor<1x1x?xsi64, {bounds = [1, 1, 10], order = #CHW}>, tensor<4xsi32> -> tensor<1x1x?x?xsi64, {bounds = [1, 1, 10, 5], order = #NCHW}>
    return %0 : tensor<1x1x?x?xsi64, {bounds = [1, 1, 10, 5], order = #NCHW}>

    // CHECK:    [[ALLOC:%.+]] = memref.alloc() : memref<1x1x10xsi64, [@CMX_NN, 0]>
    // CHECK:    [[ALLOC_0:%.+]] = memref.alloc() : memref<3xsi32, [@CMX_NN, 0]>
    // CHECK:    [[BUFF:%.+]] = VPUIP.GroupBoundedBuffer([[ALLOC]], [[ALLOC_0:%.+]]) : memref<1x1x10xsi64, [@CMX_NN, 0]>, memref<3xsi32, [@CMX_NN, 0]> -> !VPUIP.BoundedBuffer<data=memref<1x1x10xsi64, [@CMX_NN, 0]>, dynamic_shape=memref<3xsi32, [@CMX_NN, 0]>>
    // CHECK:    [[COPY:%.+]] = VPUIP.Copy inputs([[ARG0]] : !VPUIP.BoundedBuffer<data=memref<1x1x10xsi64>, dynamic_shape=memref<3xsi32>>) outputs([[BUFF]] : !VPUIP.BoundedBuffer<data=memref<1x1x10xsi64, [@CMX_NN, 0]>, dynamic_shape=memref<3xsi32, [@CMX_NN, 0]>>) -> !VPUIP.BoundedBuffer<data=memref<1x1x10xsi64, [@CMX_NN, 0]>, dynamic_shape=memref<3xsi32, [@CMX_NN, 0]>>
    // CHECK:    [[ALLOC_1:%.+]] = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>
    // CHECK:    [[COPY_0:%.+]] = VPUIP.Copy inputs([[ARG1]] : memref<4xsi32>) outputs([[ALLOC_1]] : memref<4xsi32, [@CMX_NN, 0]>) -> memref<4xsi32, [@CMX_NN, 0]>
    // CHECK:    [[ALLOC_2:%.+]] = memref.alloc() : memref<1x1x10x5xsi64, [@CMX_NN, 0]>
    // CHECK:    [[ALLOC_3:%.+]] = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>
    // CHECK:    [[BUFF_0:%.+]] = VPUIP.GroupBoundedBuffer([[ALLOC_2]], [[ALLOC_3]]) : memref<1x1x10x5xsi64, [@CMX_NN, 0]>, memref<4xsi32, [@CMX_NN, 0]> -> !VPUIP.BoundedBuffer<data=memref<1x1x10x5xsi64, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>
    // CHECK:    [[RESULT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_DynamicTile inputs([[COPY]] as %arg2: !VPUIP.BoundedBuffer<data=memref<1x1x10xsi64, [@CMX_NN, 0]>, dynamic_shape=memref<3xsi32, [@CMX_NN, 0]>>, [[COPY_0]] as %arg3: memref<4xsi32, [@CMX_NN, 0]>) outputs([[BUFF_0]] as %arg4: !VPUIP.BoundedBuffer<data=memref<1x1x10x5xsi64, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>) on tile 0 -> !VPUIP.BoundedBuffer<data=memref<1x1x10x5xsi64, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>{
    // CHECK:      VPUIP.SW.Kernel.run {attrs = [4, [1, 1, 1, 1]]}(%arg2, %arg3, %arg4) : !VPUIP.BoundedBuffer<data=memref<1x1x10xsi64, [@CMX_NN, 0]>, dynamic_shape=memref<3xsi32, [@CMX_NN, 0]>>, memref<4xsi32, [@CMX_NN, 0]>, !VPUIP.BoundedBuffer<data=memref<1x1x10x5xsi64, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>
    // CHECK:    }
    // CHECK:    [[ALLOC_4:%.+]] = memref.alloc() : memref<1x1x10x5xsi64>
    // CHECK:    [[ALLOC_5:%.+]] = memref.alloc() : memref<4xsi32>
    // CHECK:    [[BUFF_1:%.+]] = VPUIP.GroupBoundedBuffer([[ALLOC_4]], [[ALLOC_5]]) : memref<1x1x10x5xsi64>, memref<4xsi32> -> !VPUIP.BoundedBuffer<data=memref<1x1x10x5xsi64>, dynamic_shape=memref<4xsi32>>
    // CHECK:    [[COPY_1:%.+]] = VPUIP.Copy inputs([[RESULT]] : !VPUIP.BoundedBuffer<data=memref<1x1x10x5xsi64, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>) outputs([[BUFF_1]] : !VPUIP.BoundedBuffer<data=memref<1x1x10x5xsi64>, dynamic_shape=memref<4xsi32>>) -> !VPUIP.BoundedBuffer<data=memref<1x1x10x5xsi64>, dynamic_shape=memref<4xsi32>>
    // CHECK:    return [[COPY_1]] : !VPUIP.BoundedBuffer<data=memref<1x1x10x5xsi64>, dynamic_shape=memref<4xsi32>>
}
