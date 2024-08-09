//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --introduce-init-function %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @InternalConstants
module @InternalConstants {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output1" : tensor<32x16x3x3xf16>
        DataInfo "output2" : tensor<48x32x1x1xf16>
    }

    // Constants that are found in the @npu_bin section are created during compilation and should not be processed in the @init function
    const.Data @npu_bin {
        const.Rodata @value dense<1.000000e+00> : tensor<32x16x3x3xf16>
    }
    func.func @main() -> (memref<32x16x3x3xf16>, memref<48x32x1x1xf16>) {
        %cst1 = const.Declare memref<32x16x3x3xf16> = ref<@npu_bin::@value> : tensor<32x16x3x3xf16>
        %cst2 = const.Declare memref<48x32x1x1xf16> = dense<2.000000e+00> : tensor<48x32x1x1xf16>
        return %cst1, %cst2 : memref<32x16x3x3xf16>, memref<48x32x1x1xf16>
    }

    // CHECK:  const.Data @npu_bin {
    // CHECK:      const.Rodata [[CST_SYM:@.+]] dense<1.000000e+00> : tensor<32x16x3x3xf16>
    // CHECK:  }
    // CHECK:  func.func @main() -> (memref<32x16x3x3xf16>, memref<48x32x1x1xf16>) {
    // CHECK:      [[CST1:%.+]] = const.Declare memref<32x16x3x3xf16> = ref<@npu_bin::[[CST_SYM]]> : tensor<32x16x3x3xf16>
    // CHECK:      [[CST2:%.+]] = const.Declare memref<48x32x1x1xf16> = dense<2.000000e+00> : tensor<48x32x1x1xf16>
    // CHECK:      return [[CST1]], [[CST2]]
    // CHECK:  }
}

// -----

// CHECK-LABEL: @NoTransformation
module @NoTransformation {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output" : tensor<32x16x3x3xf16>
    }

    const.Data @ov_bin {
        const.Rodata @value dense<1.000000e+00> : tensor<32x16x3x3xf16>
    }
    func.func @main() -> memref<32x16x3x3xf16> {
        %cst = const.Declare memref<32x16x3x3xf16> = ref<@ov_bin::@value> : tensor<32x16x3x3xf16>
        return %cst : memref<32x16x3x3xf16>
    }

    // CHECK:  const.Data @ov_bin {
    // CHECK:      const.Rodata [[CST_SYM:@.+]] dense<1.000000e+00> : tensor<32x16x3x3xf16>
    // CHECK:  }
    // CHECK:  const.Data @init_res {
    // CHECK:      const.Ref [[FOLDED_SYM:@.+]] : tensor<32x16x3x3xf16>
    // CHECK:  }
    // CHECK:  func.func @init() {
    // CHECK:      [[LOAD:%.+]] = const.Load @ov_bin::[[CST_SYM]] -> tensor<32x16x3x3xf16>
    // CHECK:      const.Store [[LOAD]], @init_res::[[FOLDED_SYM]] : tensor<32x16x3x3xf16>
    // CHECK:      return
    // CHECK:  }
    // CHECK:  func.func @main() -> memref<32x16x3x3xf16> {
    // CHECK:      [[CST:%.+]] = const.Load @init_res::[[FOLDED_SYM]] -> memref<32x16x3x3xf16>
    // CHECK:      return [[CST]]
    // CHECK:  }
}

// -----

// CHECK-LABEL: @TransformationAdd
module @TransformationAdd {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output" : tensor<32x16x3x3xf16>
    }

    const.Data @ov_bin {
        const.Rodata @value dense<1.000000e+00> : tensor<32x16x3x3xf16>
    }
    func.func @main() -> memref<32x16x3x3xf16> {
        %cst = const.Declare memref<32x16x3x3xf16> = ref<@ov_bin::@value> : tensor<32x16x3x3xf16>, [#const.Add<1.0>]
        return %cst : memref<32x16x3x3xf16>
    }

    // CHECK:  const.Data @ov_bin {
    // CHECK:      const.Rodata [[CST_SYM:@.+]] dense<1.000000e+00> : tensor<32x16x3x3xf16>
    // CHECK:  }
    // CHECK:  const.Data @init_res {
    // CHECK:      const.Ref [[FOLDED_SYM:@.+]] : tensor<32x16x3x3xf16>
    // CHECK:  }
    // CHECK:  func.func @init() {
    // CHECK:      [[LOAD:%.+]] = const.Load @ov_bin::[[CST_SYM]] -> tensor<32x16x3x3xf16>
    // CHECK:      [[BIAS:%.+]] = const.Declare tensor<1xf32> = dense<1.000000e+00> : tensor<1xf32>
    // CHECK:      [[ADD:%.+]] = IE.Add([[LOAD]], [[BIAS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<32x16x3x3xf16>, tensor<1xf32> -> tensor<32x16x3x3xf16>
    // CHECK:      const.Store [[ADD]], @init_res::[[FOLDED_SYM]] : tensor<32x16x3x3xf16>
    // CHECK:      return
    // CHECK:  }
    // CHECK:  func.func @main() -> memref<32x16x3x3xf16> {
    // CHECK:      [[CST:%.+]] = const.Load @init_res::[[FOLDED_SYM]] -> memref<32x16x3x3xf16>
    // CHECK:      return [[CST]]
    // CHECK:  }
}

// -----

// CHECK-LABEL: @TransformationBroadcast
module @TransformationBroadcast {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output" : tensor<32x32x3x3xf16>
    }

    const.Data @ov_bin {
        const.Rodata @value dense<1.000000e+00> : tensor<32x16x3x3xf16>
    }
    func.func @main() -> memref<32x32x3x3xf16> {
        %cst = const.Declare memref<32x32x3x3xf16> = ref<@ov_bin::@value> : tensor<32x16x3x3xf16>, [#const.Broadcast<1 : i64, 32 : i64>]
        return %cst : memref<32x32x3x3xf16>
    }

    // CHECK:  const.Data @ov_bin {
    // CHECK:      const.Rodata [[CST_SYM:@.+]] dense<1.000000e+00> : tensor<32x16x3x3xf16>
    // CHECK:  }
    // CHECK:  const.Data @init_res {
    // CHECK:      const.Ref [[FOLDED_SYM:@.+]] : tensor<32x32x3x3xf16>
    // CHECK:  }
    // CHECK:  func.func @init() {
    // CHECK:      [[LOAD:%.+]] = const.Load @ov_bin::[[CST_SYM]] -> tensor<32x16x3x3xf16>
    // CHECK:      [[SHAPE:%.+]] = const.Declare tensor<4xi64> = dense<[32, 32, 3, 3]> : tensor<4xi64>
    // CHECK:      [[BROADCAST:%.+]] = IE.Broadcast([[LOAD]], [[SHAPE]]) : tensor<32x16x3x3xf16>, tensor<4xi64> -> tensor<32x32x3x3xf16>
    // CHECK:      const.Store [[BROADCAST]], @init_res::[[FOLDED_SYM]] : tensor<32x32x3x3xf16>
    // CHECK:      return
    // CHECK:  }
    // CHECK:  func.func @main() -> memref<32x32x3x3xf16> {
    // CHECK:      [[CST:%.+]] = const.Load @init_res::[[FOLDED_SYM]] -> memref<32x32x3x3xf16>
    // CHECK:      return [[CST]]
    // CHECK:  }
}

// -----

!qElemType_in = !quant.uniform<u8:f16:1, {0.002174197219488189,0.0013370063361220473}>
!qElemType_out = !quant.uniform<u8:f16:0, {0.002174197219488189,0.0013370063361220473}>
// CHECK-DAG:  [[QELEMTYPE_IN:!.+]] = !quant.uniform<u8:f16:1, {0.002174197219488189,0.0013370063361220473}>
// CHECK-DAG:  [[QELEMTYPE_OUT:!.+]] = !quant.uniform<u8:f16:0, {0.002174197219488189,0.0013370063361220473}>

// CHECK: @TransformationChangeShapeAndElemType
module @TransformationChangeShapeAndElemType {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output" : tensor<2x1x1x1xui8>
    }

    const.Data @ov_bin {
        const.Rodata @value dense<1> : tensor<1x2x1x1xui8>
    }
    func.func @main() -> memref<2x1x1x1x!qElemType_out> {
        %cst = const.Declare memref<2x1x1x1x!qElemType_out> = ref<@ov_bin::@value> : tensor<1x2x1x1xui8>, [#const.QuantCast<!qElemType_in>, #const.ChangeShapeAndElemType<[2, 1, 1, 1], !qElemType_out>]
        return %cst : memref<2x1x1x1x!qElemType_out>
    }

    // CHECK:  const.Data @ov_bin {
    // CHECK:      const.Rodata [[CST_SYM:@.+]] dense<1> : tensor<1x2x1x1xui8>
    // CHECK:  }
    // CHECK:  const.Data @init_res {
    // CHECK:      const.Ref [[FOLDED_SYM:@.+]] : tensor<2x1x1x1x[[QELEMTYPE_OUT]]>
    // CHECK:  }
    // CHECK:  func.func @init() {
    // CHECK:      [[LOAD:%.+]] = const.Load @ov_bin::[[CST_SYM]] -> tensor<1x2x1x1xui8>
    // CHECK:      [[QUANT_CAST:%.+]] = IE.QuantizeCast([[LOAD]]) {dstElemType = [[QELEMTYPE_IN]]} : tensor<1x2x1x1xui8> -> tensor<1x2x1x1x[[QELEMTYPE_IN]]>
    // CHECK:      [[RESHAPE:%.+]] = IE.AffineReshape([[QUANT_CAST]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [2, 1, 1, 1]}
    // CHECK-SAME:         : tensor<1x2x1x1x[[QELEMTYPE_IN]]> -> tensor<2x1x1x1x[[QELEMTYPE_OUT]]>
    // CHECK:      const.Store [[RESHAPE]], @init_res::[[FOLDED_SYM]] : tensor<2x1x1x1x[[QELEMTYPE_OUT]]>
    // CHECK:      return
    // CHECK:  }
    // CHECK:  func.func @main() -> memref<2x1x1x1x[[QELEMTYPE_OUT]]> {
    // CHECK:      [[CST:%.+]] = const.Load @init_res::[[FOLDED_SYM]] -> memref<2x1x1x1x[[QELEMTYPE_OUT]]>
    // CHECK:      return [[CST]]
    // CHECK:  }
}

// -----

// CHECK-LABEL: @TransformationConvertElemType
module @TransformationConvertElemType {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output" : tensor<32x16x3x3xui8>
    }

    const.Data @ov_bin {
        const.Rodata @value dense<1.000000e+00> : tensor<32x16x3x3xf16>
    }
    func.func @main() -> memref<32x16x3x3xui8> {
        %cst = const.Declare memref<32x16x3x3xui8> = ref<@ov_bin::@value> : tensor<32x16x3x3xf16>, [#const.ConvertElemType<ui8>]
        return %cst : memref<32x16x3x3xui8>
    }

    // CHECK:  const.Data @ov_bin {
    // CHECK:      const.Rodata [[CST_SYM:@.+]] dense<1.000000e+00> : tensor<32x16x3x3xf16>
    // CHECK:  }
    // CHECK:  const.Data @init_res {
    // CHECK:      const.Ref [[FOLDED_SYM:@.+]] : tensor<32x16x3x3xui8>
    // CHECK:  }
    // CHECK:  func.func @init() {
    // CHECK:      [[LOAD:%.+]] = const.Load @ov_bin::[[CST_SYM]] -> tensor<32x16x3x3xf16>
    // CHECK:      [[CONVERT:%.+]] = IE.Convert([[LOAD]]) {dstElemType = ui8} : tensor<32x16x3x3xf16> -> tensor<32x16x3x3xui8>
    // CHECK:      const.Store [[CONVERT]], @init_res::[[FOLDED_SYM]] : tensor<32x16x3x3xui8>
    // CHECK:      return
    // CHECK:  }
    // CHECK:  func.func @main() -> memref<32x16x3x3xui8> {
    // CHECK:      [[CST:%.+]] = const.Load @init_res::[[FOLDED_SYM]] -> memref<32x16x3x3xui8>
    // CHECK:      return [[CST]]
    // CHECK:  }
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>
// CHECK:  [[QELEMTYPE:!.+]] = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK: @TransformationDequantize
module @TransformationDequantize {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output" : tensor<32x16x3x3xf16>
    }

    const.Data @ov_bin {
        const.Rodata @value dense<1> : tensor<32x16x3x3xui8>
    }
    func.func @main() -> memref<32x16x3x3xf16> {
        %cst = const.Declare memref<32x16x3x3xf16> = ref<@ov_bin::@value> : tensor<32x16x3x3xui8>, [#const.QuantCast<!qElemType>, #const.Dequantize]
        return %cst : memref<32x16x3x3xf16>
    }

    // CHECK:  const.Data @ov_bin {
    // CHECK:      const.Rodata [[CST_SYM:@.+]] dense<1> : tensor<32x16x3x3xui8>
    // CHECK:  }
    // CHECK:  const.Data @init_res {
    // CHECK:      const.Ref [[FOLDED_SYM:@.+]] : tensor<32x16x3x3xf16>
    // CHECK:  }
    // CHECK:  func.func @init() {
    // CHECK:      [[LOAD:%.+]] = const.Load @ov_bin::[[CST_SYM]] -> tensor<32x16x3x3xui8>
    // CHECK:      [[QUANT_CAST:%.+]] = IE.QuantizeCast([[LOAD]]) {dstElemType = [[QELEMTYPE]]} : tensor<32x16x3x3xui8> -> tensor<32x16x3x3x[[QELEMTYPE]]>
    // CHECK:      [[DEQUANTIZE:%.+]] = IE.Dequantize([[QUANT_CAST]]) {dstElemType = f16} : tensor<32x16x3x3x[[QELEMTYPE]]> -> tensor<32x16x3x3xf16>
    // CHECK:      const.Store [[DEQUANTIZE]], @init_res::[[FOLDED_SYM]] : tensor<32x16x3x3xf16>
    // CHECK:      return
    // CHECK:  }
    // CHECK:  func.func @main() -> memref<32x16x3x3xf16> {
    // CHECK:      [[CST:%.+]] = const.Load @init_res::[[FOLDED_SYM]] -> memref<32x16x3x3xf16>
    // CHECK:      return [[CST]]
    // CHECK:  }
}

// -----

// CHECK-LABEL: @TransformationExpandDilated
module @TransformationExpandDilated {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output" : tensor<32x16x5x5xf16>
    }

    const.Data @ov_bin {
        const.Rodata @value dense<1.000000e+00> : tensor<32x16x3x3xf16>
    }
    func.func @main() -> memref<32x16x5x5xf16> {
        %cst = const.Declare memref<32x16x5x5xf16> = ref<@ov_bin::@value> : tensor<32x16x3x3xf16>, [#const.ExpandDilated<[2, 2]>]
        return %cst : memref<32x16x5x5xf16>
    }

    // CHECK:  const.Data @ov_bin {
    // CHECK:      const.Rodata [[CST_SYM:@.+]] dense<1.000000e+00> : tensor<32x16x3x3xf16>
    // CHECK:  }
    // CHECK:  const.Data @init_res {
    // CHECK:      const.Ref [[FOLDED_SYM:@.+]] : tensor<32x16x5x5xf16>
    // CHECK:  }
    // CHECK:  func.func @init() {
    // CHECK:      [[LOAD:%.+]] = const.Load @ov_bin::[[CST_SYM]] -> tensor<32x16x3x3xf16>
    // CHECK:      [[EXPAND_DILATED:%.+]] = IE.ExpandDilated([[LOAD]]) {dilations = [2, 2]} : tensor<32x16x3x3xf16> -> tensor<32x16x5x5xf16>
    // CHECK:      const.Store [[EXPAND_DILATED]], @init_res::[[FOLDED_SYM]] : tensor<32x16x5x5xf16>
    // CHECK:      return
    // CHECK:  }
    // CHECK:  func.func @main() -> memref<32x16x5x5xf16> {
    // CHECK:      [[CST:%.+]] = const.Load @init_res::[[FOLDED_SYM]] -> memref<32x16x5x5xf16>
    // CHECK:      return [[CST]]
    // CHECK:  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TransformationExpandDilated
module @TransformationExpandDilated {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output" : tensor<32x16x3x3xf16>
    }

    const.Data @ov_bin {
        const.Rodata @value dense<1.000000e+00> : tensor<32x16x3x3xf16>
    }
    func.func @main() -> memref<32x16x3x3xf16, #NHWC> {
        %cst = const.Declare memref<32x16x3x3xf16, #NHWC> = ref<@ov_bin::@value> : tensor<32x16x3x3xf16>, [#const.LayoutCast<#NHWC>]
        return %cst : memref<32x16x3x3xf16, #NHWC>
    }

    // CHECK:  const.Data @ov_bin {
    // CHECK:      const.Rodata [[CST_SYM:@.+]] dense<1.000000e+00> : tensor<32x16x3x3xf16>
    // CHECK:  }
    // CHECK:  const.Data @init_res {
    // CHECK:      const.Ref [[FOLDED_SYM:@.+]] : tensor<32x16x3x3xf16, {order = #NHWC}>
    // CHECK:  }
    // CHECK:  func.func @init() {
    // CHECK:      [[LOAD:%.+]] = const.Load @ov_bin::[[CST_SYM]] -> tensor<32x16x3x3xf16>
    // CHECK:      [[EXPAND_DILATED:%.+]] = IE.LayoutCast([[LOAD]]) {dst_order = #NHWC} : tensor<32x16x3x3xf16> -> tensor<32x16x3x3xf16, {order = #NHWC}>
    // CHECK:      const.Store [[EXPAND_DILATED]], @init_res::[[FOLDED_SYM]] : tensor<32x16x3x3xf16, {order = #NHWC}>
    // CHECK:      return
    // CHECK:  }
    // CHECK:  func.func @main() -> memref<32x16x3x3xf16, #NHWC> {
    // CHECK:      [[CST:%.+]] = const.Load @init_res::[[FOLDED_SYM]] -> memref<32x16x3x3xf16, #NHWC>
    // CHECK:      return [[CST]]
    // CHECK:  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TransformationMemPermute
module @TransformationMemPermute {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output" : tensor<32x16x3x3xf16>
    }

    const.Data @ov_bin {
        const.Rodata @value dense<1.000000e+00> : tensor<32x16x3x3xf16>
    }
    func.func @main() -> memref<32x16x3x3xf16, #NHWC> {
        %cst = const.Declare memref<32x16x3x3xf16, #NHWC> = ref<@ov_bin::@value> : tensor<32x16x3x3xf16>, [#const.MemPermute<#NHWC, #NHWC>]
        return %cst : memref<32x16x3x3xf16, #NHWC>
    }

    // CHECK:  const.Data @ov_bin {
    // CHECK:      const.Rodata [[CST_SYM:@.+]] dense<1.000000e+00> : tensor<32x16x3x3xf16>
    // CHECK:  }
    // CHECK:  const.Data @init_res {
    // CHECK:      const.Ref [[FOLDED_SYM:@.+]] : tensor<32x16x3x3xf16, {order = #NHWC}>
    // CHECK:  }
    // CHECK:  func.func @init() {
    // CHECK:      [[LOAD:%.+]] = const.Load @ov_bin::[[CST_SYM]] -> tensor<32x16x3x3xf16>
    // CHECK:      [[MEM_PERMUTE:%.+]] = IE.MemPermute([[LOAD]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<32x16x3x3xf16> -> tensor<32x16x3x3xf16, {order = #NHWC}>
    // CHECK:      const.Store [[MEM_PERMUTE]], @init_res::[[FOLDED_SYM]] : tensor<32x16x3x3xf16, {order = #NHWC}>
    // CHECK:      return
    // CHECK:  }
    // CHECK:  func.func @main() -> memref<32x16x3x3xf16, #NHWC> {
    // CHECK:      [[CST:%.+]] = const.Load @init_res::[[FOLDED_SYM]] -> memref<32x16x3x3xf16, #NHWC>
    // CHECK:      return [[CST]]
    // CHECK:  }
}

// -----

// CHECK-LABEL: @TransformationPadWithZero
module @TransformationPadWithZero {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output" : tensor<32x16x4x4xf16>
    }

    const.Data @ov_bin {
        const.Rodata @value dense<1.000000e+00> : tensor<32x16x3x3xf16>
    }
    func.func @main() -> memref<32x16x4x4xf16> {
        %cst = const.Declare memref<32x16x4x4xf16> = ref<@ov_bin::@value> : tensor<32x16x3x3xf16>, [#const.PadWithZero<[0, 0, 0, 0], [0, 0, 1, 1]>]
        return %cst : memref<32x16x4x4xf16>
    }

    // CHECK:  const.Data @ov_bin {
    // CHECK:      const.Rodata [[CST_SYM:@.+]] dense<1.000000e+00> : tensor<32x16x3x3xf16>
    // CHECK:  }
    // CHECK:  const.Data @init_res {
    // CHECK:      const.Ref [[FOLDED_SYM:@.+]] : tensor<32x16x4x4xf16>
    // CHECK:  }
    // CHECK:  func.func @init() {
    // CHECK:      [[LOAD:%.+]] = const.Load @ov_bin::[[CST_SYM]] -> tensor<32x16x3x3xf16>
    // CHECK:      [[PAD:%.+]] = IE.Pad([[LOAD]]) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 0, 0, 0], pads_end_attr = [0, 0, 1, 1]} : tensor<32x16x3x3xf16> -> tensor<32x16x4x4xf16>
    // CHECK:      const.Store [[PAD]], @init_res::[[FOLDED_SYM]] : tensor<32x16x4x4xf16>
    // CHECK:      return
    // CHECK:  }
    // CHECK:  func.func @main() -> memref<32x16x4x4xf16> {
    // CHECK:      [[CST:%.+]] = const.Load @init_res::[[FOLDED_SYM]] -> memref<32x16x4x4xf16>
    // CHECK:      return [[CST]]
    // CHECK:  }
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>
// CHECK:  [[QELEMTYPE:!.+]] = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK: @TransformationQuantCast
module @TransformationQuantCast {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output" : tensor<32x16x3x3xui8>
    }

    const.Data @ov_bin {
        const.Rodata @value dense<1> : tensor<32x16x3x3xui8>
    }
    func.func @main() -> memref<32x16x3x3x!qElemType> {
        %cst = const.Declare memref<32x16x3x3x!qElemType> = ref<@ov_bin::@value> : tensor<32x16x3x3xui8>, [#const.QuantCast<!qElemType>]
        return %cst : memref<32x16x3x3x!qElemType>
    }

    // CHECK:  const.Data @ov_bin {
    // CHECK:      const.Rodata [[CST_SYM:@.+]] dense<1> : tensor<32x16x3x3xui8>
    // CHECK:  }
    // CHECK:  const.Data @init_res {
    // CHECK:      const.Ref [[FOLDED_SYM:@.+]] : tensor<32x16x3x3x[[QELEMTYPE]]>
    // CHECK:  }
    // CHECK:  func.func @init() {
    // CHECK:      [[LOAD:%.+]] = const.Load @ov_bin::[[CST_SYM]] -> tensor<32x16x3x3xui8>
    // CHECK:      [[QUANT_CAST:%.+]] = IE.QuantizeCast([[LOAD]]) {dstElemType = [[QELEMTYPE]]} : tensor<32x16x3x3xui8> -> tensor<32x16x3x3x[[QELEMTYPE]]>
    // CHECK:      const.Store [[QUANT_CAST]], @init_res::[[FOLDED_SYM]] : tensor<32x16x3x3x[[QELEMTYPE]]>
    // CHECK:      return
    // CHECK:  }
    // CHECK:  func.func @main() -> memref<32x16x3x3x[[QELEMTYPE]]> {
    // CHECK:      [[CST:%.+]] = const.Load @init_res::[[FOLDED_SYM]] -> memref<32x16x3x3x[[QELEMTYPE]]>
    // CHECK:      return [[CST]]
    // CHECK:  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TransformationReorder
module @TransformationReorder {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output" : tensor<32x16x3x3xf16>
    }

    const.Data @ov_bin {
        const.Rodata @value dense<1.000000e+00> : tensor<32x16x3x3xf16>
    }
    func.func @main() -> memref<32x16x3x3xf16, #NHWC> {
        %cst = const.Declare memref<32x16x3x3xf16, #NHWC> = ref<@ov_bin::@value> : tensor<32x16x3x3xf16>, [#const.Reorder<#NHWC>]
        return %cst : memref<32x16x3x3xf16, #NHWC>
    }

    // CHECK:  const.Data @ov_bin {
    // CHECK:      const.Rodata [[CST_SYM:@.+]] dense<1.000000e+00> : tensor<32x16x3x3xf16>
    // CHECK:  }
    // CHECK:  const.Data @init_res {
    // CHECK:      const.Ref [[FOLDED_SYM:@.+]] : tensor<32x16x3x3xf16, {order = #NHWC}>
    // CHECK:  }
    // CHECK:  func.func @init() {
    // CHECK:      [[LOAD:%.+]] = const.Load @ov_bin::[[CST_SYM]] -> tensor<32x16x3x3xf16>
    // CHECK:      [[REORDER:%.+]] = IE.Reorder([[LOAD]]) {dstOrder = #NHWC} : tensor<32x16x3x3xf16> -> tensor<32x16x3x3xf16, {order = #NHWC}>
    // CHECK:      const.Store [[REORDER]], @init_res::[[FOLDED_SYM]] : tensor<32x16x3x3xf16, {order = #NHWC}>
    // CHECK:      return
    // CHECK:  }
    // CHECK:  func.func @main() -> memref<32x16x3x3xf16, #NHWC> {
    // CHECK:      [[CST:%.+]] = const.Load @init_res::[[FOLDED_SYM]] -> memref<32x16x3x3xf16, #NHWC>
    // CHECK:      return [[CST]]
    // CHECK:  }
}

// -----

// CHECK-LABEL: @TransformationRescale
module @TransformationRescale {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output" : tensor<32x16x3x3xf16>
    }

    const.Data @ov_bin {
        const.Rodata @value dense<1.000000e+00> : tensor<32x16x3x3xf16>
    }
    func.func @main() -> memref<32x16x3x3xf16> {
        %cst = const.Declare memref<32x16x3x3xf16> = ref<@ov_bin::@value> : tensor<32x16x3x3xf16>, [#const.Rescale<2.0 : f32>]
        return %cst : memref<32x16x3x3xf16>
    }

    // CHECK:  const.Data @ov_bin {
    // CHECK:      const.Rodata [[CST_SYM:@.+]] dense<1.000000e+00> : tensor<32x16x3x3xf16>
    // CHECK:  }
    // CHECK:  const.Data @init_res {
    // CHECK:      const.Ref [[FOLDED_SYM:@.+]] : tensor<32x16x3x3xf16>
    // CHECK:  }
    // CHECK:  func.func @init() {
    // CHECK:      [[LOAD:%.+]] = const.Load @ov_bin::[[CST_SYM]] -> tensor<32x16x3x3xf16>
    // CHECK:      [[SCALE:%.+]] = const.Declare tensor<1xf32> = dense<2.000000e+00> : tensor<1xf32>
    // CHECK:      [[MULTIPLY:%.+]] = IE.Multiply([[LOAD]], [[SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<32x16x3x3xf16>, tensor<1xf32> -> tensor<32x16x3x3xf16>
    // CHECK:      const.Store [[MULTIPLY]], @init_res::[[FOLDED_SYM]] : tensor<32x16x3x3xf16>
    // CHECK:      return
    // CHECK:  }
    // CHECK:  func.func @main() -> memref<32x16x3x3xf16> {
    // CHECK:      [[CST:%.+]] = const.Load @init_res::[[FOLDED_SYM]] -> memref<32x16x3x3xf16>
    // CHECK:      return [[CST]]
    // CHECK:  }
}

// -----

// CHECK-LABEL: @TransformationReshape
module @TransformationReshape {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output" : tensor<32x16x1x9xf16>
    }

    const.Data @ov_bin {
        const.Rodata @value dense<1.000000e+00> : tensor<32x16x3x3xf16>
    }
    func.func @main() -> memref<32x16x1x9xf16> {
        %cst = const.Declare memref<32x16x1x9xf16> = ref<@ov_bin::@value> : tensor<32x16x3x3xf16>, [#const.Reshape<[32, 16, 1, 9]>]
        return %cst : memref<32x16x1x9xf16>
    }

    // CHECK:  const.Data @ov_bin {
    // CHECK:      const.Rodata [[CST_SYM:@.+]] dense<1.000000e+00> : tensor<32x16x3x3xf16>
    // CHECK:  }
    // CHECK:  const.Data @init_res {
    // CHECK:      const.Ref [[FOLDED_SYM:@.+]] : tensor<32x16x1x9xf16>
    // CHECK:  }
    // CHECK:  func.func @init() {
    // CHECK:      [[LOAD:%.+]] = const.Load @ov_bin::[[CST_SYM]] -> tensor<32x16x3x3xf16>
    // CHECK:      [[RESHAPE:%.+]] = IE.Reshape([[LOAD]]) {shape_value = [32, 16, 1, 9]} : tensor<32x16x3x3xf16> -> tensor<32x16x1x9xf16>
    // CHECK:      const.Store [[RESHAPE]], @init_res::[[FOLDED_SYM]] : tensor<32x16x1x9xf16>
    // CHECK:      return
    // CHECK:  }
    // CHECK:  func.func @main() -> memref<32x16x1x9xf16> {
    // CHECK:      [[CST:%.+]] = const.Load @init_res::[[FOLDED_SYM]] -> memref<32x16x1x9xf16>
    // CHECK:      return [[CST]]
    // CHECK:  }
}

// -----

// CHECK-LABEL: @TransformationScalarMultInverse
module @TransformationScalarMultInverse {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output" : tensor<32x16x3x3xf16>
    }

    const.Data @ov_bin {
        const.Rodata @value dense<1.000000e+00> : tensor<32x16x3x3xf16>
    }
    func.func @main() -> memref<32x16x3x3xf16> {
        %cst = const.Declare memref<32x16x3x3xf16> = ref<@ov_bin::@value> : tensor<32x16x3x3xf16>, [#const.ScalarMultInverse]
        return %cst : memref<32x16x3x3xf16>
    }

    // CHECK:  const.Data @ov_bin {
    // CHECK:      const.Rodata [[CST_SYM:@.+]] dense<1.000000e+00> : tensor<32x16x3x3xf16>
    // CHECK:  }
    // CHECK:  const.Data @init_res {
    // CHECK:      const.Ref [[FOLDED_SYM:@.+]] : tensor<32x16x3x3xf16>
    // CHECK:  }
    // CHECK:  func.func @init() {
    // CHECK:      [[LOAD:%.+]] = const.Load @ov_bin::[[CST_SYM]] -> tensor<32x16x3x3xf16>
    // CHECK:      [[INVERSE:%.+]] = const.Declare tensor<1xf16> = dense<1.000000e+00> : tensor<1xf16>
    // CHECK:      [[DIVIDE:%.+]] = IE.Divide([[INVERSE]], [[LOAD]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xf16>, tensor<32x16x3x3xf16> -> tensor<32x16x3x3xf16>
    // CHECK:      const.Store [[DIVIDE]], @init_res::[[FOLDED_SYM]] : tensor<32x16x3x3xf16>
    // CHECK:      return
    // CHECK:  }
    // CHECK:  func.func @main() -> memref<32x16x3x3xf16> {
    // CHECK:      [[CST:%.+]] = const.Load @init_res::[[FOLDED_SYM]] -> memref<32x16x3x3xf16>
    // CHECK:      return [[CST]]
    // CHECK:  }
}

// -----

// CHECK-LABEL: @TransformationSubView
module @TransformationSubView {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output" : tensor<8x8x3x3xf16>
    }

    const.Data @ov_bin {
        const.Rodata @value dense<1.000000e+00> : tensor<32x16x3x3xf16>
    }
    func.func @main() -> memref<8x8x3x3xf16> {
        %cst = const.Declare memref<8x8x3x3xf16> = ref<@ov_bin::@value> : tensor<32x16x3x3xf16>, [#const.SubView<[16, 0, 0, 0], [8, 8, 3,3]>]
        return %cst : memref<8x8x3x3xf16>
    }

    // CHECK:  const.Data @ov_bin {
    // CHECK:      const.Rodata [[CST_SYM:@.+]] dense<1.000000e+00> : tensor<32x16x3x3xf16>
    // CHECK:  }
    // CHECK:  const.Data @init_res {
    // CHECK:      const.Ref [[FOLDED_SYM:@.+]] : tensor<8x8x3x3xf16>
    // CHECK:  }
    // CHECK:  func.func @init() {
    // CHECK:      [[LOAD:%.+]] = const.Load @ov_bin::[[CST_SYM]] -> tensor<32x16x3x3xf16>
    // CHECK:      [[SLICE:%.+]] = IE.Slice [[LOAD]] [16, 0, 0, 0] [8, 8, 3, 3] : tensor<32x16x3x3xf16> to tensor<8x8x3x3xf16>
    // CHECK:      const.Store [[SLICE]], @init_res::[[FOLDED_SYM]] : tensor<8x8x3x3xf16>
    // CHECK:      return
    // CHECK:  }
    // CHECK:  func.func @main() -> memref<8x8x3x3xf16> {
    // CHECK:      [[CST:%.+]] = const.Load @init_res::[[FOLDED_SYM]] -> memref<8x8x3x3xf16>
    // CHECK:      return [[CST]]
    // CHECK:  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TransformationSubView
module @TransformationSubView {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output" : tensor<32x3x3x16xf16>
    }

    const.Data @ov_bin {
        const.Rodata @value dense<1.000000e+00> : tensor<32x16x3x3xf16>
    }
    func.func @main() -> memref<32x3x3x16xf16> {
        %cst = const.Declare memref<32x3x3x16xf16> = ref<@ov_bin::@value> : tensor<32x16x3x3xf16>, [#const.Transpose<#NHWC>]
        return %cst : memref<32x3x3x16xf16>
    }

    // CHECK:  const.Data @ov_bin {
    // CHECK:      const.Rodata [[CST_SYM:@.+]] dense<1.000000e+00> : tensor<32x16x3x3xf16>
    // CHECK:  }
    // CHECK:  const.Data @init_res {
    // CHECK:      const.Ref [[FOLDED_SYM:@.+]] : tensor<32x3x3x16xf16>
    // CHECK:  }
    // CHECK:  func.func @init() {
    // CHECK:      [[LOAD:%.+]] = const.Load @ov_bin::[[CST_SYM]] -> tensor<32x16x3x3xf16>
    // CHECK:      [[TRANSPOSE:%.+]] = IE.Transpose([[LOAD]]) {order_value = #NHWC} : tensor<32x16x3x3xf16> -> tensor<32x3x3x16xf16>
    // CHECK:      const.Store [[TRANSPOSE]], @init_res::[[FOLDED_SYM]] : tensor<32x3x3x16xf16>
    // CHECK:      return
    // CHECK:  }
    // CHECK:  func.func @main() -> memref<32x3x3x16xf16> {
    // CHECK:      [[CST:%.+]] = const.Load @init_res::[[FOLDED_SYM]] -> memref<32x3x3x16xf16>
    // CHECK:      return [[CST]]
    // CHECK:  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MultipleConstants
module @MultipleConstants {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output1" : tensor<32x16x3x3xf16>
        DataInfo "output2" : tensor<8x8x3x3xf16>
        DataInfo "output3" : tensor<1x16x20x20xf16>
    }

    const.Data @ov_bin {
        const.Rodata @value1 dense<1.000000e+00> : tensor<32x16x3x3xf16>
        const.Rodata @value2 dense<2.000000e+00> : tensor<1x16x20x20xf16>
    }
    const.Data @npu_bin {
        const.Rodata @value dense<0.000000e+00> : tensor<32x16x3x3xf16>
    }
    func.func @main() -> (memref<32x16x3x3xf16>, memref<8x8x3x3xf16, #NHWC>, memref<1x16x20x20xf16>) {
        %cst0 = const.Declare memref<32x16x3x3xf16> = ref<@npu_bin::@value> : tensor<32x16x3x3xf16>, [#const.Add<1.0>]
        %cst1 = const.Declare memref<8x8x3x3xf16, #NHWC> = ref<@ov_bin::@value1> : tensor<32x16x3x3xf16>, [#const.Add<1.0>, #const.SubView<[16, 0, 0, 0], [8, 8, 3,3]>, #const.Reorder<#NHWC>]
        %cst2 = const.Declare memref<1x16x20x20xf16> = ref<@ov_bin::@value2> : tensor<1x16x20x20xf16>, [#const.Rescale<2.0 : f32>]

        return %cst0, %cst1, %cst2 : memref<32x16x3x3xf16>, memref<8x8x3x3xf16, #NHWC>, memref<1x16x20x20xf16>
    }

    // CHECK:  const.Data @ov_bin {
    // CHECK:      const.Rodata [[CST_OV_SYM1:@.+]] dense<1.000000e+00> : tensor<32x16x3x3xf16>
    // CHECK:      const.Rodata [[CST_OV_SYM2:@.+]] dense<2.000000e+00> : tensor<1x16x20x20xf16>
    // CHECK:  }
    // CHECK:  const.Data @init_res {
    // CHECK:      const.Ref [[FOLDED_SYM1:@.+]] : tensor<8x8x3x3xf16, {order = #NHWC}>
    // CHECK:      const.Ref [[FOLDED_SYM2:@.+]] : tensor<1x16x20x20xf16>
    // CHECK:  }
    // CHECK:  const.Data @npu_bin {
    // CHECK:      const.Rodata [[CST_NPU_SYM:@.+]] dense<0.000000e+00> : tensor<32x16x3x3xf16>
    // CHECK:  }

    // CHECK:  func.func @init() {
    // CHECK:      [[LOAD1:%.+]] = const.Load @ov_bin::[[CST_OV_SYM1]] -> tensor<32x16x3x3xf16>
    // CHECK:      [[BIAS:%.+]] = const.Declare tensor<1xf32> = dense<1.000000e+00> : tensor<1xf32>
    // CHECK:      [[ADD:%.+]] = IE.Add([[LOAD1]], [[BIAS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<32x16x3x3xf16>, tensor<1xf32> -> tensor<32x16x3x3xf16>
    // CHECK:      [[SLICE:%.+]] = IE.Slice [[ADD]] [16, 0, 0, 0] [8, 8, 3, 3] : tensor<32x16x3x3xf16> to tensor<8x8x3x3xf16>
    // CHECK:      [[REORDER:%.+]] = IE.Reorder([[SLICE]]) {dstOrder = #NHWC} : tensor<8x8x3x3xf16> -> tensor<8x8x3x3xf16, {order = #NHWC}>
    // CHECK:      const.Store [[REORDER]], @init_res::[[FOLDED_SYM1]] : tensor<8x8x3x3xf16, {order = #NHWC}>

    // CHECK:      [[LOAD2:%.+]] = const.Load @ov_bin::@value2 -> tensor<1x16x20x20xf16>
    // CHECK:      [[SCALE:%.+]] = const.Declare tensor<1xf32> = dense<2.000000e+00> : tensor<1xf32>
    // CHECK:      [[MULTIPLY:%.+]] = IE.Multiply([[LOAD2]], [[SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x20x20xf16>, tensor<1xf32> -> tensor<1x16x20x20xf16>
    // CHECK:      const.Store [[MULTIPLY]], @init_res::[[FOLDED_SYM2]] : tensor<1x16x20x20xf16>
    // CHECK:      return
    // CHECK:  }

    // CHECK:  func.func @main() -> (memref<32x16x3x3xf16>, memref<8x8x3x3xf16, #NHWC>, memref<1x16x20x20xf16>) {
    // CHECK:      [[CST0:%.+]] = const.Declare memref<32x16x3x3xf16> = ref<@npu_bin::@value> : tensor<32x16x3x3xf16>, [#const.Add<1.000000e+00 : f64>]
    // CHECK:      [[CST1:%.+]] = const.Load @init_res::[[FOLDED_SYM1]] -> memref<8x8x3x3xf16, #NHWC>
    // CHECK:      [[CST2:%.+]] = const.Load @init_res::[[FOLDED_SYM2]] -> memref<1x16x20x20xf16>
    // CHECK:      return [[CST0]], [[CST1]], [[CST2]]
    // CHECK:  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReusedSymbol
module @ReusedSymbol {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output1" : tensor<32x16x3x3xf16>
        DataInfo "output2" : tensor<8x8x3x3xf16>
    }

    const.Data @ov_bin {
        const.Rodata @value dense<1.000000e+00> : tensor<32x16x3x3xf16>
    }

    func.func @main() -> (memref<32x16x3x3xf16, #NHWC>, memref<8x8x3x3xf16>) {
        %cst0 = const.Declare memref<32x16x3x3xf16, #NHWC> = ref<@ov_bin::@value> : tensor<32x16x3x3xf16>, [#const.Reorder<#NHWC>]
        %cst1 = const.Declare memref<8x8x3x3xf16> = ref<@ov_bin::@value> : tensor<32x16x3x3xf16>, [#const.SubView<[16, 0, 0, 0], [8, 8, 3, 3]>]

        return %cst0, %cst1 : memref<32x16x3x3xf16, #NHWC>, memref<8x8x3x3xf16>
    }

    // CHECK:  const.Data @ov_bin {
    // CHECK:      const.Rodata [[CST_SYM:@.+]] dense<1.000000e+00> : tensor<32x16x3x3xf16>
    // CHECK:  }
    // CHECK:  const.Data @init_res {
    // CHECK:      const.Ref [[FOLDED_SYM1:@.+]] : tensor<32x16x3x3xf16, {order = #NHWC}>
    // CHECK:      const.Ref [[FOLDED_SYM2:@.+]] : tensor<8x8x3x3xf16>
    // CHECK:  }

    // CHECK:  func.func @init() {
    // CHECK:      [[LOAD1:%.+]] = const.Load @ov_bin::[[CST_SYM]] -> tensor<32x16x3x3xf16>
    // CHECK:      [[REORDER:%.+]] = IE.Reorder([[LOAD1]]) {dstOrder = #NHWC} : tensor<32x16x3x3xf16> -> tensor<32x16x3x3xf16, {order = #NHWC}>
    // CHECK:      const.Store [[REORDER]], @init_res::[[FOLDED_SYM1]] : tensor<32x16x3x3xf16, {order = #NHWC}>
    // CHECK:      [[LOAD2:%.+]] = const.Load @ov_bin::@value -> tensor<32x16x3x3xf16>
    // CHECK:      [[SLICE:%.+]] = IE.Slice [[LOAD2]] [16, 0, 0, 0] [8, 8, 3, 3] : tensor<32x16x3x3xf16> to tensor<8x8x3x3xf16>
    // CHECK:      const.Store [[SLICE]], @init_res::[[FOLDED_SYM2]] : tensor<8x8x3x3xf16>
    // CHECK:      return
    // CHECK:  }

    // CHECK:  func.func @main() -> (memref<32x16x3x3xf16, #NHWC>, memref<8x8x3x3xf16>) {
    // CHECK:      [[CST0:%.+]] = const.Load @init_res::[[FOLDED_SYM1]] -> memref<32x16x3x3xf16, #NHWC>
    // CHECK:      [[CST1:%.+]] = const.Load @init_res::[[FOLDED_SYM2]] -> memref<8x8x3x3xf16>
    // CHECK:      return [[CST0]], [[CST1]]
    // CHECK:  }
}
