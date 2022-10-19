// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --convert-sw-layers-to-VPUIP %s | FileCheck %s

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @Test attributes {VPU.compilationMode = "ReferenceHW"} {

// CHECK: VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]


// CHECK: module @VPU.SW {
// CHECK-NEXT:   func private @builtin_SoftMax(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, i64) attributes {VPU.kernel_code = "single_shave_softmax.cpp", VPU.kernel_entry = "singleShaveSoftmax"}
// CHECK-NEXT:   func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT: }

func @SingleSWLayer(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 3} : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    return %0: tensor<1x1x1x1000xf16>

// CHECK: [[ARG0:%.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<1x1x1x1000xf16> to memref<1x1x1x1000xf16>
// CHECK: [[VAR0:%.*]] = memref.alloc() : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
// CHECK: [[VAR1:%.*]] = VPUIP.Copy inputs([[ARG0]] : memref<1x1x1x1000xf16>) outputs([[VAR0]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>

// CHECK: [[VAR2:%.*]] = memref.alloc() : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
// CHECK: [[VAR3:%.*]] = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_SoftMax inputs([[VAR1]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs([[VAR2]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>  {
// CHECK: ^bb0(%arg1: memref<1x1x1x1000xf16, [@CMX_NN, 0]>, %arg2: memref<1x1x1x1000xf16, [@CMX_NN, 0]>):
// CHECK:   VPUIP.SW.Kernel.run {attrs = [0]}(%arg1, %arg2) : memref<1x1x1x1000xf16, [@CMX_NN, 0]>, memref<1x1x1x1000xf16, [@CMX_NN, 0]>
// CHECK: }

// CHECK: [[VAR4:%.*]] = memref.alloc() : memref<1x1x1x1000xf16>
// CHECK: [[VAR5:%.*]] = VPUIP.Copy inputs([[VAR3]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs([[VAR4]] : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>
// CHECK: [[VAR6:%.*]] = builtin.unrealized_conversion_cast [[VAR5]] : memref<1x1x1x1000xf16> to tensor<1x1x1x1000xf16>
// CHECK: return [[VAR6]] : tensor<1x1x1x1000xf16>

}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @Test attributes {VPU.compilationMode = "ReferenceHW"} {

// CHECK: VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]

// CHECK: module @VPU.SW {
// CHECK-NEXT:   func private @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder_fp16.cpp", VPU.kernel_entry = "reorder_fp16"}
// CHECK-NEXT:   func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT: }

func @MemPermuteSWLayer(%arg0: tensor<1x2x3x4xf16>) -> tensor<1x3x4x2xf16> {
    %0 = VPU.MemPermute(%arg0) {mem_perm = #NHWC, dst_order = #NHWC} : tensor<1x2x3x4xf16> -> tensor<1x3x4x2xf16>
    return %0: tensor<1x3x4x2xf16>

// CHECK: [[ARG0:%.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<1x2x3x4xf16> to memref<1x2x3x4xf16>
// CHECK: [[VAR0:%.*]] = memref.alloc() : memref<1x2x3x4xf16, [@CMX_NN, 0]>
// CHECK: [[VAR1:%.*]] = VPUIP.Copy inputs([[ARG0]] : memref<1x2x3x4xf16>) outputs([[VAR0]] : memref<1x2x3x4xf16, [@CMX_NN, 0]>) -> memref<1x2x3x4xf16, [@CMX_NN, 0]>

// CHECK: [[VAR2:%.*]] = memref.alloc() : memref<1x3x4x2xf16, [@CMX_NN, 0]>
// CHECK: [[VAR3:%.*]] = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_MemPermute inputs([[VAR1]] : memref<1x2x3x4xf16, [@CMX_NN, 0]>) outputs([[VAR2]] : memref<1x3x4x2xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x3x4x2xf16, [@CMX_NN, 0]>  {
// CHECK: ^bb0(%arg1: memref<1x2x3x4xf16, [@CMX_NN, 0]>, %arg2: memref<1x3x4x2xf16, [@CMX_NN, 0]>):  // no predecessors
// CHECK:   VPUIP.SW.Kernel.run {attrs = [
// CHECK:   [2, 0, 1, 3]
// CHECK:   ]}(%arg1, %arg2) : memref<1x2x3x4xf16, [@CMX_NN, 0]>, memref<1x3x4x2xf16, [@CMX_NN, 0]>
// CHECK: }

// CHECK: [[VAR4:%.*]] = memref.alloc() : memref<1x3x4x2xf16>
// CHECK: [[VAR5:%.*]] = VPUIP.Copy inputs([[VAR3]] : memref<1x3x4x2xf16, [@CMX_NN, 0]>) outputs([[VAR4]] : memref<1x3x4x2xf16>) -> memref<1x3x4x2xf16>
// CHECK: [[VAR6:%.*]] = builtin.unrealized_conversion_cast %5 : memref<1x3x4x2xf16> to tensor<1x3x4x2xf16>
// CHECK: return [[VAR6]] : tensor<1x3x4x2xf16>

}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @Test attributes {VPU.compilationMode = "ReferenceHW"} {

// CHECK: VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]


// CHECK: module @VPU.SW {
// CHECK-NEXT:   func private @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder_fp16.cpp", VPU.kernel_entry = "reorder_fp16"}
// CHECK-NEXT:   func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT: }

func @ReorderSWLayer(%arg0: tensor<1x2x3x4xf16>) -> tensor<1x2x3x4xf16, {order = #NHWC}> {
    %0 = VPU.MemPermute(%arg0) {mem_perm = #NHWC, dst_order = #NHWC} : tensor<1x2x3x4xf16> -> tensor<1x2x3x4xf16, {order = #NHWC}>
    return %0: tensor<1x2x3x4xf16, {order = #NHWC}>

// CHECK: [[ARG0:%.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<1x2x3x4xf16> to memref<1x2x3x4xf16>
// CHECK: [[VAR0:%.*]] = memref.alloc() : memref<1x2x3x4xf16, [@CMX_NN, 0]>
// CHECK: [[VAR1:%.*]] = VPUIP.Copy inputs([[ARG0]] : memref<1x2x3x4xf16>) outputs([[VAR0]] : memref<1x2x3x4xf16, [@CMX_NN, 0]>) -> memref<1x2x3x4xf16, [@CMX_NN, 0]>

// CHECK: [[VAR2:%.*]] = memref.alloc() : memref<1x2x3x4xf16, #NHWC, [@CMX_NN, 0]>
// CHECK: [[VAR3:%.*]] = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_MemPermute inputs([[VAR1]] : memref<1x2x3x4xf16, [@CMX_NN, 0]>) outputs([[VAR2]] : memref<1x2x3x4xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x2x3x4xf16, #NHWC, [@CMX_NN, 0]>  {
// CHECK: ^bb0(%arg1: memref<1x2x3x4xf16, [@CMX_NN, 0]>, %arg2: memref<1x2x3x4xf16, #NHWC, [@CMX_NN, 0]>):  // no predecessors
// CHECK:   VPUIP.SW.Kernel.run {attrs = [
// CHECK:   [2, 0, 1, 3]
// CHECK:   ]}(%arg1, %arg2) : memref<1x2x3x4xf16, [@CMX_NN, 0]>, memref<1x2x3x4xf16, #NHWC, [@CMX_NN, 0]>
// CHECK: }

// CHECK: [[VAR4:%.*]] = memref.alloc() : memref<1x2x3x4xf16, #NHWC>
// CHECK: [[VAR5:%.*]] = VPUIP.Copy inputs([[VAR3]] : memref<1x2x3x4xf16, #NHWC, [@CMX_NN, 0]>) outputs([[VAR4]] : memref<1x2x3x4xf16, #NHWC>) -> memref<1x2x3x4xf16, #NHWC>
// CHECK: [[VAR6:%.*]] = builtin.unrealized_conversion_cast [[VAR5]] : memref<1x2x3x4xf16, #NHWC> to tensor<1x2x3x4xf16, {order = #NHWC}>
// CHECK: return [[VAR6]] : tensor<1x2x3x4xf16, {order = #NHWC}>

}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @Test attributes {VPU.compilationMode = "ReferenceHW"} {
// CHECK: VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
// CHECK: module @VPU.SW  {
// CHECK-NEXT: func private @builtin_Sigmoid(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>) attributes {VPU.kernel_code = "sigmoid_fp16.c", VPU.kernel_entry = "sigmoid_fp16"}
// CHECK-NEXT: func private @builtin_SoftMax(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, i64) attributes {VPU.kernel_code = "single_shave_softmax.cpp", VPU.kernel_entry = "singleShaveSoftmax"}
// CHECK-NEXT: func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT: }

// CHECK: func @ThreeSWLayers(%arg0: tensor<1x1x1x2000xf16>) -> tensor<1x1x1x1000xf16> {
func @ThreeSWLayers(%arg0: tensor<1x1x1x2000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 3} : tensor<1x1x1x2000xf16> -> tensor<1x1x1x2000xf16>
    %1 = VPU.Sigmoid(%0) {axisInd = 3} : tensor<1x1x1x2000xf16> -> tensor<1x1x1x2000xf16>
    %2 = VPU.Slice %1 [0, 0, 0, 1000] [1, 1, 1, 1000] : tensor<1x1x1x2000xf16> to tensor<1x1x1x1000xf16>
    %3 = VPU.SoftMax(%2) {axisInd = 3} : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>

    return %3 : tensor<1x1x1x1000xf16>

// CHECK: [[ARG0:%.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<1x1x1x2000xf16> to memref<1x1x1x2000xf16>
// CHECK: [[VAR2:%.*]] = memref.alloc() : memref<1x1x1x2000xf16, [@CMX_NN, 0]>
// CHECK: [[VAR3:%.*]] = VPUIP.Copy inputs([[ARG0]] : memref<1x1x1x2000xf16>) outputs([[VAR2]] : memref<1x1x1x2000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x2000xf16, [@CMX_NN, 0]>
// CHECK: [[VAR4:%.*]] = memref.alloc() : memref<1x1x1x2000xf16, [@CMX_NN, 0]>
// CHECK: [[VAR5:%.*]] = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_SoftMax inputs([[VAR3]] : memref<1x1x1x2000xf16, [@CMX_NN, 0]>) outputs([[VAR4]] : memref<1x1x1x2000xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x2000xf16, [@CMX_NN, 0]>  {
// CHECK:       ^bb0(%arg1: memref<1x1x1x2000xf16, [@CMX_NN, 0]>, %arg2: memref<1x1x1x2000xf16, [@CMX_NN, 0]>):  // no predecessors
// CHECK:         VPUIP.SW.Kernel.run {attrs = [0]}(%arg1, %arg2) : memref<1x1x1x2000xf16, [@CMX_NN, 0]>, memref<1x1x1x2000xf16, [@CMX_NN, 0]>
// CHECK:       }
// CHECK: [[VAR20:%.*]] = memref.alloc() : memref<1x1x1x2000xf16>
// CHECK: [[VAR6:%.*]] = VPUIP.Copy inputs([[VAR5]] : memref<1x1x1x2000xf16, [@CMX_NN, 0]>) outputs([[VAR20]] : memref<1x1x1x2000xf16>) -> memref<1x1x1x2000xf16>

// CHECK: [[VAR7:%.*]] = memref.alloc() : memref<1x1x1x2000xf16, [@CMX_NN, 0]>
// CHECK: [[VAR8:%.*]] = VPUIP.Copy inputs([[VAR6]] : memref<1x1x1x2000xf16>) outputs([[VAR7]] : memref<1x1x1x2000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x2000xf16, [@CMX_NN, 0]>
// CHECK: [[VAR9:%.*]] = memref.alloc() : memref<1x1x1x2000xf16, [@CMX_NN, 0]>
// CHECK: [[VAR10:%.*]] = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_Sigmoid inputs([[VAR8]] : memref<1x1x1x2000xf16, [@CMX_NN, 0]>) outputs([[VAR9]] : memref<1x1x1x2000xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x2000xf16, [@CMX_NN, 0]>  {
// CHECK:      ^bb0(%arg1: memref<1x1x1x2000xf16, [@CMX_NN, 0]>, %arg2: memref<1x1x1x2000xf16, [@CMX_NN, 0]>):  // no predecessors
// CHECK:       VPUIP.SW.Kernel.run {attrs = []}(%arg1, %arg2) : memref<1x1x1x2000xf16, [@CMX_NN, 0]>, memref<1x1x1x2000xf16, [@CMX_NN, 0]>
// CHECK:  }
// CHECK: [[VAR21:%.*]] = memref.alloc() : memref<1x1x1x2000xf16>
// CHECK: [[VAR11:%.*]] = VPUIP.Copy inputs([[VAR10]] : memref<1x1x1x2000xf16, [@CMX_NN, 0]>) outputs([[VAR21]] : memref<1x1x1x2000xf16>) -> memref<1x1x1x2000xf16>

// CHECK: [[VAR22:%.*]] = builtin.unrealized_conversion_cast [[VAR11]] : memref<1x1x1x2000xf16> to tensor<1x1x1x2000xf16>
// CHECK: [[VAR23:%.*]] = VPU.Slice [[VAR22]] [0, 0, 0, 1000] [1, 1, 1, 1000] : tensor<1x1x1x2000xf16> to tensor<1x1x1x1000xf16>
// CHECK: [[VAR24:%.*]] = builtin.unrealized_conversion_cast [[VAR23]] : tensor<1x1x1x1000xf16> to memref<1x1x1x1000xf16>

// CHECK: [[VAR13:%.*]] = memref.alloc() : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
// CHECK: [[VAR14:%.*]] = VPUIP.Copy inputs([[VAR24]] : memref<1x1x1x1000xf16>) outputs([[VAR13]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
// CHECK: [[VAR15:%.*]] = memref.alloc() : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
// CHECK: [[VAR16:%.*]] = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_SoftMax inputs([[VAR14]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs([[VAR15]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>  {
// CHECK:  ^bb0(%arg1: memref<1x1x1x1000xf16, [@CMX_NN, 0]>, %arg2: memref<1x1x1x1000xf16, [@CMX_NN, 0]>):  // no predecessors
// CHECK:  VPUIP.SW.Kernel.run {attrs = [0]}(%arg1, %arg2) : memref<1x1x1x1000xf16, [@CMX_NN, 0]>, memref<1x1x1x1000xf16, [@CMX_NN, 0]>
// CHECK:  }
// CHECK: [[VAR25:%.*]] = memref.alloc() : memref<1x1x1x1000xf16>
// CHECK: [[VAR17:%.*]] = VPUIP.Copy inputs([[VAR16]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs([[VAR25]] : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>
// CHECK: [[VAR26:%.*]] = builtin.unrealized_conversion_cast [[VAR17]] : memref<1x1x1x1000xf16> to tensor<1x1x1x1000xf16>

// CHECK:  return [[VAR26]] : tensor<1x1x1x1000xf16>

}

}
