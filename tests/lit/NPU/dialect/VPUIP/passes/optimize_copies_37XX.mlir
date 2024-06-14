//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-copies %s | FileCheck %s
// REQUIRES: arch-VPUX37XX

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW  {
    func.func private @builtin_Sigmoid(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "activation_sigmoid.cpp", VPU.kernel_entry = "activation_sigmoid"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @OptimizeLastCopy(%arg0: memref<1x2x4x4xf16>, %arg1: memref<1x2x4x4xf16>, %arg2: memref<1x2x4x4xf16>)
                            -> (memref<1x2x4x4xf16>, memref<1x2x4x4xf16>) {
    %0 = const.Declare memref<1x2x4x4xf16> = dense<1.000000e+00> : tensor<1x2x4x4xf16>
    %1 = memref.alloc() : memref<1x2x4x4xf16>
    %2 = memref.alloc() : memref<1x2x4x4xf16>

    %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
        @VPU.SW::@builtin_Sigmoid inputs(%arg0 as %arg3: memref<1x2x4x4xf16>) outputs(%1 as %arg4: memref<1x2x4x4xf16>) on tile 0 -> memref<1x2x4x4xf16>  {
            VPUIP.SW.Kernel.run {attrs = [0]}(%arg3, %arg4) : memref<1x2x4x4xf16>, memref<1x2x4x4xf16>
        }
    %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
        @VPU.SW::@builtin_Sigmoid inputs(%0 as %arg3: memref<1x2x4x4xf16>) outputs(%2 as %arg4: memref<1x2x4x4xf16>) on tile 0 -> memref<1x2x4x4xf16>  {
            VPUIP.SW.Kernel.run {attrs = [0]}(%arg3, %arg4) : memref<1x2x4x4xf16>, memref<1x2x4x4xf16>
        }

    %5 = VPUIP.Copy inputs(%3 : memref<1x2x4x4xf16>) outputs(%arg1 : memref<1x2x4x4xf16>) -> memref<1x2x4x4xf16>
    %6 = VPUIP.Copy inputs(%4 : memref<1x2x4x4xf16>) outputs(%arg2 : memref<1x2x4x4xf16>) -> memref<1x2x4x4xf16>

    return %5, %6 : memref<1x2x4x4xf16>, memref<1x2x4x4xf16>

    // CHECK-DAG: [[VAR0:%.*]] = const.Declare

    // CHECK-NOT: memref.alloc() : memref<1x2x4x4xf16>
    // CHECK-NOT: memref.alloc() : memref<1x2x4x4xf16>

    // CHECK: [[VAR1:%.*]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Sigmoid inputs(%arg0 as [[ARG3:%.*]]: memref<1x2x4x4xf16>) outputs(%arg1 as [[ARG4:%.*]]: memref<1x2x4x4xf16>)
    // CHECK-SAME:           -> memref<1x2x4x4xf16>
    // CHECK: [[VAR2:%.*]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Sigmoid inputs([[VAR0]] as [[ARG3:%.*]]: memref<1x2x4x4xf16>) outputs(%arg2 as [[ARG4:%.*]]: memref<1x2x4x4xf16>)
    // CHECK-SAME:           -> memref<1x2x4x4xf16>
    // CHECK: return [[VAR1]], [[VAR2]] : memref<1x2x4x4xf16>, memref<1x2x4x4xf16>
}

// -----

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW  {
    func.func private @builtin_Sigmoid(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "activation_sigmoid.cpp", VPU.kernel_entry = "activation_sigmoid"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @NoChangesTypeMismatch(%arg0: memref<1x50x1x1xf16>, %arg1: memref<1x50xf16>) -> memref<1x50xf16> {
    %0 = memref.alloc() : memref<1x50x1x1xf16>
    %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
        @VPU.SW::@builtin_Sigmoid inputs(%arg0 as %arg2: memref<1x50x1x1xf16>) outputs(%0 as %arg3: memref<1x50x1x1xf16>) on tile 0 -> memref<1x50x1x1xf16>  {
            VPUIP.SW.Kernel.run {attrs = [0]}(%arg2, %arg3) : memref<1x50x1x1xf16>, memref<1x50x1x1xf16>
        }
    %2 = VPUIP.GenericReshape inputs(%1 : memref<1x50x1x1xf16>) -> memref<1x50xf16>
    %3 = VPUIP.Copy inputs(%2 : memref<1x50xf16>) outputs(%arg1 : memref<1x50xf16>) -> memref<1x50xf16>
    return %3 : memref<1x50xf16>

    // CHECK:       memref.alloc()
    // CHECK:       VPUIP.SW.Kernel
    // CHECK-SAME:      @VPU.SW::@builtin_Sigmoid
    // CHECK:       VPUIP.GenericReshape
    // CHECK:       [[VAR0:%.*]] = VPUIP.Copy
    // CHECK:       return [[VAR0]]
}

// -----

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW  {
    func.func private @builtin_Sigmoid(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "activation_sigmoid.cpp", VPU.kernel_entry = "activation_sigmoid"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @NoChangesInputIsBlockArgument(%arg0: memref<1x2x4x4xf16>, %arg1: memref<1x2x4x4xf16>,
                                    %arg2: memref<1x2x4x4xf16>, %arg3: memref<1x2x4x4xf16>) ->
                                    (memref<1x2x4x4xf16>, memref<1x2x4x4xf16>, memref<1x2x4x4xf16>) {
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x2x4x4xf16>) outputs(%arg1 : memref<1x2x4x4xf16>) -> memref<1x2x4x4xf16>

    %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
        @VPU.SW::@builtin_Sigmoid inputs(%arg0 as %arg4: memref<1x2x4x4xf16>) outputs(%arg2 as %arg5: memref<1x2x4x4xf16>) on tile 0 -> memref<1x2x4x4xf16>  {
            VPUIP.SW.Kernel.run {attrs = [0]}(%arg4, %arg5) : memref<1x2x4x4xf16>, memref<1x2x4x4xf16>
        }
    %2 = VPUIP.Copy inputs(%1 : memref<1x2x4x4xf16>) outputs(%arg3 : memref<1x2x4x4xf16>) -> memref<1x2x4x4xf16>

    return %0, %1, %2 : memref<1x2x4x4xf16>, memref<1x2x4x4xf16>, memref<1x2x4x4xf16>

    // CHECK:       [[VAR0:%.*]] = VPUIP.Copy
    // CHECK:       [[VAR1:%.*]] = VPUIP.SW.Kernel
    // CHECK-SAME:      @VPU.SW::@builtin_Sigmoid
    // CHECK:       [[VAR2:%.*]] = VPUIP.Copy
    // CHECK:       return [[VAR0]], [[VAR1]], [[VAR2]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW  {
    func.func private @builtin_Sigmoid(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "activation_sigmoid.cpp", VPU.kernel_entry = "activation_sigmoid"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @FuseWithPermuteCast(%arg0: memref<1x50x1x1xf16>, %arg1: memref<1x50x1x1xf16, #NHWC>) -> memref<1x50x1x1xf16, #NHWC> {
    %0 = memref.alloc() : memref<1x50x1x1xf16>
    %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
        @VPU.SW::@builtin_Sigmoid inputs(%arg0 as %arg2: memref<1x50x1x1xf16>) outputs(%0 as %arg3: memref<1x50x1x1xf16>) on tile 0 -> memref<1x50x1x1xf16>  {
            VPUIP.SW.Kernel.run {attrs = [0]}(%arg2, %arg3) : memref<1x50x1x1xf16>, memref<1x50x1x1xf16>
        }
    %2 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%1 : memref<1x50x1x1xf16>) -> memref<1x50x1x1xf16, #NHWC>
    %3 = VPUIP.Copy inputs(%2 : memref<1x50x1x1xf16, #NHWC>) outputs(%arg1 : memref<1x50x1x1xf16, #NHWC>) -> memref<1x50x1x1xf16, #NHWC>
    return %3 : memref<1x50x1x1xf16, #NHWC>

    // CHECK-NOT:   memref.alloc()
    // CHECK:       VPUIP.PermuteCast
    // CHECK:       VPUIP.SW.Kernel
    // CHECK-SAME:      @VPU.SW::@builtin_Sigmoid
    // CHECK:       return %arg1
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!InputDistributedType = !VPUIP.DistributedBuffer<
    1x30x120x120xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2
}>

!InputStub_CMX = memref<1x30x120x120xf16, #NHWC, [@CMX_NN, 0]>
!SpilledOutput_DDR = memref<1x3x120x120xf16, #NHWC, @DDR>

func.func @NotFuseCMXCopyToTheFrontOfTillingCopyDueToCMXSizeLimitation() -> !InputStub_CMX {
  %0 = VPURT.AllocDistributed -> !InputDistributedType
  %1 = memref.alloc() : !SpilledOutput_DDR
  %2 = VPUIP.NCEClusterTiling inputs(%0 as %arg0: memref<1x30x120x120xf16, #NHWC, @CMX_NN>) outputs(%1 as %arg1: !SpilledOutput_DDR) -> !SpilledOutput_DDR {
      VPUIP.Copy inputs(%arg0: memref<1x30x120x120xf16, #NHWC, @CMX_NN>) outputs(%arg1: !SpilledOutput_DDR) -> !SpilledOutput_DDR
  }

  %3 = memref.alloc() : !InputStub_CMX
  %4 = VPUIP.Copy inputs(%2 : !SpilledOutput_DDR) outputs(%3 : !InputStub_CMX) -> !InputStub_CMX

  return %4 : !InputStub_CMX

  // CHECK:   [[BUF_0:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x30x120x120xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
  // CHECK:   [[BUF_1:%.*]] = memref.alloc() : memref<1x3x120x120xf16, #NHWC, @DDR>
  // CHECK:   [[COPY_0:%.*]] = VPUIP.NCEClusterTiling inputs([[BUF_0]] as %arg0: memref<1x30x120x120xf16, #NHWC, @CMX_NN>) outputs([[BUF_1]] as %arg1: memref<1x3x120x120xf16, #NHWC, @DDR>) -> memref<1x3x120x120xf16, #NHWC, @DDR> {
  // CHECK:       VPUIP.Copy inputs(%arg0 : memref<1x30x120x120xf16, #NHWC, @CMX_NN>) outputs(%arg1 : memref<1x3x120x120xf16, #NHWC, @DDR>) -> memref<1x3x120x120xf16, #NHWC, @DDR>
  // CHECK:   }
  // CHECK:   [[BUF_2:%.*]] = memref.alloc() : memref<1x30x120x120xf16, #NHWC, [@CMX_NN, 0]>
  // CHECK:   [[COPY_1:%.*]] = VPUIP.Copy inputs([[COPY_0]] : memref<1x3x120x120xf16, #NHWC, @DDR>) outputs([[BUF_2]] : memref<1x30x120x120xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x30x120x120xf16, #NHWC, [@CMX_NN, 0]>
  // CHECK:   return [[COPY_1]] : memref<1x30x120x120xf16, #NHWC, [@CMX_NN, 0]>
}
