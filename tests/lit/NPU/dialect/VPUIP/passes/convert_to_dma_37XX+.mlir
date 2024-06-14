//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --convert-to-dma --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX37XX || arch-VPUX40XX

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @TestConvertDMAForDml attributes {VPU.directML} {

  VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
    module @VPU.SW {
      func.func private @builtin_DepthToSpace(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "depth_to_space.cpp", VPU.kernel_entry = "depth_to_space"}
      func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    }


  // CHECK-LABEL: @ConvertSWDepthToSpaceToDMAIfDML
  func.func @ConvertSWDepthToSpaceToDMAIfDML(%arg0: memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]> {
      %outBuffer = memref.alloc() : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>
      %depthToSpace = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_DepthToSpace
                          inputs(%arg0 as %arg1: memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>)
                          outputs(%outBuffer as %arg2: memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>{
                      VPUIP.SW.Kernel.run {attrs = [2, 0]}(%arg1, %arg2) : memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>, memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>
      }

      return %depthToSpace : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>

      //CHECK:    [[OUTBUFFER:%.*]] = memref.alloc() : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>
      //CHECK:    [[DepthToSpaceDMAOUT:%.*]] = VPUIP.DepthToSpaceDMA {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>}
      //CHECK:            inputs(%arg0 : memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>)
      //CHECK:            outputs([[OUTBUFFER]] : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>

      //CHECK:    return [[DepthToSpaceDMAOUT]] : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @TestConvertDMAForDml attributes {VPU.directML} {

  VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
    module @VPU.SW {
      func.func private @builtin_DepthToSpace(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "depth_to_space.cpp", VPU.kernel_entry = "depth_to_space"}
      func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    }


  // CHECK-LABEL: @ConvertSWDepthToSpaceToDMAIfDMLNotBeneficial
  func.func @ConvertSWDepthToSpaceToDMAIfDMLNotBeneficial(%arg0: memref<1x128x2x3xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x8x8x12xf16, #NHWC, [@CMX_NN, 0]> {
      %outBuffer = memref.alloc() : memref<1x8x8x12xf16, #NHWC, [@CMX_NN, 0]>
      %depthToSpace = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_DepthToSpace
                          inputs(%arg0 as %arg1: memref<1x128x2x3xf16, #NHWC, [@CMX_NN, 0]>)
                          outputs(%outBuffer as %arg2: memref<1x8x8x12xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x8x8x12xf16, #NHWC, [@CMX_NN, 0]>{
                      VPUIP.SW.Kernel.run {attrs = [4, 1]}(%arg1, %arg2) : memref<1x128x2x3xf16, #NHWC, [@CMX_NN, 0]>, memref<1x8x8x12xf16, #NHWC, [@CMX_NN, 0]>
      }

      return %depthToSpace : memref<1x8x8x12xf16, #NHWC, [@CMX_NN, 0]>

      //CHECK:    [[OUTBUFFER:%.*]] = memref.alloc() : memref<1x8x8x12xf16, #NHWC, [@CMX_NN, 0]>
      //CHECK:    [[D2S:%.*]] = VPUIP.DepthToSpaceDMA {block_size = 4 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>}
      //CHECK:            inputs(%arg0 : memref<1x128x2x3xf16, #NHWC, [@CMX_NN, 0]>)
      //CHECK:            outputs([[OUTBUFFER]] : memref<1x8x8x12xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x8x8x12xf16, #NHWC, [@CMX_NN, 0]>

      //CHECK:    return [[D2S]] : memref<1x8x8x12xf16, #NHWC, [@CMX_NN, 0]>
  }
}
