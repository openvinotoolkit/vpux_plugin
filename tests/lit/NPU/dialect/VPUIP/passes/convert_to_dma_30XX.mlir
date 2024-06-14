//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//


// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --convert-to-dma %s | FileCheck %s
// REQUIRES: arch-VPUX30XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertDepthToSpace_BLOCKS_FIRST
func.func @ConvertDepthToSpace_BLOCKS_FIRST(%arg0: memref<1x8x2x3xf16, #NHWC, @DDR>) -> memref<1x2x4x6xf16, #NHWC, @DDR> {
    %outBuffer = memref.alloc() : memref<1x2x4x6xf16, #NHWC, @DDR>
    %depthToSpace = VPUIP.DepthToSpaceUPA {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>}
                inputs(%arg0 : memref<1x8x2x3xf16, #NHWC, @DDR>)
                outputs(%outBuffer : memref<1x2x4x6xf16, #NHWC, @DDR>) -> memref<1x2x4x6xf16, #NHWC, @DDR>

    return %depthToSpace : memref<1x2x4x6xf16, #NHWC, @DDR>

    //CHECK:    [[OUTBUFFER:%.*]] = memref.alloc() : memref<1x2x4x6xf16, #NHWC, @DDR>
    //CHECK:    [[COPYBUFFER_0:%.*]] = memref.alloc() : memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[COPYOUT_0:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x8x2x3xf16, #NHWC, @DDR>)
    //CHECK:            outputs([[COPYBUFFER_0]] : memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    [[OPBUFFER:%.*]] = memref.alloc() : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[DepthToSpaceDMAOUT:%.*]] = VPUIP.DepthToSpaceDMA {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>}
    //CHECK:            inputs([[COPYOUT_0]] : memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OPBUFFER]] : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    [[COPYBUFFER_1:%.*]] = memref.alloc() : memref<1x2x4x6xf16, #NHWC, @DDR>
    //CHECK:    [[COPYOUT_1:%.*]] = VPUIP.Copy inputs([[DepthToSpaceDMAOUT]] : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:            outputs([[COPYBUFFER_1]] : memref<1x2x4x6xf16, #NHWC, @DDR>) -> memref<1x2x4x6xf16, #NHWC, @DDR>

    //CHECK:    return [[COPYOUT_1]] : memref<1x2x4x6xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertDepthToSpace_DEPTH_FIRST
func.func @ConvertDepthToSpace_DEPTH_FIRST(%arg0: memref<1x8x2x3xf16, #NHWC, @DDR>) -> memref<1x2x4x6xf16, #NHWC, @DDR> {
    %outBuffer = memref.alloc() : memref<1x2x4x6xf16, #NHWC, @DDR>
    %depthToSpace = VPUIP.DepthToSpaceUPA {block_size = 2 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>}
                inputs(%arg0 : memref<1x8x2x3xf16, #NHWC, @DDR>)
                outputs(%outBuffer : memref<1x2x4x6xf16, #NHWC, @DDR>) -> memref<1x2x4x6xf16, #NHWC, @DDR>

    return %depthToSpace : memref<1x2x4x6xf16, #NHWC, @DDR>

    //CHECK:    [[OUTBUFFER:%.*]] = memref.alloc() : memref<1x2x4x6xf16, #NHWC, @DDR>
    //CHECK:    [[COPYBUFFER_0:%.*]] = memref.alloc() : memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[COPYOUT_0:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x8x2x3xf16, #NHWC, @DDR>)
    //CHECK:            outputs([[COPYBUFFER_0]] : memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    [[OPBUFFER:%.*]] = memref.alloc() : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[DepthToSpaceDMAOUT:%.*]] = VPUIP.DepthToSpaceDMA {block_size = 2 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>}
    //CHECK:            inputs([[COPYOUT_0]] : memref<1x8x2x3xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:            outputs([[OPBUFFER]] : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:    [[COPYBUFFER_1:%.*]] = memref.alloc() : memref<1x2x4x6xf16, #NHWC, @DDR>
    //CHECK:    [[COPYOUT_1:%.*]] = VPUIP.Copy inputs([[DepthToSpaceDMAOUT]] : memref<1x2x4x6xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:            outputs([[COPYBUFFER_1]] : memref<1x2x4x6xf16, #NHWC, @DDR>) -> memref<1x2x4x6xf16, #NHWC, @DDR>

    //CHECK:    return [[COPYOUT_1]] : memref<1x2x4x6xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

func.func @ConvertMemPermute(%arg0: memref<1x8x16x16xf16, #NHWC, @DDR>)
        -> memref<1x8x16x16xf16, @DDR> {
    %buf1 = memref.alloc() : memref<1x8x16x16xf16, #NCHW, @DDR>
    %0 = VPUIP.PermuteUPA {order_value = #NWCH} inputs(%arg0 : memref<1x8x16x16xf16, #NHWC, @DDR>) outputs(%buf1 : memref<1x8x16x16xf16, #NCHW, @DDR>) -> memref<1x8x16x16xf16, @DDR>
    return %0: memref<1x8x16x16xf16, @DDR>
    // CHECK:   [[VAR0:%.*]] = memref.alloc() : memref<1x8x16x16xf16, @DDR>
    // CHECK:   [[VAR1:%.*]] = memref.alloc() : memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[VAR2:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x8x16x16xf16, #NHWC, @DDR>) outputs([[VAR1]] : memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[VAR3:%.*]] = memref.alloc() : memref<1x8x16x16xf16, [@CMX_NN, 0]>
    // CHECK:   [[VAR4:%.*]] = VPUIP.PermuteDMA {mem_perm = #NWCH} inputs([[VAR2]] : memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs([[VAR3]] : memref<1x8x16x16xf16, [@CMX_NN, 0]>) -> memref<1x8x16x16xf16, [@CMX_NN, 0]>
    // CHECK:   [[VAR5:%.*]] = memref.alloc() : memref<1x8x16x16xf16, @DDR>
    // CHECK:   [[VAR6:%.*]] = VPUIP.Copy inputs([[VAR4]] : memref<1x8x16x16xf16, [@CMX_NN, 0]>) outputs([[VAR5]] : memref<1x8x16x16xf16, @DDR>) -> memref<1x8x16x16xf16, @DDR>
    // CHECK:   return [[VAR6]] : memref<1x8x16x16xf16, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

func.func @ConvertOutputMemPermute(%arg0: memref<1x8x16x16xf16, #NHWC, @DDR>, %arg1: memref<1x8x16x16xf16>)
        -> memref<1x8x16x16xf16> {
    %0 = VPUIP.PermuteUPA {order_value = #NWCH} inputs(%arg0 : memref<1x8x16x16xf16, #NHWC, @DDR>) outputs(%arg1 : memref<1x8x16x16xf16>) -> memref<1x8x16x16xf16>
    return %arg1: memref<1x8x16x16xf16>
    // CHECK:   [[VAR0:%.*]] = memref.alloc() : memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[VAR1:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x8x16x16xf16, #NHWC, @DDR>) outputs([[VAR0]] : memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[VAR2:%.*]] = memref.alloc() : memref<1x8x16x16xf16, [@CMX_NN, 0]>
    // CHECK:   [[VAR3:%.*]] = VPUIP.PermuteDMA {mem_perm = #NWCH} inputs([[VAR1]] : memref<1x8x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs([[VAR2]] : memref<1x8x16x16xf16, [@CMX_NN, 0]>) -> memref<1x8x16x16xf16, [@CMX_NN, 0]>
    // CHECK:   [[VAR6:%.*]] = VPUIP.Copy inputs([[VAR3]] : memref<1x8x16x16xf16, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x8x16x16xf16>) -> memref<1x8x16x16xf16>
    // CHECK:   return %arg1 : memref<1x8x16x16xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

func.func @CanNotMoveMemPermuteFromDDRToCMX(%arg0: memref<1x8x16x16xf16, #NHWC, @DDR>)
        -> memref<1x8x16x16xf16, #NCWH, @DDR> {
    %buf = memref.alloc() : memref<1x8x16x16xf16, #NCWH, @DDR>
    %0 = VPUIP.PermuteUPA {order_value = #NWHC} inputs(%arg0 : memref<1x8x16x16xf16, #NHWC, @DDR>) outputs(%buf : memref<1x8x16x16xf16, #NCWH, @DDR>) -> memref<1x8x16x16xf16, #NCWH, @DDR>
    return %0: memref<1x8x16x16xf16, #NCWH, @DDR>

    // CHECK-NOT:   VPUIP.PermuteDMA
    // CHECK:       [[VAR:%.*]] = memref.alloc() : memref<1x8x16x16xf16, #NCWH, @DDR>
    // CHECK:       [[PERMUTEUPA:%.*]] = VPUIP.PermuteUPA {order_value = #NWHC}
    // CHECK-SAME:      inputs(%arg0 : memref<1x8x16x16xf16, #NHWC, @DDR>) outputs([[VAR]] : memref<1x8x16x16xf16, #NCWH, @DDR>) -> memref<1x8x16x16xf16, #NCWH, @DDR>
    // CHECK:       return [[PERMUTEUPA]] : memref<1x8x16x16xf16, #NCWH, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

func.func @CanNotMoveMemPermuteFromDDRToCMXWithLargeDMANumber(%arg0: memref<1x8x256x256xf16, #NHWC, @DDR>)
        -> memref<1x8x256x256xf16, @DDR> {
    %buf = memref.alloc() : memref<1x8x256x256xf16, @DDR>
    %0 = VPUIP.PermuteUPA {order_value = #NWCH} inputs(%arg0 : memref<1x8x256x256xf16, #NHWC, @DDR>) outputs(%buf : memref<1x8x256x256xf16, @DDR>) -> memref<1x8x256x256xf16, @DDR>
    return %0: memref<1x8x256x256xf16, @DDR>

    // CHECK-NOT:   VPUIP.PermuteDMA
    // CHECK:       [[VAR:%.*]] = memref.alloc() : memref<1x8x256x256xf16, @DDR>
    // CHECK:       [[PERMUTEUPA:%.*]] = VPUIP.PermuteUPA {order_value = #NWCH}
    // CHECK-SAME:      inputs(%arg0 : memref<1x8x256x256xf16, #NHWC, @DDR>) outputs([[VAR]] : memref<1x8x256x256xf16, @DDR>) -> memref<1x8x256x256xf16, @DDR>
    // CHECK:       return [[PERMUTEUPA]] : memref<1x8x256x256xf16, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

func.func @CanNotMoveMemPermuteFromDDRToCMXWithLargeLenth(%arg0: memref<1x512x32x32xf16, #NHWC, @DDR>)
        -> memref<1x512x32x32xf16, @DDR> {
    %buf = memref.alloc() : memref<1x512x32x32xf16, @DDR>
    %0 = VPUIP.PermuteUPA {order_value = #NWCH} inputs(%arg0 : memref<1x512x32x32xf16, #NHWC, @DDR>) outputs(%buf : memref<1x512x32x32xf16, @DDR>) -> memref<1x512x32x32xf16, @DDR>
    return %0: memref<1x512x32x32xf16, @DDR>

    // CHECK-NOT:   VPUIP.PermuteDMA
    // CHECK:       [[VAR:%.*]] = memref.alloc() : memref<1x512x32x32xf16, @DDR>
    // CHECK:       [[PERMUTEUPA:%.*]] = VPUIP.PermuteUPA {order_value = #NWCH}
    // CHECK-SAME:      inputs(%arg0 : memref<1x512x32x32xf16, #NHWC, @DDR>) outputs([[VAR]] : memref<1x512x32x32xf16, @DDR>) -> memref<1x512x32x32xf16, @DDR>
    // CHECK:       return [[PERMUTEUPA]] : memref<1x512x32x32xf16, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

func.func @ConvertMemPermuteWithThreeAxis(%arg0: memref<1x16x4x128xf16, @DDR>)
        -> memref<1x4x16x128xf16, #NHWC, @DDR> {
    %buf1 = memref.alloc() : memref<1x4x16x128xf16, #NHWC, @DDR>
    %0 = VPUIP.PermuteUPA {order_value = #NHCW} inputs(%arg0 : memref<1x16x4x128xf16, @DDR>) outputs(%buf1 : memref<1x4x16x128xf16, #NHWC, @DDR>) -> memref<1x4x16x128xf16, #NHWC, @DDR>
    return %0: memref<1x4x16x128xf16, #NHWC, @DDR>
    // CHECK:   [[VAR0:%.*]] = memref.alloc() : memref<1x4x16x128xf16, #NHWC, @DDR>
    // CHECK:   [[VAR1:%.*]] = memref.alloc() : memref<1x16x4x128xf16, [@CMX_NN, 0]>
    // CHECK:   [[VAR2:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x16x4x128xf16, @DDR>) outputs([[VAR1]] : memref<1x16x4x128xf16, [@CMX_NN, 0]>) -> memref<1x16x4x128xf16, [@CMX_NN, 0]>
    // CHECK:   [[VAR3:%.*]] = memref.alloc() : memref<1x4x16x128xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[VAR4:%.*]] = VPUIP.PermuteDMA {mem_perm = #NHCW} inputs([[VAR2]] : memref<1x16x4x128xf16, [@CMX_NN, 0]>) outputs([[VAR3]] : memref<1x4x16x128xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x4x16x128xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[VAR5:%.*]] = memref.alloc() : memref<1x4x16x128xf16, #NHWC, @DDR>
    // CHECK:   [[VAR6:%.*]] = VPUIP.Copy inputs([[VAR4]] : memref<1x4x16x128xf16, #NHWC, [@CMX_NN, 0]>) outputs([[VAR5]] : memref<1x4x16x128xf16, #NHWC, @DDR>) -> memref<1x4x16x128xf16, #NHWC, @DDR>
    // CHECK:   return [[VAR6]] : memref<1x4x16x128xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0173492431640625:114>

// CHECK-LABEL: @WrapExpandandPermuteWithoutClusterTiling
func.func @WrapExpandandPermuteWithoutClusterTiling(%arg0: memref<1x3x24x24x!qElemType>) -> memref<1x16x24x24x!qElemType> {
   %0 = memref.alloc() : memref<1x16x24x24x!qElemType>
   %1 = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} inputs(%arg0 : memref<1x3x24x24x!qElemType>) outputs(%0 : memref<1x16x24x24x!qElemType>) -> memref<1x16x24x24x!qElemType>

   return %1 : memref<1x16x24x24x!qElemType>

   //CHECK:   [[VAR0:%.*]] = memref.alloc() : memref<1x16x24x24x!qElemType>
   //CHECK:   [[VAR1:%.*]] = memref.alloc() : memref<1x16x24x24x!qElemType>
   //CHECK:   [[VAR2:%.*]] = VPUIP.ExpandDMA {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} inputs(%arg0 : memref<1x3x24x24x!qElemType>) outputs([[VAR1]] : memref<1x16x24x24x!qElemType>) -> memref<1x16x24x24x!qElemType>
   //CHECK:   return [[VAR2]] : memref<1x16x24x24x!qElemType>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @ConvertPerAxisTileToDMA(%arg0: memref<1x1x1x1xf16, #NHWC, @DDR>)
        -> memref<1x512x1x1xf16, #NHWC, @DDR> {
    %outBuffer = memref.alloc() : memref<1x512x1x1xf16, #NHWC, @DDR>
    %0 = VPUIP.PerAxisTileUPA {axis = 1 : i64, tiles = 512 : i64} inputs(%arg0 : memref<1x1x1x1xf16, #NHWC, @DDR>) outputs(%outBuffer : memref<1x512x1x1xf16, #NHWC, @DDR>) -> memref<1x512x1x1xf16, #NHWC, @DDR>
    return %0: memref<1x512x1x1xf16, #NHWC, @DDR>

    // CHECK:   [[VAR0:%.*]] = memref.alloc() : memref<1x512x1x1xf16, #NHWC, @DDR>
    // CHECK:   [[VAR1:%.*]] = memref.alloc() : memref<1x1x1x1xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[VAR2:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x1x1x1xf16, #NHWC, @DDR>) outputs([[VAR1]] : memref<1x1x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x1x1x1xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[VAR3:%.*]] = memref.alloc() : memref<1x512x1x1xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[VAR4:%.*]] = VPUIP.PerAxisTileDMA {axis = 1 : i64, tiles = 512 : i64} inputs([[VAR2]] : memref<1x1x1x1xf16, #NHWC, [@CMX_NN, 0]>) outputs([[VAR3]] : memref<1x512x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x512x1x1xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[VAR5:%.*]] = memref.alloc() : memref<1x512x1x1xf16, #NHWC, @DDR>
    // CHECK:   [[VAR6:%.*]] = VPUIP.Copy inputs([[VAR4]] : memref<1x512x1x1xf16, #NHWC, [@CMX_NN, 0]>) outputs([[VAR5]] : memref<1x512x1x1xf16, #NHWC, @DDR>) -> memref<1x512x1x1xf16, #NHWC, @DDR>
    // CHECK:   return [[VAR6]] : memref<1x512x1x1xf16, #NHWC, @DDR>
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func private @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

func.func @CannotConvertToPermuteDMA(%arg0: memref<1x16x4x76xf16, #NWHC, @DDR>)
        -> memref<1x16x4x76xf16, @DDR> {
    %0 = memref.alloc() : memref<1x16x4x76xf16, #NWHC, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x4x76xf16, #NWHC, @DDR>) outputs(%0 : memref<1x16x4x76xf16, #NWHC, [@CMX_NN, 0]>) -> memref<1x16x4x76xf16, #NWHC, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x16x4x76xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MemPermute inputs(%1 as %arg2: memref<1x16x4x76xf16, #NWHC, [@CMX_NN, 0]>) outputs(%2 as %arg3: memref<1x16x4x76xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x4x76xf16, [@CMX_NN, 0]>{
       VPUIP.SW.Kernel.run {attrs = [[2, 1, 0, 3]]}(%arg2, %arg3) : memref<1x16x4x76xf16, #NWHC, [@CMX_NN, 0]>, memref<1x16x4x76xf16, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x16x4x76xf16, @DDR>
    %4 = VPUIP.Copy inputs(%results : memref<1x16x4x76xf16, [@CMX_NN, 0]>) outputs(%3 : memref<1x16x4x76xf16, @DDR>) -> memref<1x16x4x76xf16, @DDR>
    return %4: memref<1x16x4x76xf16, @DDR>

    // CHECK:   [[VAR0:%.*]] = memref.alloc() : memref<1x16x4x76xf16, #NWHC, [@CMX_NN, 0]>
    // CHECK:   [[VAR1:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x16x4x76xf16, #NWHC, @DDR>) outputs([[VAR0]] : memref<1x16x4x76xf16, #NWHC, [@CMX_NN, 0]>) -> memref<1x16x4x76xf16, #NWHC, [@CMX_NN, 0]>
    // CHECK:   [[VAR2:%.*]] = memref.alloc() : memref<1x16x4x76xf16, [@CMX_NN, 0]>
    // CHECK:   [[RESULT:%.*]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MemPermute inputs([[VAR1]] as %arg1: memref<1x16x4x76xf16, #NWHC, [@CMX_NN, 0]>) outputs([[VAR2]] as %arg2: memref<1x16x4x76xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x4x76xf16, [@CMX_NN, 0]>{
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [
    // CHECK:     [2, 1, 0, 3]
    // CHECK:     ]}(%arg1, %arg2) : memref<1x16x4x76xf16, #NWHC, [@CMX_NN, 0]>, memref<1x16x4x76xf16, [@CMX_NN, 0]>
    // CHECK    }
    // CHECK:   [[VAR3:%.*]] = memref.alloc() : memref<1x16x4x76xf16, @DDR>
    // CHECK:   [[VAR4:%.*]] = VPUIP.Copy inputs([[RESULT]] : memref<1x16x4x76xf16, [@CMX_NN, 0]>) outputs([[VAR3]] : memref<1x16x4x76xf16, @DDR>) -> memref<1x16x4x76xf16, @DDR>
    // CHECK:   return [[VAR4]] : memref<1x16x4x76xf16, @DDR>
}

// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @MemPermuteWithChildInputIsBlockArg(%arg0: memref<1x2x71x1xf16, @DDR>, %arg1: memref<1x1x1x142xf16, @DDR>)
        -> memref<1x1x1x142xf16, @DDR> {
    %0 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%arg1 : memref<1x1x1x142xf16, @DDR>) -> memref<1x71x2x1xf16, @DDR>
    %1 = VPUIP.PermuteUPA {order_value = #NHCW} inputs(%arg0 : memref<1x2x71x1xf16, @DDR>) outputs(%0 : memref<1x71x2x1xf16, @DDR>) -> memref<1x71x2x1xf16, @DDR>
    return %arg1 : memref<1x1x1x142xf16, @DDR>

     // CHECK:   [[VAR0:%.*]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%arg1 : memref<1x1x1x142xf16, @DDR>) -> memref<1x71x2x1xf16, @DDR>
     // CHECK:   [[VAR1:%.*]] = memref.alloc() : memref<1x2x71x1xf16, [@CMX_NN, 0]>
     // CHECK:   [[VAR2:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x2x71x1xf16, @DDR>) outputs([[VAR1]] : memref<1x2x71x1xf16, [@CMX_NN, 0]>) -> memref<1x2x71x1xf16, [@CMX_NN, 0]>
     // CHECK:   [[VAR3:%.*]] = memref.alloc() : memref<1x71x2x1xf16, [@CMX_NN, 0]>
     // CHECK:   [[VAR4:%.*]] = VPUIP.PermuteDMA {mem_perm = #NHCW} inputs([[VAR2]] : memref<1x2x71x1xf16, [@CMX_NN, 0]>) outputs([[VAR3]] : memref<1x71x2x1xf16, [@CMX_NN, 0]>) -> memref<1x71x2x1xf16, [@CMX_NN, 0]>
     // CHECK:   [[VAR5:%.*]] = VPUIP.Copy inputs([[VAR4]] : memref<1x71x2x1xf16, [@CMX_NN, 0]>) outputs([[VAR0]] : memref<1x71x2x1xf16, @DDR>) -> memref<1x71x2x1xf16, @DDR>
     // CHECK:   return %arg1 : memref<1x1x1x142xf16, @DDR>
}
