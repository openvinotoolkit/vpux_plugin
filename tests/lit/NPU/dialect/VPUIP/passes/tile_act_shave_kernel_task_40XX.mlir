//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" --tile-act-shave-kernel-task --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

IE.TileResource 4 of @NCE at 1.700000e+03 MHz {
    IE.ExecutorResource 1 of @DPU
}

module @VPU.SW {
    func.func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "mvn1.cpp", VPU.kernel_entry = "mvn1"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileSegmentedShaveWithCAlignment(%arg0: memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x128x64x32xf16>) outputs(%0 as %arg2: memref<1x128x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
      %2 = VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16>) outputs(%arg2 : memref<1x128x64x32xf16, @CMX_NN>) -> memref<1x128x64x32xf16, @CMX_NN>
    }

    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], alignment = [1, 16, 1, 1]}>
    %4 = VPUIP.NCEClusterTiling inputs(%1 as %arg1: memref<1x128x64x32xf16, @CMX_NN>) outputs(%3 as %arg2: memref<1x128x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN inputs(%arg1 as %arg3: memref<1x128x64x32xf16, @CMX_NN>) outputs(%arg2 as %arg4: memref<1x128x64x32xf16, @CMX_NN>) on tile 0 -> memref<1x128x64x32xf16, @CMX_NN>{
        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg3, %arg4) : memref<1x128x64x32xf16, @CMX_NN>, memref<1x128x64x32xf16, @CMX_NN>
      }
    }

    %5 = memref.alloc() : memref<1x128x64x32xf16>
    %6 = VPUIP.NCEClusterTiling inputs(%4 as %arg1: memref<1x128x64x32xf16, @CMX_NN>) outputs(%5 as %arg2: memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16> {
      %results = VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    }

    return %6: memref<1x128x64x32xf16>

    // CHECK:    [[INPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[COPY0:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x128x64x32xf16>) outputs([[INPUT_CMX]] as %arg2: memref<1x128x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
    // CHECK:            VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16>) outputs(%arg2 : memref<1x128x64x32xf16, @CMX_NN>) -> memref<1x128x64x32xf16, @CMX_NN>
    // CHECK:    }
    // CHECK:    [[SUBVIEW0:%.*]] = VPUIP.SubView [[COPY0]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[SUBVIEW1:%.*]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[OUTPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[SUBVIEW2:%.*]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[SUBVIEW3:%.*]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[MVN:%.*]]:2 = VPUIP.NCEClusterTiling inputs([[SUBVIEW1]] as %arg1: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, [[SUBVIEW0]] as %arg2: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) outputs([[SUBVIEW3]] as %arg3: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, [[SUBVIEW2]] as %arg4: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) -> (!VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>) {
    // CHECK{LITERAL}:      results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN inputs(%arg1 as %arg5: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, %arg2 as %arg6: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) outputs(%arg3 as %arg7: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, %arg4 as %arg8: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) strides([[65536, 2048, 32, 1], [65536, 2048, 32, 1], [65536, 2048, 32, 1], [65536, 2048, 32, 1]]) on tile 0 -> (memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>){
    // CHECK:                          VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg5, %arg7) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>
    // CHECK:                          VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg6, %arg8) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>
    // CHECK:               }
    // CHECK:    }
    // CHECK:    [[CONCAT:%.*]]  = VPUIP.ConcatView inputs([[MVN]]#0, [[MVN]]#1 : !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>) outputs(%4 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[OUTPUT_DDR:%.*]] = memref.alloc() : memref<1x128x64x32xf16>
    // CHECK:    [[COPY1:%.*]] = VPUIP.NCEClusterTiling inputs([[CONCAT]] as %arg1: memref<1x128x64x32xf16, @CMX_NN>) outputs([[OUTPUT_DDR]] as %arg2: memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16> {
    // CHECK:                          VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    // CHECK:    }
    // CHECK:    return [[COPY1]] : memref<1x128x64x32xf16>
}

// -----

// CHECK-LABEL: @TileGather

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#NC = affine_map<(d0, d1) -> (d0, d1)>

module @VPU.SW {
  func.func private @builtin_Gather(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "gather.cpp", VPU.kernel_entry = "gather"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileGather(%arg0: memref<387072x1xf16>, %arg1: memref<1x96768xsi32>)
        -> memref<1x96768x1xf16> {
    %0 = memref.alloc() : memref<387072x1xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<387072x1xf16>) outputs(%0 : memref<387072x1xf16, [@CMX_NN, 0]>) -> memref<387072x1xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x96768xsi32, [@CMX_NN, 0]>
    %3 = VPUIP.Copy inputs(%arg1 : memref<1x96768xsi32>) outputs(%2 : memref<1x96768xsi32, [@CMX_NN, 0]>) -> memref<1x96768xsi32, [@CMX_NN, 0]>
    %4 = memref.alloc() : memref<1x96768x1xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Gather inputs(%1 as %arg2: memref<387072x1xf16, [@CMX_NN, 0]>, %3 as %arg3: memref<1x96768xsi32, [@CMX_NN, 0]>) outputs(%4 as %arg4: memref<1x96768x1xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x96768x1xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [1, 0]}(%arg2, %arg3, %arg4) : memref<387072x1xf16, [@CMX_NN, 0]>, memref<1x96768xsi32, [@CMX_NN, 0]>, memref<1x96768x1xf16, [@CMX_NN, 0]>
    }
    %5 = memref.alloc() : memref<1x96768x1xf16>
    %6 = VPUIP.Copy inputs(%results : memref<1x96768x1xf16, [@CMX_NN, 0]>) outputs(%5 : memref<1x96768x1xf16>) -> memref<1x96768x1xf16>
    return %6: memref<1x96768x1xf16>

    // CHECK:    [[ALLOC0:%.*]] = memref.alloc() : memref<387072x1xf16, [@CMX_NN, 0]>
    // CHECK:    [[COPY0:%.*]] = VPUIP.Copy inputs({{[^:]+}} : memref<387072x1xf16>) outputs([[ALLOC0]] : memref<387072x1xf16, [@CMX_NN, 0]>) -> memref<387072x1xf16, [@CMX_NN, 0]>
    // CHECK:    [[ALLOC1:%.*]] = memref.alloc() : memref<1x96768xsi32, [@CMX_NN, 0]>
    // CHECK:    [[COPY1:%.*]] = VPUIP.Copy inputs({{[^:]+}} : memref<1x96768xsi32>) outputs([[ALLOC1]] : memref<1x96768xsi32, [@CMX_NN, 0]>) -> memref<1x96768xsi32, [@CMX_NN, 0]>
    // CHECK:    [[ALLOC2:%.*]] = memref.alloc() : memref<1x96768x1xf16, [@CMX_NN, 0]>

    // CHECK:    [[SUBVIEW0:%.*]] = VPUIP.SubView [[COPY1]] [0, 0] [1, 48384] : memref<1x96768xsi32, [@CMX_NN, 0]> to memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW1:%.*]] = VPUIP.SubView [[ALLOC2]] [0, 0, 0] [1, 48384, 1] : memref<1x96768x1xf16, [@CMX_NN, 0]> to memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW2:%.*]] = VPUIP.SubView [[COPY1]] [0, 48384] [1, 48384] : memref<1x96768xsi32, [@CMX_NN, 0]> to memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW3:%.*]] = VPUIP.SubView [[ALLOC2]] [0, 48384, 0] [1, 48384, 1] : memref<1x96768x1xf16, [@CMX_NN, 0]> to memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>

    // CHECK:    [[GATHER:%.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Gather
    // CHECK-SAME:  inputs([[COPY0]] as {{[^:]+}}:  memref<387072x1xf16, [@CMX_NN, 0]>,
    // CHECK-SAME:  [[SUBVIEW0]] as {{[^:]+}}: memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>,
    // CHECK-SAME:  [[COPY0]] as {{[^:]+}}:  memref<387072x1xf16, [@CMX_NN, 0]>,
    // CHECK-SAME:  [[SUBVIEW2]] as {{[^:]+}}: memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>)
    // CHECK-SAME:  outputs([[SUBVIEW1]] as {{[^:]+}}: memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>,
    // CHECK-SAME:  memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>) on tile 0 ->
    // CHECK-SAME:  (memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>, memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>){
    // CHECK:      VPUIP.SW.Kernel.run {attrs = [1, 0]}({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<387072x1xf16, [@CMX_NN, 0]>, memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>,
    // CHECK-SAME:  memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:      VPUIP.SW.Kernel.run {attrs = [1, 0]}({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<387072x1xf16, [@CMX_NN, 0]>, memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>,
    // CHECK-SAME:  memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:    }

    // CHECK:    [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[GATHER]]#0, [[GATHER]]#1 : memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>,
    // CHECK-SAME:  memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>)  outputs([[ALLOC2]] : memref<1x96768x1xf16, [@CMX_NN, 0]>) -> memref<1x96768x1xf16, [@CMX_NN, 0]>
    // CHECK:    [[ALLOC3:%.*]] = memref.alloc() : memref<1x96768x1xf16>
    // CHECK:    [[COPY03:%.*]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x96768x1xf16, [@CMX_NN, 0]>) outputs([[ALLOC3]] : memref<1x96768x1xf16>) -> memref<1x96768x1xf16>

    // CHECK:    return [[COPY03]] : memref<1x96768x1xf16>

}

// -----

// CHECK-LABEL: @DontTileGatherWithStrideInput

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#NC = affine_map<(d0, d1) -> (d0, d1)>

module @VPU.SW {
  func.func private @builtin_Gather(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "gather.cpp", VPU.kernel_entry = "gather"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @DontTileGatherWithStrideInput(%arg0: memref<3996x160xf16>, %arg1: memref<1xsi32>)
        -> memref<1x160xf16> {
    %0 = memref.alloc() : memref<3996x160xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<3996x160xf16>) outputs(%0 : memref<3996x160xf16, [@CMX_NN, 0]>) -> memref<3996x160xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1xsi32, [@CMX_NN, 0]>
    %3 = VPUIP.Copy inputs(%arg1 : memref<1xsi32>) outputs(%2 : memref<1xsi32, [@CMX_NN, 0]>) -> memref<1xsi32, [@CMX_NN, 0]>
    %4 = memref.alloc() : memref<1x160xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Gather
      inputs(%1 as %arg3: memref<3996x160xf16, [@CMX_NN, 0]>,
      %3 as %arg4: memref<1xsi32, [@CMX_NN, 0]>)
      outputs(%4 as %arg5: memref<1x160xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x160xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [1, 0]}(%arg3, %arg4, %arg5) : memref<3996x160xf16, [@CMX_NN, 0]>, memref<1xsi32, [@CMX_NN, 0]>, memref<1x160xf16, [@CMX_NN, 0]>
    }

    %5 = memref.alloc() : memref<1x160xf16>
    %6 = VPUIP.Copy inputs(%results : memref<1x160xf16, [@CMX_NN, 0]>) outputs(%5 : memref<1x160xf16>) -> memref<1x160xf16>
    return %6: memref<1x160xf16>

    // CHECK:    [[ALLOC0:%.*]] = memref.alloc() : memref<3996x160xf16, [@CMX_NN, 0]>
    // CHECK:    [[COPY0:%.*]] = VPUIP.Copy inputs({{[^:]+}} : memref<3996x160xf16>) outputs([[ALLOC0]] : memref<3996x160xf16, [@CMX_NN, 0]>) -> memref<3996x160xf16, [@CMX_NN, 0]>
    // CHECK:    [[ALLOC1:%.*]] = memref.alloc() : memref<1xsi32, [@CMX_NN, 0]>
    // CHECK:    [[COPY1:%.*]] = VPUIP.Copy inputs({{[^:]+}} : memref<1xsi32>) outputs([[ALLOC1]] : memref<1xsi32, [@CMX_NN, 0]>) -> memref<1xsi32, [@CMX_NN, 0]>
    // CHECK:    [[ALLOC2:%.*]] = memref.alloc() : memref<1x160xf16, [@CMX_NN, 0]>
    // CHECK:    [[GATHER:%.*]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Gather
    // CHECK-SAME:  inputs([[COPY0]] as {{[^:]+}}: memref<3996x160xf16, [@CMX_NN, 0]>, [[COPY1]] as {{[^:]+}}: memref<1xsi32, [@CMX_NN, 0]>)
    // CHECK-SAME:  outputs([[ALLOC2]] as {{[^:]+}}: memref<1x160xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x160xf16, [@CMX_NN, 0]>{
    // CHECK:        VPUIP.SW.Kernel.run {attrs = [1, 0]}({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<3996x160xf16, [@CMX_NN, 0]>, memref<1xsi32, [@CMX_NN, 0]>, memref<1x160xf16, [@CMX_NN, 0]>
    // CHECK:      }
    // CHECK:    [[ALLOC3:%.*]] = memref.alloc() : memref<1x160xf16>
    // CHECK:    [[COPY2:%.*]] = VPUIP.Copy inputs([[GATHER]] : memref<1x160xf16, [@CMX_NN, 0]>) outputs([[ALLOC3]] : memref<1x160xf16>) -> memref<1x160xf16>

    // CHECK:    return [[COPY2]] : memref<1x160xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

IE.TileResource 4 of @NCE at 1.700000e+03 MHz {
    IE.ExecutorResource 1 of @DPU
}

module @VPU.SW {
    func.func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "mvn1.cpp", VPU.kernel_entry = "mvn1"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileSegmentedShaveWithProperCAlignment(%arg0: memref<1x64x64x32xf16>) -> memref<1x64x64x32xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x64x64x32xf16>) outputs(%0 as %arg2: memref<1x64x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
      %2 = VPUIP.Copy inputs(%arg1 : memref<1x64x64x32xf16>) outputs(%arg2 : memref<1x64x64x32xf16, @CMX_NN>) -> memref<1x64x64x32xf16, @CMX_NN>
    }

    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], alignment = [1, 16, 1, 1]}>
    %4 = VPUIP.NCEClusterTiling inputs(%1 as %arg1: memref<1x64x64x32xf16, @CMX_NN>) outputs(%3 as %arg2: memref<1x64x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN inputs(%arg1 as %arg3: memref<1x64x64x32xf16, @CMX_NN>) outputs(%arg2 as %arg4: memref<1x64x64x32xf16, @CMX_NN>) on tile 0 -> memref<1x64x64x32xf16, @CMX_NN>{
        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg3, %arg4) : memref<1x64x64x32xf16, @CMX_NN>, memref<1x64x64x32xf16, @CMX_NN>
      }
    }

    %5 = memref.alloc() : memref<1x64x64x32xf16>
    %6 = VPUIP.NCEClusterTiling inputs(%4 as %arg1: memref<1x64x64x32xf16, @CMX_NN>) outputs(%5 as %arg2: memref<1x64x64x32xf16>) -> memref<1x64x64x32xf16> {
      %results = VPUIP.Copy inputs(%arg1 : memref<1x64x64x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x64x64x32xf16>) -> memref<1x64x64x32xf16>
    }

    return %6: memref<1x64x64x32xf16>

    // CHECK:    [[INPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[COPY0:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x64x64x32xf16>) outputs([[INPUT_CMX]] as %arg2: memref<1x64x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
    // CHECK:            VPUIP.Copy inputs(%arg1 : memref<1x64x64x32xf16>) outputs(%arg2 : memref<1x64x64x32xf16, @CMX_NN>) -> memref<1x64x64x32xf16, @CMX_NN>
    // CHECK:    }
    // CHECK:    [[SUBVIEW0:%.*]] = VPUIP.SubView [[COPY0]] [0, 32, 0, 0] [1, 32, 64, 32] : !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[SUBVIEW1:%.*]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 32, 64, 32] : !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[OUTPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[SUBVIEW2:%.*]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 32, 0, 0] [1, 32, 64, 32] : !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[SUBVIEW3:%.*]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 0, 0, 0] [1, 32, 64, 32] : !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[MVN:%.*]]:2 = VPUIP.NCEClusterTiling inputs([[SUBVIEW1]] as %arg1: memref<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN>, [[SUBVIEW0]] as %arg2: memref<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN>) outputs([[SUBVIEW3]] as %arg3: memref<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN>, [[SUBVIEW2]] as %arg4: memref<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN>) -> (!VPUIP.DistributedBuffer<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>, !VPUIP.DistributedBuffer<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>) {
    // CHECK{LITERAL}:      %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN inputs(%arg1 as %arg5: memref<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN>, %arg2 as %arg6: memref<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN>) outputs(%arg3 as %arg7: memref<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN>, %arg4 as %arg8: memref<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN>) strides([[32768, 2048, 32, 1], [32768, 2048, 32, 1], [32768, 2048, 32, 1], [32768, 2048, 32, 1]]) on tile 0 -> (memref<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN>, memref<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN>){
    // CHECK:                 VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg5, %arg7) : memref<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN>, memref<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN>
    // CHECK:                 VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg6, %arg8) : memref<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN>, memref<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN>
    // CHECK:               }
    // CHECK:    }
    // CHECK:    [[CONCAT:%.*]]  = VPUIP.ConcatView inputs([[MVN]]#0, [[MVN]]#1 : !VPUIP.DistributedBuffer<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>, !VPUIP.DistributedBuffer<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>) outputs(%4 : !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>) -> !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[OUTPUT_DDR:%.*]] = memref.alloc() : memref<1x64x64x32xf16>
    // CHECK:    [[COPY1:%.*]] = VPUIP.NCEClusterTiling inputs([[CONCAT]] as %arg1: memref<1x64x64x32xf16, @CMX_NN>) outputs([[OUTPUT_DDR]] as %arg2: memref<1x64x64x32xf16>) -> memref<1x64x64x32xf16> {
    // CHECK:                          VPUIP.Copy inputs(%arg1 : memref<1x64x64x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x64x64x32xf16>) -> memref<1x64x64x32xf16>
    // CHECK:    }
    // CHECK:    return [[COPY1]] : memref<1x64x64x32xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

IE.TileResource 4 of @NCE at 1.700000e+03 MHz {
    IE.ExecutorResource 1 of @DPU
}

module @VPU.SW {
    func.func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "mvn1.cpp", VPU.kernel_entry = "mvn1"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileDuplicatedShaveWithCAlignment(%arg0: memref<1x32x64x32xf16>) -> memref<1x32x64x32xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x32x64x32xf16>) outputs(%0 as %arg2: memref<1x32x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
      %2 = VPUIP.Copy inputs(%arg1 : memref<1x32x64x32xf16>) outputs(%arg2 : memref<1x32x64x32xf16, @CMX_NN>) -> memref<1x32x64x32xf16, @CMX_NN>
    }

    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1], alignment = [1, 16, 1, 1]}>
    %4 = VPUIP.NCEClusterTiling inputs(%1 as %arg1: memref<1x32x64x32xf16, @CMX_NN>) outputs(%3 as %arg2: memref<1x32x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN inputs(%arg1 as %arg3: memref<1x32x64x32xf16, @CMX_NN>) outputs(%arg2 as %arg4: memref<1x32x64x32xf16, @CMX_NN>) on tile 0 -> memref<1x32x64x32xf16, @CMX_NN>{
        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg3, %arg4) : memref<1x32x64x32xf16, @CMX_NN>, memref<1x32x64x32xf16, @CMX_NN>
      }
    }

    %5 = memref.alloc() : memref<1x32x64x32xf16>
    %6 = VPUIP.NCEClusterTiling inputs(%4 as %arg1: memref<1x32x64x32xf16, @CMX_NN>) outputs(%5 as %arg2: memref<1x32x64x32xf16>) -> memref<1x32x64x32xf16> {
      %results = VPUIP.Copy inputs(%arg1 : memref<1x32x64x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x32x64x32xf16>) -> memref<1x32x64x32xf16>
    }

    return %6: memref<1x32x64x32xf16>

    // CHECK:    [[INPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[COPY0:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x32x64x32xf16>) outputs([[INPUT_CMX]] as %arg2: memref<1x32x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
    // CHECK:            VPUIP.Copy inputs(%arg1 : memref<1x32x64x32xf16>) outputs(%arg2 : memref<1x32x64x32xf16, @CMX_NN>) -> memref<1x32x64x32xf16, @CMX_NN>
    // CHECK:    }
    // CHECK:    [[SUBVIEW0:%.*]] = VPUIP.SubView [[COPY0]] [0, 16, 0, 0] [1, 16, 64, 32] : !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[SUBVIEW1:%.*]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 16, 64, 32] : !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[OUTPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[SUBVIEW2:%.*]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 16, 0, 0] [1, 16, 64, 32] : !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[SUBVIEW3:%.*]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 0, 0, 0] [1, 16, 64, 32] : !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[MVN:%.*]]:2 = VPUIP.NCEClusterTiling inputs([[SUBVIEW1]] as %arg1: memref<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN>, [[SUBVIEW0]] as %arg2: memref<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN>) outputs([[SUBVIEW3]] as %arg3: memref<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN>, [[SUBVIEW2]] as %arg4: memref<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN>) -> (!VPUIP.DistributedBuffer<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>, !VPUIP.DistributedBuffer<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>) {
    // CHECK{LITERAL}:      results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN inputs(%arg1 as %arg5: memref<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN>, %arg2 as %arg6: memref<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN>) outputs(%arg3 as %arg7: memref<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN>, %arg4 as %arg8: memref<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN>) strides([[65536, 2048, 32, 1], [65536, 2048, 32, 1], [65536, 2048, 32, 1], [65536, 2048, 32, 1]]) on tile 0 -> (memref<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN>, memref<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN>){
    // CHECK:                          VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg5, %arg7) : memref<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN>, memref<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN>
    // CHECK:                          VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg6, %arg8) : memref<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN>, memref<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN>
    // CHECK:               }
    // CHECK:    }
    // CHECK:    [[CONCAT:%.*]]  = VPUIP.ConcatView inputs([[MVN]]#0, [[MVN]]#1 : !VPUIP.DistributedBuffer<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>, !VPUIP.DistributedBuffer<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>) outputs(%4 : !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>) -> !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[OUTPUT_DDR:%.*]] = memref.alloc() : memref<1x32x64x32xf16>
    // CHECK:    [[COPY1:%.*]] = VPUIP.NCEClusterTiling inputs([[CONCAT]] as %arg1: memref<1x32x64x32xf16, @CMX_NN>) outputs([[OUTPUT_DDR]] as %arg2: memref<1x32x64x32xf16>) -> memref<1x32x64x32xf16> {
    // CHECK:                          VPUIP.Copy inputs(%arg1 : memref<1x32x64x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x32x64x32xf16>) -> memref<1x32x64x32xf16>
    // CHECK:    }
    // CHECK:    return [[COPY1]] : memref<1x32x64x32xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

IE.TileResource 4 of @NCE at 1.700000e+03 MHz {
    IE.ExecutorResource 1 of @DPU
}

module @VPU.SW {
    func.func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "mvn1.cpp", VPU.kernel_entry = "mvn1"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileDuplicatedShaveWithProperCAlignment(%arg0: memref<1x16x64x32xf16>) -> memref<1x16x64x32xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x16x64x32xf16>) outputs(%0 as %arg2: memref<1x16x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
      %2 = VPUIP.Copy inputs(%arg1 : memref<1x16x64x32xf16>) outputs(%arg2 : memref<1x16x64x32xf16, @CMX_NN>) -> memref<1x16x64x32xf16, @CMX_NN>
    }

    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1], alignment = [1, 16, 1, 1]}>
    %4 = VPUIP.NCEClusterTiling inputs(%1 as %arg1: memref<1x16x64x32xf16, @CMX_NN>) outputs(%3 as %arg2: memref<1x16x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN inputs(%arg1 as %arg3: memref<1x16x64x32xf16, @CMX_NN>) outputs(%arg2 as %arg4: memref<1x16x64x32xf16, @CMX_NN>) on tile 0 -> memref<1x16x64x32xf16, @CMX_NN>{
        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg3, %arg4) : memref<1x16x64x32xf16, @CMX_NN>, memref<1x16x64x32xf16, @CMX_NN>
      }
    }

    %5 = memref.alloc() : memref<1x16x64x32xf16>
    %6 = VPUIP.NCEClusterTiling inputs(%4 as %arg1: memref<1x16x64x32xf16, @CMX_NN>) outputs(%5 as %arg2: memref<1x16x64x32xf16>) -> memref<1x16x64x32xf16> {
      %results = VPUIP.Copy inputs(%arg1 : memref<1x16x64x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x16x64x32xf16>) -> memref<1x16x64x32xf16>
    }

    return %6: memref<1x16x64x32xf16>

    // CHECK:    [[INPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[COPY0:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x16x64x32xf16>) outputs([[INPUT_CMX]] as %arg2: memref<1x16x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
    // CHECK:            VPUIP.Copy inputs(%arg1 : memref<1x16x64x32xf16>) outputs(%arg2 : memref<1x16x64x32xf16, @CMX_NN>) -> memref<1x16x64x32xf16, @CMX_NN>
    // CHECK:    }
    // CHECK:    [[SUBVIEW0:%.*]] = VPUIP.SubView [[COPY0]] [0, 8, 0, 0] [1, 8, 64, 32] : !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[SUBVIEW1:%.*]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 8, 64, 32] : !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[OUTPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[SUBVIEW2:%.*]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 8, 0, 0] [1, 8, 64, 32] : !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[SUBVIEW3:%.*]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 0, 0, 0] [1, 8, 64, 32] : !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[MVN:%.*]]:2 = VPUIP.NCEClusterTiling inputs([[SUBVIEW1]] as %arg1: memref<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN>, [[SUBVIEW0]] as %arg2: memref<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN>) outputs([[SUBVIEW3]] as %arg3: memref<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN>, [[SUBVIEW2]] as %arg4: memref<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN>) -> (!VPUIP.DistributedBuffer<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>, !VPUIP.DistributedBuffer<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>) {
    // CHECK{LITERAL}:      %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN inputs(%arg1 as %arg5: memref<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN>, %arg2 as %arg6: memref<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN>) outputs(%arg3 as %arg7: memref<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN>, %arg4 as %arg8: memref<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN>) strides([[32768, 2048, 32, 1], [32768, 2048, 32, 1], [32768, 2048, 32, 1], [32768, 2048, 32, 1]]) on tile 0 -> (memref<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN>, memref<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN>){
    // CHECK:                 VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg5, %arg7) : memref<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN>, memref<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN>
    // CHECK:                 VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg6, %arg8) : memref<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN>, memref<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN>
    // CHECK:               }
    // CHECK:    }
    // CHECK:    [[CONCAT:%.*]]  = VPUIP.ConcatView inputs([[MVN]]#0, [[MVN]]#1 : !VPUIP.DistributedBuffer<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>, !VPUIP.DistributedBuffer<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>) outputs(%4 : !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>) -> !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[OUTPUT_DDR:%.*]] = memref.alloc() : memref<1x16x64x32xf16>
    // CHECK:    [[COPY1:%.*]] = VPUIP.NCEClusterTiling inputs([[CONCAT]] as %arg1: memref<1x16x64x32xf16, @CMX_NN>) outputs([[OUTPUT_DDR]] as %arg2: memref<1x16x64x32xf16>) -> memref<1x16x64x32xf16> {
    // CHECK:                          VPUIP.Copy inputs(%arg1 : memref<1x16x64x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x16x64x32xf16>) -> memref<1x16x64x32xf16>
    // CHECK:    }
    // CHECK:    return [[COPY1]] : memref<1x16x64x32xf16>
}

// -----

IE.TileResource 6 of @NCE at 1.300000e+03 MHz {
    IE.ExecutorResource 1 of @DPU
}

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
  func.func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "mvn1.cpp", VPU.kernel_entry = "mvn1"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileUnevenClusterMVNWithAlignment(%arg0: memref<1x128x16x1xf16>)
        -> memref<1x128x16x1xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x128x16x1xf16>) outputs(%0 as %arg2: memref<1x128x16x1xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}> {
      %2 = VPUIP.Copy inputs(%arg1 : memref<1x128x16x1xf16>) outputs(%arg2 : memref<1x128x16x1xf16, @CMX_NN>) -> memref<1x128x16x1xf16, @CMX_NN>
    }
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>
    %4 = VPUIP.NCEClusterTiling inputs(%1 as %arg1: memref<1x128x16x1xf16, @CMX_NN>) outputs(%3 as %arg2: memref<1x128x16x1xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN inputs(%arg1 as %arg3: memref<1x128x16x1xf16, @CMX_NN>) outputs(%arg2 as %arg4: memref<1x128x16x1xf16, @CMX_NN>) on tile 0 -> memref<1x128x16x1xf16, @CMX_NN>{
        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg3, %arg4) : memref<1x128x16x1xf16, @CMX_NN>, memref<1x128x16x1xf16, @CMX_NN>
      }
    }
    %5 = memref.alloc() : memref<1x128x16x1xf16>
    %6 = VPUIP.NCEClusterTiling inputs(%4 as %arg1: memref<1x128x16x1xf16, @CMX_NN>) outputs(%5 as %arg2: memref<1x128x16x1xf16>) -> memref<1x128x16x1xf16> {
      %7 = VPUIP.Copy inputs(%arg1 : memref<1x128x16x1xf16, @CMX_NN>) outputs(%arg2 : memref<1x128x16x1xf16>) -> memref<1x128x16x1xf16>
    }
    return %6: memref<1x128x16x1xf16>
    // CHECK:    [[ALLOC_INPUT:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>
    // CHECK:    [[COPY_0:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x128x16x1xf16>) outputs([[ALLOC_INPUT]] as %arg2: memref<1x128x16x1xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}> {
    // CHECK:                            VPUIP.Copy inputs(%arg1 : memref<1x128x16x1xf16>) outputs(%arg2 : memref<1x128x16x1xf16, @CMX_NN>) -> memref<1x128x16x1xf16, @CMX_NN>
    // CHECK:    }
    // CHECK:    [[SUBVIEW_0:%.*]] = VPUIP.SubView [[COPY_0]] [0, 48, 0, 0] [1, 80, 16, 1]
    // CHECK-SAME{LITERAL}:                       {explicit_output_shapes = [[1, 24, 16, 1], [1, 24, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1]]} :
    // CHECK-SAME:                   !VPUIP.DistributedBuffer<1x128x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>
    // CHECK-SAME:                   to !VPUIP.DistributedBuffer<1x80x16x1xf16, {order = #NCHW, strides = [2048, 16, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 24, 16, 1], [1, 24, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 24, 0, 0], [0, 48, 0, 0], [0, 56, 0, 0], [0, 64, 0, 0], [0, 72, 0, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 24, 16, 1], [1, 24, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 24, 0, 0], [0, 48, 0, 0], [0, 56, 0, 0], [0, 64, 0, 0], [0, 72, 0, 0]]
    // CHECK:    [[SUBVIEW_1:%.*]] = VPUIP.SubView [[COPY_0]] [0, 0, 0, 0] [1, 48, 16, 1]
    // CHECK-SAME:                     !VPUIP.DistributedBuffer<1x128x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>
    // CHECK-SAME:                     to !VPUIP.DistributedBuffer<1x48x16x1xf16, {order = #NCHW, strides = [2048, 16, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments}>
    // CHECK:    [[ALLOC_OUTPUT:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>
    // CHECK:    [[SUBVIEW_2:%.*]] = VPUIP.SubView [[ALLOC_OUTPUT]] [0, 48, 0, 0] [1, 80, 16, 1]
    // CHECK-SAME{LITERAL}:                       {explicit_output_shapes = [[1, 24, 16, 1], [1, 24, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1]]} :
    // CHECK-SAME:                    !VPUIP.DistributedBuffer<1x128x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>
    // CHECK-SAME:                    to !VPUIP.DistributedBuffer<1x80x16x1xf16, {order = #NCHW, strides = [2048, 16, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 24, 16, 1], [1, 24, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 24, 0, 0], [0, 48, 0, 0], [0, 56, 0, 0], [0, 64, 0, 0], [0, 72, 0, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 24, 16, 1], [1, 24, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 24, 0, 0], [0, 48, 0, 0], [0, 56, 0, 0], [0, 64, 0, 0], [0, 72, 0, 0]]
    // CHECK:    [[SUBVIEW_3:%.*]] = VPUIP.SubView [[ALLOC_OUTPUT]] [0, 0, 0, 0] [1, 48, 16, 1]
    // CHECK-SAME:                    !VPUIP.DistributedBuffer<1x128x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>
    // CHECK-SAME:                     to !VPUIP.DistributedBuffer<1x48x16x1xf16, {order = #NCHW, strides = [2048, 16, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments}>
    // CHECK:    [[CLUSTER_MVN:%.*]]:2 = VPUIP.NCEClusterTiling inputs([[SUBVIEW_1]] as %arg1: memref<1x48x16x1xf16, {order = #NCHW, strides = [2048, 16, 1, 1]}, @CMX_NN>, [[SUBVIEW_0]] as %arg2: memref<1x80x16x1xf16, {order = #NCHW, strides = [2048, 16, 1, 1]}, @CMX_NN>) outputs([[SUBVIEW_3]] as %arg3: memref<1x48x16x1xf16, {order = #NCHW, strides = [2048, 16, 1, 1]}, @CMX_NN>, [[SUBVIEW_2]] as %arg4: memref<1x80x16x1xf16, {order = #NCHW, strides = [2048, 16, 1, 1]}, @CMX_NN>)
    // CHECK-SAME:                     (!VPUIP.DistributedBuffer<1x48x16x1xf16, {order = #NCHW, strides = [2048, 16, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments}>
    // CHECK-SAME{LITERAL}:               !VPUIP.DistributedBuffer<1x80x16x1xf16, {order = #NCHW, strides = [2048, 16, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 24, 16, 1], [1, 24, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 24, 0, 0], [0, 48, 0, 0], [0, 56, 0, 0], [0, 64, 0, 0], [0, 72, 0, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 24, 16, 1], [1, 24, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 24, 0, 0], [0, 48, 0, 0], [0, 56, 0, 0], [0, 64, 0, 0], [0, 72, 0, 0]]
    // CHECK:                              VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN inputs(%arg1 as %arg5: memref<1x48x16x1xf16, {order = #NCHW, strides = [2048, 16, 1, 1]}, @CMX_NN>, %arg2 as %arg6: memref<1x80x16x1xf16, {order = #NCHW, strides = [2048, 16, 1, 1]}, @CMX_NN>) outputs(%arg3 as %arg7: memref<1x48x16x1xf16, {order = #NCHW, strides = [2048, 16, 1, 1]}, @CMX_NN>, %arg4 as %arg8: memref<1x80x16x1xf16, {order = #NCHW, strides = [2048, 16, 1, 1]}, @CMX_NN>)
    // CHECK-SAME{LITERAL}:                       strides([[512, 16, 1, 1], [512, 16, 1, 1], [256, 16, 1, 1], [256, 16, 1, 1], [256, 16, 1, 1], [256, 16, 1, 1]]) on tile 0 -> (memref<1x48x16x1xf16, {order = #NCHW, strides = [2048, 16, 1, 1]}, @CMX_NN>, memref<1x80x16x1xf16, {order = #NCHW, strides = [2048, 16, 1, 1]}, @CMX_NN>){
    // CHECK:                                   VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg5, %arg7) : memref<1x48x16x1xf16, {order = #NCHW, strides = [2048, 16, 1, 1]}, @CMX_NN>, memref<1x48x16x1xf16, {order = #NCHW, strides = [2048, 16, 1, 1]}, @CMX_NN>
    // CHECK:                                   VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg6, %arg8) : memref<1x80x16x1xf16, {order = #NCHW, strides = [2048, 16, 1, 1]}, @CMX_NN>, memref<1x80x16x1xf16, {order = #NCHW, strides = [2048, 16, 1, 1]}, @CMX_NN>
    // CHECK:                              }
    // CHECK:    }
    // CHECK:    [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[CLUSTER_MVN]]#0, [[CLUSTER_MVN]]#1 :
    // CHECK-SAME:    outputs([[ALLOC_OUTPUT]] : !VPUIP.DistributedBuffer<1x128x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>) -> !VPUIP.DistributedBuffer<1x128x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>
    // CHECK:    [[ALLOC_DDR:%.*]] = memref.alloc() : memref<1x128x16x1xf16>
    // CHECK:    [[COPY_1:%.*]] = VPUIP.NCEClusterTiling inputs([[CONCAT]] as %arg1: memref<1x128x16x1xf16, @CMX_NN>) outputs([[ALLOC_DDR]] as %arg2: memref<1x128x16x1xf16>) -> memref<1x128x16x1xf16> {
    // CHECK:                           VPUIP.Copy inputs(%arg1 : memref<1x128x16x1xf16, @CMX_NN>) outputs(%arg2 : memref<1x128x16x1xf16>) -> memref<1x128x16x1xf16>
    // CHECK:    }
    // CHECK:    return [[COPY_1]] : memref<1x128x16x1xf16>

}

// -----

IE.TileResource 4 of @NCE at 1.300000e+03 MHz {
    IE.ExecutorResource 1 of @DPU
}

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW  {
    func.func private @builtin_TanhOp(memref<*xf16>, memref<*xf16>, i64) attributes {VPU.kernel_code = "activation_tanh.cpp", VPU.kernel_entry = "activation_tanh"}
}

func.func @TileClusterTanHWithDifferentDims(%arg0: memref<1x16x64x128xf16>)
        -> memref<1x16x64x128xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x64x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x16x64x128xf16>) outputs(%0 as %arg2: memref<1x16x64x128xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x16x64x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}> {
      %2 = VPUIP.Copy inputs(%arg1 : memref<1x16x64x128xf16>) outputs(%arg2 : memref<1x16x64x128xf16, @CMX_NN>) -> memref<1x16x64x128xf16, @CMX_NN>
    }
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x64x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>
    %4 = VPUIP.NCEClusterTiling inputs(%1 as %arg1: memref<1x16x64x128xf16>) outputs(%3 as %arg2: memref<1x16x64x128xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x16x64x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}> {
          %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_TanhOp inputs(%arg1 as %arg3: memref<1x16x64x128xf16, @CMX_NN>) outputs(%arg2 as %arg4: memref<1x16x64x128xf16, @CMX_NN>) on tile 0 -> memref<1x16x64x128xf16, @CMX_NN> {
               VPUIP.SW.Kernel.run(%arg3, %arg4) : memref<1x16x64x128xf16, @CMX_NN>, memref<1x16x64x128xf16, @CMX_NN>
          }
    }
    %5 = memref.alloc() : memref<1x16x64x1xf16>
    %6 = VPUIP.NCEClusterTiling inputs(%4 as %arg1: memref<1x16x64x128xf16, @CMX_NN>) outputs(%5 as %arg2: memref<1x16x64x128xf16>) -> memref<1x16x64x128xf16> {
      %7 = VPUIP.Copy inputs(%arg1 : memref<1x16x64x128xf16, @CMX_NN>) outputs(%arg2 : memref<1x16x64x128xf16>) -> memref<1x16x64x128xf16>
    }
    return %6: memref<1x16x64x128xf16>

}

// -----

IE.TileResource 4 of @NCE at 1.300000e+03 MHz {
    IE.ExecutorResource 1 of @DPU
}

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!Distributed = !VPUIP.DistributedBuffer<
    4x10x5x17xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[4, 10, 2, 17], [4, 10, 1, 17], [4, 10, 1, 17], [4, 10, 1, 17]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0]],
    memory_shapes = [[4, 10, 2, 17], [4, 10, 1, 17], [4, 10, 1, 17], [4, 10, 1, 17]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0]]
}>

module @VPU.SW {
    func.func private @builtin_MVN6(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i64, f64, i64, none) attributes {VPU.kernel_code = "mvn6.cpp", VPU.kernel_entry = "mvn6", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL:  func.func @TileMVN6OnDimCAndDistributedOnDimH
// CHECK-SAME:    ([[INPUT:%.*]]: memref<4x10x5x17xf16>)
func.func @TileMVN6OnDimCAndDistributedOnDimH(%arg0: memref<4x10x5x17xf16>) -> memref<4x10x5x17xf16> {
    %0 = VPURT.AllocDistributed -> !Distributed
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg2: memref<4x10x5x17xf16>) outputs(%0 as %arg3: memref<4x10x5x17xf16, @CMX_NN>) -> !Distributed {
      %6 = VPUIP.Copy inputs(%arg2 : memref<4x10x5x17xf16>) outputs(%arg3 : memref<4x10x5x17xf16, @CMX_NN>) -> memref<4x10x5x17xf16, @CMX_NN>
    }
    %2 = VPURT.AllocDistributed -> !Distributed
    %3 = VPUIP.NCEClusterTiling inputs(%1 as %arg2: memref<4x10x5x17xf16, @CMX_NN>) outputs(%2 as %arg3: memref<4x10x5x17xf16, @CMX_NN>) -> !Distributed {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN6 inputs(%arg2 as %arg4: memref<4x10x5x17xf16, @CMX_NN>) outputs(%arg3 as %arg5: memref<4x10x5x17xf16, @CMX_NN>) on tile 0 -> memref<4x10x5x17xf16, @CMX_NN>{
        VPUIP.SW.Kernel.run {attrs = [true, 0, 5.000000e-01, 1, [3]]}(%arg4, %arg5) : memref<4x10x5x17xf16, @CMX_NN>, memref<4x10x5x17xf16, @CMX_NN>
      }
    }
    %alloc = memref.alloc() : memref<4x10x5x17xf16>
    %4 = VPUIP.NCEClusterTiling inputs(%3 as %arg2: memref<4x10x5x17xf16, @CMX_NN>) outputs(%alloc as %arg3: memref<4x10x5x17xf16>) -> memref<4x10x5x17xf16> {
      %6 = VPUIP.Copy inputs(%arg2 : memref<4x10x5x17xf16, @CMX_NN>) outputs(%arg3 : memref<4x10x5x17xf16>) -> memref<4x10x5x17xf16>
    }

    return %4 : memref<4x10x5x17xf16>

  // CHECK:       [[SUBVIEW0:%.*]] = VPUIP.SubView [[INPUT]] [0, 5, 0, 0] [4, 5, 5, 17] : memref<4x10x5x17xf16> to memref<4x5x5x17xf16, {order = #NCHW, strides = [850, 85, 17, 1]}>
  // CHECK:       [[ALLOC_INPUT0:%.*]] = VPURT.AllocDistributed ->
  // CHECK-SAME:                        !VPUIP.DistributedBuffer<
  // CHECK-SAME:                          4x5x5x17xf16, #NCHW, @CMX_NN, {
  // CHECK-SAME:                          mode = "SEGMENTED",
  // CHECK-SAME:                          num_tiles = [1, 1, 4, 1],
  // CHECK-SAME:                          num_clusters = 4 : i64,
  // CHECK-SAME:                          uniform_distributed_segments,
  // CHECK-SAME{LITERAL}:                 compute_shapes = [[4, 5, 2, 17], [4, 5, 1, 17], [4, 5, 1, 17], [4, 5, 1, 17]],
  // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0]],
  // CHECK-SAME{LITERAL}:                 memory_shapes = [[4, 5, 2, 17], [4, 5, 1, 17], [4, 5, 1, 17], [4, 5, 1, 17]],
  // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0]]}>
  // CHECK:       [[COPY_INPUT0:%.*]] = VPUIP.NCEClusterTiling
  // CHECK-SAME:                          inputs([[SUBVIEW0]] as [[ARG0:%[^:]+]]: memref<4x5x5x17xf16, {order = #NCHW, strides = [850, 85, 17, 1]}>)
  // CHECK-SAME:                          outputs([[ALLOC_INPUT0]] as [[ARG1:%[^:]+]]: memref<4x5x5x17xf16, @CMX_NN>) ->
  // CHECK-SAME:                        !VPUIP.DistributedBuffer<
  // CHECK-SAME:                          4x5x5x17xf16, #NCHW, @CMX_NN, {
  // CHECK-SAME:                          mode = "SEGMENTED",
  // CHECK-SAME:                          num_tiles = [1, 1, 4, 1],
  // CHECK-SAME:                          num_clusters = 4 : i64,
  // CHECK-SAME:                          uniform_distributed_segments,
  // CHECK-SAME{LITERAL}:                 compute_shapes = [[4, 5, 2, 17], [4, 5, 1, 17], [4, 5, 1, 17], [4, 5, 1, 17]],
  // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0]],
  // CHECK-SAME{LITERAL}:                 memory_shapes = [[4, 5, 2, 17], [4, 5, 1, 17], [4, 5, 1, 17], [4, 5, 1, 17]],
  // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0]]}> {
  // CHECK:                                       VPUIP.Copy
  // CHECK-SAME:                                      inputs([[ARG0]] : memref<4x5x5x17xf16, {order = #NCHW, strides = [850, 85, 17, 1]}>)
  // CHECK-SAME:                                      outputs([[ARG1]] : memref<4x5x5x17xf16, @CMX_NN>) -> memref<4x5x5x17xf16, @CMX_NN>
  // CHECK:       }
  // CHECK:       [[SUBVIEW1:%.*]] = VPUIP.SubView [[INPUT]] [0, 0, 0, 0] [4, 5, 5, 17] : memref<4x10x5x17xf16> to memref<4x5x5x17xf16, {order = #NCHW, strides = [850, 85, 17, 1]}>
  // CHECK:       [[ALLOC_INPUT1:%.*]] = VPURT.AllocDistributed ->
  // CHECK-SAME:                        !VPUIP.DistributedBuffer<
  // CHECK-SAME:                          4x5x5x17xf16, #NCHW, @CMX_NN, {
  // CHECK-SAME:                          mode = "SEGMENTED",
  // CHECK-SAME:                          num_tiles = [1, 1, 4, 1],
  // CHECK-SAME:                          num_clusters = 4 : i64,
  // CHECK-SAME:                          uniform_distributed_segments,
  // CHECK-SAME{LITERAL}:                 compute_shapes = [[4, 5, 2, 17], [4, 5, 1, 17], [4, 5, 1, 17], [4, 5, 1, 17]],
  // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0]],
  // CHECK-SAME{LITERAL}:                 memory_shapes = [[4, 5, 2, 17], [4, 5, 1, 17], [4, 5, 1, 17], [4, 5, 1, 17]],
  // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0]]}>
  // CHECK:       [[COPY_INPUT1:%.*]] = VPUIP.NCEClusterTiling
  // CHECK-SAME:                          inputs([[SUBVIEW1]] as [[ARG2:%[^:]+]]: memref<4x5x5x17xf16, {order = #NCHW, strides = [850, 85, 17, 1]}>)
  // CHECK-SAME:                          outputs([[ALLOC_INPUT1]] as [[ARG3:%[^:]+]]: memref<4x5x5x17xf16, @CMX_NN>) ->
  // CHECK-SAME:                        !VPUIP.DistributedBuffer<
  // CHECK-SAME:                          4x5x5x17xf16, #NCHW, @CMX_NN, {
  // CHECK-SAME:                          mode = "SEGMENTED",
  // CHECK-SAME:                          num_tiles = [1, 1, 4, 1],
  // CHECK-SAME:                          num_clusters = 4 : i64,
  // CHECK-SAME:                          uniform_distributed_segments,
  // CHECK-SAME{LITERAL}:                 compute_shapes = [[4, 5, 2, 17], [4, 5, 1, 17], [4, 5, 1, 17], [4, 5, 1, 17]],
  // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0]],
  // CHECK-SAME{LITERAL}:                 memory_shapes = [[4, 5, 2, 17], [4, 5, 1, 17], [4, 5, 1, 17], [4, 5, 1, 17]],
  // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0]]}> {
  // CHECK:                                       VPUIP.Copy
  // CHECK-SAME:                                      inputs([[ARG2]] : memref<4x5x5x17xf16, {order = #NCHW, strides = [850, 85, 17, 1]}>)
  // CHECK-SAME:                                      outputs([[ARG3]] : memref<4x5x5x17xf16, @CMX_NN>) -> memref<4x5x5x17xf16, @CMX_NN>
  // CHECK:       }
  // CHECK:       [[ALLOC_OUTPUT0:%.*]] = VPURT.AllocDistributed ->
  // CHECK-SAME:                        !VPUIP.DistributedBuffer<
  // CHECK-SAME:                          4x5x5x17xf16, #NCHW, @CMX_NN, {
  // CHECK-SAME:                          mode = "SEGMENTED",
  // CHECK-SAME:                          num_tiles = [1, 1, 4, 1],
  // CHECK-SAME:                          num_clusters = 4 : i64,
  // CHECK-SAME:                          uniform_distributed_segments,
  // CHECK-SAME{LITERAL}:                 compute_shapes = [[4, 5, 2, 17], [4, 5, 1, 17], [4, 5, 1, 17], [4, 5, 1, 17]],
  // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0]],
  // CHECK-SAME{LITERAL}:                 memory_shapes = [[4, 5, 2, 17], [4, 5, 1, 17], [4, 5, 1, 17], [4, 5, 1, 17]],
  // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0]]}>
  // CHECK:       [[ALLOC_OUTPUT1:%.*]] = VPURT.AllocDistributed ->
  // CHECK-SAME:                        !VPUIP.DistributedBuffer<
  // CHECK-SAME:                          4x5x5x17xf16, #NCHW, @CMX_NN, {
  // CHECK-SAME:                          mode = "SEGMENTED",
  // CHECK-SAME:                          num_tiles = [1, 1, 4, 1],
  // CHECK-SAME:                          num_clusters = 4 : i64,
  // CHECK-SAME:                          uniform_distributed_segments,
  // CHECK-SAME{LITERAL}:                 compute_shapes = [[4, 5, 2, 17], [4, 5, 1, 17], [4, 5, 1, 17], [4, 5, 1, 17]],
  // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0]],
  // CHECK-SAME{LITERAL}:                 memory_shapes = [[4, 5, 2, 17], [4, 5, 1, 17], [4, 5, 1, 17], [4, 5, 1, 17]],
  // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0]]}>

  // CHECK:       [[MVN:%.*]]:2 = VPUIP.NCEClusterTiling
  // CHECK-SAME:                          inputs([[COPY_INPUT1]] as [[ARG4:%[^:]+]]: memref<4x5x5x17xf16, @CMX_NN>, [[COPY_INPUT0]] as [[ARG5:%[^:]+]]: memref<4x5x5x17xf16, @CMX_NN>)
  // CHECK-SAME:                          outputs([[ALLOC_OUTPUT1]] as [[ARG6:%[^:]+]]: memref<4x5x5x17xf16, @CMX_NN>, [[ALLOC_OUTPUT0]] as [[ARG7:%[^:]+]]: memref<4x5x5x17xf16, @CMX_NN>) -> (
  // CHECK-SAME:                        !VPUIP.DistributedBuffer<
  // CHECK-SAME:                          4x5x5x17xf16, #NCHW, @CMX_NN, {
  // CHECK-SAME:                          mode = "SEGMENTED",
  // CHECK-SAME:                          num_tiles = [1, 1, 4, 1],
  // CHECK-SAME:                          num_clusters = 4 : i64,
  // CHECK-SAME:                          uniform_distributed_segments,
  // CHECK-SAME{LITERAL}:                 compute_shapes = [[4, 5, 2, 17], [4, 5, 1, 17], [4, 5, 1, 17], [4, 5, 1, 17]],
  // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0]],
  // CHECK-SAME{LITERAL}:                 memory_shapes = [[4, 5, 2, 17], [4, 5, 1, 17], [4, 5, 1, 17], [4, 5, 1, 17]],
  // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0]]}>,
  // CHECK-SAME:                        !VPUIP.DistributedBuffer<
  // CHECK-SAME:                          4x5x5x17xf16, #NCHW, @CMX_NN, {
  // CHECK-SAME:                          mode = "SEGMENTED",
  // CHECK-SAME:                          num_tiles = [1, 1, 4, 1],
  // CHECK-SAME:                          num_clusters = 4 : i64,
  // CHECK-SAME:                          uniform_distributed_segments,
  // CHECK-SAME{LITERAL}:                 compute_shapes = [[4, 5, 2, 17], [4, 5, 1, 17], [4, 5, 1, 17], [4, 5, 1, 17]],
  // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0]],
  // CHECK-SAME{LITERAL}:                 memory_shapes = [[4, 5, 2, 17], [4, 5, 1, 17], [4, 5, 1, 17], [4, 5, 1, 17]],
  // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0]]}>) {
  // CHECK:                                       VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN6
  // CHECK-SAME:                                      inputs([[ARG4]] as [[ARG8:%[^:]+]]: memref<4x5x5x17xf16, @CMX_NN>, [[ARG5]] as [[ARG9:%[^:]+]]: memref<4x5x5x17xf16, @CMX_NN>)
  // CHECK-SAME:                                      outputs([[ARG6]] as [[ARG10:%[^:]+]]: memref<4x5x5x17xf16, @CMX_NN>, [[ARG7]] as [[ARG11:%[^:]+]]: memref<4x5x5x17xf16, @CMX_NN>) on tile 0 -> (memref<4x5x5x17xf16, @CMX_NN>, memref<4x5x5x17xf16, @CMX_NN>){
  // CHECK:                                               VPUIP.SW.Kernel.run {attrs = [true, 0, 5.000000e-01, 1, [3]]}([[ARG8]], [[ARG10]]) : memref<4x5x5x17xf16, @CMX_NN>, memref<4x5x5x17xf16, @CMX_NN>
  // CHECK:                                               VPUIP.SW.Kernel.run {attrs = [true, 0, 5.000000e-01, 1, [3]]}([[ARG9]], [[ARG11]]) : memref<4x5x5x17xf16, @CMX_NN>, memref<4x5x5x17xf16, @CMX_NN>
  // CHECK:                                       }
  // CHECK:       }

  // CHECK:       [[ALLOC:%.*]] = memref.alloc() : memref<4x10x5x17xf16>
  // CHECK:       [[SUBVIEW2:%.*]] = VPUIP.SubView [[ALLOC]] [0, 0, 0, 0] [4, 5, 5, 17] : memref<4x10x5x17xf16> to memref<4x5x5x17xf16, {order = #NCHW, strides = [850, 85, 17, 1]}>
  // CHECK:       [[COPY_OUT_0:%.*]] = VPUIP.NCEClusterTiling
  // CHECK-SAME:                          inputs([[MVN]]#0 as [[ARG12:%[^:]+]]: memref<4x5x5x17xf16, @CMX_NN>)
  // CHECK-SAME:                          outputs([[SUBVIEW2]] as [[ARG13:%[^:]+]]: memref<4x5x5x17xf16, {order = #NCHW, strides = [850, 85, 17, 1]}>) -> memref<4x5x5x17xf16, {order = #NCHW, strides = [850, 85, 17, 1]}> {
  // CHECK:                                       VPUIP.Copy
  // CHECK-SAME:                                      inputs([[ARG12]] : memref<4x5x5x17xf16, @CMX_NN>)
  // CHECK-SAME:                                      outputs([[ARG13]] : memref<4x5x5x17xf16, {order = #NCHW, strides = [850, 85, 17, 1]}>) -> memref<4x5x5x17xf16, {order = #NCHW, strides = [850, 85, 17, 1]}>
  // CHECK:       }
  // CHECK:       [[SUBVIEW3:%.*]] = VPUIP.SubView [[ALLOC]] [0, 5, 0, 0] [4, 5, 5, 17] : memref<4x10x5x17xf16> to memref<4x5x5x17xf16, {order = #NCHW, strides = [850, 85, 17, 1]}>
  // CHECK:       [[COPY_OUT_1:%.*]] = VPUIP.NCEClusterTiling
  // CHECK-SAME:                          inputs([[MVN]]#1 as [[ARG14:%[^:]+]]: memref<4x5x5x17xf16, @CMX_NN>)
  // CHECK-SAME:                          outputs([[SUBVIEW3]] as [[ARG15:%[^:]+]]: memref<4x5x5x17xf16, {order = #NCHW, strides = [850, 85, 17, 1]}>) -> memref<4x5x5x17xf16, {order = #NCHW, strides = [850, 85, 17, 1]}> {
  // CHECK:                                       VPUIP.Copy
  // CHECK-SAME:                                      inputs([[ARG14]] : memref<4x5x5x17xf16, @CMX_NN>)
  // CHECK-SAME:                                      outputs([[ARG15]] : memref<4x5x5x17xf16, {order = #NCHW, strides = [850, 85, 17, 1]}>) -> memref<4x5x5x17xf16, {order = #NCHW, strides = [850, 85, 17, 1]}>
  // CHECK:       }
  // CHECK:       [[CONCAT:%.*]] = VPUIP.ConcatView
  // CHECK-SAME:                          inputs([[COPY_OUT_0]], [[COPY_OUT_1]] : memref<4x5x5x17xf16, {order = #NCHW, strides = [850, 85, 17, 1]}>, memref<4x5x5x17xf16, {order = #NCHW, strides = [850, 85, 17, 1]}>)
  // CHECK-SAME:                          outputs([[ALLOC]] : memref<4x10x5x17xf16>) -> memref<4x10x5x17xf16>
  // CHECK:       return [[CONCAT:%.*]] : memref<4x10x5x17xf16>
}

// -----

IE.TileResource 4 of @NCE at 1.300000e+03 MHz {
    IE.ExecutorResource 1 of @DPU
}

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!Distributed = !VPUIP.DistributedBuffer<
    4x1x10x17xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[4, 1, 3, 17], [4, 1, 3, 17], [4, 1, 2, 17], [4, 1, 2, 17]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 8, 0]],
    memory_shapes = [[4, 1, 3, 17], [4, 1, 3, 17], [4, 1, 2, 17], [4, 1, 2, 17]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 8, 0]]
}>

module @VPU.SW {
    func.func private @builtin_MVN6(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i64, f64, i64, none) attributes {VPU.kernel_code = "mvn6.cpp", VPU.kernel_entry = "mvn6", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL:  func.func @TileMVN6OnDimHAndDistributedOnDimH
// CHECK-SAME:    ([[INPUT:%.*]]: memref<4x1x10x17xf16>)
func.func @TileMVN6OnDimHAndDistributedOnDimH(%arg0: memref<4x1x10x17xf16>) -> memref<4x1x10x17xf16> {
    %0 = VPURT.AllocDistributed -> !Distributed
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg2: memref<4x1x10x17xf16>) outputs(%0 as %arg3: memref<4x1x10x17xf16, @CMX_NN>) -> !Distributed {
      %6 = VPUIP.Copy inputs(%arg2 : memref<4x1x10x17xf16>) outputs(%arg3 : memref<4x1x10x17xf16, @CMX_NN>) -> memref<4x1x10x17xf16, @CMX_NN>
    }
    %2 = VPURT.AllocDistributed -> !Distributed
    %3 = VPUIP.NCEClusterTiling inputs(%1 as %arg2: memref<4x1x10x17xf16, @CMX_NN>) outputs(%2 as %arg3: memref<4x1x10x17xf16, @CMX_NN>) -> !Distributed {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN6 inputs(%arg2 as %arg4: memref<4x1x10x17xf16, @CMX_NN>) outputs(%arg3 as %arg5: memref<4x1x10x17xf16, @CMX_NN>) on tile 0 -> memref<4x1x10x17xf16, @CMX_NN>{
        VPUIP.SW.Kernel.run {attrs = [true, 0, 5.000000e-01, 1, [3]]}(%arg4, %arg5) : memref<4x1x10x17xf16, @CMX_NN>, memref<4x1x10x17xf16, @CMX_NN>
      }
    }
    %alloc = memref.alloc() : memref<4x1x10x17xf16>
    %4 = VPUIP.NCEClusterTiling inputs(%3 as %arg2: memref<4x1x10x17xf16, @CMX_NN>) outputs(%alloc as %arg3: memref<4x1x10x17xf16>) -> memref<4x1x10x17xf16> {
      %6 = VPUIP.Copy inputs(%arg2 : memref<4x1x10x17xf16, @CMX_NN>) outputs(%arg3 : memref<4x1x10x17xf16>) -> memref<4x1x10x17xf16>
    }
    return %4 : memref<4x1x10x17xf16>

    // CHECK:       [[SUBVIEW0:%.*]] = VPUIP.SubView [[INPUT]] [0, 0, 6, 0] [4, 1, 4, 17] : memref<4x1x10x17xf16> to memref<4x1x4x17xf16, {order = #NCHW, strides = [170, 170, 17, 1]}>
    // CHECK:       [[ALLOC_INPUT0:%.*]] = VPURT.AllocDistributed ->
    // CHECK-SAME:                        !VPUIP.DistributedBuffer<
    // CHECK-SAME:                          4x1x4x17xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:                          mode = "SEGMENTED",
    // CHECK-SAME:                          num_tiles = [1, 1, 4, 1],
    // CHECK-SAME:                          num_clusters = 4 : i64,
    // CHECK-SAME:                          uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[4, 1, 1, 17], [4, 1, 1, 17], [4, 1, 1, 17], [4, 1, 1, 17]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[4, 1, 1, 17], [4, 1, 1, 17], [4, 1, 1, 17], [4, 1, 1, 17]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 0]]}>
    // CHECK:       [[COPY_INPUT0:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:                          inputs([[SUBVIEW0]] as [[ARG0:%[^:]+]]: memref<4x1x4x17xf16, {order = #NCHW, strides = [170, 170, 17, 1]}>)
    // CHECK-SAME:                          outputs([[ALLOC_INPUT0]] as [[ARG1:%[^:]+]]: memref<4x1x4x17xf16, @CMX_NN>) ->
    // CHECK-SAME:                        !VPUIP.DistributedBuffer<
    // CHECK-SAME:                          4x1x4x17xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:                          mode = "SEGMENTED",
    // CHECK-SAME:                          num_tiles = [1, 1, 4, 1],
    // CHECK-SAME:                          num_clusters = 4 : i64,
    // CHECK-SAME:                          uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[4, 1, 1, 17], [4, 1, 1, 17], [4, 1, 1, 17], [4, 1, 1, 17]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[4, 1, 1, 17], [4, 1, 1, 17], [4, 1, 1, 17], [4, 1, 1, 17]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 0]]}> {
    // CHECK:                                       VPUIP.Copy
    // CHECK-SAME:                                      inputs([[ARG0]] : memref<4x1x4x17xf16, {order = #NCHW, strides = [170, 170, 17, 1]}>)
    // CHECK-SAME:                                      outputs([[ARG1]] : memref<4x1x4x17xf16, @CMX_NN>) -> memref<4x1x4x17xf16, @CMX_NN>
    // CHECK:       }
    // CHECK:       [[SUBVIEW1:%.*]] = VPUIP.SubView [[INPUT]] [0, 0, 0, 0] [4, 1, 6, 17] : memref<4x1x10x17xf16> to memref<4x1x6x17xf16, {order = #NCHW, strides = [170, 170, 17, 1]}>
    // CHECK:       [[ALLOC_INPUT1:%.*]] = VPURT.AllocDistributed ->
    // CHECK-SAME:                        !VPUIP.DistributedBuffer<
    // CHECK-SAME:                          4x1x6x17xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:                          mode = "SEGMENTED",
    // CHECK-SAME:                          num_tiles = [1, 1, 4, 1],
    // CHECK-SAME:                          num_clusters = 4 : i64,
    // CHECK-SAME:                          uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[4, 1, 2, 17], [4, 1, 1, 17], [4, 1, 2, 17], [4, 1, 1, 17]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 5, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[4, 1, 2, 17], [4, 1, 1, 17], [4, 1, 2, 17], [4, 1, 1, 17]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 5, 0]]}>
    // CHECK:       [[COPY_INPUT1:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:                          inputs([[SUBVIEW1]] as [[ARG2:%[^:]+]]: memref<4x1x6x17xf16, {order = #NCHW, strides = [170, 170, 17, 1]}>)
    // CHECK-SAME:                          outputs([[ALLOC_INPUT1]] as [[ARG3:%[^:]+]]: memref<4x1x6x17xf16, @CMX_NN>) ->
    // CHECK-SAME:                        !VPUIP.DistributedBuffer<
    // CHECK-SAME:                          4x1x6x17xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:                          mode = "SEGMENTED",
    // CHECK-SAME:                          num_tiles = [1, 1, 4, 1],
    // CHECK-SAME:                          num_clusters = 4 : i64,
    // CHECK-SAME:                          uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[4, 1, 2, 17], [4, 1, 1, 17], [4, 1, 2, 17], [4, 1, 1, 17]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 5, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[4, 1, 2, 17], [4, 1, 1, 17], [4, 1, 2, 17], [4, 1, 1, 17]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 5, 0]]}> {
    // CHECK:                                       VPUIP.Copy
    // CHECK-SAME:                                      inputs([[ARG2]] : memref<4x1x6x17xf16, {order = #NCHW, strides = [170, 170, 17, 1]}>)
    // CHECK-SAME:                                      outputs([[ARG3]] : memref<4x1x6x17xf16, @CMX_NN>) -> memref<4x1x6x17xf16, @CMX_NN>
    // CHECK:       }

    // CHECK:       [[ALLOC_OUTPUT0:%.*]] = VPURT.AllocDistributed ->
    // CHECK-SAME:                        !VPUIP.DistributedBuffer<
    // CHECK-SAME:                          4x1x4x17xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:                          mode = "SEGMENTED",
    // CHECK-SAME:                          num_tiles = [1, 1, 4, 1],
    // CHECK-SAME:                          num_clusters = 4 : i64,
    // CHECK-SAME:                          uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[4, 1, 1, 17], [4, 1, 1, 17], [4, 1, 1, 17], [4, 1, 1, 17]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[4, 1, 1, 17], [4, 1, 1, 17], [4, 1, 1, 17], [4, 1, 1, 17]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 0]]}>
    // CHECK:       [[ALLOC_OUTPUT1:%.*]] = VPURT.AllocDistributed ->
    // CHECK-SAME:                        !VPUIP.DistributedBuffer<
    // CHECK-SAME:                          4x1x6x17xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:                          mode = "SEGMENTED",
    // CHECK-SAME:                          num_tiles = [1, 1, 4, 1],
    // CHECK-SAME:                          num_clusters = 4 : i64,
    // CHECK-SAME:                          uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[4, 1, 2, 17], [4, 1, 1, 17], [4, 1, 2, 17], [4, 1, 1, 17]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 5, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[4, 1, 2, 17], [4, 1, 1, 17], [4, 1, 2, 17], [4, 1, 1, 17]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 5, 0]]}>

    // CHECK:       [[MVN:%.*]]:2 = VPUIP.NCEClusterTiling
    // CHECK-SAME:                          inputs([[COPY_INPUT1]] as [[ARG4:%[^:]+]]: memref<4x1x6x17xf16, @CMX_NN>, [[COPY_INPUT0]] as [[ARG5:%[^:]+]]: memref<4x1x4x17xf16, @CMX_NN>)
    // CHECK-SAME:                          outputs([[ALLOC_OUTPUT1]] as [[ARG6:%[^:]+]]: memref<4x1x6x17xf16, @CMX_NN>, [[ALLOC_OUTPUT0]] as [[ARG7:%[^:]+]]: memref<4x1x4x17xf16, @CMX_NN>) -> (
    // CHECK-SAME:                        !VPUIP.DistributedBuffer<
    // CHECK-SAME:                          4x1x6x17xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:                          mode = "SEGMENTED",
    // CHECK-SAME:                          num_tiles = [1, 1, 4, 1],
    // CHECK-SAME:                          num_clusters = 4 : i64,
    // CHECK-SAME:                          uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[4, 1, 2, 17], [4, 1, 1, 17], [4, 1, 2, 17], [4, 1, 1, 17]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 5, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[4, 1, 2, 17], [4, 1, 1, 17], [4, 1, 2, 17], [4, 1, 1, 17]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 5, 0]]}>,
    // CHECK-SAME:                        !VPUIP.DistributedBuffer<
    // CHECK-SAME:                          4x1x4x17xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:                          mode = "SEGMENTED",
    // CHECK-SAME:                          num_tiles = [1, 1, 4, 1],
    // CHECK-SAME:                          num_clusters = 4 : i64,
    // CHECK-SAME:                          uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[4, 1, 1, 17], [4, 1, 1, 17], [4, 1, 1, 17], [4, 1, 1, 17]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[4, 1, 1, 17], [4, 1, 1, 17], [4, 1, 1, 17], [4, 1, 1, 17]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 0]]}>) {
    // CHECK:                                       VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN6
    // CHECK-SAME:                                      inputs([[ARG4]] as [[ARG8:%[^:]+]]: memref<4x1x6x17xf16, @CMX_NN>, [[ARG5]] as [[ARG9:%[^:]+]]: memref<4x1x4x17xf16, @CMX_NN>)
    // CHECK-SAME:                                      outputs([[ARG6]] as [[ARG10:%[^:]+]]: memref<4x1x6x17xf16, @CMX_NN>, [[ARG7]] as [[ARG11:%[^:]+]]: memref<4x1x4x17xf16, @CMX_NN>) on tile 0 -> (memref<4x1x6x17xf16, @CMX_NN>, memref<4x1x4x17xf16, @CMX_NN>){
    // CHECK:                                               VPUIP.SW.Kernel.run {attrs = [true, 0, 5.000000e-01, 1, [3]]}([[ARG8]], [[ARG10]]) : memref<4x1x6x17xf16, @CMX_NN>, memref<4x1x6x17xf16, @CMX_NN>
    // CHECK:                                               VPUIP.SW.Kernel.run {attrs = [true, 0, 5.000000e-01, 1, [3]]}([[ARG9]], [[ARG11]]) : memref<4x1x4x17xf16, @CMX_NN>, memref<4x1x4x17xf16, @CMX_NN>
    // CHECK:                                           }
    // CHECK:       }

    // CHECK:       [[ALLOC:%.*]] = memref.alloc() : memref<4x1x10x17xf16>
    // CHECK:       [[SUBVIEW2:%.*]] = VPUIP.SubView [[ALLOC]] [0, 0, 0, 0] [4, 1, 6, 17] : memref<4x1x10x17xf16> to memref<4x1x6x17xf16, {order = #NCHW, strides = [170, 170, 17, 1]}>
    // CHECK:       [[COPY_OUT_0:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:                          inputs([[MVN]]#0 as [[ARG12:%[^:]+]]: memref<4x1x6x17xf16, @CMX_NN>)
    // CHECK-SAME:                          outputs([[SUBVIEW2]] as [[ARG13:%[^:]+]]: memref<4x1x6x17xf16, {order = #NCHW, strides = [170, 170, 17, 1]}>) -> memref<4x1x6x17xf16, {order = #NCHW, strides = [170, 170, 17, 1]}> {
    // CHECK:                                       VPUIP.Copy
    // CHECK-SAME:                                      inputs([[ARG12]] : memref<4x1x6x17xf16, @CMX_NN>)
    // CHECK-SAME:                                      outputs([[ARG13]] : memref<4x1x6x17xf16, {order = #NCHW, strides = [170, 170, 17, 1]}>) -> memref<4x1x6x17xf16, {order = #NCHW, strides = [170, 170, 17, 1]}>
    // CHECK:       }
    // CHECK:       [[SUBVIEW3:%.*]] = VPUIP.SubView [[ALLOC]] [0, 0, 6, 0] [4, 1, 4, 17] : memref<4x1x10x17xf16> to memref<4x1x4x17xf16, {order = #NCHW, strides = [170, 170, 17, 1]}>
    // CHECK:       [[COPY_OUT_1:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:                          inputs([[MVN]]#1 as [[ARG14:%[^:]+]]: memref<4x1x4x17xf16, @CMX_NN>)
    // CHECK-SAME:                          outputs([[SUBVIEW3]] as [[ARG15:%[^:]+]]: memref<4x1x4x17xf16, {order = #NCHW, strides = [170, 170, 17, 1]}>) -> memref<4x1x4x17xf16, {order = #NCHW, strides = [170, 170, 17, 1]}> {
    // CHECK:                                       VPUIP.Copy
    // CHECK-SAME:                                      inputs([[ARG14]] : memref<4x1x4x17xf16, @CMX_NN>)
    // CHECK-SAME:                                      outputs([[ARG15]] : memref<4x1x4x17xf16, {order = #NCHW, strides = [170, 170, 17, 1]}>) -> memref<4x1x4x17xf16, {order = #NCHW, strides = [170, 170, 17, 1]}>
    // CHECK:       }

    // CHECK:       [[CONCAT:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:                          inputs([[COPY_OUT_0]], [[COPY_OUT_1]] : memref<4x1x6x17xf16, {order = #NCHW, strides = [170, 170, 17, 1]}>, memref<4x1x4x17xf16, {order = #NCHW, strides = [170, 170, 17, 1]}>)
    // CHECK-SAME:                          outputs(%alloc : memref<4x1x10x17xf16>) -> memref<4x1x10x17xf16>
    // CHECK:       return [[CONCAT]] : memref<4x1x10x17xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
  func.func private @builtin_SoftMax(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL:  func.func @TileClusterSoftmaxWithAlignment
// CHECK-SAME:     ([[INPUT:%.+]]: memref<1x30x12x1xf16>)
func.func @TileClusterSoftmaxWithAlignment(%arg0: memref<1x30x12x1xf16>)
        -> memref<1x30x12x1xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x30x12x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x30x12x1xf16>) outputs(%0 as %arg2: memref<1x30x12x1xf16, #NCHW, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x30x12x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %2 = VPUIP.Copy inputs(%arg1 : memref<1x30x12x1xf16>) outputs(%arg2 : memref<1x30x12x1xf16, @CMX_NN>) -> memref<1x30x12x1xf16, @CMX_NN>
    }
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x30x12x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %4 = VPUIP.NCEClusterTiling inputs(%1 as %arg1: memref<1x30x12x1xf16, #NCHW, @CMX_NN>) outputs(%3 as %arg2: memref<1x30x12x1xf16, #NCHW, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x30x12x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%arg1 as %arg6: memref<1x30x12x1xf16, #NCHW, @CMX_NN>) outputs(%arg2 as %arg7: memref<1x30x12x1xf16, #NCHW, @CMX_NN>) on tile 0 -> memref<1x30x12x1xf16, #NCHW, @CMX_NN> {
        VPUIP.SW.Kernel.run {attrs = [0]}(%arg6, %arg7) : memref<1x30x12x1xf16, @CMX_NN>, memref<1x30x12x1xf16, @CMX_NN>
      }
    }
    %5 = memref.alloc() : memref<1x30x12x1xf16>
    %6 = VPUIP.NCEClusterTiling inputs(%4 as %arg1: memref<1x30x12x1xf16, @CMX_NN>) outputs(%5 as %arg2: memref<1x30x12x1xf16>) -> memref<1x30x12x1xf16> {
      %7 = VPUIP.Copy inputs(%arg1 : memref<1x30x12x1xf16, @CMX_NN>) outputs(%arg2 : memref<1x30x12x1xf16>) -> memref<1x30x12x1xf16>
    }
    return %6: memref<1x30x12x1xf16>

    // CHECK:     [[INPUT_BUFFER:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x30x12x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:     [[INPUT_COPY:%.*]] = VPUIP.NCEClusterTiling inputs([[INPUT]] as [[ARG0:%.*]]: memref<1x30x12x1xf16>) outputs([[INPUT_BUFFER]] as [[ARG1:%.*]]: memref<1x30x12x1xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x30x12x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:                         VPUIP.Copy inputs([[ARG0]] : memref<1x30x12x1xf16>) outputs([[ARG1]] : memref<1x30x12x1xf16, @CMX_NN>) -> memref<1x30x12x1xf16, @CMX_NN>
    // CHECK:     }
    // CHECK:     [[INPUT_SUBVIEW0:%.*]] = VPUIP.SubView [[INPUT_COPY]] [0, 16, 0, 0] [1, 14, 12, 1] : !VPUIP.DistributedBuffer<1x30x12x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x14x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:     [[INPUT_SUBVIEW1:%.*]] = VPUIP.SubView [[INPUT_COPY]] [0, 0, 0, 0] [1, 16, 12, 1] : !VPUIP.DistributedBuffer<1x30x12x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x16x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:     [[OUTPUT_BUFFER:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x30x12x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:     [[OUTPUT_SUBVIEW0:%.*]] = VPUIP.SubView [[OUTPUT_BUFFER]] [0, 16, 0, 0] [1, 14, 12, 1] : !VPUIP.DistributedBuffer<1x30x12x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x14x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:     [[OUTPUT_SUBVIEW1:%.*]] = VPUIP.SubView [[OUTPUT_BUFFER]] [0, 0, 0, 0] [1, 16, 12, 1] : !VPUIP.DistributedBuffer<1x30x12x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x16x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:     [[SOFTMAX:%.*]]:2 = VPUIP.NCEClusterTiling
    // CHECK-SAME:                  inputs([[INPUT_SUBVIEW1]] as [[ARG0:%.*]]: memref<1x16x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN>, [[INPUT_SUBVIEW0]] as [[ARG1:%.*]]: memref<1x14x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN>) outputs([[OUTPUT_SUBVIEW1]] as [[ARG2:%.*]]: memref<1x16x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN>, [[OUTPUT_SUBVIEW0]] as [[ARG3:%.*]]: memref<1x14x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN>)
    // CHECK-SAME:                  -> (!VPUIP.DistributedBuffer<1x16x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x14x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) {
    // CHECK:                          VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_SoftMax
    // CHECK-SAME:                         inputs([[ARG0]] as [[ARG4:%.*]]: memref<1x16x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN>, [[ARG1]] as [[ARG5:%.*]]: memref<1x14x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN>) outputs([[ARG2]] as [[ARG6:%.*]]: memref<1x16x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN>, [[ARG3]] as [[ARG7:%.*]]: memref<1x14x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN>)
    // CHECK-SAME{LITERAL}:                strides([[180, 6, 1, 1], [180, 6, 1, 1]]) on tile 0 -> (memref<1x16x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN>, memref<1x14x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN>){
    // CHECK:                               VPUIP.SW.Kernel.run {attrs = [0]}([[ARG4]], [[ARG6]]) : memref<1x16x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN>, memref<1x16x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN>
    // CHECK:                               VPUIP.SW.Kernel.run {attrs = [0]}([[ARG5]], [[ARG7]]) : memref<1x14x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN>, memref<1x14x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN>
    // CHECK:                          }
    // CHECK:     }
    // CHECK:     [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[SOFTMAX]]#0, [[SOFTMAX]]#1 : !VPUIP.DistributedBuffer<1x16x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x14x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%4 : !VPUIP.DistributedBuffer<1x30x12x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x30x12x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:     [[OUTPUT_BUFFER_DDR:%.*]] = memref.alloc() : memref<1x30x12x1xf16>
    // CHECK:     [[OUTPUT_COPY:%.*]] = VPUIP.NCEClusterTiling inputs([[CONCAT]] as [[ARG0:%.*]]: memref<1x30x12x1xf16, @CMX_NN>) outputs([[OUTPUT_BUFFER_DDR]] as [[ARG1:%.*]]: memref<1x30x12x1xf16>) -> memref<1x30x12x1xf16> {
    // CHECK:                           VPUIP.Copy inputs([[ARG0]] : memref<1x30x12x1xf16, @CMX_NN>) outputs([[ARG1]] : memref<1x30x12x1xf16>) -> memref<1x30x12x1xf16>
    // CHECK:     }
    // CHECK:     return [[OUTPUT_COPY]] : memref<1x30x12x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module @VPU.SW {
    func.func private @builtin_Multiply(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "eltwise_mul.cpp", VPU.kernel_entry = "eltwise_mul", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

!origDistType = !VPUIP.DistributedBuffer<1x12x128x512xf16, #NCHW, @CMX_NN, {
                                mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
                                compute_shapes = [[1, 3, 128, 512], [1, 3, 128, 512], [1, 3, 128, 512], [1, 3, 128, 512]],
                                compute_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0]],
                                memory_shapes = [[1, 3, 128, 512], [1, 3, 128, 512], [1, 3, 128, 512], [1, 3, 128, 512]],
                                memory_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0]]}>
!origDistType1 = !VPUIP.DistributedBuffer<1x12x128x1xf16, #NCHW, @CMX_NN, {
                                mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
                                compute_shapes = [[1, 3, 128, 1], [1, 3, 128, 1], [1, 3, 128, 1], [1, 3, 128, 1]],
                                compute_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0]],
                                memory_shapes = [[1, 3, 128, 1], [1, 3, 128, 1], [1, 3, 128, 1], [1, 3, 128, 1]],
                                memory_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0]]}>

func.func @BalanceTileMultiply(%input0: !origDistType1, %input1: !origDistType)
        -> !origDistType {

    %0 = VPURT.AllocDistributed -> !origDistType
    %1 = VPUIP.NCEClusterTiling inputs(
        %input0 as %arg0: memref<1x12x128x1xf16, @CMX_NN>,
        %input1 as %arg1: memref<1x12x128x512xf16, @CMX_NN>)
        outputs(%0 as %arg2: memref<1x12x128x512xf16, @CMX_NN>)
            -> !origDistType {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Multiply inputs(
        %arg0 as %arg3: memref<1x12x128x1xf16, @CMX_NN>,
        %arg1 as %arg4: memref<1x12x128x512xf16, @CMX_NN>)
        outputs(%arg2 as %arg5: memref<1x12x128x512xf16, @CMX_NN>) on tile 0 -> memref<1x12x128x512xf16, @CMX_NN>{
        VPUIP.SW.Kernel.run(%arg3, %arg4, %arg5) : memref<1x12x128x1xf16, @CMX_NN>, memref<1x12x128x512xf16, @CMX_NN>, memref<1x12x128x512xf16, @CMX_NN>
      }
    }

    return %1: !origDistType

    // CHECK:       [[IN_SHAPECAST_0:%.+]] = VPUIP.ShapeCast
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_offsets = [[0, 0, 0, 0], [0, 384, 0, 0], [0, 768, 0, 0], [0, 1152, 0, 0]]
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_shapes = [[1, 384, 1, 1], [1, 384, 1, 1], [1, 384, 1, 1], [1, 384, 1, 1]]
    // CHECK-SAME-DAG{LITERAL}:      shape = [1, 1536, 1, 1]
    // CHECK-DAG:   [[SUBVIEW_0_0:%.+]] = VPUIP.SubView [[IN_SHAPECAST_0]] [0, 768, 0, 0] [1, 768, 1, 1]
    // CHECK-DAG:   [[SUBVIEW_0_1:%.+]] = VPUIP.SubView [[IN_SHAPECAST_0]] [0, 0, 0, 0] [1, 768, 1, 1]

    // CHECK:       [[IN_SHAPECAST_1:%.+]] = VPUIP.ShapeCast
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_offsets = [[0, 0, 0, 0], [0, 384, 0, 0], [0, 768, 0, 0], [0, 1152, 0, 0]]
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_shapes = [[1, 384, 1, 512], [1, 384, 1, 512], [1, 384, 1, 512], [1, 384, 1, 512]]
    // CHECK-SAME-DAG{LITERAL}:      shape = [1, 1536, 1, 512]
    // CHECK-DAG:   [[SUBVIEW_1_0:%.+]] = VPUIP.SubView [[IN_SHAPECAST_1]] [0, 768, 0, 0] [1, 768, 1, 512]
    // CHECK-DAG:   [[SUBVIEW_1_1:%.+]] = VPUIP.SubView [[IN_SHAPECAST_1]] [0, 0, 0, 0] [1, 768, 1, 512]

    // CHECK:       [[OUT_BUFF:%.+]] = VPURT.AllocDistributed
    // CHECK-DAG:   [[SUBVIEW_OUT_0:%.+]] = VPUIP.SubView [[OUT_BUFF]] [0, 768, 0, 0] [1, 768, 1, 512]
    // CHECK-DAG:   [[SUBVIEW_OUT_1:%.+]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 0, 0] [1, 768, 1, 512]

    // CHECK:       [[NCECluster:%.+]]:2 = VPUIP.NCEClusterTiling
    // CHECK-SAME{LITERAL}:         compute_shapes = [[1, 192, 1, 512], [1, 192, 1, 512], [1, 192, 1, 512], [1, 192, 1, 512]]
    // CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 192, 0, 0], [0, 384, 0, 0], [0, 576, 0, 0]]
    // CHECK-SAME{LITERAL}:         memory_shapes = [[1, 192, 1, 512], [1, 192, 1, 512], [1, 192, 1, 512], [1, 192, 1, 512]]
    // CHECK-SAME{LITERAL}:         memory_offsets = [[0, 0, 0, 0], [0, 192, 0, 0], [0, 384, 0, 0], [0, 576, 0, 0]]
    // CHECK:           [[SW_KERNEL:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              memref<1x768x1x1xf16, {order = #NCHW, strides = [1536, 1, 1, 1]}, @CMX_NN>
    // CHECK-SAME:              memref<1x768x1x512xf16, {order = #NCHW, strides = [786432, 512, 512, 1]}, @CMX_NN>
    // CHECK-SAME:              memref<1x768x1x512xf16, {order = #NCHW, strides = [786432, 512, 512, 1]}, @CMX_NN>
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              memref<1x768x1x1xf16, {order = #NCHW, strides = [1536, 1, 1, 1]}, @CMX_NN>
    // CHECK-SAME:              memref<1x768x1x512xf16, {order = #NCHW, strides = [786432, 512, 512, 1]}, @CMX_NN>
    // CHECK-SAME:              memref<1x768x1x512xf16, {order = #NCHW, strides = [786432, 512, 512, 1]}, @CMX_NN>

    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK:       [[OUT_SHAPECAST:%.+]] = VPUIP.ShapeCast
    // CHECK-SAME-DAG{LITERAL}:       explicit_output_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0]]
    // CHECK-SAME-DAG{LITERAL}:       explicit_output_shapes = [[1, 3, 128, 512], [1, 3, 128, 512], [1, 3, 128, 512], [1, 3, 128, 512]]
    // CHECK-SAME-DAG{LITERAL}:       shape = [1, 12, 128, 512]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @VPU.SW {
    func.func private @builtin_SoftMax(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64) attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

!origDistType = !VPUIP.DistributedBuffer<1x4096x30x4xf16, #NHWC, @CMX_NN, {
                    mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
                    compute_shapes = [[1, 4096, 5, 4], [1, 4096, 5, 4], [1, 4096, 5, 4], [1, 4096, 5, 4], [1, 4096, 5, 4], [1, 4096, 5, 4]],
                    compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 25, 0]],
                    memory_shapes = [[1, 4096, 5, 4], [1, 4096, 5, 4], [1, 4096, 5, 4], [1, 4096, 5, 4], [1, 4096, 5, 4], [1, 4096, 5, 4]],
                    memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 25, 0]]}>

func.func @BalanceTileSoftmax(%input: !origDistType)
        -> !origDistType {

    %0 = VPURT.AllocDistributed -> !origDistType

    %1 = VPUIP.NCEClusterTiling inputs(%input as %arg4: memref<1x4096x30x4xf16, #NHWC, @CMX_NN>)
            outputs(%0 as %arg5: memref<1x4096x30x4xf16, #NHWC, @CMX_NN>) -> !origDistType {
      %results_2080 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax
            inputs(%arg4 as %arg6: memref<1x4096x30x4xf16, #NHWC, @CMX_NN>)
            outputs(%arg5 as %arg7: memref<1x4096x30x4xf16, #NHWC, @CMX_NN>) on tile 0 -> memref<1x4096x30x4xf16, #NHWC, @CMX_NN>{
        VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg6, %arg7) : memref<1x4096x30x4xf16, #NHWC, @CMX_NN>, memref<1x4096x30x4xf16, #NHWC, @CMX_NN>
      }
    }

    return %1: !origDistType

    // CHECK:       [[IN_SHAPECAST_0:%.+]] = VPUIP.ShapeCast
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_offsets = [[0, 0, 0, 0], [0, 0, 20, 0], [0, 0, 40, 0], [0, 0, 60, 0], [0, 0, 80, 0], [0, 0, 100, 0]]
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_shapes = [[1, 4096, 20, 1], [1, 4096, 20, 1], [1, 4096, 20, 1], [1, 4096, 20, 1], [1, 4096, 20, 1], [1, 4096, 20, 1]]
    // CHECK-SAME-DAG{LITERAL}:      shape = [1, 4096, 120, 1]
    // CHECK-DAG:   [[SUBVIEW_0_0:%.+]] = VPUIP.SubView [[IN_SHAPECAST_0]] [0, 0, 60, 0] [1, 4096, 60, 1]
    // CHECK-DAG:   [[SUBVIEW_0_1:%.+]] = VPUIP.SubView [[IN_SHAPECAST_0]] [0, 0, 0, 0] [1, 4096, 60, 1]

    // CHECK:       [[OUT_BUFF:%.+]] = VPURT.AllocDistributed
    // CHECK-DAG:   [[SUBVIEW_OUT_0:%.+]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 60, 0] [1, 4096, 60, 1]
    // CHECK-DAG:   [[SUBVIEW_OUT_1:%.+]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 0, 0] [1, 4096, 60, 1]

    // CHECK:       [[NCECluster:%.+]]:2 = VPUIP.NCEClusterTiling
    // CHECK-SAME{LITERAL}:         compute_shapes = [[1, 4096, 10, 1], [1, 4096, 10, 1], [1, 4096, 10, 1], [1, 4096, 10, 1], [1, 4096, 10, 1], [1, 4096, 10, 1]]
    // CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 20, 0], [0, 0, 30, 0], [0, 0, 40, 0], [0, 0, 50, 0]]
    // CHECK-SAME{LITERAL}:         memory_shapes = [[1, 4096, 10, 1], [1, 4096, 10, 1], [1, 4096, 10, 1], [1, 4096, 10, 1], [1, 4096, 10, 1], [1, 4096, 10, 1]]
    // CHECK-SAME{LITERAL}:         memory_offsets = [[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 20, 0], [0, 0, 30, 0], [0, 0, 40, 0], [0, 0, 50, 0]]
    // CHECK:           [[SW_KERNEL:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              memref<1x4096x60x1xf16, {order = #NHWC, strides = [491520, 1, 4096, 4096]}, @CMX_NN>
    // CHECK-SAME:              memref<1x4096x60x1xf16, {order = #NHWC, strides = [491520, 1, 4096, 4096]}, @CMX_NN>
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              memref<1x4096x60x1xf16, {order = #NHWC, strides = [491520, 1, 4096, 4096]}, @CMX_NN>
    // CHECK-SAME:              memref<1x4096x60x1xf16, {order = #NHWC, strides = [491520, 1, 4096, 4096]}, @CMX_NN>

    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK:       [[OUT_SHAPECAST:%.+]] = VPUIP.ShapeCast
    // CHECK-SAME-DAG{LITERAL}:       explicit_output_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 25, 0]]
    // CHECK-SAME-DAG{LITERAL}:       explicit_output_shapes = [[1, 4096, 5, 4], [1, 4096, 5, 4], [1, 4096, 5, 4], [1, 4096, 5, 4], [1, 4096, 5, 4], [1, 4096, 5, 4]]
    // CHECK-SAME-DAG{LITERAL}:       shape = [1, 4096, 30, 4]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
    func.func private @builtin_Sin(memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "activation_sin.cpp", VPU.kernel_entry = "activation_sin"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

!Distributed = !VPUIP.DistributedBuffer<1x4x12x481xf16, #NHWC, @CMX_NN, {
                                mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
                                compute_shapes = [[1, 4, 3, 481], [1, 4, 3, 481], [1, 4, 3, 481], [1, 4, 3, 481]],
                                compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0]],
                                memory_shapes = [[1, 4, 3, 481], [1, 4, 3, 481], [1, 4, 3, 481], [1, 4, 3, 481]],
                                memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0]]}>

// CHECK-LABEL:   @BalanceTileSinOp
func.func @BalanceTileSinOp(%input: !Distributed) -> !Distributed {

    %0 = VPURT.AllocDistributed -> !Distributed
    %1 = VPUIP.NCEClusterTiling inputs(%input as %arg0: memref<1x4x12x481xf16, #NHWC, @CMX_NN>)
                                outputs(%0 as %arg1: memref<1x4x12x481xf16, #NHWC, @CMX_NN>) -> !Distributed {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Sin
        inputs(%arg0 as %arg2: memref<1x4x12x481xf16, #NHWC, @CMX_NN>)
        outputs(%arg1 as %arg3: memref<1x4x12x481xf16, #NHWC, @CMX_NN>) on tile 0 -> memref<1x4x12x481xf16, #NHWC, @CMX_NN>{
          VPUIP.SW.Kernel.run(%arg2, %arg3) : memref<1x4x12x481xf16, #NHWC, @CMX_NN>, memref<1x4x12x481xf16, #NHWC, @CMX_NN>
      }
    }

    return %1: !Distributed

    // CHECK:       [[IN_SHAPECAST:%.+]] = VPUIP.ShapeCast
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_offsets = [[0, 0, 0, 0], [0, 0, 5772, 0], [0, 0, 11544, 0], [0, 0, 17316, 0]]
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_shapes = [[1, 1, 5772, 1], [1, 1, 5772, 1], [1, 1, 5772, 1], [1, 1, 5772, 1]]
    // CHECK-SAME-DAG{LITERAL}:      shape = [1, 23088, 1, 1]
    // CHECK-DAG:   [[SUBVIEW_0:%.+]] = VPUIP.SubView [[IN_SHAPECAST]] [0, 0, 11584, 0] [1, 1, 11504, 1]
    // CHECK-DAG:   [[SUBVIEW_1:%.+]] = VPUIP.SubView [[IN_SHAPECAST]] [0, 0, 0, 0] [1, 1, 11584, 1]

    // CHECK:       [[OUT_BUFF:%.+]] = VPURT.AllocDistributed
    // CHECK-DAG:   [[SUBVIEW_OUT_0:%.+]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 11584, 0] [1, 1, 11504, 1]
    // CHECK-DAG:   [[SUBVIEW_OUT_1:%.+]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 0, 0] [1, 1, 11584, 1]

    // CHECK:       [[NCECluster:%.+]]:2 = VPUIP.NCEClusterTiling
    // CHECK-SAME{LITERAL}:         compute_shapes = [[1, 1, 2896, 1], [1, 1, 2896, 1], [1, 1, 2896, 1], [1, 1, 2896, 1]]
    // CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 0, 2896, 0], [0, 0, 5792, 0], [0, 0, 8688, 0]]
    // CHECK-SAME{LITERAL}:         memory_shapes = [[1, 1, 2896, 1], [1, 1, 2896, 1], [1, 1, 2896, 1], [1, 1, 2896, 1]]
    // CHECK-SAME{LITERAL}:         memory_offsets = [[0, 0, 0, 0], [0, 0, 2896, 0], [0, 0, 5792, 0], [0, 0, 8688, 0]]
    // CHECK:           [[SW_KERNEL:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              memref<1x1x11584x1xf16, {order = #NHWC, strides = [23088, 1, 1, 1]}, @CMX_NN>
    // CHECK-SAME:              memref<1x1x11584x1xf16, {order = #NHWC, strides = [23088, 1, 1, 1]}, @CMX_NN>
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              memref<1x1x11504x1xf16, {order = #NHWC, strides = [23088, 1, 1, 1]}, @CMX_NN>
    // CHECK-SAME:              memref<1x1x11504x1xf16, {order = #NHWC, strides = [23088, 1, 1, 1]}, @CMX_NN>

    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK:       [[OUT_SHAPECAST:%.+]] = VPUIP.ShapeCast
    // CHECK-SAME-DAG{LITERAL}:       explicit_output_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0]]
    // CHECK-SAME-DAG{LITERAL}:       explicit_output_shapes = [[1, 4, 3, 481], [1, 4, 3, 481], [1, 4, 3, 481], [1, 4, 3, 481]]
    // CHECK-SAME-DAG{LITERAL}:       shape = [1, 4, 12, 481]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
    func.func private @builtin_Cos(memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "activation_cos.cpp", VPU.kernel_entry = "activation_cos"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

!Distributed = !VPUIP.DistributedBuffer<1x32x12x1xf16, #NHWC, @CMX_NN, {
                                mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
                                compute_shapes = [[1, 32, 3, 1], [1, 32, 3, 1], [1, 32, 3, 1], [1, 32, 3, 1]],
                                compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0]],
                                memory_shapes = [[1, 32, 3, 1], [1, 32, 3, 1], [1, 32, 3, 1], [1, 32, 3, 1]],
                                memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0]]}>

// CHECK-LABEL:   @BalanceTileCosOp
func.func @BalanceTileCosOp(%input: !Distributed) -> !Distributed {

    %0 = VPURT.AllocDistributed -> !Distributed
    %1 = VPUIP.NCEClusterTiling inputs(%input as %arg0: memref<1x32x12x1xf16, #NHWC, @CMX_NN>)
                                outputs(%0 as %arg1: memref<1x32x12x1xf16, #NHWC, @CMX_NN>) -> !Distributed {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Cos
        inputs(%arg0 as %arg2: memref<1x32x12x1xf16, #NHWC, @CMX_NN>)
        outputs(%arg1 as %arg3: memref<1x32x12x1xf16, #NHWC, @CMX_NN>) on tile 0 -> memref<1x32x12x1xf16, #NHWC, @CMX_NN>{
          VPUIP.SW.Kernel.run(%arg2, %arg3) : memref<1x32x12x1xf16, #NHWC, @CMX_NN>, memref<1x32x12x1xf16, #NHWC, @CMX_NN>
      }
    }

    return %1: !Distributed

    // CHECK:       [[IN_SHAPECAST:%.+]] = VPUIP.ShapeCast
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_offsets = [[0, 0, 0, 0], [0, 0, 96, 0], [0, 0, 192, 0], [0, 0, 288, 0]]
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_shapes = [[1, 1, 96, 1], [1, 1, 96, 1], [1, 1, 96, 1], [1, 1, 96, 1]]
    // CHECK-SAME-DAG{LITERAL}:      shape = [1, 1, 384, 1]
    // CHECK-DAG:   [[SUBVIEW_0:%.+]] = VPUIP.SubView [[IN_SHAPECAST]] [0, 0, 192, 0] [1, 1, 192, 1]
    // CHECK-DAG:   [[SUBVIEW_1:%.+]] = VPUIP.SubView [[IN_SHAPECAST]] [0, 0, 0, 0] [1, 1, 192, 1]

    // CHECK:       [[OUT_BUFF:%.+]] = VPURT.AllocDistributed
    // CHECK-DAG:   [[SUBVIEW_OUT_0:%.+]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 192, 0] [1, 1, 192, 1]
    // CHECK-DAG:   [[SUBVIEW_OUT_1:%.+]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 0, 0] [1, 1, 192, 1]

    // CHECK:       [[NCECluster:%.+]]:2 = VPUIP.NCEClusterTiling
    // CHECK-SAME{LITERAL}:         compute_shapes = [[1, 1, 48, 1], [1, 1, 48, 1], [1, 1, 48, 1], [1, 1, 48, 1]]
    // CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 0, 48, 0], [0, 0, 96, 0], [0, 0, 144, 0]]
    // CHECK-SAME{LITERAL}:         memory_shapes = [[1, 1, 48, 1], [1, 1, 48, 1], [1, 1, 48, 1], [1, 1, 48, 1]]
    // CHECK-SAME{LITERAL}:         memory_offsets = [[0, 0, 0, 0], [0, 0, 48, 0], [0, 0, 96, 0], [0, 0, 144, 0]]
    // CHECK:           [[SW_KERNEL:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              memref<1x1x192x1xf16, {order = #NHWC, strides = [384, 1, 1, 1]}, @CMX_NN>
    // CHECK-SAME:              memref<1x1x192x1xf16, {order = #NHWC, strides = [384, 1, 1, 1]}, @CMX_NN>
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              memref<1x1x192x1xf16, {order = #NHWC, strides = [384, 1, 1, 1]}, @CMX_NN>
    // CHECK-SAME:              memref<1x1x192x1xf16, {order = #NHWC, strides = [384, 1, 1, 1]}, @CMX_NN>

    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK:       [[OUT_SHAPECAST:%.+]] = VPUIP.ShapeCast
    // CHECK-SAME-DAG{LITERAL}:       explicit_output_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0]]
    // CHECK-SAME-DAG{LITERAL}:       explicit_output_shapes = [[1, 32, 3, 1], [1, 32, 3, 1], [1, 32, 3, 1], [1, 32, 3, 1]]
    // CHECK-SAME-DAG{LITERAL}:       shape = [1, 32, 12, 1]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

module @VPU.SW {
    func.func private @builtin_Gelu(memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "activation_gelu.cpp", VPU.kernel_entry = "activation_gelu"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

!Distributed = !VPUIP.DistributedBuffer<1x352x3x4xf16, #NHWC, @CMX_NN, {
                                mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
                                compute_shapes = [[1, 96, 3, 4], [1, 96, 3, 4], [1, 80, 3, 4], [1, 80, 3, 4]],
                                compute_offsets = [[0, 0, 0, 0], [0, 96, 0, 0], [0, 192, 0, 0], [0, 272, 0, 0]],
                                memory_shapes = [[1, 96, 3, 4], [1, 96, 3, 4], [1, 80, 3, 4], [1, 80, 3, 4]],
                                memory_offsets = [[0, 0, 0, 0], [0, 96, 0, 0], [0, 192, 0, 0], [0, 272, 0, 0]]}>

// CHECK-LABEL:   @AdjustGeluLayoutToInsertSubviewOnly
func.func @AdjustGeluLayoutToInsertSubviewOnly(%input: !Distributed) -> !Distributed {

    %0 = VPURT.AllocDistributed -> !Distributed
    %1 = VPUIP.NCEClusterTiling inputs(%input as %arg0: memref<1x352x3x4xf16, #NHWC, @CMX_NN>)
                                outputs(%0 as %arg1: memref<1x352x3x4xf16, #NHWC, @CMX_NN>) -> !Distributed {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Gelu
        inputs(%arg0 as %arg2: memref<1x352x3x4xf16, #NHWC, @CMX_NN>)
        outputs(%arg1 as %arg3: memref<1x352x3x4xf16, #NHWC, @CMX_NN>) on tile 0 -> memref<1x352x3x4xf16, #NHWC, @CMX_NN>{
          VPUIP.SW.Kernel.run(%arg2, %arg3) : memref<1x352x3x4xf16, #NHWC, @CMX_NN>, memref<1x352x3x4xf16, #NHWC, @CMX_NN>
      }
    }

    return %1: !Distributed

    // CHECK:       [[IN_PERMUTECAST:%.+]] = VPUIP.PermuteCast {dst_order = #NCWH, mem_perm = #NCWH}
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x352x3x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x352x3x4xf16, #NCWH, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]
    // CHECK-DAG:   [[SUBVIEW_0:%.+]] = VPUIP.SubView [[IN_PERMUTECAST]] [0, 192, 0, 0] [1, 160, 3, 4]
    // CHECK-DAG:   [[SUBVIEW_1:%.+]] = VPUIP.SubView [[IN_PERMUTECAST]] [0, 0, 0, 0] [1, 192, 3, 4]

    // CHECK:       [[OUT_BUFF:%.+]] = VPURT.AllocDistributed
    // CHECK-DAG:   [[SUBVIEW_OUT_0:%.+]] = VPUIP.SubView [[OUT_BUFF]] [0, 192, 0, 0] [1, 160, 3, 4]
    // CHECK-DAG:   [[SUBVIEW_OUT_1:%.+]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 0, 0] [1, 192, 3, 4]

    // CHECK:       [[NCECluster:%.+]]:2 = VPUIP.NCEClusterTiling
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x192x3x4xf16, {order = #NCWH, strides = [4224, 12, 1, 3]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64
    // CHECK-SAME{LITERAL}:         compute_shapes = [[1, 48, 3, 4], [1, 48, 3, 4], [1, 48, 3, 4], [1, 48, 3, 4]]
    // CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 48, 0, 0], [0, 96, 0, 0], [0, 144, 0, 0]]
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x160x3x4xf16, {order = #NCWH, strides = [4224, 12, 1, 3]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64
    // CHECK-SAME{LITERAL}:         compute_shapes = [[1, 48, 3, 4], [1, 48, 3, 4], [1, 32, 3, 4], [1, 32, 3, 4]]
    // CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 48, 0, 0], [0, 96, 0, 0], [0, 128, 0, 0]]
    // CHECK:           [[SW_KERNEL:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              memref<1x192x3x4xf16, {order = #NCWH, strides = [4224, 12, 1, 3]}, @CMX_NN>
    // CHECK-SAME:              memref<1x192x3x4xf16, {order = #NCWH, strides = [4224, 12, 1, 3]}, @CMX_NN>
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              memref<1x160x3x4xf16, {order = #NCWH, strides = [4224, 12, 1, 3]}, @CMX_NN>
    // CHECK-SAME:              memref<1x160x3x4xf16, {order = #NCWH, strides = [4224, 12, 1, 3]}, @CMX_NN>

    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK:       [[OUT_PERMUTECAST:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x352x3x4xf16, #NCWH, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x352x3x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

module @VPU.SW {
    func.func private @builtin_Multiply(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "eltwise_mul.cpp", VPU.kernel_entry = "eltwise_mul", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

!Distributed = !VPUIP.DistributedBuffer<1x1280x113x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
                compute_shapes = [[1, 224, 113, 4], [1, 224, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4]],
                compute_offsets =  [[0, 0, 0, 0], [0, 224, 0, 0], [0, 448, 0, 0], [0, 656, 0, 0], [0, 864, 0, 0], [0, 1072, 0, 0]],
                memory_shapes = [[1, 224, 113, 4], [1, 224, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4]],
                memory_offsets = [[0, 0, 0, 0], [0, 224, 0, 0], [0, 448, 0, 0], [0, 656, 0, 0], [0, 864, 0, 0], [0, 1072, 0, 0]]}>

// CHECK-LABEL:   @AdjustMultiplyLayoutToInsertSubviewOnly
func.func @AdjustMultiplyLayoutToInsertSubviewOnly(%input1: !Distributed, %input2: !Distributed) -> !Distributed {
    %alloc = VPURT.AllocDistributed -> !Distributed
    %ncecluster = VPUIP.NCEClusterTiling
        inputs(%input1 as %arg4: memref<1x1280x113x4xf16, #NHWC, @CMX_NN>, %input2 as %arg5: memref<1x1280x113x4xf16, #NHWC, @CMX_NN>)
        outputs(%alloc as %arg6: memref<1x1280x113x4xf16, #NHWC, @CMX_NN>) -> !Distributed {
      %swkernel = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Multiply inputs(%arg4 as %arg7: memref<1x1280x113x4xf16, #NHWC, @CMX_NN>,
                %arg5 as %arg8: memref<1x1280x113x4xf16, #NHWC, @CMX_NN>) outputs(%arg6 as %arg9: memref<1x1280x113x4xf16, #NHWC, @CMX_NN>) on tile 0 -> memref<1x1280x113x4xf16, #NHWC, @CMX_NN>{
        VPUIP.SW.Kernel.run(%arg7, %arg8, %arg9) : memref<1x1280x113x4xf16, #NHWC, @CMX_NN>, memref<1x1280x113x4xf16, #NHWC, @CMX_NN>, memref<1x1280x113x4xf16, #NHWC, @CMX_NN>
      }
    }
    return %ncecluster : !Distributed

    // CHECK:       [[IN_PERMUTECAST_0:%.+]] = VPUIP.PermuteCast {dst_order = #NCWH, mem_perm = #NCWH}
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x1280x113x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1]
    // CHECK-SAME{LITERAL}:              compute_shapes = [[1, 224, 113, 4], [1, 224, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4]]
    // CHECK-SAME{LITERAL}:              compute_offsets = [[0, 0, 0, 0], [0, 224, 0, 0], [0, 448, 0, 0], [0, 656, 0, 0], [0, 864, 0, 0], [0, 1072, 0, 0]]
    // CHECK-SAME{LITERAL}:              memory_shapes = [[1, 224, 113, 4], [1, 224, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4]]
    // CHECK-SAME{LITERAL}:              memory_offsets = [[0, 0, 0, 0], [0, 224, 0, 0], [0, 448, 0, 0], [0, 656, 0, 0], [0, 864, 0, 0], [0, 1072, 0, 0]]
    // CHECK-SAME:              -> !VPUIP.DistributedBuffer<1x1280x113x4xf16, #NCWH, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1]
    // CHECK-SAME{LITERAL}:              compute_shapes = [[1, 224, 113, 4], [1, 224, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4]]
    // CHECK-SAME{LITERAL}:              compute_offsets = [[0, 0, 0, 0], [0, 224, 0, 0], [0, 448, 0, 0], [0, 656, 0, 0], [0, 864, 0, 0], [0, 1072, 0, 0]]
    // CHECK-SAME{LITERAL}:              memory_shapes = [[1, 224, 113, 4], [1, 224, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4]]
    // CHECK-SAME{LITERAL}:              memory_offsets = [[0, 0, 0, 0], [0, 224, 0, 0], [0, 448, 0, 0], [0, 656, 0, 0], [0, 864, 0, 0], [0, 1072, 0, 0]]
    // CHECK-DAG:   [[SUBVIEW_0_0:%.+]] = VPUIP.SubView [[IN_PERMUTECAST_0]] [0, 672, 0, 0] [1, 608, 113, 4]
    // CHECK-DAG:   [[SUBVIEW_0_1:%.+]] = VPUIP.SubView [[IN_PERMUTECAST_0]] [0, 0, 0, 0] [1, 672, 113, 4]

    // CHECK:       [[IN_PERMUTECAST_1:%.+]] = VPUIP.PermuteCast {dst_order = #NCWH, mem_perm = #NCWH}
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x1280x113x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1]
    // CHECK-SAME{LITERAL}:              compute_shapes = [[1, 224, 113, 4], [1, 224, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4]]
    // CHECK-SAME{LITERAL}:              compute_offsets = [[0, 0, 0, 0], [0, 224, 0, 0], [0, 448, 0, 0], [0, 656, 0, 0], [0, 864, 0, 0], [0, 1072, 0, 0]]
    // CHECK-SAME{LITERAL}:              memory_shapes = [[1, 224, 113, 4], [1, 224, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4]]
    // CHECK-SAME{LITERAL}:              memory_offsets = [[0, 0, 0, 0], [0, 224, 0, 0], [0, 448, 0, 0], [0, 656, 0, 0], [0, 864, 0, 0], [0, 1072, 0, 0]]
    // CHECK-SAME:              -> !VPUIP.DistributedBuffer<1x1280x113x4xf16, #NCWH, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1]
    // CHECK-SAME{LITERAL}:              compute_shapes = [[1, 224, 113, 4], [1, 224, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4]]
    // CHECK-SAME{LITERAL}:              compute_offsets = [[0, 0, 0, 0], [0, 224, 0, 0], [0, 448, 0, 0], [0, 656, 0, 0], [0, 864, 0, 0], [0, 1072, 0, 0]]
    // CHECK-SAME{LITERAL}:              memory_shapes = [[1, 224, 113, 4], [1, 224, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4]]
    // CHECK-SAME{LITERAL}:              memory_offsets = [[0, 0, 0, 0], [0, 224, 0, 0], [0, 448, 0, 0], [0, 656, 0, 0], [0, 864, 0, 0], [0, 1072, 0, 0]]
    // CHECK-DAG:   [[SUBVIEW_1_0:%.+]] = VPUIP.SubView [[IN_PERMUTECAST_1]] [0, 672, 0, 0] [1, 608, 113, 4]
    // CHECK-DAG:   [[SUBVIEW_1_1:%.+]] = VPUIP.SubView [[IN_PERMUTECAST_1]] [0, 0, 0, 0] [1, 672, 113, 4]

    // CHECK:       [[ALLOC:%.+]] = VPURT.AllocDistributed
    // CHECK:       [[ALC_SUBVIEW_0:%.+]] VPUIP.SubView %6 [0, 672, 0, 0] [1, 608, 113, 4]
    // CHECK:       [[ALC_SUBVIEW_0:%.+]] VPUIP.SubView %6 [0, 0, 0, 0] [1, 672, 113, 4]

    // CHECK:       [[NCECluster:%.+]]:2 = VPUIP.NCEClusterTiling
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x672x113x4xf16, {order = #NCWH, strides = [578560, 452, 1, 113]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64,
    // CHECK-SAME{LITERAL}:         compute_shapes = [[1, 112, 113, 4], [1, 112, 113, 4], [1, 112, 113, 4], [1, 112, 113, 4], [1, 112, 113, 4], [1, 112, 113, 4]],
    // CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 112, 0, 0], [0, 224, 0, 0], [0, 336, 0, 0], [0, 448, 0, 0], [0, 560, 0, 0]],
    // CHECK-SAME{LITERAL}:         memory_shapes = [[1, 112, 113, 4], [1, 112, 113, 4], [1, 112, 113, 4], [1, 112, 113, 4], [1, 112, 113, 4], [1, 112, 113, 4]],
    // CHECK-SAME{LITERAL}:         memory_offsets = [[0, 0, 0, 0], [0, 112, 0, 0], [0, 224, 0, 0], [0, 336, 0, 0], [0, 448, 0, 0], [0, 560, 0, 0]]}>,
    // CHECK-SAME{LITERAL}:     !VPUIP.DistributedBuffer<1x608x113x4xf16, {order = #NCWH, strides = [578560, 452, 1, 113]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64,
    // CHECK-SAME{LITERAL}:         compute_shapes = [[1, 112, 113, 4], [1, 112, 113, 4], [1, 96, 113, 4], [1, 96, 113, 4], [1, 96, 113, 4], [1, 96, 113, 4]],
    // CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 112, 0, 0], [0, 224, 0, 0], [0, 320, 0, 0], [0, 416, 0, 0], [0, 512, 0, 0]],
    // CHECK-SAME{LITERAL}:         memory_shapes = [[1, 112, 113, 4], [1, 112, 113, 4], [1, 96, 113, 4], [1, 96, 113, 4], [1, 96, 113, 4], [1, 96, 113, 4]],
    // CHECK-SAME{LITERAL}:         memory_offsets = [[0, 0, 0, 0], [0, 112, 0, 0], [0, 224, 0, 0], [0, 320, 0, 0], [0, 416, 0, 0], [0, 512, 0, 0]]}>)
    // CHECK:           [[SW_KERNEL:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              memref<1x672x113x4xf16, {order = #NCWH, strides = [578560, 452, 1, 113]}, @CMX_NN>
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              memref<1x608x113x4xf16, {order = #NCWH, strides = [578560, 452, 1, 113]}, @CMX_NN>

    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[NCECluster]]#0, [[NCECluster]]#1
    // CHECK:       [[OUT_PERMUTECAST:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:              inputs([[CONCAT]] : !VPUIP.DistributedBuffer<1x1280x113x4xf16, #NCWH, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1],
    // CHECK-SAME{LITERAL}:              compute_shapes = [[1, 224, 113, 4], [1, 224, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4]]
    // CHECK-SAME{LITERAL}:              compute_offsets = [[0, 0, 0, 0], [0, 224, 0, 0], [0, 448, 0, 0], [0, 656, 0, 0], [0, 864, 0, 0], [0, 1072, 0, 0]]
    // CHECK-SAME{LITERAL}:              memory_shapes = [[1, 224, 113, 4], [1, 224, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4]]
    // CHECK-SAME{LITERAL}:              memory_offsets = [[0, 0, 0, 0], [0, 224, 0, 0], [0, 448, 0, 0], [0, 656, 0, 0], [0, 864, 0, 0], [0, 1072, 0, 0]]
    // CHECK-SAME:              -> !VPUIP.DistributedBuffer<1x1280x113x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1]
    // CHECK-SAME{LITERAL}:              compute_shapes = [[1, 224, 113, 4], [1, 224, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4]]
    // CHECK-SAME{LITERAL}:              compute_offsets = [[0, 0, 0, 0], [0, 224, 0, 0], [0, 448, 0, 0], [0, 656, 0, 0], [0, 864, 0, 0], [0, 1072, 0, 0]]
    // CHECK-SAME{LITERAL}:              memory_shapes = [[1, 224, 113, 4], [1, 224, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4]]
    // CHECK-SAME{LITERAL}:              memory_offsets = [[0, 0, 0, 0], [0, 224, 0, 0], [0, 448, 0, 0], [0, 656, 0, 0], [0, 864, 0, 0], [0, 1072, 0, 0]]

    // CHECK:       return [[OUT_PERMUTECAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
    func.func private @builtin_Multiply(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "eltwise_mul.cpp", VPU.kernel_entry = "eltwise_mul", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

!Distributed = !VPUIP.DistributedBuffer<1x1280x113x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
                compute_shapes = [[1, 224, 113, 4], [1, 224, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4]],
                compute_offsets =  [[0, 0, 0, 0], [0, 224, 0, 0], [0, 448, 0, 0], [0, 656, 0, 0], [0, 864, 0, 0], [0, 1072, 0, 0]],
                memory_shapes = [[1, 224, 113, 4], [1, 224, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4]],
                memory_offsets = [[0, 0, 0, 0], [0, 224, 0, 0], [0, 448, 0, 0], [0, 656, 0, 0], [0, 864, 0, 0], [0, 1072, 0, 0]]}>

!Distributed1 = !VPUIP.DistributedBuffer<1x1280x113x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
                compute_shapes = [[1, 224, 113, 1], [1, 224, 113, 1], [1, 208, 113, 1], [1, 208, 113, 1], [1, 208, 113, 1], [1, 208, 113, 1]],
                compute_offsets =  [[0, 0, 0, 0], [0, 224, 0, 0], [0, 448, 0, 0], [0, 656, 0, 0], [0, 864, 0, 0], [0, 1072, 0, 0]],
                memory_shapes = [[1, 224, 113, 1], [1, 224, 113, 1], [1, 208, 113, 1], [1, 208, 113, 1], [1, 208, 113, 1], [1, 208, 113, 1]],
                memory_offsets = [[0, 0, 0, 0], [0, 224, 0, 0], [0, 448, 0, 0], [0, 656, 0, 0], [0, 864, 0, 0], [0, 1072, 0, 0]]}>

// CHECK-LABEL:   @CantAdjustBroadcastMultiplyLayout
// CHECK-SAME:      [[INPUT_0:%.+]]: !VPUIP.DistributedBuffer<1x1280x113x4xf16, #NHWC
// CHECK-SAME:      [[INPUT_1:%.+]]: !VPUIP.DistributedBuffer<1x1280x113x1xf16, #NHWC
func.func @CantAdjustBroadcastMultiplyLayout(%input1: !Distributed, %input2: !Distributed1) -> !Distributed {
    %alloc = VPURT.AllocDistributed -> !Distributed
    %ncecluster = VPUIP.NCEClusterTiling
        inputs(%input1 as %arg4: memref<1x1280x113x4xf16, #NHWC, @CMX_NN>, %input2 as %arg5: memref<1x1280x113x1xf16, #NHWC, @CMX_NN>)
        outputs(%alloc as %arg6: memref<1x1280x113x4xf16, #NHWC, @CMX_NN>) -> !Distributed {
      %swkernel = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Multiply inputs(%arg4 as %arg7: memref<1x1280x113x4xf16, #NHWC, @CMX_NN>,
                %arg5 as %arg8: memref<1x1280x113x4xf16, #NHWC, @CMX_NN>) outputs(%arg6 as %arg9: memref<1x1280x113x4xf16, #NHWC, @CMX_NN>) on tile 0 -> memref<1x1280x113x4xf16, #NHWC, @CMX_NN>{
        VPUIP.SW.Kernel.run(%arg7, %arg8, %arg9) : memref<1x1280x113x4xf16, #NHWC, @CMX_NN>, memref<1x1280x113x4xf16, #NHWC, @CMX_NN>, memref<1x1280x113x4xf16, #NHWC, @CMX_NN>
      }
    }
    return %ncecluster : !Distributed

    // Can't avoid spilling by layout change, because the broadcast dimension is between the tileDim and highest dim
    // CHECK-NOT:       VPUIP.PermuteCast
    // CHECK:           [[ALLOC_IN_0:%.+]] = memref.alloc() : memref<1x1280x113x1xf16, #NHWC, @DDR>
    // CHECK:           [[NCE_COPY_IN_0:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[INPUT_1]] as %arg2: memref<1x1280x113x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:          outputs([[ALLOC_IN_0]] as %arg3: memref<1x1280x113x1xf16, #NHWC, @DDR>)
    // CHECK:           [[SUBVIEW_IN_0:%.+]] = VPUIP.SubView [[NCE_COPY_IN_0]] [0, 656, 0, 0] [1, 624, 113, 1]
    // CHECK:           [[ALLOC_IN_0_1:%.+]] = VPURT.AllocDistributed
    // CHECK:           [[NCE_COPY_IN_0_1:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[SUBVIEW_IN_0]] as %arg2: memref<1x624x113x1xf16, {order = #NHWC, strides = [144640, 1, 1280, 1280]}, @DDR>)
    // CHECK-SAME:          outputs([[ALLOC_IN_0_1]] as %arg3: memref<1x624x113x1xf16, #NHWC, @CMX_NN>)

    // CHECK:           [[ALLOC_IN_1:%.+]] = memref.alloc() : memref<1x1280x113x4xf16, #NHWC, @DDR>
    // CHECK:           [[NCE_COPY_IN_1:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[INPUT_0]] as %arg2: memref<1x1280x113x4xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:          outputs([[ALLOC_IN_1]] as %arg3: memref<1x1280x113x4xf16, #NHWC, @DDR>)
    // CHECK:           [[SUBVIEW_IN_1:%.+]] = VPUIP.SubView [[NCE_COPY_IN_1]] [0, 656, 0, 0] [1, 624, 113, 4]
    // CHECK:           [[ALLOC_IN_1_1:%.+]] = VPURT.AllocDistributed
    // CHECK:           [[NCE_COPY_IN_1_1:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[SUBVIEW_IN_1]] as %arg2: memref<1x624x113x4xf16, {order = #NHWC, strides = [578560, 1, 5120, 1280]}, @DDR>)
    // CHECK-SAME:          outputs([[ALLOC_IN_1_1]] as %arg3: memref<1x624x113x4xf16, #NHWC, @CMX_NN>)

    // CHECK:           [[ALLOC_IN_0_0:%.+]] = memref.alloc() : memref<1x1280x113x1xf16, #NHWC, @DDR>
    // CHECK:           [[NCE_COPY_IN_0_0:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[INPUT_1]] as %arg2: memref<1x1280x113x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:          outputs([[ALLOC_IN_0_0]] as %arg3: memref<1x1280x113x1xf16, #NHWC, @DDR>)
    // CHECK:           [[SUBVIEW_IN_0_0:%.+]] = VPUIP.SubView [[NCE_COPY_IN_0_0]] [0, 0, 0, 0] [1, 656, 113, 1]
    // CHECK:           [[ALLOC_IN_0_1_0:%.+]] = VPURT.AllocDistributed
    // CHECK:           [[NCE_COPY_IN_0_1_0:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[SUBVIEW_IN_0_0]] as %arg2: memref<1x656x113x1xf16, {order = #NHWC, strides = [144640, 1, 1280, 1280]}, @DDR>)
    // CHECK-SAME:          outputs([[ALLOC_IN_0_1_0]] as %arg3: memref<1x656x113x1xf16, #NHWC, @CMX_NN>)

    // CHECK:           [[ALLOC_IN_1_0:%.+]] = memref.alloc() : memref<1x1280x113x4xf16, #NHWC, @DDR>
    // CHECK:           [[NCE_COPY_IN_1_0:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[INPUT_0]] as %arg2: memref<1x1280x113x4xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:          outputs([[ALLOC_IN_1_0]] as %arg3: memref<1x1280x113x4xf16, #NHWC, @DDR>)
    // CHECK:           [[SUBVIEW_IN_1_0:%.+]] = VPUIP.SubView [[NCE_COPY_IN_1_0]] [0, 0, 0, 0] [1, 656, 113, 4]
    // CHECK:           [[ALLOC_IN_1_1_0:%.+]] = VPURT.AllocDistributed
    // CHECK:           [[NCE_COPY_IN_1_1_0:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[SUBVIEW_IN_1_0]] as %arg2: memref<1x656x113x4xf16, {order = #NHWC, strides = [578560, 1, 5120, 1280]}, @DDR>)
    // CHECK-SAME:          outputs([[ALLOC_IN_1_1_0]] as %arg3: memref<1x656x113x4xf16, #NHWC, @CMX_NN>)

    // CHECK:           [[NCE_SW:%.+]]:2 = VPUIP.NCEClusterTiling
    // CHECK:               [[SW_KERNEL:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Multiply
    // CHECK:                   VPUIP.SW.Kernel.run
    // CHECK-SAME:                  memref<1x656x113x4xf16, #NHWC, @CMX_NN>
    // CHECK:                   VPUIP.SW.Kernel.run
    // CHECK-SAME:                  memref<1x624x113x4xf16, #NHWC, @CMX_NN>

    // CHECK:           [[SUBVIEW_OUT_0:%.+]] VPUIP.SubView
    // CHECK:           [[NCE_COPY_OUT_0:%.+]] = VPUIP.NCEClusterTiling
    // CHECK:               [[COPY_OUT_0:%.+]] = VPUIP.Copy
    // CHECK:           [[SUBVIEW_OUT_1:%.+]] VPUIP.SubView
    // CHECK:           [[NCE_COPY_OUT_1:%.+]] = VPUIP.NCEClusterTiling
    // CHECK:               [[COPY_OUT_1:%.+]] = VPUIP.Copy
    // CHECK:           [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[NCE_COPY_OUT_0]], [[NCE_COPY_OUT_1]]

    // CHECK:           [[NCE_COPY_OUT:%.+]] = VPUIP.NCEClusterTiling
    // CHECK:               [[COPY_OUT:%.+]] = VPUIP.Copy
    // CHECK:           return [[NCE_COPY_OUT]]
}
