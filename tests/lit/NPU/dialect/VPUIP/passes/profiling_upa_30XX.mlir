//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --upa-profiling %s | FileCheck %s
// REQUIRES: arch-VPUX30XX

// CHECK-LABEL: @UpaProfiling
#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
module @UpaProfiling {
    IE.CNNNetwork entryPoint : @main inputsInfo :  {
        DataInfo "in" : tensor<1x48x30x30xf16>
    } outputsInfo :  {
        DataInfo "out" : tensor<1x48x30x30xf32>
    } profilingOutputsInfo :  {
    }
    func.func @main(%arg0: memref<1x48x30x30xf16>, %arg1: memref<1x48x30x30xf32>) -> memref<1x48x30x30xf32> {
        %2 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x48x30x30xf16, @DDR>
        %3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        %4 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        %5 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        VPURT.Task waits(%3 : !VPURT.Barrier) updates(%4 : !VPURT.Barrier) {
            %6 = VPUIP.PermuteUPA {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>} inputs(%arg0 : memref<1x48x30x30xf16>) outputs(%2 : memref<1x48x30x30xf16, @DDR>) -> memref<1x48x30x30xf16, @DDR>
        }
        VPURT.Task waits(%4 : !VPURT.Barrier) updates(%5 : !VPURT.Barrier) {
            %6 = VPUIP.ConvertUPA inputs(%2 : memref<1x48x30x30xf16, @DDR>) outputs(%arg1 : memref<1x48x30x30xf32>) -> memref<1x48x30x30xf32>
        }
        return %arg1 : memref<1x48x30x30xf32>
    }

    //CHECK:        profilingOutputsInfo
    //CHECK-NEXT:   DataInfo "upa" : tensor<12xui32>
    //CHECK:        func.func @main(%arg0: memref<1x48x30x30xf16>, %arg1: memref<1x48x30x30xf32>, %arg2: memref<12xui32>) -> (memref<1x48x30x30xf32>, memref<12xui32>)
    //CHECK:        [[VAR0:%.+]] = VPURT.DeclareBuffer <ProfilingOutput> [0] <0> -> memref<6xui32>
    //CHECK:        VPURT.Task
    //CHECK-SAME:   profiling_data([[VAR0]] : memref<6xui32>)
    //CHECK:        [[VAR1:%.+]] = VPURT.DeclareBuffer <ProfilingOutput> [0] <24> -> memref<6xui32>
    //CHECK:        VPURT.Task
    //CHECK-SAME:   profiling_data([[VAR1]] : memref<6xui32>)
    //CHECK:        return %arg1, %arg2 : memref<1x48x30x30xf32>, memref<12xui32>
}
