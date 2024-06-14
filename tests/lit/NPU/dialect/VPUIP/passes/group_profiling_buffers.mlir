//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --group-profiling-buffers %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX || arch-VPUX40XX

// CHECK-LABEL: @GroupProfilingBuffers
module @GroupProfilingBuffers {
    IE.CNNNetwork entryPoint : @main inputsInfo :  {
    DataInfo "in" : tensor<1x48x30x30xf16>
    } outputsInfo :  {
        DataInfo "out" : tensor<1x48x30x30xf32>
    } profilingOutputsInfo :  {
        DataInfo "dpu" : tensor<4xui64>
        DataInfo "dma" : tensor<14xui32>
        DataInfo "upa" : tensor<24xui32>
    }
    func.func @main(%arg0: memref<1x48x30x30xf16>, %arg1: memref<1x48x30x30xf32>, %arg2: memref<4xui64>, %arg3: memref<14xui32>, %arg4: memref<24xui32>) -> (memref<1x48x30x30xf32>, memref<4xui64>, memref<14xui32>, memref<24xui32>) {
        %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        %2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        %3 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<4xui64, [@CMX_NN, 0]>
        %4 = VPURT.DeclareBuffer <CMX_NN> [0] <24> -> memref<14xui32,[@CMX_NN, 0]>
        %5 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x48x30x30xf32, @DDR>
        %6 = VPURT.DeclareBuffer <ProfilingOutput> [2] <72> -> memref<6xui32>
        VPURT.Task profiling_data(%6 : memref<6xui32>) updates(%0 : !VPURT.Barrier) {
            %62 = VPUIP.ConvertUPA inputs(%arg0 : memref<1x48x30x30xf16>) outputs(%5 : memref<1x48x30x30xf32, @DDR>) -> memref<1x48x30x30xf32, @DDR>
        }
        VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
            %62 = VPUIP.NNDMA {set_crit = false, set_ord = true} inputs(%3 : memref<4xui64, [@CMX_NN, 0]>) outputs(%arg2 : memref<4xui64>) -> memref<4xui64>
        }
        VPURT.Task waits(%2 : !VPURT.Barrier) {
            %62 = VPUIP.NNDMA {set_crit = false, set_ord = true} inputs(%4 : memref<14xui32, [@CMX_NN, 0]>) outputs(%arg3 : memref<14xui32>) -> memref<14xui32>
        }
        return %arg1, %arg2, %arg3, %arg4 : memref<1x48x30x30xf32>, memref<4xui64>, memref<14xui32>, memref<24xui32>
    }

    //CHECK:        profilingOutputsInfo
    //CHECK-NEXT:   DataInfo "profilingOutput" {
    //CHECK-NEXT:   VPUIP.ProfilingSection  type 1 : 32 bytes from 0
    //CHECK-NEXT:   VPUIP.ProfilingSection  type 4 : 56 bytes from 64
    //CHECK-NEXT:   VPUIP.ProfilingSection  type 2 : 96 bytes from 128
    //CHECK-NEXT:   } : tensor<64xui32>
    //CHECK:        %arg0: memref<1x48x30x30xf16>, %arg1: memref<1x48x30x30xf32>, %arg2: memref<64xui32>) -> (memref<1x48x30x30xf32>, memref<64xui32>)

    //CHECK:        [[VAR0:%.+]] = VPURT.DeclareBuffer <ProfilingOutput> [0] <200> -> memref<6xui32>
    //CHECK:        VPURT.Task
    //CHECK-SAME:   profiling_data([[VAR0]] : memref<6xui32>)

    //CHECK:        [[VAR1:%.+]] = VPURT.DeclareBuffer <ProfilingOutput> [0] <0> -> memref<4xui64>
    //CHECK:        VPUIP.NNDMA
    //CHECK-SAME:   outputs([[VAR1]] : memref<4xui64>)

    //CHECK:        [[VAR2:%.+]] = VPURT.DeclareBuffer <ProfilingOutput> [0] <64> -> memref<14xui32>
    //CHECK:        VPUIP.NNDMA
    //CHECK-SAME:   outputs([[VAR2]] : memref<14xui32>)
    //CHECK:        return %arg1, %arg2 : memref<1x48x30x30xf32>, memref<64xui32>
}
