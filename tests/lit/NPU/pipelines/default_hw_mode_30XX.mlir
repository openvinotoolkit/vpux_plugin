//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --vpu-arch=%arch% --split-input-file --mlir-elide-elementsattrs-if-larger 8 --default-hw-mode %s | FileCheck %s --strict-whitespace
// REQUIRES: arch-VPUX30XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @Convolution
module @Convolution {
    // CHECK-DAG: {{  }}module @UsedMemory
    // CHECK-DAG: {{    }}IE.MemoryResource {{[0-9]+}} bytes of @DDR
    // CHECK-DAG: {{  }}IE.PipelineOptions @Options
    // CHECK-DAG: {{    }}IE.Option @VPU.BarrierMaxVariantSum : 512
    // CHECK-DAG: {{    }}IE.Option @VPU.BarrierMaxVariantCount : 512

    // CHECK-DAG: {{  }}IE.ExecutorResource 1 of @DMA_NN
    // CHECK-DAG: {{  }}IE.ExecutorResource 16 of @SHAVE_UPA
    // CHECK-DAG: {{  }}IE.TileResource 4 of @NCE at 7.000000e+02 MHz
    // CHECK-DAG: {{    }}IE.ExecutorResource 5 of @DPU
    // CHECK-DAG: {{    }}builtin.module @UsedMemory
    // CHECK-DAG: {{      }}IE.MemoryResource {{[0-9]+}} bytes of @CMX_NN

    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x3x62x62xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf16>
    }

    // CHECK:       func.func @main(
    // CHECK-SAME:      [[ARG0:%.+]]: memref<1x3x62x62xf16, @DDR>,
    // CHECK-SAME:      [[ARG1:%.+]]: memref<1x48x60x60xf16, @DDR>) -> memref<1x48x60x60xf16, @DDR> {
    func.func @main(%arg: tensor<1x3x62x62xf32>) -> tensor<1x48x60x60xf32> {
        %cst = const.Declare tensor<48x3x3x3xf32> = dense<1.0> : tensor<48x3x3x3xf32>
        %1 = IE.Convolution(%arg, %cst) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
        return %1 : tensor<1x48x60x60xf32>

        // CHECK-DAG:   const.Declare memref<1x1x1x7296xf16>

        // CHECK-DAG:   [[OUT0_DDR:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x48x60x60xf16, @DDR>
        // CHECK-DAG:   [[OUT1_DDR:%.+]] = VPURT.DeclareBuffer <DDR> <123008> -> memref<1x48x15x60xf16, #NHWC, @DDR>
        // CHECK-DAG:   [[OUT2_DDR:%.+]] = VPURT.DeclareBuffer <DDR> <209408> -> memref<1x48x15x60xf16, #NHWC, @DDR>
        // CHECK-DAG:   [[OUT3_DDR:%.+]] = VPURT.DeclareBuffer <DDR> <295808> -> memref<1x48x15x60xf16, #NHWC, @DDR>
        // CHECK-DAG:   [[OUT4_DDR:%.+]] = VPURT.DeclareBuffer <DDR> <382208> -> memref<1x48x15x60xf16, #NHWC, @DDR>

        // CHECK-DAG:   [[IN0_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <14592> -> memref<1x48x15x60xf16, #NHWC, [@CMX_NN, 0]>
        // CHECK-DAG:   [[IN1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <14592> -> memref<1x48x15x60xf16, #NHWC, [@CMX_NN, 1]>
        // CHECK-DAG:   [[IN2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <14592> -> memref<1x48x15x60xf16, #NHWC, [@CMX_NN, 2]>
        // CHECK-DAG:   [[IN3_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [3] <14592> -> memref<1x48x15x60xf16, #NHWC, [@CMX_NN, 3]>
        // CHECK-DAG:   [[IN0_DDR:%.+]] = VPURT.DeclareBuffer <DDR> <123008> -> memref<1x48x60x60xf16, #NHWC, @DDR>

        // CHECK:       VPURT.Task waits([[barrier_0:%.*]] : !VPURT.Barrier) updates([[barrier_1:%.*]] : !VPURT.Barrier)
        // CHECK:       VPUIP.NCEClusterTask
        // CHECK-SAME:          task_type = #VPUIP.nce_task_type<CONV>
        // CHECK-SAME:      [[input_0:%.*]] : memref<1x16x16x62xf16, #NHWC, [@CMX_NN, 0]>)
        // CHECK-SAME:      [[weight_0:%.*]] : memref<48x16x3x3xf16, #NHWC, [@CMX_NN, 0]>)
        // CHECK-SAME:      [[weight_table_0:%.*]] : memref<48x1x1x4xsi32, [@CMX_NN, 0]>)
        // CHECK-SAME:      [[parent_input:%.*]] : !VPUIP.DistributedBuffer<1x16x62x62xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
        // CHECK-SAME:      [[parent_output:%.*]] : !VPUIP.DistributedBuffer<1x48x60x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
        // CHECK-SAME:      [[output_0:%.*]] : memref<1x48x15x60xf16, #NHWC, [@CMX_NN, 0]>)
        // CHECK:               DPUTask

        // CHECK:       VPURT.Task waits([[barrier_0:%.*]] : !VPURT.Barrier) updates([[barrier_1:%.*]] : !VPURT.Barrier)
        // CHECK:       VPUIP.NCEClusterTask
        // CHECK-SAME:          task_type = #VPUIP.nce_task_type<CONV>
        // CHECK-SAME:      [[input_0:%.*]] : memref<1x16x16x62xf16, #NHWC, [@CMX_NN, 1]>)
        // CHECK-SAME:      [[weight_0:%.*]] : memref<48x16x3x3xf16, #NHWC, [@CMX_NN, 1]>)
        // CHECK-SAME:      [[weight_table_0:%.*]] : memref<48x1x1x4xsi32, [@CMX_NN, 1]>)
        // CHECK-SAME:      [[parent_input:%.*]] : !VPUIP.DistributedBuffer<1x16x62x62xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
        // CHECK-SAME:      [[parent_output:%.*]] : !VPUIP.DistributedBuffer<1x48x60x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
        // CHECK-SAME:      [[output_0:%.*]] : memref<1x48x15x60xf16, #NHWC, [@CMX_NN, 1]>)
        // CHECK:               DPUTask

        // CHECK:       VPURT.Task waits([[barrier_0:%.*]] : !VPURT.Barrier) updates([[barrier_1:%.*]] : !VPURT.Barrier)
        // CHECK:       VPUIP.NCEClusterTask
        // CHECK-SAME:          task_type = #VPUIP.nce_task_type<CONV>
        // CHECK-SAME:      [[input_0:%.*]] : memref<1x16x16x62xf16, #NHWC, [@CMX_NN, 2]>)
        // CHECK-SAME:      [[weight_0:%.*]] : memref<48x16x3x3xf16, #NHWC, [@CMX_NN, 2]>)
        // CHECK-SAME:      [[weight_table_0:%.*]] : memref<48x1x1x4xsi32, [@CMX_NN, 2]>)
        // CHECK-SAME:      [[parent_input:%.*]] : !VPUIP.DistributedBuffer<1x16x62x62xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
        // CHECK-SAME:      [[parent_output:%.*]] : !VPUIP.DistributedBuffer<1x48x60x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
        // CHECK-SAME:      [[output_0:%.*]] : memref<1x48x15x60xf16, #NHWC, [@CMX_NN, 2]>)
        // CHECK:               DPUTask

        // CHECK:       VPURT.Task waits([[barrier_0:%.*]] : !VPURT.Barrier) updates([[barrier_1:%.*]] : !VPURT.Barrier)
        // CHECK:       VPUIP.NCEClusterTask
        // CHECK-SAME:          task_type = #VPUIP.nce_task_type<CONV>
        // CHECK-SAME:      [[input_3:%.*]] : memref<1x16x14x62xf16, #NHWC, [@CMX_NN, 3]>)
        // CHECK-SAME:      [[weight_3:%.*]] : memref<48x16x3x3xf16, #NHWC, [@CMX_NN, 3]>)
        // CHECK-SAME:      [[weight_table_3:%.*]] : memref<48x1x1x4xsi32, [@CMX_NN, 3]>)
        // CHECK-SAME:      [[parent_input:%.*]] : !VPUIP.DistributedBuffer<1x16x62x62xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
        // CHECK-SAME:      [[parent_output:%.*]] : !VPUIP.DistributedBuffer<1x48x60x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
        // CHECK-SAME:      [[output_3:%.*]] : memref<1x48x15x60xf16, #NHWC, [@CMX_NN, 3]>)
        // CHECK:               DPUTask

        // CHECK:       VPURT.Task waits([[barrier_1:%.*]] : !VPURT.Barrier)
        // CHECK:       VPUIP.NNDMA {port = 0 : i64} inputs([[IN0_CMX]] : memref<1x48x15x60xf16, #NHWC, [@CMX_NN, 0]>)
        // CHECK-SAME:      outputs([[OUT1_DDR]] : memref<1x48x15x60xf16, #NHWC, @DDR>)

        // CHECK:       VPURT.Task
        // CHECK:       VPUIP.NNDMA {port = 0 : i64} inputs([[IN1_CMX]] : memref<1x48x15x60xf16, #NHWC, [@CMX_NN, 1]>)
        // CHECK-SAME:      outputs([[OUT2_DDR]] : memref<1x48x15x60xf16, #NHWC, @DDR>)

        // CHECK:       VPURT.Task
        // CHECK:       VPUIP.NNDMA {port = 0 : i64} inputs([[IN2_CMX]] : memref<1x48x15x60xf16, #NHWC, [@CMX_NN, 2]>)
        // CHECK-SAME:      outputs([[OUT3_DDR]] : memref<1x48x15x60xf16, #NHWC, @DDR>)

        // CHECK:       VPURT.Task updates([[barrier_2:%.*]] : !VPURT.Barrier)
        // CHECK:       VPUIP.NNDMA {port = 0 : i64} inputs([[IN3_CMX]] : memref<1x48x15x60xf16, #NHWC, [@CMX_NN, 3]>)
        // CHECK-SAME:      outputs([[OUT4_DDR]] : memref<1x48x15x60xf16, #NHWC, @DDR>)

        // CHECK:       VPURT.Task waits([[barrier_2:%.*]] : !VPURT.Barrier)
        // CHECK:       VPUIP.PermuteUPA {order_value = #NWCH} inputs([[IN0_DDR]] : memref<1x48x60x60xf16, #NHWC, @DDR>)
        // CHECK-SAME:      outputs([[OUT0_DDR]] : memref<1x48x60x60xf16, @DDR>)

        // CHECK:   return %arg1 : memref<1x48x60x60xf16, @DDR>
  }
}

// -----

// CHECK-LABEL: @SingleLayer
module @SingleLayer {

// CHECK-DAG: {{  }}module @UsedMemory
// CHECK-DAG: {{    }}IE.MemoryResource {{[0-9]+}} bytes of @DDR
// CHECK-DAG: {{  }}IE.PipelineOptions @Options
// CHECK-DAG: {{    }}IE.Option @VPU.BarrierMaxVariantSum : 512
// CHECK-DAG: {{    }}IE.Option @VPU.BarrierMaxVariantCount : 512
// CHECK-DAG: {{  }}IE.ExecutorResource 1 of @DMA_NN
// CHECK-DAG: {{  }}IE.ExecutorResource 16 of @SHAVE_UPA
// CHECK-DAG: {{  }}IE.TileResource 4 of @NCE at 7.000000e+02 MHz
// CHECK-DAG: {{    }}IE.ExecutorResource 5 of @DPU
// CHECK-DAG: {{    }}builtin.module @UsedMemory
// CHECK-DAG: {{      }}IE.MemoryResource {{[0-9]+}} bytes of @CMX_NN

// CHECK: IE.CNNNetwork
IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        // CHECK: DataInfo "input" : tensor<1x1000xf16>
        DataInfo "input" : tensor<1x1000xf16>
    }
    outputsInfo : {
        // CHECK: DataInfo "softmax" : tensor<1x1000xf16>
        DataInfo "softmax" : tensor<1x1000xf16>
    }

// CHECK:       func.func @main(
// CHECK-SAME:      [[ARG0:%.+]]: memref<1x1000xf16, @DDR>,
// CHECK-SAME:      [[ARG1:%.+]]: memref<1x1000xf16, @DDR>) -> memref<1x1000xf16, @DDR> {
func.func @main(%arg0: tensor<1x1000xf16>) -> tensor<1x1000xf16> {
    %0 = IE.SoftMax(%arg0) {axisInd = 1} : tensor<1x1000xf16> -> tensor<1x1000xf16>
    return %0 : tensor<1x1000xf16>

    // CHECK-DAG:   [[IN1:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x1x1x1000xf16, @DDR>
    // CHECK-DAG:   [[OUT1:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x1000xf16, @DDR>

    // CHECK-DAG:   [[VAR1:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1000xf16, @DDR>
    // CHECK-DAG:   [[VAR2:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1000xf16, @DDR>

    // CHECK-DAG:   [[VAR3:%.+]] = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    // CHECK-NEXT:  VPURT.Task
    // CHECK-SAME:              updates([[VAR3]] : !VPURT.Barrier)
    // CHECK-NEXT:  VPUIP.SoftMaxUPA
    // CHECK-SAME:              axisInd = 3
    // CHECK-SAME:              inputs([[IN1]] : memref<1x1x1x1000xf16, @DDR>
    // CHECK-SAME:              outputs([[VAR1]] : memref<1x1x1x1000xf16, @DDR>

    // CHECK:  VPURT.Task
    // CHECK-SAME:              waits([[VAR3]] : !VPURT.Barrier)
    // CHECK-NEXT:  VPUIP.NNDMA
    // CHECK-SAME:              inputs([[VAR2]] : memref<1x1000xf16, @DDR>)
    // CHECK-SAME:              outputs([[OUT1]] : memref<1x1000xf16, @DDR>)

    // CHECK:  return [[ARG1]] : memref<1x1000xf16, @DDR>
}

}
