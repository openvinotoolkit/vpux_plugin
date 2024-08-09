//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --vpu-arch=%arch% --split-input-file --mlir-elide-elementsattrs-if-larger 8 --default-hw-mode %s | FileCheck %s --strict-whitespace
// REQUIRES: arch-NPU37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @Convolution
module @Convolution {
    // CHECK:   {{  }}module @UsedMemory
    // CHECK-DAG:  {{    }}IE.MemoryResource {{[0-9]+}} bytes of @DDR
    // CHECK-DAG:  {{  }}IE.ExecutorResource 2 of @DMA_NN
    // CHECK-DAG:  {{  }}IE.TileResource {activity_factor = {{[0-9]+.[0-9]+}} : f64} 2 of @NCE at 1.300000e+03 MHz
    // CHECK-DAG:  {{    }}builtin.module @UsedMemory
    // CHECK-DAG:  {{      }}IE.MemoryResource {{[0-9]+}} bytes of @CMX_NN
    // CHECK-DAG:  {{    }}IE.ExecutorResource 1 of @DPU
    // CHECK-DAG:  {{    }}IE.ExecutorResource 2 of @SHAVE_ACT
    // CHECK-DAG:  {{    }}IE.ExecutorResource 1 of @SHAVE_NN

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

        // CHECK-DAG:   const.Declare memref<1x1x1x2688xf16>

        // CHECK:       VPURT.Task waits([[barrier_0:%.*]] : !VPURT.Barrier) updates([[barrier_1:%.*]] : !VPURT.Barrier)
        // CHECK:       VPUIP.NCEClusterTask
        // CHECK-SAME:          task_type = #VPUIP.nce_task_type<ELTWISE>
        // CHECK:       VPUIP.NCEClusterTask
        // CHECK-SAME:          task_type = #VPUIP.nce_task_type<ELTWISE>

        // CHECK:       VPUIP.NCEClusterTask
        // CHECK-SAME:          task_type = #VPUIP.nce_task_type<CONV>
        // CHECK-SAME:      [[input_0:%.*]] : memref<1x16x32x62xf16, #NHWC, [@CMX_NN, 0]>)
        // CHECK-SAME:      [[weight_0:%.*]] : memref<48x16x3x3xf16, {order = #NHWC, sparsityCompression = #VPUIP.SparsityCompressionAttr<axis = 0 : i64, numElems = dense<27> : tensor<48xi64>, alignment = 16 : i64>}, [@CMX_NN, 0]>)
        // CHECK-SAME:      [[weight_sm_0:%.*]] : memref<48x1x1x256xi1, [@CMX_NN, 0]>)
        // CHECK-SAME:      [[weight_table_0:%.*]] : memref<48x1x1x4xsi32, [@CMX_NN, 0]>)
        // CHECK-SAME:      [[parent_input_0:%.*]] : !VPUIP.DistributedBuffer<1x16x62x62xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>)
        // CHECK-SAME:      [[parent_output_0:%.*]] : !VPUIP.DistributedBuffer<1x48x60x60xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        // CHECK-SAME:      [[output_0:%.*]] : memref<1x48x30x60xf16, [@CMX_NN, 0]>)
        // CHECK:               DPUTask

        // CHECK:       VPURT.Task waits([[barrier_0:%.*]] : !VPURT.Barrier) updates([[barrier_1:%.*]] : !VPURT.Barrier)
        // CHECK:       VPUIP.NCEClusterTask
        // CHECK-SAME:          task_type = #VPUIP.nce_task_type<CONV>
        // CHECK-SAME:      [[input_1:%.*]] : memref<1x16x30x62xf16, #NHWC, [@CMX_NN, 1]>)
        // CHECK-SAME:      [[weight_1:%.*]] : memref<48x16x3x3xf16, {order = #NHWC, sparsityCompression = #VPUIP.SparsityCompressionAttr<axis = 0 : i64, numElems = dense<27> : tensor<48xi64>, alignment = 16 : i64>}, [@CMX_NN, 1]>)
        // CHECK-SAME:      [[weight_sm_1:%.*]] : memref<48x1x1x256xi1, [@CMX_NN, 1]>
        // CHECK-SAME:      [[weight_table_1:%.*]] : memref<48x1x1x4xsi32, [@CMX_NN, 1]>)
        // CHECK-SAME:      [[parent_input_1:%.*]] : !VPUIP.DistributedBuffer<1x16x62x62xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>)
        // CHECK-SAME:      [[parent_output_1:%.*]] : !VPUIP.DistributedBuffer<1x48x60x60xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        // CHECK-SAME:      [[output_1:%.*]] : memref<1x48x30x60xf16, [@CMX_NN, 1]>)
        // CHECK:               DPUTask
  }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!qElemType2 = !quant.uniform<u8:f16, 0.047852594712201289:128>

// CHECK-LABEL: @ScaleShiftSubgraph
module @ScaleShiftSubgraph {
    // CHECK:   {{  }}module @UsedMemory
    // CHECK-DAG:  {{    }}IE.MemoryResource {{[0-9]+}} bytes of @DDR
    // CHECK-DAG:  {{  }}IE.ExecutorResource 2 of @DMA_NN
    // CHECK-DAG:  {{  }}IE.TileResource {activity_factor = {{[0-9]+.[0-9]+}} : f64} 2 of @NCE at 1.300000e+03 MHz
    // CHECK-DAG:  {{    }}builtin.module @UsedMemory
    // CHECK-DAG:  {{      }}IE.MemoryResource {{[0-9]+}} bytes of @CMX_NN
    // CHECK-DAG:  {{    }}IE.ExecutorResource 1 of @DPU
    // CHECK-DAG:  {{    }}IE.ExecutorResource 2 of @SHAVE_ACT
    // CHECK-DAG:  {{    }}IE.ExecutorResource 1 of @SHAVE_NN

    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x256x20x20xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<1x512x20x20xf32>
    }

    // CHECK:       func.func @main(
    // CHECK-SAME:      [[ARG0:%.+]]: memref<1x256x20x20xf32, @DDR>,
    // CHECK-SAME:      [[ARG1:%.+]]: memref<1x512x20x20xf32, @DDR>) -> memref<1x512x20x20xf32, @DDR> {
    func.func @main(%arg: tensor<1x256x20x20xf32>) -> tensor<1x512x20x20xf32> {

        %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<-2.05229545> : tensor<1x1x1x1xf32>
        %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<2.0362618> : tensor<1x1x1x1xf32>
        %cst_2 = const.Declare tensor<1x1x1x1xf32> = dense<-2.05229545> : tensor<1x1x1x1xf32>
        %cst_3 = const.Declare tensor<1x1x1x1xf32> = dense<2.0362618> : tensor<1x1x1x1xf32>
        %0 = IE.FakeQuantize(%arg, %cst_0, %cst_1, %cst_2, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x256x20x20xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x256x20x20xf32>

        %1 = IE.Concat(%0, %0) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x256x20x20xf32>, tensor<1x256x20x20xf32> -> tensor<1x512x20x20xf32>

        %cst_4 = const.Declare tensor<1x512x1x1xf32> = dense<1.0> : tensor<1x512x1x1xf32>
        %2 = IE.Multiply(%1, %cst_4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x20x20xf32>, tensor<1x512x1x1xf32> -> tensor<1x512x20x20xf32>

        %cst_5 = const.Declare tensor<1x512x1x1xf32> = dense<1.0> : tensor<1x512x1x1xf32>
        %3 = IE.Add(%2, %cst_5) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x20x20xf32>, tensor<1x512x1x1xf32> -> tensor<1x512x20x20xf32>

        %cst_6 = const.Declare tensor<1xf32> = dense<1.000000e-01> : tensor<1xf32>
        %4 = IE.PRelu(%3, %cst_6) : tensor<1x512x20x20xf32>, tensor<1xf32> -> tensor<1x512x20x20xf32>

        %cst_7 = const.Declare tensor<1x1x1x1xf32> = dense<-6.12513208> : tensor<1x1x1x1xf32>
        %cst_8 = const.Declare tensor<1x1x1x1xf32> = dense<6.07727957> : tensor<1x1x1x1xf32>
        %cst_9 = const.Declare tensor<1x1x1x1xf32> = dense<-6.12513208> : tensor<1x1x1x1xf32>
        %cst_10 = const.Declare tensor<1x1x1x1xf32> = dense<6.07727957> : tensor<1x1x1x1xf32>
        %5 = IE.FakeQuantize(%4, %cst_7, %cst_8, %cst_9, %cst_10) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x512x20x20xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x20x20xf32>

        return %5 : tensor<1x512x20x20xf32>

        // CHECK-DAG:   const.Declare memref<1x1x1x12288xf16>

        // CHECK:       VPURT.Task waits([[barrier_0:%.*]] : !VPURT.Barrier) updates([[barrier_1:%.*]] : !VPURT.Barrier)
        // CHECK:       VPUIP.NCEClusterTask {is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<DWCONV>}
        // CHECK-SAME:      input([[input_0:%.*]] : memref<1x512x10x20xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>)
        // CHECK-SAME:      weights([[weight_0:%.*]] : memref<512x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
        // CHECK-SAME:      weight_table([[weight_table_0:%.*]] : memref<512x1x1x4xsi32, [@CMX_NN, 0]>)
        // CHECK-SAME:      parent_input([[parent_input_0:%.*]] : !VPUIP.DistributedBuffer<1x512x20x20xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        // CHECK-SAME:      parent_output([[parent_output_0:%.*]] : !VPUIP.DistributedBuffer<1x512x20x20x!qElemType2, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        // CHECK-SAME:      outputs([[output_0:%.*]] : memref<1x512x10x20x!qElemType2, #NHWC, [@CMX_NN, 0]>) -> memref<1x512x10x20x!qElemType2, #NHWC, [@CMX_NN, 0]> variants : {
        // CHECK:               DPUTask
        // CHECK:               DPUTask
        // CHECK:               DPUTask
        // CHECK:               DPUTask
        // CHECK:               DPUTask
        // CHECK:               DPUTask
        // CHECK:               DPUTask
        // CHECK:               DPUTask
        // CHECK:       } PPE : {
        // CHECK:               PPETask <LPRELU>

        // CHECK:       VPURT.Task waits([[barrier_0:%.*]] : !VPURT.Barrier) updates([[barrier_1:%.*]] : !VPURT.Barrier)
        // CHECK:       VPUIP.NCEClusterTask {is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<DWCONV>}
        // CHECK-SAME:      input([[input_1:%.*]] : memref<1x512x10x20xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 1]>)
        // CHECK-SAME:      weights([[weight_1:%.*]] : memref<512x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
        // CHECK-SAME:      weight_table([[weight_table_1:%.*]] : memref<512x1x1x4xsi32, [@CMX_NN, 1]>)
        // CHECK-SAME:      parent_input([[parent_input_0:%.*]] : !VPUIP.DistributedBuffer<1x512x20x20xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        // CHECK-SAME:      parent_output([[parent_output_0:%.*]] : !VPUIP.DistributedBuffer<1x512x20x20x!qElemType2, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        // CHECK-SAME:      outputs([[output_1:%.*]] : memref<1x512x10x20x!qElemType2, #NHWC, [@CMX_NN, 1]>) -> memref<1x512x10x20x!qElemType2, #NHWC, [@CMX_NN, 1]> variants : {
        // CHECK:               DPUTask
        // CHECK:               DPUTask
        // CHECK:               DPUTask
        // CHECK:               DPUTask
        // CHECK:               DPUTask
        // CHECK:               DPUTask
        // CHECK:               DPUTask
        // CHECK:               DPUTask
        // CHECK:       } PPE : {
        // CHECK:               PPETask <LPRELU>
  }
}
