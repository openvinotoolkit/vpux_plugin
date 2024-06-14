//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-parallel-copies %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @OptimizeParallelNonConstCopies(
        %input: memref<1x16x112x112xf32, #NHWC>,
        %output1: memref<1x16x112x112xf16, #NHWC, @DDR>,
        %output2: memref<1x16x112x112xf16, #NHWC, @DDR>)
         -> (memref<1x16x112x112xf16, #NHWC, @DDR>, memref<1x16x112x112xf16, #NHWC, @DDR>){
    %wt = const.Declare memref<16x1x1x4xsi32, @CMX_NN> = dense<1> : tensor<16x1x1x4xsi32>
    %act_win = const.Declare memref<1x1x1x16xui8, @CMX_NN> = dense<1> : tensor<1x1x1x16xui8>
    %0 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @DDR>
    %1 = VPUIP.ConvertUPA
        inputs(%input : memref<1x16x112x112xf32, #NHWC>)
        outputs(%0 : memref<1x16x112x112xf16, #NHWC, @DDR>)
        -> memref<1x16x112x112xf16, #NHWC, @DDR>
    %2 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %3 = VPUIP.Copy
            inputs(%1 : memref<1x16x112x112xf16, #NHWC, @DDR>)
            outputs(%2 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
             -> memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %4 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %5 = VPUIP.NCEClusterTask {
            activation_window_channel_length = 27 : i64,
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<MAXPOOL>
        }
        input(%3 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
        weight_table(%wt : memref<16x1x1x4xsi32, @CMX_NN>)
        activation_window(%act_win : memref<1x1x1x16xui8, @CMX_NN>)
        parent_input(%3 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
        parent_output(%4 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
        outputs(%4 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>) -> memref<1x16x112x112xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask { outEnd = [16, 112, 112], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
        }
        PPE : {
        }
    %6 = VPUIP.Copy
            inputs(%5 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
            outputs(%output1 : memref<1x16x112x112xf16, #NHWC, @DDR>)
            -> memref<1x16x112x112xf16, #NHWC, @DDR>

    %7 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %8 = VPUIP.Copy
            inputs(%1 : memref<1x16x112x112xf16, #NHWC, @DDR>)
            outputs(%7 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
             -> memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %9 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %10 = VPUIP.NCEClusterTask {
            activation_window_channel_length = 27 : i64,
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<MAXPOOL>
        }
        input(%8 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
        weight_table(%wt : memref<16x1x1x4xsi32, @CMX_NN>)
        activation_window(%act_win : memref<1x1x1x16xui8, @CMX_NN>)
        parent_input(%8 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
        parent_output(%9 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
        outputs(%9 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>) -> memref<1x16x112x112xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask { outEnd = [16, 112, 112], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
        }
        PPE : {
        }
    %11 = VPUIP.Copy
            inputs(%10 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
            outputs(%output1 : memref<1x16x112x112xf16, #NHWC, @DDR>)
            -> memref<1x16x112x112xf16, #NHWC, @DDR>

    return %6, %11 : memref<1x16x112x112xf16, #NHWC, @DDR>, memref<1x16x112x112xf16, #NHWC, @DDR>

}

// CHECK-LABEL: func.func @OptimizeParallelNonConstCopies

// CHECK:       [[VAR0:%.*]] =  VPUIP.ConvertUPA inputs(%arg0 : memref<1x16x112x112xf32, #NHWC>)
// CHECK:       [[VAR1:%.*]] =  VPUIP.Copy inputs([[VAR0]] : memref<1x16x112x112xf16, #NHWC, @DDR>)
// CHECK:       [[VAR2:%.+]] = VPUIP.NCEClusterTask
// CHECK-SAME:       input([[VAR1]] : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
// CHECK:       [[VAR3:%.*]] =  VPUIP.Copy inputs([[VAR2]] : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)

// CHECK-NOT:   VPUIP.Copy
// CHECK:       [[VAR4:%.+]] = VPUIP.NCEClusterTask
// CHECK-SAME:       input([[VAR1]] : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
// CHECK:       [[VAR5:%.*]] =  VPUIP.Copy inputs([[VAR4]] : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @OptimizeParallelSubViewPatternCopies(
        %input: memref<1x16x112x113xf32, #NHWC>,
        %output1: memref<1x16x112x112xf16, #NHWC, @DDR>,
        %output2: memref<1x16x112x112xf16, #NHWC, @DDR>)
         -> (memref<1x16x112x112xf16, #NHWC, @DDR>, memref<1x16x112x112xf16, #NHWC, @DDR>){
    %wt = const.Declare memref<16x1x1x4xsi32, @CMX_NN> = dense<1> : tensor<16x1x1x4xsi32>
    %act_win = const.Declare memref<1x1x1x16xui8, @CMX_NN> = dense<1> : tensor<1x1x1x16xui8>
    %0 = memref.alloc() : memref<1x16x112x113xf16, #NHWC, @DDR>

    %1 = VPUIP.ConvertUPA
        inputs(%input : memref<1x16x112x113xf32, #NHWC>)
        outputs(%0 : memref<1x16x112x113xf16, #NHWC, @DDR>)
        -> memref<1x16x112x113xf16, #NHWC, @DDR>
    %2 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %3 = VPUIP.SubView %1 [0, 0, 0, 0] [1, 16, 112, 112] :
                memref<1x16x112x113xf16, #NHWC, @DDR> to memref<1x16x112x112xf16, {order = #NHWC, strides = [202496, 1, 1808, 16]}, @DDR>
    %4 = VPUIP.Copy
            inputs(%3 : memref<1x16x112x112xf16, {order = #NHWC, strides = [202496, 1, 1808, 16]}, @DDR>)
            outputs(%2 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
             -> memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %5 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %6 = VPUIP.NCEClusterTask {
            activation_window_channel_length = 27 : i64,
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<MAXPOOL>
        }
        input(%4 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
        weight_table(%wt : memref<16x1x1x4xsi32, @CMX_NN>)
        activation_window(%act_win : memref<1x1x1x16xui8, @CMX_NN>)
        parent_input(%4 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
        parent_output(%5 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
        outputs(%5 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>) -> memref<1x16x112x112xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask { outEnd = [16, 112, 112], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
        }
        PPE : {
        }
    %7 = VPUIP.Copy
            inputs(%6 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
            outputs(%output1 : memref<1x16x112x112xf16, #NHWC, @DDR>)
            -> memref<1x16x112x112xf16, #NHWC, @DDR>

    %8 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %9 = VPUIP.SubView %1 [0, 0, 0, 0] [1, 16, 112, 112] :
                memref<1x16x112x113xf16, #NHWC, @DDR> to memref<1x16x112x112xf16, {order = #NHWC, strides = [202496, 1, 1808, 16]}, @DDR>
    %10 = VPUIP.Copy
            inputs(%9 : memref<1x16x112x112xf16, {order = #NHWC, strides = [202496, 1, 1808, 16]}, @DDR>)
            outputs(%8 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
             -> memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %11 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %12 = VPUIP.NCEClusterTask {
            activation_window_channel_length = 27 : i64,
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<MAXPOOL>
        }
        input(%10 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
        weight_table(%wt : memref<16x1x1x4xsi32, @CMX_NN>)
        activation_window(%act_win : memref<1x1x1x16xui8, @CMX_NN>)
        parent_input(%10 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
        parent_output(%11 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
        outputs(%11 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>) -> memref<1x16x112x112xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask { outEnd = [16, 112, 112], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
        }
        PPE : {
        }
    %13 = VPUIP.Copy
            inputs(%12 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
            outputs(%output1 : memref<1x16x112x112xf16, #NHWC, @DDR>)
            -> memref<1x16x112x112xf16, #NHWC, @DDR>

    return %7, %13 : memref<1x16x112x112xf16, #NHWC, @DDR>, memref<1x16x112x112xf16, #NHWC, @DDR>

}

// CHECK-LABEL: func.func @OptimizeParallelSubViewPatternCopies

// CHECK:       [[VAR0:%.*]] =  VPUIP.ConvertUPA inputs(%arg0 : memref<1x16x112x113xf32, #NHWC>)
// CHECK:       [[VAR1:%.*]] =  VPUIP.SubView [[VAR0]] [0, 0, 0, 0] [1, 16, 112, 112]
// CHECK:       [[VAR2:%.*]] =  VPUIP.Copy
// CHECK-SAME       inputs([[VAR1]] : memref<1x16x112x112xf16, {order = #NCHW, strides = [202496, 1, 1808, 16]}, @DDR>)
// CHECK:       [[VAR3:%.+]] = VPUIP.NCEClusterTask
// CHECK-SAME:       input([[VAR2]] : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
// CHECK:       [[VAR4:%.*]] =  VPUIP.Copy inputs([[VAR3]] : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)

// CHECK-NOT:   VPUIP.SubView
// CHECK-NOT:   VPUIP.Copy
// CHECK:       [[VAR5:%.+]] = VPUIP.NCEClusterTask
// CHECK-SAME:       input([[VAR2]] : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
// CHECK:       [[VAR6:%.*]] =  VPUIP.Copy inputs([[VAR5]] : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
