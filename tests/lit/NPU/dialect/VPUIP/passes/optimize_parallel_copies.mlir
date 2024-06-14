//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-parallel-copies %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX || arch-VPUX40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @OptimizeParallelConstCopies(
        %output1: memref<1x16x112x112xf16, #NHWC, @DDR>,
        %output2: memref<1x16x112x112xf16, #NHWC, @DDR>)
         -> (memref<1x16x112x112xf16, #NHWC, @DDR>, memref<1x16x112x112xf16, #NHWC, @DDR>){
    %wt = const.Declare memref<16x1x1x4xsi32, @CMX_NN> = dense<1> : tensor<16x1x1x4xsi32>
    %act_win = const.Declare memref<1x1x1x16xui8, @CMX_NN> = dense<1> : tensor<1x1x1x16xui8>
    %0 = const.Declare memref<1x16x112x112xf16, #NHWC, @DDR> = dense<1.000000e+00> : tensor<1x16x112x112xf16>, [#const.Reorder<#NHWC>]
    %1 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %2 = VPUIP.Copy
            inputs(%0 : memref<1x16x112x112xf16, #NHWC, @DDR>)
            outputs(%1 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
             -> memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %4 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %5 = VPUIP.NCEClusterTask {
            activation_window_channel_length = 27 : i64,
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<MAXPOOL>
        }
        input(%2 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
        weight_table(%wt : memref<16x1x1x4xsi32, @CMX_NN>)
        activation_window(%act_win : memref<1x1x1x16xui8, @CMX_NN>)
        parent_input(%2 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
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
            inputs(%0 : memref<1x16x112x112xf16, #NHWC, @DDR>)
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

// CHECK-LABEL: func.func @OptimizeParallelConstCopies

// CHECK:       [[VAR0:%.*]] =  const.Declare memref<1x16x112x112xf16, #NHWC, @DDR>
// CHECK:       [[VAR1:%.*]] =  VPUIP.Copy inputs([[VAR0]] : memref<1x16x112x112xf16, #NHWC, @DDR>)
// CHECK:       [[VAR2:%.+]] = VPUIP.NCEClusterTask
// CHECK-SAME:       input([[VAR1]] : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
// CHECK:       [[VAR3:%.*]] =  VPUIP.Copy inputs([[VAR2]] : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)

// CHECK:       [[VAR4:%.*]] =  VPUIP.Copy inputs([[VAR0]] : memref<1x16x112x112xf16, #NHWC, @DDR>)
// CHECK:       [[VAR5:%.+]] = VPUIP.NCEClusterTask
// CHECK-SAME:       input([[VAR4]] : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
// CHECK:       [[VAR6:%.*]] =  VPUIP.Copy inputs([[VAR5]] : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @SkipInputSubViewPatternCopies(
        %input: memref<1x16x112x113xf16, #NHWC, @DDR>,
        %output1: memref<1x16x112x112xf16, #NHWC, @DDR>,
        %output2: memref<1x16x112x112xf16, #NHWC, @DDR>)
         -> (memref<1x16x112x112xf16, #NHWC, @DDR>, memref<1x16x112x112xf16, #NHWC, @DDR>){
    %wt = const.Declare memref<16x1x1x4xsi32, @CMX_NN> = dense<1> : tensor<16x1x1x4xsi32>
    %act_win = const.Declare memref<1x1x1x16xui8, @CMX_NN> = dense<1> : tensor<1x1x1x16xui8>
    %0 = memref.alloc() : memref<1x16x112x113xf16, #NHWC, @DDR>

    %2 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %3 = VPUIP.SubView %input [0, 0, 0, 0] [1, 16, 112, 112] :
                memref<1x16x112x113xf16, #NHWC, @DDR> to memref<1x16x112x112xf16, {
                    order = #NHWC, strides = [202496, 1, 1808, 16]
                }, @DDR>
    %4 = VPUIP.Copy
            inputs(%3 : memref<1x16x112x112xf16, {
                order = #NHWC, strides = [202496, 1, 1808, 16]
            }, @DDR>)
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
    %9 = VPUIP.SubView %input [0, 0, 0, 0] [1, 16, 112, 112] :
                memref<1x16x112x113xf16, #NHWC, @DDR> to memref<1x16x112x112xf16, {
                    order = #NHWC, strides = [202496, 1, 1808, 16]
                }, @DDR>
    %10 = VPUIP.Copy
            inputs(%9 : memref<1x16x112x112xf16, {
                order = #NHWC, strides = [202496, 1, 1808, 16]
            }, @DDR>)
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

// CHECK-LABEL: func.func @SkipInputSubViewPatternCopies

// CHECK:       %[[ALLOC_CMX_1:.*]] = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>

// CHECK:       %[[SUBVIEW_1:.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 16, 112, 112] :
// CHECK-SAME:        memref<1x16x112x113xf16, #NHWC, @DDR> to memref<1x16x112x112xf16, {order = #NHWC, strides = [202496, 1, 1808, 16]}, @DDR>

// CHECK:       %[[DDR2CMX_1:.*]] = VPUIP.Copy inputs(%[[SUBVIEW_1]] :
// CHECK-SAME:        memref<1x16x112x112xf16, {order = #NHWC, strides = [202496, 1, 1808, 16]}, @DDR>)
// CHECK-SAME:        outputs(%[[ALLOC_CMX_1]] : memref<1x16x112x112xf16, #NHWC, @CMX_NN>) -> memref<1x16x112x112xf16, #NHWC, @CMX_NN>

// CHECK:       %[[ALLOC_NCE_CMX:.*]] = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>
// CHECK:       %[[NCE_1:.*]] = VPUIP.NCEClusterTask
// CHECK-SAME:        input(%[[DDR2CMX_1]] : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
// CHECK-SAME:        outputs(%[[ALLOC_NCE_CMX:.*]] : memref<1x16x112x112xf16, #NHWC, @CMX_NN>) -> memref<1x16x112x112xf16, #NHWC, @CMX_NN>

// CHECK:       %[[CMX2DDR_1:.*]] = VPUIP.Copy
// CHECK-SAME:        inputs(%[[NCE_1]] : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
// CHECK-SAME:        outputs(%arg1 : memref<1x16x112x112xf16, #NHWC, @DDR>) -> memref<1x16x112x112xf16, #NHWC, @DDR>

// CHECK:       %[[ALLOC_CMX_2:.*]] = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>
// CHECK:       %[[SUBVIEW_2:.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 16, 112, 112] :
// CHECK-SAME:        memref<1x16x112x113xf16, #NHWC, @DDR> to memref<1x16x112x112xf16, {order = #NHWC, strides = [202496, 1, 1808, 16]}, @DDR>

// CHECK:       %[[DDR2CMX_2:.*]] = VPUIP.Copy
// CHECK-SAME:        inputs(%[[SUBVIEW_2]] : memref<1x16x112x112xf16, {order = #NHWC, strides = [202496, 1, 1808, 16]}, @DDR>)
// CHECK-SAME:        outputs(%[[ALLOC_CMX_2]] : memref<1x16x112x112xf16, #NHWC, @CMX_NN>) -> memref<1x16x112x112xf16, #NHWC, @CMX_NN>

// CHECK:       %[[ALLOC_CMX_3:.*]] = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>
// CHECK:       %[[NCE_2:.*]] = VPUIP.NCEClusterTask
// CHECK-SAME:        input(%[[DDR2CMX_2]] : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
// CHECK-SAME:        outputs(%[[ALLOC_CMX_3]] : memref<1x16x112x112xf16, #NHWC, @CMX_NN>) -> memref<1x16x112x112xf16, #NHWC, @CMX_NN>

// CHECK:       %[[CMX2DDR_2:.*]] = VPUIP.Copy
// CHECK-SAME:        inputs(%[[NCE_2]] : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
// CHECK-SAME:        outputs(%arg1 : memref<1x16x112x112xf16, #NHWC, @DDR>) -> memref<1x16x112x112xf16, #NHWC, @DDR>

// CHECK:       return %[[CMX2DDR_1]], %[[CMX2DDR_2]] : memref<1x16x112x112xf16, #NHWC, @DDR>, memref<1x16x112x112xf16, #NHWC, @DDR>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Output_DDR = memref<1x32x28x28xf16, #NHWC, @DDR>
!Output_CMX = memref<1x32x28x28xf16, #NHWC, @CMX_NN>
!Output = memref<1x32x28x28xf16,  #NHWC>
!Weights_CMX = memref<128x32x1x1xf16, #NHWC, @CMX_NN>
!Output_CONV = memref<1x128x28x28xf16, #NHWC, @CMX_NN>
!Weights_table_CMX = memref<128x1x1x4xsi32, @CMX_NN>

!CopyOutput_Distributed = !VPUIP.DistributedBuffer<
  1x32x28x28xf16, #NHWC, @CMX_NN, {
  mode = DUPLICATED,
  num_clusters = 4 : i64
}>

!ConvOutput_Distributed = !VPUIP.DistributedBuffer<
  1x128x28x28xf16, #NHWC, @CMX_NN, {
    mode = DUPLICATED,
    num_clusters = 4 : i64
}>

func.func @OptimizeParallelMulticlusterCopies() -> (!ConvOutput_Distributed, !ConvOutput_Distributed) {
    %0 = memref.alloc() : !Output_DDR
    %1 = VPURT.AllocDistributed -> !CopyOutput_Distributed
    %2 = VPURT.AllocDistributed -> !CopyOutput_Distributed
    %3 = memref.alloc() : !Output_CMX
    %4 = VPUIP.NCEClusterTiling inputs(%3 as %arg0: !Output_CMX ) outputs(%0 as %arg3: !Output) -> !Output_DDR {
      %inner = VPUIP.Copy inputs(%arg0 : !Output_CMX ) outputs(%arg3 : !Output) -> !Output
    }
    %5 = VPUIP.NCEClusterTiling inputs(%4 as %arg0: !Output) outputs(%1 as %arg3: !Output_CMX ) -> !CopyOutput_Distributed {
      %inner = VPUIP.Copy inputs(%arg0 : !Output) outputs(%arg3 : !Output_CMX ) -> !Output_CMX
    }
    %6 = memref.alloc() : !Output_CONV
    %7 = memref.alloc() : !Weights_CMX
    %8 = memref.alloc() : !Weights_table_CMX
    %9 = VPUIP.NCEClusterTiling
      inputs(%5 as %arg2: !Output_CMX, %7 as %arg3: !Weights_CMX, %8 as %arg4:  !Weights_table_CMX)
      outputs(%6 as %arg5: !Output_CONV)
       -> !ConvOutput_Distributed {
      %inner = VPUIP.NCEClusterTask {
        kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [1, 1],
        kernel_strides = [1, 1],
        task_type = #VPUIP.nce_task_type<CONV>
          } input(%arg2 : !Output_CMX)
          weights(%arg3 : !Weights_CMX)
          weight_table(%arg4 : !Weights_table_CMX)
          parent_input(%arg2 : !Output_CMX)
          parent_output(%arg5 : !Output_CONV)
          outputs(%arg5 : !Output_CONV)
            -> !Output_CONV variants :  {
              DPUTask {cluster_id = 0 : i64, outEnd = [15, 5, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
       } PPE :  {
      }
    }
    %10 = VPUIP.NCEClusterTiling inputs(%4 as %arg0: !Output) outputs(%2 as %arg1: !Output_CMX ) -> !CopyOutput_Distributed {
      %inner = VPUIP.Copy inputs(%arg0 : !Output) outputs(%arg1 : !Output_CMX ) -> !Output_CMX
    }
    %12 = VPUIP.NCEClusterTiling
      inputs(%10 as %arg2: !Output_CMX, %7 as %arg3: !Weights_CMX, %8 as %arg4:  !Weights_table_CMX)
      outputs(%6 as %arg5: !Output_CONV)
       -> !ConvOutput_Distributed {
      %inner = VPUIP.NCEClusterTask {
        kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [1, 1],
        kernel_strides = [1, 1],
        task_type = #VPUIP.nce_task_type<CONV>
          } input(%arg2 : !Output_CMX)
          weights(%arg3 : !Weights_CMX)
          weight_table(%arg4 : !Weights_table_CMX)
          parent_input(%arg2 : !Output_CMX)
          parent_output(%arg5 : !Output_CONV)
          outputs(%arg5 : !Output_CONV)
            -> !Output_CONV variants :  {
              DPUTask {cluster_id = 0 : i64, outEnd = [15, 5, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
       } PPE :  {
      }
    }
    return %9, %12: !ConvOutput_Distributed, !ConvOutput_Distributed
}

// CHECK-LABEL: @OptimizeParallelMulticlusterCopies

//CHECK: [[DISTR_BUFFER0:%.*]] = VPURT.AllocDistributed
//CHECK: [[COMMON_ROOT:%.*]] = VPUIP.NCEClusterTiling

//CHECK: [[BRANCH1_COPY:%.*]] = VPUIP.NCEClusterTiling
//CHECK-SAME:  inputs([[COMMON_ROOT]] as %arg0: memref<1x32x28x28xf16, #NHWC>)
//CHECK-SAME:  outputs([[DISTR_BUFFER0]] as %arg1: memref<1x32x28x28xf16, #NHWC, @CMX_NN>)

//CHECK: [[BRANCH1_CONSUMER:%.*]] = VPUIP.NCEClusterTiling
//CHECK-SAME: inputs([[BRANCH1_COPY]] as %arg0: memref<1x32x28x28xf16, #NHWC, @CMX_NN>

//CHECK-NOT: VPUIP.NCEClusterTiling
//CHECK-NOT: VPUIP.Copy

//CHECK: [[BRANCH2_CONSUMER:%.*]] = VPUIP.NCEClusterTiling
//CHECK-SAME: inputs([[BRANCH1_COPY]] as %arg0: memref<1x32x28x28xf16, #NHWC, @CMX_NN>

//CHECK:  return [[BRANCH1_CONSUMER]], [[BRANCH2_CONSUMER]]

// -----

// CHECK-LABEL: @OptimizeParallelMultiShaveCopies

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#NC = affine_map<(d0, d1) -> (d0, d1)>

module @VPU.SW {
    func.func private @builtin_Gather(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "gather.cpp", VPU.kernel_entry = "gather"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @OptimizeParallelMultiShaveCopies(%arg0: memref<387072x3xf16, @DDR>,
                                            %arg1: memref<1x96768xsi32, @DDR>,
                                            %arg2: memref<1x96768xsi32, @DDR>,
                                            %arg3: memref<1x96768x1xf16, @DDR>,
                                            %arg4: memref<1x96768x1xf16, @DDR>)
                                            -> (memref<1x96768x1xf16, @DDR>, memref<1x96768x1xf16, @DDR>) {
    %0 = VPUIP.SubView %arg0 [0, 0] [387072, 1] : memref<387072x3xf16, @DDR> to memref<387072x1xf16, {order = #NC, strides = [3, 1]}, @DDR>
    %alloc = memref.alloc() : memref<387072x1xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%0 : memref<387072x1xf16, {order = #NC, strides = [3, 1]}, @DDR>) outputs(%alloc : memref<387072x1xf16, [@CMX_NN, 0]>) -> memref<387072x1xf16, [@CMX_NN, 0]>
    %alloc_0 = memref.alloc() : memref<1x96768xsi32, [@CMX_NN, 0]>
    %2 = VPUIP.Copy inputs(%arg1 : memref<1x96768xsi32, @DDR>) outputs(%alloc_0 : memref<1x96768xsi32, [@CMX_NN, 0]>) -> memref<1x96768xsi32, [@CMX_NN, 0]>
    %alloc_1 = memref.alloc() : memref<1x96768x1xf16, [@CMX_NN, 0]>
    %3 = VPUIP.SubView %2 [0, 0] [1, 48384] : memref<1x96768xsi32, [@CMX_NN, 0]> to memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>
    %4 = VPUIP.SubView %alloc_1 [0, 0, 0] [1, 48384, 1] : memref<1x96768x1xf16, [@CMX_NN, 0]> to memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>
    %5 = VPUIP.SubView %2 [0, 48384] [1, 48384] : memref<1x96768xsi32, [@CMX_NN, 0]> to memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>
    %6 = VPUIP.SubView %alloc_1 [0, 48384, 0] [1, 48384, 1] : memref<1x96768x1xf16, [@CMX_NN, 0]> to memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>

    %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Gather
    inputs( %1 as %arg5: memref<387072x1xf16, [@CMX_NN, 0]>,
            %3 as %arg6: memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>,
            %1 as %arg7: memref<387072x1xf16, [@CMX_NN, 0]>,
            %5 as %arg8: memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>)
    outputs(%4 as %arg9: memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>,
            %6 as %arg10: memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>) on tile 0 ->
    (memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>, memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>){
      VPUIP.SW.Kernel.run {attrs = [1, 0]}(%arg5, %arg6, %arg9) : memref<387072x1xf16, [@CMX_NN, 0]>, memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>,
      memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>
      VPUIP.SW.Kernel.run {attrs = [1, 0]}(%arg7, %arg8, %arg10) : memref<387072x1xf16, [@CMX_NN, 0]>, memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>,
      memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>
    }

    %7 = VPUIP.ConcatView inputs(%results#0, %results#1 : memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>, memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>)
        outputs(%alloc_1 : memref<1x96768x1xf16, [@CMX_NN, 0]>) -> memref<1x96768x1xf16, [@CMX_NN, 0]>
    %8 = VPUIP.Copy inputs(%7 : memref<1x96768x1xf16, [@CMX_NN, 0]>) outputs(%arg3 : memref<1x96768x1xf16, @DDR>) -> memref<1x96768x1xf16, @DDR>

    %alloc_2 = memref.alloc() : memref<387072x1xf16, [@CMX_NN, 0]>
    %10 = VPUIP.Copy inputs(%0 : memref<387072x1xf16, {order = #NC, strides = [3, 1]}, @DDR>) outputs(%alloc_2 : memref<387072x1xf16, [@CMX_NN, 0]>) -> memref<387072x1xf16, [@CMX_NN, 0]>
    %alloc_3 = memref.alloc() : memref<1x96768xsi32, [@CMX_NN, 0]>
    %11 = VPUIP.Copy inputs(%arg2 : memref<1x96768xsi32, @DDR>) outputs(%alloc_3 : memref<1x96768xsi32, [@CMX_NN, 0]>) -> memref<1x96768xsi32, [@CMX_NN, 0]>
    %alloc_4 = memref.alloc() : memref<1x96768x1xf16, [@CMX_NN, 0]>
    %12 = VPUIP.SubView %11 [0, 0] [1, 48384] : memref<1x96768xsi32, [@CMX_NN, 0]> to memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>
    %13 = VPUIP.SubView %alloc_4 [0, 0, 0] [1, 48384, 1] : memref<1x96768x1xf16, [@CMX_NN, 0]> to memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>
    %14 = VPUIP.SubView %11 [0, 48384] [1, 48384] : memref<1x96768xsi32, [@CMX_NN, 0]> to memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>
    %15 = VPUIP.SubView %alloc_4 [0, 48384, 0] [1, 48384, 1] : memref<1x96768x1xf16, [@CMX_NN, 0]> to memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>

    %results_5:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Gather
    inputs( %10 as %arg5: memref<387072x1xf16, [@CMX_NN, 0]>,
            %12 as %arg6: memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>,
            %10 as %arg7: memref<387072x1xf16, [@CMX_NN, 0]>,
            %14 as %arg8: memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>)
    outputs(%13 as %arg9: memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>,
    %15 as %arg10: memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>) on tile 0 ->
    (memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>, memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>){
      VPUIP.SW.Kernel.run {attrs = [1, 0]}(%arg5, %arg6, %arg9) : memref<387072x1xf16, [@CMX_NN, 0]>, memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>,
      memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>
      VPUIP.SW.Kernel.run {attrs = [1, 0]}(%arg7, %arg8, %arg10) : memref<387072x1xf16, [@CMX_NN, 0]>, memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>,
      memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>
    }

    %16 = VPUIP.ConcatView inputs(%results_5#0, %results_5#1 : memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>, memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>)
        outputs(%alloc_4 : memref<1x96768x1xf16, [@CMX_NN, 0]>) -> memref<1x96768x1xf16, [@CMX_NN, 0]>
    %17 = VPUIP.Copy inputs(%16 : memref<1x96768x1xf16, [@CMX_NN, 0]>) outputs(%arg4 : memref<1x96768x1xf16, @DDR>) -> memref<1x96768x1xf16, @DDR>
    return %8, %17 : memref<1x96768x1xf16, @DDR>, memref<1x96768x1xf16, @DDR>

    // CHECK:       [[SUBVIEW0:%.*]] = VPUIP.SubView {{[^:]+}} [0, 0] [387072, 1] : memref<387072x3xf16, @DDR> to memref<387072x1xf16, {order = #NC, strides = [3, 1]}, @DDR>
    // CHECK:       [[ALLOC_CMX0:%.*]] = memref.alloc() : memref<387072x1xf16, [@CMX_NN, 0]>
    // CHECK:       [[COPY0:%.*]] = VPUIP.Copy inputs([[SUBVIEW0]] : memref<387072x1xf16, {order = #NC, strides = [3, 1]}, @DDR>) outputs([[ALLOC_CMX0]] : memref<387072x1xf16, [@CMX_NN, 0]>) -> memref<387072x1xf16, [@CMX_NN, 0]>

    // CHECK:       [[ALLOC_CMX1:%.*]] = memref.alloc() : memref<1x96768xsi32, [@CMX_NN, 0]>
    // CHECK:       [[COPY1:%.*]] = VPUIP.Copy inputs({{[^:]+}} : memref<1x96768xsi32, @DDR>) outputs([[ALLOC_CMX1]] : memref<1x96768xsi32, [@CMX_NN, 0]>) -> memref<1x96768xsi32, [@CMX_NN, 0]>
    // CHECK:       [[ALLOC_CMX2:%.*]] = memref.alloc() : memref<1x96768x1xf16, [@CMX_NN, 0]>
    // CHECK:       [[SUBVIEW1:%.*]] = VPUIP.SubView [[COPY1]] [0, 0] [1, 48384] : memref<1x96768xsi32, [@CMX_NN, 0]> to memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>
    // CHECK:       [[SUBVIEW2:%.*]] = VPUIP.SubView [[ALLOC_CMX2]] [0, 0, 0] [1, 48384, 1] : memref<1x96768x1xf16, [@CMX_NN, 0]> to memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:       [[SUBVIEW3:%.*]] = VPUIP.SubView [[COPY1]] [0, 48384] [1, 48384] : memref<1x96768xsi32, [@CMX_NN, 0]> to memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>
    // CHECK:       [[SUBVIEW4:%.*]] = VPUIP.SubView [[ALLOC_CMX2]] [0, 48384, 0] [1, 48384, 1] : memref<1x96768x1xf16, [@CMX_NN, 0]> to memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>

    // CHECK:       [[GATHER0:%.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Gather
    // CHECK-SAME:      inputs([[COPY0]] as {{[^:]+}}: memref<387072x1xf16, [@CMX_NN, 0]>,
    // CHECK-SAME:      [[SUBVIEW1]] as {{[^:]+}}: memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>,
    // CHECK-SAME:      [[COPY0]] as {{[^:]+}}: memref<387072x1xf16, [@CMX_NN, 0]>,
    // CHECK-SAME:      [[SUBVIEW3]] as {{[^:]+}}: memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>)
    // CHECK-SAME:      outputs([[SUBVIEW2]] as {{[^:]+}}: memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>,
    // CHECK-SAME:      [[SUBVIEW4]] as {{[^:]+}}: memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>) on tile 0 ->
    // CHECK-SAME:      (memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>, memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>){
    // CHECK:         VPUIP.SW.Kernel.run {attrs = [1, 0]}({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<387072x1xf16, [@CMX_NN, 0]>, memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>,
    // CHECK-SAME:      memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:         VPUIP.SW.Kernel.run {attrs = [1, 0]}({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<387072x1xf16, [@CMX_NN, 0]>, memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>,
    // CHECK-SAME:      memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:       }

    // CHECK:       [[CONCAT0:%.*]] = VPUIP.ConcatView inputs([[GATHER0]]#0, [[GATHER0]]#1 : memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>,
    // CHECK-SAME:      memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>) outputs(%alloc_1 : memref<1x96768x1xf16, [@CMX_NN, 0]>) -> memref<1x96768x1xf16, [@CMX_NN, 0]>
    // CHECK:       [[COPY2:%.*]] = VPUIP.Copy inputs([[CONCAT0]] : memref<1x96768x1xf16, [@CMX_NN, 0]>) outputs({{[^:]+}} : memref<1x96768x1xf16, @DDR>) -> memref<1x96768x1xf16, @DDR>

    // CHECK:       [[ALLOC_CMX3:%.*]] = memref.alloc() : memref<1x96768xsi32, [@CMX_NN, 0]>
    // CHECK:       [[COPY3:%.*]] = VPUIP.Copy inputs({{[^:]+}} : memref<1x96768xsi32, @DDR>) outputs([[ALLOC_CMX3]] : memref<1x96768xsi32, [@CMX_NN, 0]>) -> memref<1x96768xsi32, [@CMX_NN, 0]>
    // CHECK:       [[ALLOC_CMX4:%.*]] = memref.alloc() : memref<1x96768x1xf16, [@CMX_NN, 0]>
    // CHECK:       [[SUBVIEW5:%.*]] = VPUIP.SubView [[COPY3]] [0, 0] [1, 48384] : memref<1x96768xsi32, [@CMX_NN, 0]> to memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>
    // CHECK:       [[SUBVIEW6:%.*]] = VPUIP.SubView [[ALLOC_CMX4]] [0, 0, 0] [1, 48384, 1] : memref<1x96768x1xf16, [@CMX_NN, 0]> to memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:       [[SUBVIEW7:%.*]] = VPUIP.SubView [[COPY3]] [0, 48384] [1, 48384] : memref<1x96768xsi32, [@CMX_NN, 0]> to memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>
    // CHECK:       [[SUBVIEW8:%.*]] = VPUIP.SubView [[ALLOC_CMX4]] [0, 48384, 0] [1, 48384, 1] : memref<1x96768x1xf16, [@CMX_NN, 0]> to memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>

    // CHECK:       [[GATHER1:%.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Gather
    // CHECK-SAME:      inputs([[COPY0]] as {{[^:]+}}: memref<387072x1xf16, [@CMX_NN, 0]>,
    // CHECK-SAME:      [[SUBVIEW5]] as {{[^:]+}}: memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>,
    // CHECK-SAME:      [[COPY0]] as {{[^:]+}}: memref<387072x1xf16, [@CMX_NN, 0]>,
    // CHECK-SAME:      [[SUBVIEW7]] as {{[^:]+}}: memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>)
    // CHECK-SAME:      outputs([[SUBVIEW6]] as {{[^:]+}}: memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>,
    // CHECK-SAME:      [[SUBVIEW8]] as {{[^:]+}}: memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>) on tile 0 ->
    // CHECK-SAME:      (memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>, memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>){
    // CHECK:         VPUIP.SW.Kernel.run {attrs = [1, 0]}({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<387072x1xf16, [@CMX_NN, 0]>, memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>,
    // CHECK-SAME:      memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:         VPUIP.SW.Kernel.run {attrs = [1, 0]}({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<387072x1xf16, [@CMX_NN, 0]>, memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>,
    // CHECK-SAME:      memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:        }

    // CHECK:       [[CONCAT1:%.*]] = VPUIP.ConcatView inputs([[GATHER1]]#0, [[GATHER1]]#1 : memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>,
    // CHECK-SAME:      memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>) outputs(%alloc_3 : memref<1x96768x1xf16, [@CMX_NN, 0]>) -> memref<1x96768x1xf16, [@CMX_NN, 0]>
    // CHECK:       [[COPY4:%.*]] = VPUIP.Copy inputs([[CONCAT1]] : memref<1x96768x1xf16, [@CMX_NN, 0]>) outputs({{[^:]+}} : memref<1x96768x1xf16, @DDR>) -> memref<1x96768x1xf16, @DDR>

    // CHECK:       return [[COPY2]], [[COPY4]] : memref<1x96768x1xf16, @DDR>, memref<1x96768x1xf16, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!OutputDistributed = !VPUIP.DistributedBuffer<
    1x144x64x128xf16, #NHWC, @CMX_NN, {
    mode = DUPLICATED,
    num_clusters = 4 : i64
}>

func.func @OptimizeParallelSubViewWithClusterTilingCopies(
        %input: memref<1x144x128x128xf16, #NHWC, @DDR>,
        %weights: memref<32x144x1x1xf16, #NHWC, @CMX_NN>,
        %weights_table : memref<144x1x1x4xsi32, @CMX_NN>)
         -> (!OutputDistributed, !OutputDistributed) {

    %0 = memref.alloc() : memref<1x144x128x128xf16, #NHWC, @DDR>
    %1 = IERT.Convert
        inputs(%input : memref<1x144x128x128xf16, #NHWC, @DDR>)
        outputs(%0 : memref<1x144x128x128xf16, #NHWC, @DDR>)
        -> memref<1x144x128x128xf16, #NHWC, @DDR>

    %2 = VPUIP.SubView %1 [0, 0, 64, 0] [1, 144, 64, 128]
            : memref<1x144x128x128xf16, #NHWC, @DDR>
            to memref<1x144x64x128xf16, {order = affine_map<(d0, d1, d2, d3)
                -> (d0, d2, d3, d1)>, strides = [2359296, 1, 18432, 144]}, @DDR>
    %3 = VPURT.AllocDistributed -> !OutputDistributed
    %4 = VPUIP.NCEClusterTiling
            inputs(%2 as %arg2: memref<1x144x64x128xf16, #NHWC>)
            outputs(%3 as %arg3: memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
                -> !OutputDistributed {
        %inner = VPUIP.Copy
                inputs(%arg2 : memref<1x144x64x128xf16, #NHWC>)
                outputs(%arg3 : memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
                    -> memref<1x144x64x128xf16, #NHWC, @CMX_NN>
    }
    %5 = VPURT.AllocDistributed -> !OutputDistributed
    %6 = VPUIP.NCEClusterTiling
            inputs(
                %4 as %arg2: memref<1x144x64x128xf16, #NHWC, @CMX_NN>,
                %weights as %arg3: memref<32x144x1x1xf16, #NHWC, @CMX_NN>,
                %weights_table as %arg4: memref<32x1x1x4xsi32, @CMX_NN>)
            outputs(
                %5 as %arg5: memref<1x32x64x128xf16, #NHWC, @CMX_NN>)
                    -> !OutputDistributed {
        %inner = VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                minimumHardwareExecutionCost = 9240 : i64, task_type = #VPUIP.nce_task_type<CONV>}
            input(
                %arg2 : memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
                weights(%arg3 : memref<32x144x1x1xf16, #NHWC, @CMX_NN>)
                weight_table(%arg4 : memref<32x1x1x4xsi32, @CMX_NN>)
                parent_input(%arg2 : memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
                parent_output(%arg5 : memref<1x32x64x128xf16, #NHWC, @CMX_NN>)
                outputs(%arg5 : memref<1x32x64x128xf16, #NHWC, @CMX_NN>)
                    -> memref<1x32x64x128xf16, #NHWC, @CMX_NN> variants :  {
                DPUTask {cluster_id = 0 : i64, outEnd = [15, 5, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
        } PPE :  {
        }
    }

    %7 = VPUIP.SubView %1 [0, 0, 64, 0] [1, 144, 64, 128]
            : memref<1x144x128x128xf16, #NHWC, @DDR>
            to memref<1x144x64x128xf16, {order = affine_map<(d0, d1, d2, d3)
                -> (d0, d2, d3, d1)>, strides = [2359296, 1, 18432, 144]}, @DDR>
    %8 = VPURT.AllocDistributed -> !OutputDistributed
    %9 = VPUIP.NCEClusterTiling
            inputs(%7 as %arg2: memref<1x144x64x128xf16, #NHWC>)
            outputs(%8 as %arg3: memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
                -> !OutputDistributed {
        %inner = VPUIP.Copy
                inputs(%arg2 : memref<1x144x64x128xf16, #NHWC>)
                outputs(%arg3 : memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
                    -> memref<1x144x64x128xf16, #NHWC, @CMX_NN>
    }
    %10 = VPURT.AllocDistributed -> !OutputDistributed
    %11 = VPUIP.NCEClusterTiling
            inputs(
                %9 as %arg2: memref<1x144x64x128xf16, #NHWC, @CMX_NN>,
                %weights as %arg3: memref<32x144x1x1xf16, #NHWC, @CMX_NN>,
                %weights_table as %arg4: memref<32x1x1x4xsi32, @CMX_NN>)
            outputs(
                %10 as %arg5: memref<1x32x64x128xf16, #NHWC, @CMX_NN>)
                    -> !OutputDistributed {
        %inner = VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                minimumHardwareExecutionCost = 9240 : i64, task_type = #VPUIP.nce_task_type<CONV>}
            input(
                %arg2 : memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
                weights(%arg3 : memref<32x144x1x1xf16, #NHWC, @CMX_NN>)
                weight_table(%arg4 : memref<32x1x1x4xsi32, @CMX_NN>)
                parent_input(%arg2 : memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
                parent_output(%arg5 : memref<1x32x64x128xf16, #NHWC, @CMX_NN>)
                outputs(%arg5 : memref<1x32x64x128xf16, #NHWC, @CMX_NN>)
                    -> memref<1x32x64x128xf16, #NHWC, @CMX_NN> variants :  {
                DPUTask {cluster_id = 0 : i64, outEnd = [15, 5, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
        } PPE :  {
        }
    }

    return %6, %11 : !OutputDistributed, !OutputDistributed

    // CHECK:       [[IN_BUFFER:%.*]] = memref.alloc() : memref<1x144x128x128xf16, #NHWC, @DDR>
    // CHECK:       [[CONVERT:%.*]] = IERT.Convert inputs(%arg0 : memref<1x144x128x128xf16, #NHWC, @DDR>) outputs([[IN_BUFFER]] : memref<1x144x128x128xf16, #NHWC, @DDR>) -> memref<1x144x128x128xf16, #NHWC, @DDR>

    // CHECK:       [[SUBVIEW_1:%.*]] = VPUIP.SubView [[CONVERT]] [0, 0, 64, 0] [1, 144, 64, 128]
    // CHECK-SAME:      memref<1x144x128x128xf16, #NHWC, @DDR> to
    // CHECK-SAME:      memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>
    // CHECK:       [[BUFFER_1:%.*]] = VPURT.AllocDistributed
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[COPY_1:%.*]] = VPUIP.NCEClusterTiling inputs([[SUBVIEW_1]] as %arg3: memref<1x144x64x128xf16, #NHWC>)
    // CHECK-SAME:      outputs([[BUFFER_1]] as %arg4: memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}> {
    // CHECK:       [[COPY_INNER_1:%.*]] = VPUIP.Copy inputs(%arg3 : memref<1x144x64x128xf16, #NHWC>)
    // CHECK-SAME:      outputs(%arg4 : memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:          -> memref<1x144x64x128xf16, #NHWC, @CMX_NN>

    // CHECK:       [[BUFFER_2:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[NCE_1:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[COPY_1]] as %arg3: memref<1x144x64x128xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:      %arg1 as %arg4: memref<32x144x1x1xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:      %arg2 as %arg5: memref<32x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:      outputs([[BUFFER_2]] as %arg6: memref<1x32x64x128xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>

    // CHECK-NOT:   VPUIP.SubView
    // CHECK:       [[BUFFER_3:%.*]] = VPURT.AllocDistributed
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK-NOT:   VPUIP.Copy
    // CHECK:       [[NCE_2:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[COPY_1]] as %arg3: memref<1x144x64x128xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:      %arg1 as %arg4: memref<32x144x1x1xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:      %arg2 as %arg5: memref<32x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:      outputs([[BUFFER_3]] as %arg6: memref<1x32x64x128xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>


    // CHECK:       return [[NCE_1]], [[NCE_2]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:                                     !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x2x512x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!SubViewDistributed = !VPUIP.DistributedBuffer<
    1x1x512x1xf16, {
    order = #NHWC,
    strides = [1024, 1, 2, 2]}, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

func.func @NotOptimizeParallelClusterTilingCopiesWithSubviewHasDiffOffset(
        %arg0: memref<1x1x512x1xf16, @DDR>) -> !OutputDistributed {
    %1 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%arg0 : memref<1x1x512x1xf16, @DDR>) -> memref<1x1x512x1xf16, #NHWC, @DDR>
    %2 = VPURT.AllocDistributed -> !OutputDistributed

    %3 = VPUIP.SubView %2 [0, 0, 0, 0] [1, 1, 512, 1] : !OutputDistributed to !SubViewDistributed
    %4 = VPUIP.NCEClusterTiling
            inputs(%1 as %arg3: memref<1x1x512x1xf16, #NHWC, @DDR>)
            outputs(%3 as %arg4: memref<1x1x512x1xf16, {order = #NHWC, strides = [1024, 1, 2, 2]}, @CMX_NN>) -> !SubViewDistributed {
      VPUIP.Copy inputs(%arg3 : memref<1x1x512x1xf16, #NHWC, @DDR>)
                 outputs(%arg4 : memref<1x1x512x1xf16, {order = #NHWC, strides = [1024, 1, 2, 2]}, @CMX_NN>) -> memref<1x1x512x1xf16, {order = #NHWC, strides = [1024, 1, 2, 2]}, @CMX_NN>
    }

    %5 = VPUIP.SubView %2 [0, 1, 0, 0] [1, 1, 512, 1] : !OutputDistributed to !SubViewDistributed
    %6 = VPUIP.NCEClusterTiling
            inputs(%1 as %arg3: memref<1x1x512x1xf16, #NHWC, @DDR>)
            outputs(%5 as %arg4: memref<1x1x512x1xf16, {order = #NHWC, strides = [1024, 1, 2, 2]}, @CMX_NN>) -> !SubViewDistributed {
      VPUIP.Copy inputs(%arg3 : memref<1x1x512x1xf16, #NHWC, @DDR>)
                 outputs(%arg4 : memref<1x1x512x1xf16, {order = #NHWC, strides = [1024, 1, 2, 2]}, @CMX_NN>) -> memref<1x1x512x1xf16, {order = #NHWC, strides = [1024, 1, 2, 2]}, @CMX_NN>
    }

    %7 = VPUIP.ConcatView
            inputs(%4, %6 : !SubViewDistributed, !SubViewDistributed)
            outputs(%2 : !OutputDistributed) -> !OutputDistributed

    return %7 : !OutputDistributed

    // CHECK:       [[PERMUTECAST:%.*]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:      inputs(%arg0 : memref<1x1x512x1xf16, @DDR>) -> memref<1x1x512x1xf16, #NHWC, @DDR>
    // CHECK:       [[OUT_BUFFER:%.*]] = VPURT.AllocDistributed

    // CHECK:       [[SUBVIEW_0:%.*]] = VPUIP.SubView [[OUT_BUFFER]] [0, 0, 0, 0] [1, 1, 512, 1]
    // CHECK:       [[CLUSTER_0:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[PERMUTECAST]] as %arg1: memref<1x1x512x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[SUBVIEW_0]] as %arg2: memref<1x1x512x1xf16, {order = #NHWC, strides = [1024, 1, 2, 2]}, @CMX_NN>)
    // CHECK:       [[COPY_0:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs(%arg1 : memref<1x1x512x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:      outputs(%arg2 : memref<1x1x512x1xf16, {order = #NHWC, strides = [1024, 1, 2, 2]}, @CMX_NN>)

    // CHECK:       [[SUBVIEW_1:%.*]] = VPUIP.SubView [[OUT_BUFFER]] [0, 1, 0, 0] [1, 1, 512, 1]
    // CHECK:       [[CLUSTER_1:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[PERMUTECAST]] as %arg1: memref<1x1x512x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[SUBVIEW_1]] as %arg2: memref<1x1x512x1xf16, {order = #NHWC, strides = [1024, 1, 2, 2]}, @CMX_NN>)
    // CHECK:       [[COPY_1:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs(%arg1 : memref<1x1x512x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:      outputs(%arg2 : memref<1x1x512x1xf16, {order = #NHWC, strides = [1024, 1, 2, 2]}, @CMX_NN>)

    // CHECK:       [[CONCATVIEW:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[CLUSTER_0]], [[CLUSTER_1]]
    // CHECK-SAME:      outputs([[OUT_BUFFER]]

    // CHECK:       return [[CONCATVIEW]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!IODataDDRType = memref<1x32x28x28xf16, #NHWC, @DDR>
!IOSMDDRType = memref<1x32x28x28xi1, #NHWC, @DDR>
!IOSparseDDRType = !VPUIP.SparseBuffer<data=!IODataDDRType, sparsity_map=!IOSMDDRType>

!IODataCMXType = memref<1x32x28x28xf16, #NHWC, @CMX_NN>
!IOSMCMXType = memref<1x32x28x28xi1, #NHWC, @CMX_NN>
!IOSparseCMXType = !VPUIP.SparseBuffer<data=!IODataCMXType, sparsity_map=!IOSMCMXType>


!Weights_CMX = memref<32x32x1x1xf16, #NHWC, @CMX_NN>
!Weights_table_CMX = memref<32x1x1x4xsi32, @CMX_NN>

!IODataDistrType = !VPUIP.DistributedBuffer<
  1x32x28x28xf16, #NHWC, @CMX_NN, {
  mode = DUPLICATED,
  num_clusters = 4 : i64
}>

!IOSMDistrType = !VPUIP.DistributedBuffer<
  1x32x28x28xi1, #NHWC, @CMX_NN, {
  mode = DUPLICATED,
  num_clusters = 4 : i64
}>

!IOSparseDistrType = !VPUIP.SparseBuffer<data=!IODataDistrType, sparsity_map=!IOSMDistrType>

// CHECK-LABEL: @OptimizeParallelMulticlusterCopiesSparse
func.func @OptimizeParallelMulticlusterCopiesSparse() -> (!IOSparseDistrType, !IOSparseDistrType) {
    %0 = memref.alloc() : !IODataCMXType
    %1 = memref.alloc() : !IOSMCMXType
    %2 = VPUIP.GroupSparseBuffer(%0, %1) -> !IOSparseCMXType

    %3 = memref.alloc() : !IODataDDRType
    %4 = memref.alloc() : !IOSMDDRType
    %5 = VPUIP.GroupSparseBuffer(%3, %4) -> !IOSparseDDRType

    %6 = VPUIP.NCEClusterTiling inputs(%2 as %arg0: !IOSparseCMXType ) outputs(%5 as %arg3: !IOSparseDDRType) -> !IOSparseDDRType {
      %inner = VPUIP.Copy inputs(%arg0 : !IOSparseCMXType ) outputs(%arg3 : !IOSparseDDRType) -> !IOSparseDDRType
    }

    %7 = VPURT.AllocDistributed -> !IODataDistrType
    %8 = VPURT.AllocDistributed -> !IOSMDistrType
    %9 = VPUIP.GroupSparseBuffer(%7, %8) -> !IOSparseDistrType

    %10 = VPUIP.NCEClusterTiling inputs(%6 as %arg0: !IOSparseDDRType) outputs(%9 as %arg3: !IOSparseCMXType) -> !IOSparseDistrType {
      %inner = VPUIP.Copy inputs(%arg0 : !IOSparseDDRType) outputs(%arg3 : !IOSparseCMXType ) -> !IOSparseCMXType
    }

    %11 = VPURT.AllocDistributed -> !IODataDistrType
    %12 = VPURT.AllocDistributed -> !IOSMDistrType
    %13 = VPUIP.GroupSparseBuffer(%11, %12) -> !IOSparseDistrType

    %14 = memref.alloc() : !Weights_CMX
    %15 = memref.alloc() : !Weights_table_CMX

    %in_data_0, %in_sm_0 = VPUIP.UngroupSparseBuffer(%10) {resultSegmentSizes = array<i32: 1, 1, 0>}
        -> !IODataDistrType, !IOSMDistrType
    %out_data_0, %out_sm_0 = VPUIP.UngroupSparseBuffer(%13) {resultSegmentSizes = array<i32: 1, 1, 0>}
        -> !IODataDistrType, !IOSMDistrType

    %16:2 = VPUIP.NCEClusterTiling
      inputs(%in_data_0 as %arg2: !IODataCMXType,
             %in_sm_0 as %arg3: !IOSMCMXType,
             %14 as %arg4: !Weights_CMX,
             %15 as %arg5:  !Weights_table_CMX)
      outputs(%out_data_0 as %arg6: !IODataCMXType,
              %out_sm_0 as %arg7: !IOSMCMXType
      ) -> (!IODataDistrType, !IOSMDistrType) {
      %inner:2 = VPUIP.NCEClusterTask {
        kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [1, 1],
        kernel_strides = [1, 1],
        task_type = #VPUIP.nce_task_type<CONV>
          }
          input(%arg2 : !IODataCMXType)
          input_sparsity_map(%arg3 : !IOSMCMXType)
          weights(%arg4 : !Weights_CMX)
          weight_table(%arg5 : !Weights_table_CMX)
          parent_input(%arg2 : !IODataCMXType)
          parent_input_sparsity_map(%arg3 : !IOSMCMXType)
          parent_output(%arg6 : !IODataCMXType)
          parent_output_sparsity_map(%arg7 : !IOSMCMXType)
          outputs(%arg6 : !IODataCMXType)
          output_sparsity_map(%arg7 : !IOSMCMXType)
            -> !IODataCMXType, !IOSMCMXType variants :  {
              DPUTask {cluster_id = 0 : i64, outEnd = [15, 5, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
       } PPE :  {
      }
    }
    %output_sparse_0 = VPUIP.GroupSparseBuffer(%16#0, %16#1) -> !IOSparseDistrType

    %17 = VPURT.AllocDistributed -> !IODataDistrType
    %18 =  VPURT.AllocDistributed -> !IOSMDistrType
    %19 = VPUIP.GroupSparseBuffer(%17, %18) -> !IOSparseDistrType

    %20 = VPUIP.NCEClusterTiling inputs(%6 as %arg0: !IOSparseDDRType) outputs(%19 as %arg3: !IOSparseCMXType) -> !IOSparseDistrType {
      %inner = VPUIP.Copy inputs(%arg0 : !IOSparseDDRType) outputs(%arg3 : !IOSparseCMXType ) -> !IOSparseCMXType
    }

    %21 = VPURT.AllocDistributed -> !IODataDistrType
    %22 =  VPURT.AllocDistributed -> !IOSMDistrType
    %23 = VPUIP.GroupSparseBuffer(%21, %22) -> !IOSparseDistrType

    %in_data_1, %in_sm_1 = VPUIP.UngroupSparseBuffer(%20) {resultSegmentSizes = array<i32: 1, 1, 0>}
        -> !IODataDistrType, !IOSMDistrType
    %out_data_1, %out_sm_1 = VPUIP.UngroupSparseBuffer(%23) {resultSegmentSizes = array<i32: 1, 1, 0>}
        -> !IODataDistrType, !IOSMDistrType

    %24:2 = VPUIP.NCEClusterTiling
      inputs(%in_data_1 as %arg2: !IODataCMXType,
             %in_sm_1 as %arg3: !IOSMCMXType,
             %14 as %arg4: !Weights_CMX,
             %15 as %arg5:  !Weights_table_CMX)
      outputs(%out_data_1 as %arg6: !IODataCMXType,
              %out_sm_1 as %arg7: !IOSMCMXType
      ) -> (!IODataDistrType, !IOSMDistrType) {
      %inner:2 = VPUIP.NCEClusterTask {
        kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [1, 1],
        kernel_strides = [1, 1],
        task_type = #VPUIP.nce_task_type<CONV>
          }
          input(%arg2 : !IODataCMXType)
          input_sparsity_map(%arg3 : !IOSMCMXType)
          weights(%arg4 : !Weights_CMX)
          weight_table(%arg5 : !Weights_table_CMX)
          parent_input(%arg2 : !IODataCMXType)
          parent_input_sparsity_map(%arg3 : !IOSMCMXType)
          parent_output(%arg6 : !IODataCMXType)
          parent_output_sparsity_map(%arg7 : !IOSMCMXType)
          outputs(%arg6 : !IODataCMXType)
          output_sparsity_map(%arg7 : !IOSMCMXType)
            -> !IODataCMXType, !IOSMCMXType variants :  {
              DPUTask {cluster_id = 0 : i64, outEnd = [15, 5, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
       } PPE :  {
      }
    }
    %output_sparse_1 = VPUIP.GroupSparseBuffer(%24#0, %24#1) -> !IOSparseDistrType

    return %output_sparse_0, %output_sparse_1: !IOSparseDistrType, !IOSparseDistrType

    // CHECK:       [[BUFF_0_DATA:%.*]] = memref.alloc() : memref<1x32x28x28xf16, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_0_SM:%.*]] = memref.alloc() : memref<1x32x28x28xi1, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_0:%.*]] = VPUIP.GroupSparseBuffer([[BUFF_0_DATA]], [[BUFF_0_SM]])
    // CHECK-SAME:                 -> !VPUIP.SparseBuffer<data=memref<1x32x28x28xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x32x28x28xi1, #NHWC, @CMX_NN>>


    // CHECK:       [[BUFF_1_DATA:%.*]] = memref.alloc() : memref<1x32x28x28xf16, #NHWC, @DDR>
    // CHECK:       [[BUFF_1_SM:%.*]] = memref.alloc() : memref<1x32x28x28xi1, #NHWC, @DDR>
    // CHECK:       [[BUFF_1:%.*]] = VPUIP.GroupSparseBuffer([[BUFF_1_DATA]], [[BUFF_1_SM]])
    // CHECK-SAME:                 -> !VPUIP.SparseBuffer<data=memref<1x32x28x28xf16, #NHWC, @DDR>, sparsity_map=memref<1x32x28x28xi1, #NHWC, @DDR>>

    // CHECK:       [[COMMON_ROOT:%.*]] = VPUIP.NCEClusterTiling inputs([[BUFF_0]] as %arg0
    // CHECK-SAME:                 outputs([[BUFF_1]] as %arg1
    // CHECK-SAME:                 -> !VPUIP.SparseBuffer<data=memref<1x32x28x28xf16, #NHWC, @DDR>, sparsity_map=memref<1x32x28x28xi1, #NHWC, @DDR>>
    // CHECK:                       VPUIP.Copy inputs(%arg0
    // CHECK-SAME:                      outputs(%arg1

    // CHECK:       [[BUFF_2_DATA:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[BUFF_2_SM:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x28x28xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[BUFF_2:%.*]] = VPUIP.GroupSparseBuffer([[BUFF_2_DATA]], [[BUFF_2_SM]])
    // CHECK-SAME:                 -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x32x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>,
    // CHECK-SAME:                  sparsity_map=!VPUIP.DistributedBuffer<1x32x28x28xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>>

    // CHECK:       [[BRANCH_0_COPY:%.*]] = VPUIP.NCEClusterTiling inputs([[COMMON_ROOT]] as %arg0
    // CHECK-SAME:                 outputs([[BUFF_2]] as %arg1
    // CHECK-SAME:                 -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x32x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>,
    // CHECK-SAME:                  sparsity_map=!VPUIP.DistributedBuffer<1x32x28x28xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>>
    // CHECK:                       VPUIP.Copy inputs(%arg0
    // CHECK-SAME:                      outputs(%arg1

    // CHECK:       [[BUFF_3_DATA:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[BUFF_3_SM:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x28x28xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[BUFF_3:%.*]] = VPUIP.GroupSparseBuffer([[BUFF_3_DATA]], [[BUFF_3_SM]])
    // CHECK-SAME:                 -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x32x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>,
    // CHECK-SAME:                  sparsity_map=!VPUIP.DistributedBuffer<1x32x28x28xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>>

    // CHECK:       [[DATA_0:%.*]], [[SM_0:%.*]] = VPUIP.UngroupSparseBuffer([[BRANCH_0_COPY]])
    // CHECK:       [[DATA_1:%.*]], [[SM_1:%.*]] = VPUIP.UngroupSparseBuffer([[BUFF_3]])

    // CHECK:       [[NCE0_U:%.*]]:2 = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[DATA_0]] as %arg0: memref<1x32x28x28xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:             [[SM_0]] as %arg1: memref<1x32x28x28xi1, #NHWC, @CMX_NN>
    // CHECK-SAME:      outputs([[DATA_1]] as %arg4: memref<1x32x28x28xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:              [[SM_1]] as %arg5: memref<1x32x28x28xi1, #NHWC, @CMX_NN>)
    // CHECK-SAME:              -> (!VPUIP.DistributedBuffer<1x32x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>,
    // CHECK-SAME:                  !VPUIP.DistributedBuffer<1x32x28x28xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)

    // CHECK:       [[NCE_0:%.*]] = VPUIP.GroupSparseBuffer([[NCE0_U]]#0, [[NCE0_U]]#1)

    // CHECK-NOT:       VPUIP.NCEClusterTiling
    // CHECK-NOT:       VPUIP.Copy

    // CHECK:       [[BUFF_4_DATA:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[BUFF_4_SM:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x28x28xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[BUFF_4:%.*]] = VPUIP.GroupSparseBuffer([[BUFF_4_DATA]], [[BUFF_4_SM]])
    // CHECK-SAME:                 -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x32x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>,
    // CHECK-SAME:                  sparsity_map=!VPUIP.DistributedBuffer<1x32x28x28xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>>


    // CHECK:       [[DATA_2:%.*]], [[SM_2:%.*]] = VPUIP.UngroupSparseBuffer([[BRANCH_0_COPY]])
    // CHECK:       [[DATA_3:%.*]], [[SM_3:%.*]] = VPUIP.UngroupSparseBuffer([[BUFF_4]])

    // CHECK:       [[NCE1_U:%.*]]:2 = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[DATA_2]] as %arg0: memref<1x32x28x28xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:             [[SM_2]] as %arg1: memref<1x32x28x28xi1, #NHWC, @CMX_NN>
    // CHECK-SAME:      outputs([[DATA_3]] as %arg4: memref<1x32x28x28xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:              [[SM_3]] as %arg5: memref<1x32x28x28xi1, #NHWC, @CMX_NN>)
    // CHECK-SAME:              -> (!VPUIP.DistributedBuffer<1x32x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>,
    // CHECK-SAME:                  !VPUIP.DistributedBuffer<1x32x28x28xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)

    // CHECK:       [[NCE_1:%.*]] = VPUIP.GroupSparseBuffer([[NCE1_U]]#0, [[NCE1_U]]#1)

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!IDataDDRType = memref<1x144x128x128xf16, #NHWC, @DDR>
!ISMDDRType = memref<1x144x128x128xi1, #NHWC, @DDR>
!ISparseDDRType = !VPUIP.SparseBuffer<data=!IDataDDRType, sparsity_map=!ISMDDRType>

!ISparseHalfDDRType = !VPUIP.SparseBuffer<
  data=memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>,
  sparsity_map=memref<1x144x64x128xi1, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>
>

!IDataHalfCMXType = memref<1x144x64x128xf16, #NHWC, @CMX_NN>
!ISMHalfCMXType = memref<1x144x64x128xi1, #NHWC, @CMX_NN>
!ISparseHalfCMXType = !VPUIP.SparseBuffer<
  data=!IDataHalfCMXType,
  sparsity_map=!ISMHalfCMXType
>

!ODistrDataType = !VPUIP.DistributedBuffer<
    1x144x64x128xf16, #NHWC, @CMX_NN, {
    mode = DUPLICATED,
    num_clusters = 4 : i64
}>
!ODistrSMType = !VPUIP.DistributedBuffer<
    1x144x64x128xi1, #NHWC, @CMX_NN, {
    mode = DUPLICATED,
    num_clusters = 4 : i64
}>
!ODistrSparseType = !VPUIP.SparseBuffer<data=!ODistrDataType, sparsity_map=!ODistrSMType>

// CHECK-LABEL: @OptimizeParallelSubViewWithClusterTilingCopiesSparse
func.func @OptimizeParallelSubViewWithClusterTilingCopiesSparse(
        %input: !ISparseDDRType,
        %weights: memref<32x144x1x1xf16, #NHWC, @CMX_NN>,
        %weights_table : memref<144x1x1x4xsi32, @CMX_NN>)
         -> (!ODistrSparseType, !ODistrSparseType) {

    %0 = memref.alloc() : !IDataDDRType
    %1 = memref.alloc() : !ISMDDRType
    %2 = VPUIP.GroupSparseBuffer(%0, %1) -> !ISparseDDRType

    %3 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC}
            inputs(%2 : !ISparseDDRType ) -> !ISparseDDRType

    %4 = VPUIP.SubView %3 [0, 0, 64, 0] [1, 144, 64, 128] : !ISparseDDRType to !ISparseHalfDDRType

    %5 = VPURT.AllocDistributed -> !ODistrDataType
    %6 = VPURT.AllocDistributed -> !ODistrSMType
    %7 = VPUIP.GroupSparseBuffer(%5, %6) -> !ODistrSparseType

    %8 = VPUIP.NCEClusterTiling
            inputs(%4 as %arg2: !ISparseHalfDDRType)
            outputs(%7 as %arg3: !ISparseHalfCMXType)
                -> !ODistrSparseType {
        %inner = VPUIP.Copy
                inputs(%arg2 : !ISparseHalfDDRType)
                outputs(%arg3 : !ISparseHalfCMXType)
                    -> !ISparseHalfCMXType
    }

    %9 = VPURT.AllocDistributed -> !ODistrDataType
    %10 = VPURT.AllocDistributed -> !ODistrSMType
    %11 = VPUIP.GroupSparseBuffer(%9, %10) -> !ODistrSparseType

    %in_data_0, %in_sm_0 = VPUIP.UngroupSparseBuffer(%8) {resultSegmentSizes = array<i32: 1, 1, 0>}
        -> !ODistrDataType, !ODistrSMType
    %out_data_0, %out_sm_0 = VPUIP.UngroupSparseBuffer(%11) {resultSegmentSizes = array<i32: 1, 1, 0>}
        -> !ODistrDataType, !ODistrSMType

    %12:2 = VPUIP.NCEClusterTiling
        inputs(%in_data_0 as %arg2: !IDataHalfCMXType,
                %in_sm_0 as %arg3: !ISMHalfCMXType,
                %weights as %arg4: memref<32x144x1x1xf16, #NHWC, @CMX_NN>,
                %weights_table as %arg5: memref<32x1x1x4xsi32, @CMX_NN>)
        outputs(%out_data_0 as %arg6: !IDataHalfCMXType,
              %out_sm_0 as %arg7: !ISMHalfCMXType
      ) -> (!ODistrDataType, !ODistrSMType) {
        %inner:2 = VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                minimumHardwareExecutionCost = 9240 : i64, task_type = #VPUIP.nce_task_type<CONV>}
          input(%arg2 : !IDataHalfCMXType)
          input_sparsity_map(%arg3 : !ISMHalfCMXType)
          weights(%arg4 : memref<32x144x1x1xf16, #NHWC, @CMX_NN>)
          weight_table(%arg5 : memref<32x1x1x4xsi32, @CMX_NN>)
          parent_input(%arg2 : !IDataHalfCMXType)
          parent_input_sparsity_map(%arg3 : !ISMHalfCMXType)
          parent_output(%arg6 : !IDataHalfCMXType)
          parent_output_sparsity_map(%arg7 : !ISMHalfCMXType)
          outputs(%arg6 : !IDataHalfCMXType)
          output_sparsity_map(%arg7 : !ISMHalfCMXType)
            -> !IDataHalfCMXType, !ISMHalfCMXType variants :  {
                DPUTask {cluster_id = 0 : i64, outEnd = [15, 5, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
        } PPE :  {
        }
    }
    %output_sparse_0 = VPUIP.GroupSparseBuffer(%12#0, %12#1) -> !ODistrSparseType

    %13 = VPUIP.SubView %3 [0, 0, 64, 0] [1, 144, 64, 128] : !ISparseDDRType to !ISparseHalfDDRType

    %14 = VPURT.AllocDistributed -> !ODistrDataType
    %15 = VPURT.AllocDistributed -> !ODistrSMType
    %16 = VPUIP.GroupSparseBuffer(%14, %15) -> !ODistrSparseType

    %17 = VPUIP.NCEClusterTiling
            inputs(%13 as %arg2: !ISparseHalfDDRType)
            outputs(%16 as %arg3: !ISparseHalfCMXType)
                -> !ODistrSparseType {
        %inner = VPUIP.Copy
                inputs(%arg2 : !ISparseHalfDDRType)
                outputs(%arg3 : !ISparseHalfCMXType)
                    -> !ISparseHalfCMXType
    }

    %18 = VPURT.AllocDistributed -> !ODistrDataType
    %19 = VPURT.AllocDistributed -> !ODistrSMType
    %20 = VPUIP.GroupSparseBuffer(%18, %19) -> !ODistrSparseType

    %in_data_1, %in_sm_1 = VPUIP.UngroupSparseBuffer(%17) {resultSegmentSizes = array<i32: 1, 1, 0>}
        -> !ODistrDataType, !ODistrSMType
    %out_data_1, %out_sm_1 = VPUIP.UngroupSparseBuffer(%20) {resultSegmentSizes = array<i32: 1, 1, 0>}
        -> !ODistrDataType, !ODistrSMType

    %21:2 = VPUIP.NCEClusterTiling
        inputs(%in_data_1 as %arg2: !IDataHalfCMXType,
                %in_sm_1 as %arg3: !ISMHalfCMXType,
                %weights as %arg4: memref<32x144x1x1xf16, #NHWC, @CMX_NN>,
                %weights_table as %arg5: memref<32x1x1x4xsi32, @CMX_NN>)
        outputs(%out_data_1 as %arg6: !IDataHalfCMXType,
              %out_sm_1 as %arg7: !ISMHalfCMXType
      ) -> (!ODistrDataType, !ODistrSMType) {
        %inner:2 = VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                minimumHardwareExecutionCost = 9240 : i64, task_type = #VPUIP.nce_task_type<CONV>}
          input(%arg2 : !IDataHalfCMXType)
          input_sparsity_map(%arg3 : !ISMHalfCMXType)
          weights(%arg4 : memref<32x144x1x1xf16, #NHWC, @CMX_NN>)
          weight_table(%arg5 : memref<32x1x1x4xsi32, @CMX_NN>)
          parent_input(%arg2 : !IDataHalfCMXType)
          parent_input_sparsity_map(%arg3 : !ISMHalfCMXType)
          parent_output(%arg6 : !IDataHalfCMXType)
          parent_output_sparsity_map(%arg7 : !ISMHalfCMXType)
          outputs(%arg6 : !IDataHalfCMXType)
          output_sparsity_map(%arg7 : !ISMHalfCMXType)
            -> !IDataHalfCMXType, !ISMHalfCMXType variants :  {
                DPUTask {cluster_id = 0 : i64, outEnd = [15, 5, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
        } PPE :  {
        }
    }
    %output_sparse_1 = VPUIP.GroupSparseBuffer(%21#0, %21#1) -> !ODistrSparseType

    return %output_sparse_0, %output_sparse_1 : !ODistrSparseType, !ODistrSparseType

    // CHECK:       [[BUFF_0_DATA:%.*]] = memref.alloc() : memref<1x144x128x128xf16, #NHWC, @DDR>
    // CHECK:       [[BUFF_0_SM:%.*]] = memref.alloc() : memref<1x144x128x128xi1, #NHWC, @DDR>
    // CHECK:       [[BUFF_0:%.*]] = VPUIP.GroupSparseBuffer([[BUFF_0_DATA]], [[BUFF_0_SM]])
    // CHECK-SAME:                 -> !VPUIP.SparseBuffer<data=memref<1x144x128x128xf16, #NHWC, @DDR>, sparsity_map=memref<1x144x128x128xi1, #NHWC, @DDR>>

    // CHECK:       [[PERMUTE:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:                  inputs([[BUFF_0]] : !VPUIP.SparseBuffer<data=memref<1x144x128x128xf16, #NHWC, @DDR>, sparsity_map=memref<1x144x128x128xi1, #NHWC, @DDR>>
    // CHECK-SAME:                  -> !VPUIP.SparseBuffer<data=memref<1x144x128x128xf16, #NHWC, @DDR>, sparsity_map=memref<1x144x128x128xi1, #NHWC, @DDR>>

    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView [[PERMUTE]] [0, 0, 64, 0] [1, 144, 64, 128] :
    // CHECK-SAME:                  !VPUIP.SparseBuffer<data=memref<1x144x128x128xf16, #NHWC, @DDR>, sparsity_map=memref<1x144x128x128xi1, #NHWC, @DDR>> to
    // CHECK-SAME:                  !VPUIP.SparseBuffer<data=memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>, sparsity_map=memref<1x144x64x128xi1, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>>

    // CHECK:       [[BUFF_1_DATA:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[BUFF_1_SM:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[BUFF_1:%.*]] = VPUIP.GroupSparseBuffer([[BUFF_1_DATA]], [[BUFF_1_SM]])
    // CHECK-SAME:                 -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>, sparsity_map=!VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>>

    // CHECK:       [[COPY_0:%.*]] = VPUIP.NCEClusterTiling inputs([[SUBVIEW_0]] as %arg3
    // CHECK-SAME:                 outputs([[BUFF_1]] as %arg4
    // CHECK-SAME:                 -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>, sparsity_map=!VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>>
    // CHECK:                       VPUIP.Copy inputs(%arg3
    // CHECK-SAME:                      outputs(%arg4

    // CHECK:       [[BUFF_2_DATA:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[BUFF_2_SM:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[BUFF_2:%.*]] = VPUIP.GroupSparseBuffer([[BUFF_2_DATA]], [[BUFF_2_SM]])
    // CHECK-SAME:                 -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>, sparsity_map=!VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>>

    // CHECK:       [[DATA_0:%.*]], [[SM_0:%.*]] = VPUIP.UngroupSparseBuffer([[COPY_0]])
    // CHECK:       [[DATA_1:%.*]], [[SM_1:%.*]] = VPUIP.UngroupSparseBuffer([[BUFF_2]])

    // CHECK:       [[NCE0_U:%.*]]:2 = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[DATA_0]] as %arg3: memref<1x144x64x128xf16, #NHWC, @CMX_NN>
    // CHECK-SAME:             [[SM_0]] as %arg4: memref<1x144x64x128xi1, #NHWC, @CMX_NN>
    // CHECK-SAME:      outputs([[DATA_1]] as %arg7: memref<1x144x64x128xf16, #NHWC, @CMX_NN>
    // CHECK-SAME:              [[SM_1]] as %arg8: memref<1x144x64x128xi1, #NHWC, @CMX_NN>)
    // CHECK-SAME:              -> (!VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>,
    // CHECK-SAME:                  !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)

    // CHECK:       [[NCE_0:%.*]] = VPUIP.GroupSparseBuffer([[NCE0_U]]#0, [[NCE0_U]]#1)

    // CHECK-NOT:   VPUIP.SubView [[PERMUTE]] [0, 0, 64, 0] [1, 144, 64, 128]
    // CHECK-NOT:   VPUIP.NCEClusterTiling
    // CHECK-NOT:   VPUIP.Copy

    // CHECK:       [[BUFF_3_DATA:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[BUFF_3_SM:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[BUFF_3:%.*]] = VPUIP.GroupSparseBuffer([[BUFF_3_DATA]], [[BUFF_3_SM]])
    // CHECK-SAME:                 -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>, sparsity_map=!VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>>

    // CHECK:       [[DATA_2:%.*]], [[SM_2:%.*]] = VPUIP.UngroupSparseBuffer([[COPY_0]])
    // CHECK:       [[DATA_3:%.*]], [[SM_3:%.*]] = VPUIP.UngroupSparseBuffer([[BUFF_3]])

    // CHECK:       [[NCE1_U:%.*]]:2 = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[DATA_2]] as %arg3: memref<1x144x64x128xf16, #NHWC, @CMX_NN>
    // CHECK-SAME:             [[SM_2]] as %arg4: memref<1x144x64x128xi1, #NHWC, @CMX_NN>
    // CHECK-SAME:      outputs([[DATA_3]] as %arg7: memref<1x144x64x128xf16, #NHWC, @CMX_NN>
    // CHECK-SAME:              [[SM_3]] as %arg8: memref<1x144x64x128xi1, #NHWC, @CMX_NN>)
    // CHECK-SAME:              -> (!VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>,
    // CHECK-SAME:                  !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)

    // CHECK:       [[NCE_1:%.*]] = VPUIP.GroupSparseBuffer([[NCE1_U]]#0, [[NCE1_U]]#1)

    // CHECK:       return [[NCE_0]], [[NCE_1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!IDataDDRType = memref<1x144x128x128xf16, #NHWC, @DDR>
!ISMDDRType = memref<1x144x128x128xi1, #NHWC, @DDR>
!ISparseDDRType = !VPUIP.SparseBuffer<data=!IDataDDRType, sparsity_map=!ISMDDRType>

!ISparseHalfDDRType = !VPUIP.SparseBuffer<
  data=memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>,
  sparsity_map=memref<1x144x64x128xi1, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>
>

!IDataHalfCMXType = memref<1x144x64x128xf16, #NHWC, @CMX_NN>
!ISMHalfCMXType = memref<1x144x64x128xi1, #NHWC, @CMX_NN>
!ISparseHalfCMXType = !VPUIP.SparseBuffer<
  data=!IDataHalfCMXType,
  sparsity_map=!ISMHalfCMXType
>

!ODistrDataType = !VPUIP.DistributedBuffer<
    1x144x64x128xf16, #NHWC, @CMX_NN, {
    mode = DUPLICATED,
    num_clusters = 4 : i64
}>
!ODistrSMType = !VPUIP.DistributedBuffer<
    1x144x64x128xi1, #NHWC, @CMX_NN, {
    mode = DUPLICATED,
    num_clusters = 4 : i64
}>
!ODistrSparseType = !VPUIP.SparseBuffer<data=!ODistrDataType, sparsity_map=!ODistrSMType>

// CHECK-LABEL: @NotOptimizeParallelClusterTilingCopiesWithSubviewHasDiffOffsetSparse
func.func @NotOptimizeParallelClusterTilingCopiesWithSubviewHasDiffOffsetSparse(
        %input: !ISparseDDRType,
        %weights: memref<32x144x1x1xf16, #NHWC, @CMX_NN>,
        %weights_table : memref<144x1x1x4xsi32, @CMX_NN>)
         -> (!ODistrSparseType, !ODistrSparseType) {

    %0 = memref.alloc() : !IDataDDRType
    %1 = memref.alloc() : !ISMDDRType
    %2 = VPUIP.GroupSparseBuffer(%0, %1) -> !ISparseDDRType

    %3 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC}
            inputs(%2 : !ISparseDDRType ) -> !ISparseDDRType

    %4 = VPUIP.SubView %3 [0, 0, 64, 0] [1, 144, 64, 128] : !ISparseDDRType to !ISparseHalfDDRType

    %5 = VPURT.AllocDistributed -> !ODistrDataType
    %6 = VPURT.AllocDistributed -> !ODistrSMType
    %7 = VPUIP.GroupSparseBuffer(%5, %6) -> !ODistrSparseType

    %8 = VPUIP.NCEClusterTiling
            inputs(%4 as %arg2: !ISparseHalfDDRType)
            outputs(%7 as %arg3: !ISparseHalfCMXType)
                -> !ODistrSparseType {
        %inner = VPUIP.Copy
                inputs(%arg2 : !ISparseHalfDDRType)
                outputs(%arg3 : !ISparseHalfCMXType)
                    -> !ISparseHalfCMXType
    }

    %9 = VPURT.AllocDistributed -> !ODistrDataType
    %10 = VPURT.AllocDistributed -> !ODistrSMType
    %11 = VPUIP.GroupSparseBuffer(%9, %10) -> !ODistrSparseType

    %in_data_0, %in_sm_0 = VPUIP.UngroupSparseBuffer(%8) {resultSegmentSizes = array<i32: 1, 1, 0>}
        -> !ODistrDataType, !ODistrSMType
    %out_data_0, %out_sm_0 = VPUIP.UngroupSparseBuffer(%11) {resultSegmentSizes = array<i32: 1, 1, 0>}
        -> !ODistrDataType, !ODistrSMType

    %12:2 = VPUIP.NCEClusterTiling
        inputs(%in_data_0 as %arg2: !IDataHalfCMXType,
                %in_sm_0 as %arg3: !ISMHalfCMXType,
                %weights as %arg4: memref<32x144x1x1xf16, #NHWC, @CMX_NN>,
                %weights_table as %arg5: memref<32x1x1x4xsi32, @CMX_NN>)
        outputs(%out_data_0 as %arg6: !IDataHalfCMXType,
              %out_sm_0 as %arg7: !ISMHalfCMXType
      ) -> (!ODistrDataType, !ODistrSMType) {
        %inner:2 = VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                minimumHardwareExecutionCost = 9240 : i64, task_type = #VPUIP.nce_task_type<CONV>}
          input(%arg2 : !IDataHalfCMXType)
          input_sparsity_map(%arg3 : !ISMHalfCMXType)
          weights(%arg4 : memref<32x144x1x1xf16, #NHWC, @CMX_NN>)
          weight_table(%arg5 : memref<32x1x1x4xsi32, @CMX_NN>)
          parent_input(%arg2 : !IDataHalfCMXType)
          parent_input_sparsity_map(%arg3 : !ISMHalfCMXType)
          parent_output(%arg6 : !IDataHalfCMXType)
          parent_output_sparsity_map(%arg7 : !ISMHalfCMXType)
          outputs(%arg6 : !IDataHalfCMXType)
          output_sparsity_map(%arg7 : !ISMHalfCMXType)
            -> !IDataHalfCMXType, !ISMHalfCMXType variants :  {
                DPUTask {cluster_id = 0 : i64, outEnd = [15, 5, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
        } PPE :  {
        }
    }
    %output_sparse_0 = VPUIP.GroupSparseBuffer(%12#0, %12#1) -> !ODistrSparseType

    %13 = VPUIP.SubView %3 [0, 1, 64, 0] [1, 144, 64, 128] : !ISparseDDRType to !ISparseHalfDDRType

    %14 = VPURT.AllocDistributed -> !ODistrDataType
    %15 = VPURT.AllocDistributed -> !ODistrSMType
    %16 = VPUIP.GroupSparseBuffer(%14, %15) -> !ODistrSparseType

    %17 = VPUIP.NCEClusterTiling
            inputs(%13 as %arg2: !ISparseHalfDDRType)
            outputs(%16 as %arg3: !ISparseHalfCMXType)
                -> !ODistrSparseType {
        %inner = VPUIP.Copy
                inputs(%arg2 : !ISparseHalfDDRType)
                outputs(%arg3 : !ISparseHalfCMXType)
                    -> !ISparseHalfCMXType
    }

    %18 = VPURT.AllocDistributed -> !ODistrDataType
    %19 = VPURT.AllocDistributed -> !ODistrSMType
    %20 = VPUIP.GroupSparseBuffer(%18, %19) -> !ODistrSparseType

    %in_data_1, %in_sm_1 = VPUIP.UngroupSparseBuffer(%17) {resultSegmentSizes = array<i32: 1, 1, 0>}
        -> !ODistrDataType, !ODistrSMType
    %out_data_1, %out_sm_1 = VPUIP.UngroupSparseBuffer(%20) {resultSegmentSizes = array<i32: 1, 1, 0>}
        -> !ODistrDataType, !ODistrSMType

    %21:2 = VPUIP.NCEClusterTiling
        inputs(%in_data_1 as %arg2: !IDataHalfCMXType,
                %in_sm_1 as %arg3: !ISMHalfCMXType,
                %weights as %arg4: memref<32x144x1x1xf16, #NHWC, @CMX_NN>,
                %weights_table as %arg5: memref<32x1x1x4xsi32, @CMX_NN>)
        outputs(%out_data_1 as %arg6: !IDataHalfCMXType,
              %out_sm_1 as %arg7: !ISMHalfCMXType
      ) -> (!ODistrDataType, !ODistrSMType) {
        %inner:2 = VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                minimumHardwareExecutionCost = 9240 : i64, task_type = #VPUIP.nce_task_type<CONV>}
          input(%arg2 : !IDataHalfCMXType)
          input_sparsity_map(%arg3 : !ISMHalfCMXType)
          weights(%arg4 : memref<32x144x1x1xf16, #NHWC, @CMX_NN>)
          weight_table(%arg5 : memref<32x1x1x4xsi32, @CMX_NN>)
          parent_input(%arg2 : !IDataHalfCMXType)
          parent_input_sparsity_map(%arg3 : !ISMHalfCMXType)
          parent_output(%arg6 : !IDataHalfCMXType)
          parent_output_sparsity_map(%arg7 : !ISMHalfCMXType)
          outputs(%arg6 : !IDataHalfCMXType)
          output_sparsity_map(%arg7 : !ISMHalfCMXType)
            -> !IDataHalfCMXType, !ISMHalfCMXType variants :  {
                DPUTask {cluster_id = 0 : i64, outEnd = [15, 5, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
        } PPE :  {
        }
    }
    %output_sparse_1 = VPUIP.GroupSparseBuffer(%21#0, %21#1) -> !ODistrSparseType

    return %output_sparse_0, %output_sparse_1 : !ODistrSparseType, !ODistrSparseType

    // CHECK:       [[BUFF_0_DATA:%.*]] = memref.alloc() : memref<1x144x128x128xf16, #NHWC, @DDR>
    // CHECK:       [[BUFF_0_SM:%.*]] = memref.alloc() : memref<1x144x128x128xi1, #NHWC, @DDR>
    // CHECK:       [[BUFF_0:%.*]] = VPUIP.GroupSparseBuffer([[BUFF_0_DATA]], [[BUFF_0_SM]])
    // CHECK-SAME:                 -> !VPUIP.SparseBuffer<data=memref<1x144x128x128xf16, #NHWC, @DDR>, sparsity_map=memref<1x144x128x128xi1, #NHWC, @DDR>>

    // CHECK:       [[PERMUTE:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:                  inputs([[BUFF_0]] : !VPUIP.SparseBuffer<data=memref<1x144x128x128xf16, #NHWC, @DDR>, sparsity_map=memref<1x144x128x128xi1, #NHWC, @DDR>>
    // CHECK-SAME:                  -> !VPUIP.SparseBuffer<data=memref<1x144x128x128xf16, #NHWC, @DDR>, sparsity_map=memref<1x144x128x128xi1, #NHWC, @DDR>>

    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView [[PERMUTE]] [0, 0, 64, 0] [1, 144, 64, 128] :
    // CHECK-SAME:                  !VPUIP.SparseBuffer<data=memref<1x144x128x128xf16, #NHWC, @DDR>, sparsity_map=memref<1x144x128x128xi1, #NHWC, @DDR>> to
    // CHECK-SAME:                  !VPUIP.SparseBuffer<data=memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>, sparsity_map=memref<1x144x64x128xi1, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>>

    // CHECK:       [[BUFF_1_DATA:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[BUFF_1_SM:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[BUFF_1:%.*]] = VPUIP.GroupSparseBuffer([[BUFF_1_DATA]], [[BUFF_1_SM]])
    // CHECK-SAME:                 -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>, sparsity_map=!VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>>

    // CHECK:       [[COPY_0:%.*]] = VPUIP.NCEClusterTiling inputs([[SUBVIEW_0]] as %arg3
    // CHECK-SAME:                 outputs([[BUFF_1]] as %arg4
    // CHECK-SAME:                 -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>, sparsity_map=!VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>>
    // CHECK:                       VPUIP.Copy inputs(%arg3
    // CHECK-SAME:                      outputs(%arg4

    // CHECK:       [[BUFF_2_DATA:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[BUFF_2_SM:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[BUFF_2:%.*]] = VPUIP.GroupSparseBuffer([[BUFF_2_DATA]], [[BUFF_2_SM]])
    // CHECK-SAME:                 -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>, sparsity_map=!VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>>

    // CHECK:       [[DATA_0:%.*]], [[SM_0:%.*]] = VPUIP.UngroupSparseBuffer([[COPY_0]])
    // CHECK:       [[DATA_1:%.*]], [[SM_1:%.*]] = VPUIP.UngroupSparseBuffer([[BUFF_2]])

    // CHECK:       [[NCE0_U:%.*]]:2 = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[DATA_0]] as %arg3: memref<1x144x64x128xf16, #NHWC, @CMX_NN>
    // CHECK-SAME:             [[SM_0]] as %arg4: memref<1x144x64x128xi1, #NHWC, @CMX_NN>
    // CHECK-SAME:      outputs([[DATA_1]] as %arg7: memref<1x144x64x128xf16, #NHWC, @CMX_NN>
    // CHECK-SAME:              [[SM_1]] as %arg8: memref<1x144x64x128xi1, #NHWC, @CMX_NN>)
    // CHECK-SAME:              -> (!VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>,
    // CHECK-SAME:                  !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)

    // CHECK:       [[NCE_0:%.*]] = VPUIP.GroupSparseBuffer([[NCE0_U]]#0, [[NCE0_U]]#1)

    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[PERMUTE]] [0, 1, 64, 0] [1, 144, 64, 128] :
    // CHECK-SAME:                  !VPUIP.SparseBuffer<data=memref<1x144x128x128xf16, #NHWC, @DDR>, sparsity_map=memref<1x144x128x128xi1, #NHWC, @DDR>> to
    // CHECK-SAME:                  !VPUIP.SparseBuffer<data=memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>, sparsity_map=memref<1x144x64x128xi1, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>>

    // CHECK:       [[BUFF_3_DATA:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[BUFF_3_SM:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[BUFF_3:%.*]] = VPUIP.GroupSparseBuffer([[BUFF_3_DATA]], [[BUFF_3_SM]])
    // CHECK-SAME:                 -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>, sparsity_map=!VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>>

    // CHECK:       [[COPY_1:%.*]] = VPUIP.NCEClusterTiling inputs([[SUBVIEW_1]] as %arg3
    // CHECK-SAME:                 outputs([[BUFF_3]] as %arg4
    // CHECK-SAME:                 -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>, sparsity_map=!VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>>
    // CHECK:                       VPUIP.Copy inputs(%arg3
    // CHECK-SAME:                      outputs(%arg4

    // CHECK:       [[BUFF_4_DATA:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[BUFF_4_SM:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[BUFF_4:%.*]] = VPUIP.GroupSparseBuffer([[BUFF_4_DATA]], [[BUFF_4_SM]])
    // CHECK-SAME:                 -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>, sparsity_map=!VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>>

    // CHECK:       [[DATA_2:%.*]], [[SM_2:%.*]] = VPUIP.UngroupSparseBuffer([[COPY_1]])
    // CHECK:       [[DATA_3:%.*]], [[SM_3:%.*]] = VPUIP.UngroupSparseBuffer([[BUFF_4]])

    // CHECK:       [[NCE1_U:%.*]]:2 = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[DATA_2]] as %arg3: memref<1x144x64x128xf16, #NHWC, @CMX_NN>
    // CHECK-SAME:             [[SM_2]] as %arg4: memref<1x144x64x128xi1, #NHWC, @CMX_NN>
    // CHECK-SAME:      outputs([[DATA_3]] as %arg7: memref<1x144x64x128xf16, #NHWC, @CMX_NN>
    // CHECK-SAME:              [[SM_3]] as %arg8: memref<1x144x64x128xi1, #NHWC, @CMX_NN>)
    // CHECK-SAME:              -> (!VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>,
    // CHECK-SAME:                  !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)

    // CHECK:       [[NCE_1:%.*]] = VPUIP.GroupSparseBuffer([[NCE1_U]]#0, [[NCE1_U]]#1)

    // CHECK:       return [[NCE_0]], [[NCE_1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!OutputDistributed = !VPUIP.DistributedBuffer<
    1x144x64x128xf16, #NHWC, @CMX_NN, {
    mode = DUPLICATED,
    num_clusters = 4 : i64
}>

func.func @OptimizeParallelSubViewWithParallelClusterTilingCopies(
        %input: memref<1x144x128x128xf16, #NHWC, @DDR>,
        %weights: memref<32x144x1x1xf16, #NHWC, @CMX_NN>,
        %weights_table : memref<144x1x1x4xsi32, @CMX_NN>)
         -> (!OutputDistributed, !OutputDistributed, !OutputDistributed) {

    %0 = memref.alloc() : memref<1x144x128x128xf16, #NHWC, @DDR>
    %1 = IERT.Convert
        inputs(%input : memref<1x144x128x128xf16, #NHWC, @DDR>)
        outputs(%0 : memref<1x144x128x128xf16, #NHWC, @DDR>)
        -> memref<1x144x128x128xf16, #NHWC, @DDR>

    %2 = VPUIP.SubView %1 [0, 0, 64, 0] [1, 144, 64, 128]
            : memref<1x144x128x128xf16, #NHWC, @DDR>
            to memref<1x144x64x128xf16, {order = affine_map<(d0, d1, d2, d3)
                -> (d0, d2, d3, d1)>, strides = [2359296, 1, 18432, 144]}, @DDR>
    %3 = VPURT.AllocDistributed -> !OutputDistributed
    %4 = VPUIP.NCEClusterTiling
            inputs(%2 as %arg2: memref<1x144x64x128xf16, #NHWC>)
            outputs(%3 as %arg3: memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
                -> !OutputDistributed {
        %inner = VPUIP.Copy
                inputs(%arg2 : memref<1x144x64x128xf16, #NHWC>)
                outputs(%arg3 : memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
                    -> memref<1x144x64x128xf16, #NHWC, @CMX_NN>
    }

    %100 = VPURT.AllocDistributed -> !OutputDistributed
    %101 = VPUIP.NCEClusterTiling
            inputs(%2 as %arg2: memref<1x144x64x128xf16, #NHWC>)
            outputs(%100 as %arg3: memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
                -> !OutputDistributed {
        %inner = VPUIP.Copy
                inputs(%arg2 : memref<1x144x64x128xf16, #NHWC>)
                outputs(%arg3 : memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
                    -> memref<1x144x64x128xf16, #NHWC, @CMX_NN>
    }


    %5 = VPURT.AllocDistributed -> !OutputDistributed
    %6 = VPUIP.NCEClusterTiling
            inputs(
                %4 as %arg2: memref<1x144x64x128xf16, #NHWC, @CMX_NN>,
                %weights as %arg3: memref<32x144x1x1xf16, #NHWC, @CMX_NN>,
                %weights_table as %arg4: memref<32x1x1x4xsi32, @CMX_NN>)
            outputs(
                %5 as %arg5: memref<1x32x64x128xf16, #NHWC, @CMX_NN>)
                    -> !OutputDistributed {
        %inner = VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                minimumHardwareExecutionCost = 9240 : i64, task_type = #VPUIP.nce_task_type<CONV>}
            input(
                %arg2 : memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
                weights(%arg3 : memref<32x144x1x1xf16, #NHWC, @CMX_NN>)
                weight_table(%arg4 : memref<32x1x1x4xsi32, @CMX_NN>)
                parent_input(%arg2 : memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
                parent_output(%arg5 : memref<1x32x64x128xf16, #NHWC, @CMX_NN>)
                outputs(%arg5 : memref<1x32x64x128xf16, #NHWC, @CMX_NN>)
                    -> memref<1x32x64x128xf16, #NHWC, @CMX_NN> variants :  {
                DPUTask {cluster_id = 0 : i64, outEnd = [15, 5, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
        } PPE :  {
        }
    }

	    %102 = VPURT.AllocDistributed -> !OutputDistributed
    %103 = VPUIP.NCEClusterTiling
            inputs(
                %101 as %arg2: memref<1x144x64x128xf16, #NHWC, @CMX_NN>,
                %weights as %arg3: memref<32x144x1x1xf16, #NHWC, @CMX_NN>,
                %weights_table as %arg4: memref<32x1x1x4xsi32, @CMX_NN>)
            outputs(
                %5 as %arg5: memref<1x32x64x128xf16, #NHWC, @CMX_NN>)
                    -> !OutputDistributed {
        %inner = VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                minimumHardwareExecutionCost = 9240 : i64, task_type = #VPUIP.nce_task_type<CONV>}
            input(
                %arg2 : memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
                weights(%arg3 : memref<32x144x1x1xf16, #NHWC, @CMX_NN>)
                weight_table(%arg4 : memref<32x1x1x4xsi32, @CMX_NN>)
                parent_input(%arg2 : memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
                parent_output(%arg5 : memref<1x32x64x128xf16, #NHWC, @CMX_NN>)
                outputs(%arg5 : memref<1x32x64x128xf16, #NHWC, @CMX_NN>)
                    -> memref<1x32x64x128xf16, #NHWC, @CMX_NN> variants :  {
                DPUTask {cluster_id = 0 : i64, outEnd = [15, 5, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
        } PPE :  {
        }
    }


    %7 = VPUIP.SubView %1 [0, 0, 64, 0] [1, 144, 64, 128]
            : memref<1x144x128x128xf16, #NHWC, @DDR>
            to memref<1x144x64x128xf16, {order = affine_map<(d0, d1, d2, d3)
                -> (d0, d2, d3, d1)>, strides = [2359296, 1, 18432, 144]}, @DDR>
    %8 = VPURT.AllocDistributed -> !OutputDistributed
    %9 = VPUIP.NCEClusterTiling
            inputs(%7 as %arg2: memref<1x144x64x128xf16, #NHWC>)
            outputs(%8 as %arg3: memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
                -> !OutputDistributed {
        %inner = VPUIP.Copy
                inputs(%arg2 : memref<1x144x64x128xf16, #NHWC>)
                outputs(%arg3 : memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
                    -> memref<1x144x64x128xf16, #NHWC, @CMX_NN>
    }
    %10 = VPURT.AllocDistributed -> !OutputDistributed
    %11 = VPUIP.NCEClusterTiling
            inputs(
                %9 as %arg2: memref<1x144x64x128xf16, #NHWC, @CMX_NN>,
                %weights as %arg3: memref<32x144x1x1xf16, #NHWC, @CMX_NN>,
                %weights_table as %arg4: memref<32x1x1x4xsi32, @CMX_NN>)
            outputs(
                %10 as %arg5: memref<1x32x64x128xf16, #NHWC, @CMX_NN>)
                    -> !OutputDistributed {
        %inner = VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                minimumHardwareExecutionCost = 9240 : i64, task_type = #VPUIP.nce_task_type<CONV>}
            input(
                %arg2 : memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
                weights(%arg3 : memref<32x144x1x1xf16, #NHWC, @CMX_NN>)
                weight_table(%arg4 : memref<32x1x1x4xsi32, @CMX_NN>)
                parent_input(%arg2 : memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
                parent_output(%arg5 : memref<1x32x64x128xf16, #NHWC, @CMX_NN>)
                outputs(%arg5 : memref<1x32x64x128xf16, #NHWC, @CMX_NN>)
                    -> memref<1x32x64x128xf16, #NHWC, @CMX_NN> variants :  {
                DPUTask {cluster_id = 0 : i64, outEnd = [15, 5, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
        } PPE :  {
        }
    }

    return %6, %11, %103 : !OutputDistributed, !OutputDistributed, !OutputDistributed

    // CHECK:       [[IN_BUFFER:%.*]] = memref.alloc() : memref<1x144x128x128xf16, #NHWC, @DDR>
    // CHECK:       [[CONVERT:%.*]] = IERT.Convert inputs(%arg0 : memref<1x144x128x128xf16, #NHWC, @DDR>) outputs([[IN_BUFFER]] : memref<1x144x128x128xf16, #NHWC, @DDR>) -> memref<1x144x128x128xf16, #NHWC, @DDR>

    // CHECK:       [[SUBVIEW_1:%.*]] = VPUIP.SubView [[CONVERT]] [0, 0, 64, 0] [1, 144, 64, 128]
    // CHECK-SAME:      memref<1x144x128x128xf16, #NHWC, @DDR> to
    // CHECK-SAME:      memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>
    // CHECK:       [[BUFFER_1:%.*]] = VPURT.AllocDistributed
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[COPY_1:%.*]] = VPUIP.NCEClusterTiling inputs([[SUBVIEW_1]] as %arg3: memref<1x144x64x128xf16, #NHWC>)
    // CHECK-SAME:      outputs([[BUFFER_1]] as %arg4: memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}> {
    // CHECK:       [[COPY_INNER_1:%.*]] = VPUIP.Copy inputs(%arg3 : memref<1x144x64x128xf16, #NHWC>)
    // CHECK-SAME:      outputs(%arg4 : memref<1x144x64x128xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:          -> memref<1x144x64x128xf16, #NHWC, @CMX_NN>

    // CHECK:       [[BUFFER_4:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[NCE_1:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[COPY_1]] as %arg3: memref<1x144x64x128xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:      %arg1 as %arg4: memref<32x144x1x1xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:      %arg2 as %arg5: memref<32x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:      outputs([[BUFFER_4]] as %arg6: memref<1x32x64x128xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>

    // CHECK-NOT:   VPUIP.SubView
    // CHECK-NOT:   VPUIP.Copy
    // CHECK:       [[NCE_2:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[COPY_1]] as %arg3: memref<1x144x64x128xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:      %arg1 as %arg4: memref<32x144x1x1xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:      %arg2 as %arg5: memref<32x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:      outputs([[BUFFER_4]] as %arg6: memref<1x32x64x128xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[BUFFER_6:%.*]] = VPURT.AllocDistributed
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK-NOT:   VPUIP.Copy
    // CHECK:       [[NCE_3:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[COPY_1]] as %arg3: memref<1x144x64x128xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:      %arg1 as %arg4: memref<32x144x1x1xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:      %arg2 as %arg5: memref<32x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:      outputs([[BUFFER_6]] as %arg6: memref<1x32x64x128xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>


    // CHECK:       return [[NCE_1]], [[NCE_3]], [[NCE_2]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributed = !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!CopyOutputDistributed = !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @DDR, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

// CHECK-LABEL: @OptimizeParallelClusterTilingCopiesWithInPlaceNCEEltwise
func.func @OptimizeParallelClusterTilingCopiesWithInPlaceNCEEltwise(%arg0:
memref<1x128x104x104xf32, #NHWC, @DDR>) -> (!CopyOutputDistributed, !OutputDistributed) {
    %0 = memref.alloc() : memref<1x128x104x104xf16, #NHWC, @DDR>
    %1 = IERT.Convert
        inputs(%arg0 : memref<1x128x104x104xf32, #NHWC, @DDR>)
        outputs(%0 : memref<1x128x104x104xf16, #NHWC, @DDR>)
        -> memref<1x128x104x104xf16, #NHWC, @DDR>
    %2 = VPUIP.SubView %1 [0, 0, 0, 0] [1, 128, 52, 104] :
                memref<1x128x104x104xf16, #NHWC, @DDR> to memref<1x128x52x104xf16, {order = #NHWC, strides = [1384448, 1, 13312, 128]}, @DDR>
    %3 = VPURT.AllocDistributed -> !OutputDistributed
    %4 = VPUIP.NCEClusterTiling
        inputs(%2 as %arg4: memref<1x128x52x104xf16, #NHWC>)
        outputs(%3 as %arg5: memref<1x128x52x104xf16, #NHWC, @CMX_NN>)
        -> !OutputDistributed {
        %inner = VPUIP.Copy
            inputs(%arg4 : memref<1x128x52x104xf16, #NHWC>)
            outputs(%arg5 : memref<1x128x52x104xf16, #NHWC, @CMX_NN>)
            -> memref<1x128x52x104xf16, #NHWC, @CMX_NN>
    }
    %5 = VPURT.AllocDistributed -> !CopyOutputDistributed
    %6 = VPUIP.NCEClusterTiling
        inputs(%4 as %arg4: memref<1x128x52x104xf16, #NHWC, @CMX_NN>)
        outputs(%5 as %arg5: memref<1x128x52x104xf16, #NHWC, @DDR>)
        -> !CopyOutputDistributed {
        %inner = VPUIP.Copy
            inputs(%arg4 : memref<1x128x52x104xf16, #NHWC, @CMX_NN>)
            outputs(%arg5 : memref<1x128x52x104xf16, #NHWC, @DDR>)
            -> memref<1x128x52x104xf16, #NHWC, @DDR>
    }
    %7 = VPUIP.SubView %1 [0, 0, 0, 0] [1, 128, 52, 104] :
        memref<1x128x104x104xf16, #NHWC, @DDR> to memref<1x128x52x104xf16, {order = #NHWC, strides = [1384448, 1, 13312, 128]}, @DDR>
    %8 = VPUIP.SubView %1 [0, 0, 52, 0] [1, 128, 52, 104] :
        memref<1x128x104x104xf16, #NHWC, @DDR> to memref<1x128x52x104xf16, {order = #NHWC, strides = [1384448, 1, 13312, 128]}, @DDR>
    %9 = VPURT.AllocDistributed -> !OutputDistributed
    %10 = VPUIP.NCEClusterTiling
        inputs(%7 as %arg4: memref<1x128x52x104xf16, #NHWC>)
        outputs(%9 as %arg5: memref<1x128x52x104xf16, #NHWC, @CMX_NN>)
        -> !OutputDistributed {
        %inner = VPUIP.Copy
            inputs(%arg4 : memref<1x128x52x104xf16, #NHWC>)
            outputs(%arg5 : memref<1x128x52x104xf16, #NHWC, @CMX_NN>)
            -> memref<1x128x52x104xf16, #NHWC, @CMX_NN>
    }
    %11 = VPURT.AllocDistributed -> !OutputDistributed
    %12 = VPUIP.NCEClusterTiling
        inputs(%8 as %arg4: memref<1x128x52x104xf16, #NHWC>)
        outputs(%11 as %arg5: memref<1x128x52x104xf16, #NHWC, @CMX_NN>)
        -> !OutputDistributed {
        %inner = VPUIP.Copy
            inputs(%arg4 : memref<1x128x52x104xf16, #NHWC>)
            outputs(%arg5 : memref<1x128x52x104xf16, #NHWC, @CMX_NN>)
            -> memref<1x128x52x104xf16, #NHWC, @CMX_NN>
    }
    %13 = VPUIP.NCEClusterTiling
        inputs(%10 as %arg4: memref<1x128x52x104xf16, #NHWC, @CMX_NN>, %12 as %arg5: memref<1x128x52x104xf16, #NHWC, @CMX_NN>)
        outputs(%9 as %arg6: memref<1x128x52x104xf16, #NHWC, @CMX_NN>)
        -> !OutputDistributed {
        %inner = VPUIP.NCEClusterTask {
                activation_window_channel_length = 0 : i64,
                is_inplace = true,
                minimumHardwareExecutionCost = 4294967400 : i64,
                task_type = #VPUIP.nce_task_type<ELTWISE>
            }
            input(%arg4 : memref<1x128x52x104xf16, #NHWC, @CMX_NN>)
            weights(%arg5 : memref<1x128x52x104xf16, #NHWC, @CMX_NN>)
            parent_input(%arg4 : memref<1x128x52x104xf16, #NHWC, @CMX_NN>)
            parent_output(%arg6 : memref<1x128x52x104xf16, #NHWC, @CMX_NN>)
            outputs(%arg6 : memref<1x128x52x104xf16, #NHWC, @CMX_NN>)
            -> memref<1x128x52x104xf16, #NHWC, @CMX_NN> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [103, 25, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [103, 51, 127], outStart = [0, 26, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        } PPE : {
            PPETask <NOOP> {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00]}
        }
    }
    return %6, %13: !CopyOutputDistributed, !OutputDistributed
    // CHECK:       [[MEM_REF:%.*]] = memref.alloc() : memref<1x128x104x104xf16, #NHWC, @DDR>
    // CHECK:       [[CONVERT:%.*]] = IERT.Convert inputs(%arg0 : memref<1x128x104x104xf32, #NHWC, @DDR>) outputs([[MEM_REF]] : memref<1x128x104x104xf16, #NHWC, @DDR>) -> memref<1x128x104x104xf16, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW:%.*]] = VPUIP.SubView [[CONVERT]] [0, 0, 0, 0] [1, 128, 52, 104] :
    // CHECK-SAME:        memref<1x128x104x104xf16, #NHWC, @DDR> to memref<1x128x52x104xf16, {order = #NHWC, strides = [1384448, 1, 13312, 128]}, @DDR>
    // CHECK:       [[DISTBUFF:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[NCE:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:        inputs([[SUBVIEW]] as %arg1: memref<1x128x52x104xf16, #NHWC>)
    // CHECK-SAME:        outputs([[DISTBUFF]] as %arg2: memref<1x128x52x104xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:       [[DISTBUFF_1:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @DDR, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[NCE_1:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:        inputs([[NCE]] as %arg1: memref<1x128x52x104xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:        outputs([[DISTBUFF_1]] as %arg2: memref<1x128x52x104xf16, #NHWC, @DDR>)
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @DDR, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:       [[SUBVIEW_0:%.*]] = VPUIP.SubView [[CONVERT]] [0, 0, 52, 0] [1, 128, 52, 104] :
    // CHECK-SAME:        memref<1x128x104x104xf16, #NHWC, @DDR> to memref<1x128x52x104xf16, {order = #NHWC, strides = [1384448, 1, 13312, 128]}, @DDR>
    // CHECK:       [[DISTBUFF_2:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[NCE_2:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:        inputs([[SUBVIEW_0]] as %arg1: memref<1x128x52x104xf16, #NHWC>)
    // CHECK-SAME:        outputs([[DISTBUFF_2]] as %arg2: memref<1x128x52x104xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:       [[NCE_3:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:        inputs([[NCE]] as %arg1: memref<1x128x52x104xf16, #NHWC, @CMX_NN>, [[NCE_2]] as %arg2: memref<1x128x52x104xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:        outputs([[DISTBUFF]] as %arg3: memref<1x128x52x104xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1x128x52x104xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:       return [[NCE_1]], [[NCE_3]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!bufDdr = memref<1x88x52x52xf16, {order = #NCHW, strides = [475904, 2704, 52, 1]}, @DDR>
!typeCst = memref<1x88x52x12xf16>
!bufDistributed = !VPUIP.DistributedBuffer<
  1x88x52x64xf16, #NCHW, @CMX_NN, {
  mode = "SEGMENTED",
  num_tiles = [1, 1, 2, 1],
  num_clusters = 2 : i64
}>

!bufDistributed0 = !VPUIP.DistributedBuffer<
  1x88x52x52xf16, {order = #NCHW, strides = [292864, 3328, 64, 1]}, @CMX_NN, {
  mode = "SEGMENTED",
  num_tiles = [1, 1, 2, 1],
  num_clusters = 2 : i64
}>
!bufCompact0 = memref<1x88x52x52xf16, {order = #NCHW, strides = [292864, 3328, 64, 1]}, @CMX_NN>

!bufDistributed1 = !VPUIP.DistributedBuffer<
  1x88x52x12xf16, {order = #NCHW, strides = [292864, 3328, 64, 1]}, @CMX_NN, {
  mode = "SEGMENTED",
  num_tiles = [1, 1, 2, 1],
  num_clusters = 2 : i64
}>
!bufCompact1 = memref<1x88x52x12xf16, {order = #NCHW, strides = [292864, 3328, 64, 1]}, @CMX_NN>

func.func @DoNotOptimizeCopiesIfUserIsConcat(%arg0: !bufDdr, %arg1: !bufDdr) -> (!bufDistributed, !bufDistributed) {

  %cst = const.Declare !typeCst = dense<0.000000e+00> : tensor<54912xf16>, [#const.Reshape<[1, 88, 52, 12]>, #const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>>]

  %buf0 = VPURT.AllocDistributed -> !bufDistributed
  %subview0_0 = VPUIP.SubView %buf0 [0, 0, 0, 0] [1, 88, 52, 52] : !bufDistributed to !bufDistributed0
  %copy0_0 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg2: !bufDdr) outputs(%subview0_0 as %arg3: !bufCompact0) -> !bufDistributed0 {
    %0 = VPUIP.Copy inputs(%arg2 : !bufDdr) outputs(%arg3 : !bufCompact0) -> !bufCompact0
  }
  %subview0_1 = VPUIP.SubView %buf0 [0, 0, 0, 52] [1, 88, 52, 12] : !bufDistributed to !bufDistributed1
  %copy0_1 = VPUIP.NCEClusterTiling inputs(%cst as %arg2: !typeCst) outputs(%subview0_1 as %arg3: !bufCompact1) -> !bufDistributed1 {
    %0 = VPUIP.Copy inputs(%arg2 : !typeCst) outputs(%arg3 : !bufCompact1) -> !bufCompact1
  }
  %concat0 = VPUIP.ConcatView inputs(%copy0_0, %copy0_1 : !bufDistributed0, !bufDistributed1) outputs(%buf0 : !bufDistributed) -> !bufDistributed

  %buf1 = VPURT.AllocDistributed -> !bufDistributed
  %subview1_0 = VPUIP.SubView %buf1 [0, 0, 0, 0] [1, 88, 52, 52] : !bufDistributed to !bufDistributed0
  %copy1_0 = VPUIP.NCEClusterTiling inputs(%arg1 as %arg2: !bufDdr) outputs(%subview1_0 as %arg3: !bufCompact0) -> !bufDistributed0 {
    %0 = VPUIP.Copy inputs(%arg2 : !bufDdr) outputs(%arg3 : !bufCompact0) -> !bufCompact0
  }
  %subview1_1 = VPUIP.SubView %buf1 [0, 0, 0, 52] [1, 88, 52, 12] : !bufDistributed to !bufDistributed1
  %copy1_1 = VPUIP.NCEClusterTiling inputs(%cst as %arg2: !typeCst) outputs(%subview1_1 as %arg3: !bufCompact1) -> !bufDistributed1 {
    %0 = VPUIP.Copy inputs(%arg2 : !typeCst) outputs(%arg3 : !bufCompact1) -> !bufCompact1
  }
  %concat1 = VPUIP.ConcatView inputs(%copy1_0, %copy1_1 : !bufDistributed0, !bufDistributed1) outputs(%buf1 : !bufDistributed) -> !bufDistributed

  return %concat0, %concat1: !bufDistributed, !bufDistributed

  // CHECK-LABEL: @DoNotOptimizeCopiesIfUserIsConcat

  // CHECK: [[CST:%.*]] = const.Declare
  // CHECK: [[BUF0:%.*]] = VPURT.AllocDistributed
  // CHECK: [[SUBVIEW0_0:%.*]] = VPUIP.SubView [[BUF0]]
  // CHECK: [[COPY0_0:%.*]] = VPUIP.NCEClusterTiling
  // CHECK-SAME: outputs([[SUBVIEW0_0]]
  // CHECK-NEXT: VPUIP.Copy

  // CHECK: [[SUBVIEW0_1:%.*]] = VPUIP.SubView [[BUF0]]
  // CHECK: [[COPY0_1:%.*]] = VPUIP.NCEClusterTiling inputs([[CST]]
  // CHECK-SAME: outputs([[SUBVIEW0_1]]
  // CHECK-NEXT: VPUIP.Copy

  // CHECK: [[CONCAT0:%.*]] = VPUIP.ConcatView inputs([[COPY0_0]], [[COPY0_1]]

  // CHECK: [[BUF1:%.*]] = VPURT.AllocDistributed
  // CHECK: [[SUBVIEW1_0:%.*]] = VPUIP.SubView [[BUF1]]
  // CHECK: [[COPY1_0:%.*]] = VPUIP.NCEClusterTiling
  // CHECK-SAME: outputs([[SUBVIEW1_0]]
  // CHECK-NEXT: VPUIP.Copy

  // CHECK: [[SUBVIEW1_1:%.*]] = VPUIP.SubView [[BUF1]]
  // CHECK: [[COPY1_1:%.*]] = VPUIP.NCEClusterTiling inputs([[CST]]
  // CHECK-SAME: outputs([[SUBVIEW1_1]]
  // CHECK-NEXT: VPUIP.Copy

  // CHECK: [[CONCAT1:%.*]] = VPUIP.ConcatView inputs([[COPY1_0]], [[COPY1_1]]

  // CHECK:  return [[CONCAT0]], [[CONCAT1]]
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DistrType = !VPUIP.DistributedBuffer<1x18x160x288xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
!ConstDistrType = !VPUIP.DistributedBuffer<1x18x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

module @VPU.SW {
  func.func private @builtin_PRelu(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "prelu_fp16.cpp", VPU.kernel_entry = "prelu_fp16"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @OptimizeConstCopyToSoftKernel() -> (!DistrType, !DistrType)  {
    %cst = const.Declare memref<1x18x1x1xf16, #NHWC> = dense<1.000000e+00> : tensor<18xf32>, [#const.ConvertElemType<f16>, #const.Reshape<[1, 18, 1, 1]>, #const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>]
    %0 = VPURT.AllocDistributed -> !DistrType
    %1 = VPURT.AllocDistributed -> !ConstDistrType
    %2 = VPURT.AllocDistributed -> !DistrType

    %3 = VPUIP.NCEClusterTiling inputs(%cst as %arg0: memref<1x18x1x1xf16, #NHWC>) outputs(%1 as %arg3: memref<1x18x1x1xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x18x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %inner = VPUIP.Copy inputs(%arg0 : memref<1x18x1x1xf16, #NHWC>) outputs(%arg3 :memref<1x18x1x1xf16, #NHWC, @CMX_NN>) -> memref<1x18x1x1xf16, #NHWC, @CMX_NN>
    }

    %4 = VPUIP.NCEClusterTiling inputs(%0 as %arg0: memref<1x18x160x288xf16, #NHWC, @CMX_NN>,
                                       %3 as %arg1: memref<1x18x1x1xf16, #NHWC, @CMX_NN>)
                                outputs(%2 as %arg2: memref<1x18x160x288xf16, #NHWC, @CMX_NN>)
                                      -> !DistrType {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_PRelu
                    inputs(%arg0 as %arg3: memref<1x18x160x288xf16, #NHWC, @CMX_NN>,
                           %arg1 as %arg4: memref<1x18x1x1xf16, #NHWC, @CMX_NN>)
                    outputs(%arg2 as %arg5: memref<1x18x160x288xf16, #NHWC, @CMX_NN>) on tile 0 -> memref<1x18x160x288xf16, #NHWC, @CMX_NN>{
        VPUIP.SW.Kernel.run(%arg3, %arg4, %arg5) : memref<1x18x160x288xf16, #NHWC, @CMX_NN>, memref<1x18x1x1xf16, #NHWC, @CMX_NN>, memref<1x18x160x288xf16, #NHWC, @CMX_NN>
      }
    }

    %5 = VPURT.AllocDistributed -> !DistrType
    %6 = VPURT.AllocDistributed -> !ConstDistrType
    %7 = VPURT.AllocDistributed -> !DistrType

    // the copy will be elimated because copy from the same const source
    %8 = VPUIP.NCEClusterTiling inputs(%cst as %arg0: memref<1x18x1x1xf16, #NHWC>) outputs(%6 as %arg3: memref<1x18x1x1xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x18x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %inner = VPUIP.Copy inputs(%arg0 : memref<1x18x1x1xf16, #NHWC>) outputs(%arg3 :memref<1x18x1x1xf16, #NHWC, @CMX_NN>) -> memref<1x18x1x1xf16, #NHWC, @CMX_NN>
    }

    %9 = VPUIP.NCEClusterTiling inputs(%5 as %arg0: memref<1x18x160x288xf16, #NHWC, @CMX_NN>,
                                       %8 as %arg1: memref<1x18x1x1xf16, #NHWC, @CMX_NN>)
                                outputs(%7 as %arg2: memref<1x18x160x288xf16, #NHWC, @CMX_NN>)
                                      -> !DistrType {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_PRelu
                    inputs(%arg0 as %arg3: memref<1x18x160x288xf16, #NHWC, @CMX_NN>,
                           %arg1 as %arg4: memref<1x18x1x1xf16, #NHWC, @CMX_NN>)
                    outputs(%arg2 as %arg5: memref<1x18x160x288xf16, #NHWC, @CMX_NN>) on tile 0 -> memref<1x18x160x288xf16, #NHWC, @CMX_NN>{
        VPUIP.SW.Kernel.run(%arg3, %arg4, %arg5) : memref<1x18x160x288xf16, #NHWC, @CMX_NN>, memref<1x18x1x1xf16, #NHWC, @CMX_NN>, memref<1x18x160x288xf16, #NHWC, @CMX_NN>
      }
    }

    return %4, %9 : !DistrType, !DistrType


    // CHECK-DAG:       [[CST:%.*]] = const.Declare memref<1x18x1x1xf16, #NHWC> = dense<1.000000e+00> : tensor<18xf32>, [#const.ConvertElemType<f16>, #const.Reshape<[1, 18, 1, 1]>, #const.Reorder<#NHWC>]
    // CHECK:           [[DISTR_BUFFER0:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x18x160x288xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[DISTR_BUFFER1:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x18x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:           [[DISTR_BUFFER2:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x18x160x288xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[COPY0:%.*]] = VPUIP.NCEClusterTiling inputs([[CST]] as %arg0: memref<1x18x1x1xf16, #NHWC>) outputs([[DISTR_BUFFER1]] as %arg1: memref<1x18x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK:           [[SWKERNEL0:%.*]] = VPUIP.NCEClusterTiling inputs([[DISTR_BUFFER0]] as %arg0: memref<1x18x160x288xf16, #NHWC, @CMX_NN>, [[COPY0]] as %arg1: memref<1x18x1x1xf16, #NHWC, @CMX_NN>)

    // CHECK:           [[DISTR_BUFFER3:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x18x160x288xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[DISTR_BUFFER4:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x18x160x288xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:           [[SWKERNEL1:%.*]] = VPUIP.NCEClusterTiling inputs([[DISTR_BUFFER3]] as %arg0: memref<1x18x160x288xf16, #NHWC, @CMX_NN>, [[COPY0]] as %arg1: memref<1x18x1x1xf16, #NHWC, @CMX_NN>)

    // CHECK:           return  [[SWKERNEL0]], [[SWKERNEL1]]
}
