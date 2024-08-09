//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --verify-diagnostics --init-compiler="vpu-arch=%arch%" --convert-eltwise-to-in-place --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0047491008160161037:146>

!qTypeCMX = memref<1x32x103x512x!qElemType, #NHWC, @CMX_NN>
!qTypeDDR = memref<1x32x103x512x!qElemType, #NHWC>

!DistributedType1 = !VPUIP.DistributedBuffer<
    1x32x103x512x!qElemType,
    #NHWC, @CMX_NN, {
        mode = "SEGMENTED",
        num_tiles = [1, 1, 2, 1],
        num_clusters = 2 : i64
}>

// CHECK-LABEL: InplaceEltwiseToEltwise
func.func @InplaceEltwiseToEltwise(%in1: !qTypeDDR, %in2: !qTypeDDR, %in3: !qTypeDDR) -> (!DistributedType1) {
    %eltwise_1_cmx_input_buf_1 = VPURT.AllocDistributed -> !DistributedType1
    %eltwise_1_cmx_input_1 = VPUIP.NCEClusterTiling inputs(%in1 as %arg4: !qTypeDDR) outputs(%eltwise_1_cmx_input_buf_1 as %arg5: !qTypeCMX) -> !DistributedType1 {
      %copy_to_cmx_1 = VPUIP.Copy inputs(%arg4 : !qTypeDDR) outputs(%arg5 : !qTypeCMX) -> !qTypeCMX
    }

    %eltwise_1_cmx_input_buf_2 = VPURT.AllocDistributed -> !DistributedType1
    %eltwise_1_cmx_input_2 = VPUIP.NCEClusterTiling inputs(%in2 as %arg4: !qTypeDDR) outputs(%eltwise_1_cmx_input_buf_2 as %arg5: !qTypeCMX) -> !DistributedType1 {
      %copy_to_cmx_2 = VPUIP.Copy inputs(%arg4 : !qTypeDDR) outputs(%arg5 : !qTypeCMX) -> !qTypeCMX
    }

    %eltwise_1_cmx_out_buf = VPURT.AllocDistributed -> !DistributedType1
    %eltwise_1 = VPUIP.NCEClusterTiling inputs(%eltwise_1_cmx_input_1 as %arg4: !qTypeCMX, %eltwise_1_cmx_input_2 as %arg5: !qTypeCMX) outputs(%eltwise_1_cmx_out_buf as %arg6: !qTypeCMX) -> !DistributedType1 {
      %eltwise_1_inner = VPUIP.NCEClusterTask {activation_window_channel_length = 0 : i64, is_inplace = true, minimumHardwareExecutionCost = 21125 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>} input(%arg4 : !qTypeCMX) weights(%arg5 : !qTypeCMX) parent_input(%arg4 : !qTypeCMX) parent_output(%arg6 : !qTypeCMX) outputs(%arg6 : !qTypeCMX) -> !qTypeCMX
        variants : {
          DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [511, 51, 31], outStart = [0, 0, 0], pad = #VPU.Padding<bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64>}
          DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [511, 102, 31], outStart = [0, 52, 0], pad = #VPU.Padding<bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64>}
        } PPE : {
          PPETask <NOOP> {clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64, in1_quant_mult = [19919], in2_quant_mult = [7511], lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [27275], quant_post_shift = 0 : i64, quant_shift = [29]}
        }
    }

    %eltwise_2_cmx_input_buf_1 = VPURT.AllocDistributed -> !DistributedType1
    %eltwise_2_cmx_input_1 = VPUIP.NCEClusterTiling inputs(%in3 as %arg4: !qTypeDDR) outputs(%eltwise_2_cmx_input_buf_1 as %arg5: !qTypeCMX) -> !DistributedType1 {
      %copy_to_cmx_2 = VPUIP.Copy inputs(%arg4 : !qTypeDDR) outputs(%arg5 : !qTypeCMX) -> !qTypeCMX
    }

    %eltwise_2_cmx_out_buf = VPURT.AllocDistributed -> !DistributedType1
    %eltwise_2 = VPUIP.NCEClusterTiling inputs(%eltwise_1 as %arg4: !qTypeCMX, %eltwise_2_cmx_input_1 as %arg5: !qTypeCMX) outputs(%eltwise_2_cmx_out_buf as %arg6: !qTypeCMX) -> !DistributedType1 {
      %eltwise_2_inner = VPUIP.NCEClusterTask {activation_window_channel_length = 0 : i64, is_inplace = true, minimumHardwareExecutionCost = 21125 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>} input(%arg4 : !qTypeCMX) weights(%arg5 : !qTypeCMX) parent_input(%arg4 : !qTypeCMX) parent_output(%arg6 : !qTypeCMX) outputs(%arg6 : !qTypeCMX) -> !qTypeCMX
        variants : {
          DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [511, 51, 31], outStart = [0, 0, 0], pad = #VPU.Padding<bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64>}
          DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [511, 102, 31], outStart = [0, 52, 0], pad = #VPU.Padding<bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64>}
        } PPE : {
          PPETask <NOOP> {clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64, in1_quant_mult = [19919], in2_quant_mult = [7511], lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [27275], quant_post_shift = 0 : i64, quant_shift = [29]}
       }
    }

    return %eltwise_2 : !DistributedType1

  // CHECK: [[ELTWISE_1_INPUT_BUF_1:%.+]] = VPURT.AllocDistributed
  // CHECK: [[ELTWISE_1_INPUT_1:%.+]] = VPUIP.NCEClusterTiling
  // CHECK: [[ELTWISE_1_INPUT_BUF_2:%.+]] = VPURT.AllocDistributed
  // CHECK: [[ELTWISE_1_INPUT_2:%.+]] = VPUIP.NCEClusterTiling
  // CHECK: [[ELTWISE_1:%.+]] = VPUIP.NCEClusterTiling inputs(
  // CHECK-SAME:  [[ELTWISE_1_INPUT_1]] as %arg3: memref<1x32x103x512x!qElemType, #NHWC, @CMX_NN>,
  // CHECK-SAME:  [[ELTWISE_1_INPUT_2]] as %arg4: memref<1x32x103x512x!qElemType, #NHWC, @CMX_NN>)
  // CHECK-SAME:  outputs([[ELTWISE_1_INPUT_BUF_1]] as %arg5: memref<1x32x103x512x!qElemType, #NHWC, @CMX_NN>)

  // CHECK: [[ELTWISE_2_INPUT_BUF_1:%.+]] = VPURT.AllocDistributed
  // CHECK: [[ELTWISE_2_INPUT_1:%.+]] = VPUIP.NCEClusterTiling
  // CHECK: [[ELTWISE_2:%.+]] = VPUIP.NCEClusterTiling inputs(
  // CHECK-SAME:  [[ELTWISE_1]] as %arg3: memref<1x32x103x512x!qElemType, #NHWC, @CMX_NN>,
  // CHECK-SAME:  [[ELTWISE_2_INPUT_1]] as %arg4: memref<1x32x103x512x!qElemType, #NHWC, @CMX_NN>)
  // CHECK-SAME:  outputs([[ELTWISE_1_INPUT_BUF_1]] as %arg5: memref<1x32x103x512x!qElemType, #NHWC, @CMX_NN>)

  // CHECK: return [[ELTWISE_2]] : !VPUIP.DistributedBuffer<1x32x103x512x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0047491008160161037:146>

!qTypeCMX = memref<1x32x103x512x!qElemType, #NHWC, @CMX_NN>
!qTypeDDR = memref<1x32x103x512x!qElemType, #NHWC>
!qTypeConvCMX = memref<1x32x206x256x!qElemType, #NHWC, @CMX_NN>
!qTypeConvDDR = memref<1x32x206x256x!qElemType, #NHWC>

!DistributedType1 = !VPUIP.DistributedBuffer<
    1x32x103x512x!qElemType,
    #NHWC, @CMX_NN, {
        mode = "SEGMENTED",
        num_tiles = [1, 1, 2, 1],
        num_clusters = 2 : i64
}>

!DistributedType2 = !VPUIP.DistributedBuffer<
    1x32x206x256x!qElemType,
    #NHWC, @CMX_NN, {
        mode = "SEGMENTED",
        num_tiles = [1, 1, 2, 1],
        num_clusters = 2 : i64
}>

// CHECK-LABEL: InPlaceEltwiseWithSiblingsOnBothInputs
func.func @InPlaceEltwiseWithSiblingsOnBothInputs(
    %weights0 : memref<32x32x1x1xf16, #NHWC, @CMX_NN>, %weightsTable0 : memref<32x1x1x4xsi32, @CMX_NN>)
      -> (!DistributedType1, !DistributedType2, !DistributedType1) {
    %in1 = memref.alloc() : !qTypeDDR
    %in2 = memref.alloc() : !qTypeDDR
    %eltwiseIn1CMXBuff = VPURT.AllocDistributed -> !DistributedType1
    %eltwiseIn1CMX = VPUIP.NCEClusterTiling inputs(%in1 as %arg4: !qTypeDDR) outputs(%eltwiseIn1CMXBuff as %arg5: !qTypeCMX) -> !DistributedType1 {
      %0 = VPUIP.Copy inputs(%arg4 : !qTypeDDR) outputs(%arg5 : !qTypeCMX) -> !qTypeCMX
    }

    %eltwiseIn2CMXBuff = VPURT.AllocDistributed -> !DistributedType1
    %eltwiseIn2CMX = VPUIP.NCEClusterTiling inputs(%in2 as %arg4: !qTypeDDR) outputs(%eltwiseIn2CMXBuff as %arg5: !qTypeCMX) -> !DistributedType1 {
      %0 = VPUIP.Copy inputs(%arg4 : !qTypeDDR) outputs(%arg5 : !qTypeCMX) -> !qTypeCMX
    }

    %eltwiseOutCMXBuff = VPURT.AllocDistributed -> !DistributedType1
    %eltwise = VPUIP.NCEClusterTiling
      inputs(%eltwiseIn1CMX as %arg4: !qTypeCMX, %eltwiseIn2CMX as %arg5: !qTypeCMX)
      outputs(%eltwiseOutCMXBuff as %arg6: !qTypeCMX)
        -> !DistributedType1 {
        %0 = VPUIP.NCEClusterTask {
          activation_window_channel_length = 0 :i64, is_inplace = true,
          minimumHardwareExecutionCost = 21125 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
            input(%arg4 : !qTypeCMX)
            weights(%arg5 : !qTypeCMX)
            parent_input(%arg4 : !qTypeCMX)
            parent_output(%arg6 : !qTypeCMX)
            outputs(%arg6 : !qTypeCMX) -> !qTypeCMX
        variants : {
          DPUTask {
            cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>,
            outEnd = [511, 51, 31], outStart = [0, 0, 0],
            pad = #VPU.Padding<bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64>}
          DPUTask {
            cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>,
            outEnd = [511, 51, 31], outStart = [0, 0, 0],
            pad = #VPU.Padding<bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64>}
        } PPE : {
          PPETask <NOOP> {
            clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64,
            in1_quant_mult = [19919], in2_quant_mult = [7511], lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
            quant_mult = [27275], quant_post_shift = 0 : i64, quant_shift = [29]}
        }
    }

    %conv0InCMXBuff = VPURT.AllocDistributed -> !DistributedType1
    %convInDDR = VPUIP.ShapeCast {shape = [1, 32, 206, 256]} inputs(%in1 : !qTypeDDR) -> !qTypeConvDDR
    %conv0InCMX = VPUIP.NCEClusterTiling inputs(%convInDDR as %arg4: !qTypeConvDDR) outputs(%conv0InCMXBuff as %arg5: !qTypeConvCMX)
      -> !DistributedType2 {
      %0 = VPUIP.Copy inputs(%arg4 : !qTypeConvDDR) outputs(%arg5 : !qTypeConvCMX) -> !qTypeConvCMX
    }

    %convOutBuff0 = VPURT.AllocDistributed -> !DistributedType2
    %conv0 = VPUIP.NCEClusterTiling
      inputs(%conv0InCMX as %arg0: !qTypeConvCMX,
              %weights0 as %arg1: memref<32x32x1x1xf16, #NHWC, @CMX_NN>,
              %weightsTable0 as %arg2: memref<32x1x1x4xsi32, @CMX_NN>)
      outputs(%convOutBuff0 as %arg3: !qTypeConvCMX)
      -> !DistributedType2 {
      %0 = VPUIP.NCEClusterTask {
          kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>
      }
      input(%arg0 : !qTypeConvCMX)
      weights(%arg1 : memref<32x32x1x1xf16, #NHWC, @CMX_NN>)
      weight_table(%arg2 : memref<32x1x1x4xsi32, @CMX_NN>)
      parent_input(%arg0 : !qTypeConvCMX)
      parent_output(%arg3 : !qTypeConvCMX)
      outputs(%arg3 : !qTypeConvCMX)
          -> !qTypeConvCMX variants : {
        DPUTask {
          cluster_id = 0 : i64,
          inEnd = [255, 102, 31], inStart = [0, 0, 0],
          mpe_mode = #VPU.mpe_mode<CUBOID_8x16>,
          outEnd = [255, 102, 31], outStart = [0, 0, 0],
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
        }
        DPUTask {
          cluster_id = 1 : i64,
          inEnd = [255, 102, 31], inStart = [0, 102, 0],
          mpe_mode = #VPU.mpe_mode<CUBOID_8x16>,
          outEnd = [255, 102, 31], outStart = [0, 0, 0],
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
        }
      } PPE : {
        PPETask <NOOP> {
          clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64,
          in1_quant_mult = [19919], in2_quant_mult = [7511], lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
          quant_mult = [27275], quant_post_shift = 0 : i64, quant_shift = [29]
        }
      }
    }


    %convOutBuff1 = VPURT.AllocDistributed -> !DistributedType1
    %conv1 = VPUIP.NCEClusterTiling
      inputs(%eltwiseIn2CMX as %arg0: !qTypeCMX,
              %weights0 as %arg1: memref<32x32x1x1xf16, #NHWC, @CMX_NN>,
              %weightsTable0 as %arg2: memref<32x1x1x4xsi32, @CMX_NN>)
      outputs(%convOutBuff0 as %arg3: !qTypeCMX)
      -> !DistributedType1 {
      %0 = VPUIP.NCEClusterTask {
          kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>
      }
      input(%arg0 : !qTypeCMX)
      weights(%arg1 : memref<32x32x1x1xf16, #NHWC, @CMX_NN>)
      weight_table(%arg2 : memref<32x1x1x4xsi32, @CMX_NN>)
      parent_input(%arg0 : !qTypeCMX)
      parent_output(%arg3 : !qTypeCMX)
      outputs(%arg3 : !qTypeCMX)
          -> !qTypeCMX variants : {
        DPUTask {
          cluster_id = 0 : i64,
          inEnd = [511, 51, 31], inStart = [0, 0, 0],
          mpe_mode = #VPU.mpe_mode<CUBOID_8x16>,
          outEnd = [511, 51, 31], outStart = [0, 0, 0],
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
        }
        DPUTask {
          cluster_id = 1 : i64,
          inEnd = [511, 51, 31], inStart = [0, 52, 0],
          mpe_mode = #VPU.mpe_mode<CUBOID_8x16>,
          outEnd = [511, 51, 31], outStart = [0, 0, 0],
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
        }
      } PPE : {
        PPETask <NOOP> {
          clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64,
          in1_quant_mult = [19919], in2_quant_mult = [7511], lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
          quant_mult = [27275], quant_post_shift = 0 : i64, quant_shift = [29]
        }
      }
    }

    return %eltwise, %conv0, %conv1 : !DistributedType1, !DistributedType2, !DistributedType1

  // CHECK: [[DDR_BUF_0:%.+]] = memref.alloc
  // CHECK: [[DDR_BUF_1:%.+]] = memref.alloc

  // CHECK: [[ELTWISE_1_INPUT_BUF_1:%.+]] = VPURT.AllocDistributed
  // CHECK: [[ELTWISE_1_INPUT_1:%.+]] = VPUIP.NCEClusterTiling
  // CHECK-SAME: inputs([[DDR_BUF_0]] as {{[^:]+}}: memref<1x32x103x512x!qElemType, #NHWC>)
  // CHECK-SAME: outputs([[ELTWISE_1_INPUT_BUF_1]] as {{[^:]+}}: memref<1x32x103x512x!qElemType, #NHWC, @CMX_NN>)
  // CHECK-NEXT:  VPUIP.Copy

  // CHECK: [[ELTWISE_1_INPUT_BUF_2:%.+]] = VPURT.AllocDistributed
  // CHECK: [[ELTWISE_1_INPUT_2:%.+]] = VPUIP.NCEClusterTiling
  // CHECK-SAME: inputs([[DDR_BUF_1]] as {{[^:]+}}: memref<1x32x103x512x!qElemType, #NHWC>)
  // CHECK-SAME: outputs([[ELTWISE_1_INPUT_BUF_2]] as {{[^:]+}}: memref<1x32x103x512x!qElemType, #NHWC, @CMX_NN>)
  // CHECK-NEXT:  VPUIP.Copy

  // CHECK: [[ELTWISE:%.+]] = VPUIP.NCEClusterTiling
  // CHECK-SAME:  inputs([[ELTWISE_1_INPUT_1]] as {{[^:]+}}: memref<1x32x103x512x!qElemType, #NHWC, @CMX_NN>,
  // CHECK-SAME:         [[ELTWISE_1_INPUT_2]] as {{[^:]+}}: memref<1x32x103x512x!qElemType, #NHWC, @CMX_NN>)
  // CHECK-SAME:  outputs([[ELTWISE_1_INPUT_BUF_2]] as {{[^:]+}}: memref<1x32x103x512x!qElemType, #NHWC, @CMX_NN>)
  // CHECK-NEXT:    VPUIP.NCEClusterTask
  // CHECK-SAME:      is_inplace = true

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0047491008160161037:146>

!qTypeCMX = memref<1x32x103x512x!qElemType, #NHWC, @CMX_NN>
!qTypeDDR = memref<1x32x103x512x!qElemType, #NHWC>
!outCMX = memref<1x32x103x512xf16, #NHWC, @CMX_NN>
!outDDR = memref<1x32x103x512xf16, #NHWC>

!DistributedType1 = !VPUIP.DistributedBuffer<
    1x32x103x512x!qElemType,
    #NHWC, @CMX_NN, {
        mode = "SEGMENTED",
        num_tiles = [1, 1, 2, 1],
        num_clusters = 2 : i64
}>

!DistributedType2 = !VPUIP.DistributedBuffer<
    1x32x103x512xf16,
    #NHWC, @CMX_NN, {
        mode = "SEGMENTED",
        num_tiles = [1, 1, 2, 1],
        num_clusters = 2 : i64
}>

// expected-error@+1 {{Failed to convert Eltwise to in-place Eltwise}}
func.func @IllegalInPlaceEltwise(
    %weights0 : memref<32x32x1x1xf16, #NHWC, @CMX_NN>, %weightsTable0 : memref<32x1x1x4xsi32, @CMX_NN>)
      -> (!DistributedType2) {
    %in1 = memref.alloc() : !qTypeDDR
    %in2 = memref.alloc() : !qTypeDDR
    %eltwiseIn1CMXBuff = VPURT.AllocDistributed -> !DistributedType1
    %eltwiseIn1CMX = VPUIP.NCEClusterTiling inputs(%in1 as %arg4: !qTypeDDR) outputs(%eltwiseIn1CMXBuff as %arg5: !qTypeCMX)
      -> !DistributedType1 {
      %0 = VPUIP.Copy inputs(%arg4 : !qTypeDDR) outputs(%arg5 : !qTypeCMX) -> !qTypeCMX
    }

    %eltwiseIn2CMXBuff = VPURT.AllocDistributed -> !DistributedType1
    %eltwiseIn2CMX = VPUIP.NCEClusterTiling inputs(%in2 as %arg4: !qTypeDDR) outputs(%eltwiseIn2CMXBuff as %arg5: !qTypeCMX)
      -> !DistributedType1 {
      %0 = VPUIP.Copy inputs(%arg4 : !qTypeDDR) outputs(%arg5 : !qTypeCMX) -> !qTypeCMX
    }

    %eltwiseOutCMXBuff = VPURT.AllocDistributed -> !DistributedType2
    %eltwise = VPUIP.NCEClusterTiling
      inputs(%eltwiseIn1CMX as %arg4: !qTypeCMX, %eltwiseIn2CMX as %arg5: !qTypeCMX)
      outputs(%eltwiseOutCMXBuff as %arg6: !outCMX)
        -> !DistributedType2 {
        %0 = VPUIP.NCEClusterTask {
          activation_window_channel_length = 0 :i64, is_inplace = true,
          minimumHardwareExecutionCost = 21125 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
            input(%arg4 : !qTypeCMX)
            weights(%arg5 : !qTypeCMX)
            parent_input(%arg4 : !qTypeCMX)
            parent_output(%arg6 : !outCMX)
            outputs(%arg6 : !outCMX) -> !outCMX
        variants : {
          DPUTask {
            cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>,
            outEnd = [511, 51, 31], outStart = [0, 0, 0],
            pad = #VPU.Padding<bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64>}
          DPUTask {
            cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>,
            outEnd = [511, 51, 31], outStart = [0, 0, 0],
            pad = #VPU.Padding<bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64>}
        } PPE : {
          PPETask <NOOP> {
            clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64,
            in1_quant_mult = [19919], in2_quant_mult = [7511], lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
            quant_mult = [27275], quant_post_shift = 0 : i64, quant_shift = [29]}
        }
    }

    return %eltwise : !DistributedType2

}
