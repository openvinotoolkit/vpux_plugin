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
      %eltwise_1_inner = VPUIP.NCEClusterTask {is_inplace = true, minimumHardwareExecutionCost = 21125 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>} input(%arg4 : !qTypeCMX) weights(%arg5 : !qTypeCMX) parent_input(%arg4 : !qTypeCMX) parent_output(%arg6 : !qTypeCMX) outputs(%arg6 : !qTypeCMX) -> !qTypeCMX
        variants : {
          DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [511, 51, 31], outStart = [0, 0, 0], pad = #VPU.Padding<bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64>}
          DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [511, 102, 31], outStart = [0, 52, 0], pad = #VPU.Padding<bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64>}
        } PPE : {
          PPETask {ppe = #VPU.PPEStub<>}
        }
    }

    %eltwise_2_cmx_input_buf_1 = VPURT.AllocDistributed -> !DistributedType1
    %eltwise_2_cmx_input_1 = VPUIP.NCEClusterTiling inputs(%in3 as %arg4: !qTypeDDR) outputs(%eltwise_2_cmx_input_buf_1 as %arg5: !qTypeCMX) -> !DistributedType1 {
      %copy_to_cmx_2 = VPUIP.Copy inputs(%arg4 : !qTypeDDR) outputs(%arg5 : !qTypeCMX) -> !qTypeCMX
    }

    %eltwise_2_cmx_out_buf = VPURT.AllocDistributed -> !DistributedType1
    %eltwise_2 = VPUIP.NCEClusterTiling inputs(%eltwise_1 as %arg4: !qTypeCMX, %eltwise_2_cmx_input_1 as %arg5: !qTypeCMX) outputs(%eltwise_2_cmx_out_buf as %arg6: !qTypeCMX) -> !DistributedType1 {
      %eltwise_2_inner = VPUIP.NCEClusterTask {is_inplace = true, minimumHardwareExecutionCost = 21125 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>} input(%arg4 : !qTypeCMX) weights(%arg5 : !qTypeCMX) parent_input(%arg4 : !qTypeCMX) parent_output(%arg6 : !qTypeCMX) outputs(%arg6 : !qTypeCMX) -> !qTypeCMX
        variants : {
          DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [511, 51, 31], outStart = [0, 0, 0], pad = #VPU.Padding<bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64>}
          DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [511, 102, 31], outStart = [0, 52, 0], pad = #VPU.Padding<bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64>}
        } PPE : {
          PPETask {ppe = #VPU.PPEStub<>}
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
          is_inplace = true,
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
          PPETask {
            ppe = #VPU.PPEStub<>
            }
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
        PPETask {
          ppe = #VPU.PPEStub<>
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
        PPETask {
          ppe = #VPU.PPEStub<>
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

!qTypeCMX1 = memref<1x1x207360x1xf16, {order = #NHWC, strides = [409600, 1, 1, 1]}, @CMX_NN>
!qTypeCMX2 = memref<1x1x202240x1xf16, {order = #NHWC, strides = [409600, 1, 1, 1]}, @CMX_NN>
!qTypeCMX3 = memref<1x16x160x160xf16, #NHWC, @CMX_NN>

!DistributedType1 = !VPUIP.DistributedBuffer<
    1x1x409600x1xf16,
    #NHWC, @CMX_NN, {
        mode = "SEGMENTED",
        num_tiles = [1, 1, 3, 1],
        num_clusters = 3 : i64,
        uniform_distributed_segments,
        compute_shapes = [[1, 1, 138240, 1], [1, 1, 135680, 1], [1, 1, 135680, 1]],
        compute_offsets = [[0, 0, 0, 0], [0, 0, 138240, 0], [0, 0, 273920, 0]],
        memory_shapes = [[1, 1, 138240, 1], [1, 1, 135680, 1], [1, 1, 135680, 1]],
        memory_offsets = [[0, 0, 0, 0], [0, 0, 138240, 0], [0, 0, 273920, 0]]
}>

!DistributedType2 = !VPUIP.DistributedBuffer<
    1x1x202240x1xf16,
    {order = #NHWC, strides = [409600, 1, 1, 1]}, @CMX_NN, {
        mode = "SEGMENTED",
        num_tiles = [1, 1, 3, 1],
        num_clusters = 3 : i64,
        uniform_distributed_segments,
        compute_shapes = [[1, 1, 69120, 1], [1, 1, 66560, 1], [1, 1, 66560, 1]],
        compute_offsets = [[0, 0, 0, 0], [0, 0, 69120, 0], [0, 0, 135680, 0]],
        memory_shapes = [[1, 1, 69120, 1], [1, 1, 66560, 1], [1, 1, 66560, 1]],
        memory_offsets = [[0, 0, 0, 0], [0, 0, 69120, 0], [0, 0, 135680, 0]]
}>

!DistributedType3 = !VPUIP.DistributedBuffer<
    1x1x207360x1xf16,
    {order = #NHWC, strides = [409600, 1, 1, 1]}, @CMX_NN, {
        mode = "SEGMENTED",
        num_tiles = [1, 1, 3, 1],
        num_clusters = 3 : i64,
        uniform_distributed_segments,
        compute_shapes = [[1, 1, 69120, 1], [1, 1, 69120, 1], [1, 1, 69120, 1]],
        compute_offsets = [[0, 0, 0, 0], [0, 0, 69120, 0], [0, 0, 138240, 0]],
        memory_shapes = [[1, 1, 69120, 1], [1, 1, 69120, 1], [1, 1, 69120, 1]],
        memory_offsets = [[0, 0, 0, 0], [0, 0, 69120, 0], [0, 0, 138240, 0]]
}>

!DistributedType4 = !VPUIP.DistributedBuffer<
    1x16x160x160xf16,
    #NHWC, @CMX_NN, {
        mode = "SEGMENTED",
        num_tiles = [1, 1, 3, 1],
        num_clusters = 3 : i64,
        uniform_distributed_segments,
        compute_shapes = [[1, 16, 54, 160], [1, 16, 53, 160], [1, 16, 53, 160]],
        compute_offsets = [[0, 0, 0, 0], [0, 0, 54, 0], [0, 0, 107, 0]],
        memory_shapes = [[1, 16, 54, 160], [1, 16, 53, 160], [1, 16, 53, 160]],
        memory_offsets = [[0, 0, 0, 0], [0, 0, 54, 0], [0, 0, 107, 0]]
}>

!DistributedType5 = !VPUIP.DistributedBuffer<
    1x16x160x160xf16,
    #NHWC, @CMX_NN, {
        mode = "OVERLAPPED",
        num_tiles = [1, 1, 3, 1],
        num_clusters = 3 : i64,
        uniform_distributed_segments,
        compute_shapes = [[1, 16, 54, 160], [1, 16, 53, 160], [1, 16, 53, 160]],
        compute_offsets = [[0, 0, 0, 0], [0, 0, 54, 0], [0, 0, 107, 0]],
        memory_shapes = [[1, 16, 54, 160], [1, 16, 53, 160], [1, 16, 53, 160]],
        memory_offsets = [[0, 0, 0, 0], [0, 0, 54, 0], [0, 0, 107, 0]]
}>

// CHECK-LABEL: CreateShapeCastOpBeforeDistributedCastOp
func.func @CreateShapeCastOpBeforeDistributedCastOp (%in1 : !DistributedType3, %in2 : !DistributedType2, %in3 : !qTypeCMX3) -> (!DistributedType5) {
    %eltwise_1_cmx_buf = VPURT.AllocDistributed -> !DistributedType1
    %eltwise_1_concat = VPUIP.ConcatView
      inputs(%in1, %in2 : !DistributedType3, !DistributedType2)
      outputs(%eltwise_1_cmx_buf : !DistributedType1) -> !DistributedType1
    %eltwise_1_shape_cast = VPUIP.ShapeCast {
      explicit_output_offsets = [[0, 0, 0, 0], [0, 0, 54, 0], [0, 0, 107, 0]],
      explicit_output_shapes = [[1, 16, 54, 160], [1, 16, 53, 160], [1, 16, 53, 160]],
      shape = [1, 16, 160, 160]}
      inputs(%eltwise_1_concat : !DistributedType1) -> !DistributedType4
    %eltwise_1_distributed_cast = VPUIP.DistributedCast inputs(%eltwise_1_shape_cast : !DistributedType4) -> !DistributedType5
    %eltwise_2_cmx_buf = VPURT.AllocDistributed -> !DistributedType5
    %eltwise_2 = VPUIP.NCEClusterTiling
      inputs(
        %in3 as %arg2: !qTypeCMX3,
        %eltwise_1_distributed_cast as %arg3: !qTypeCMX3)
      outputs(%eltwise_2_cmx_buf as %arg4: !qTypeCMX3) -> !DistributedType5 {
        %eltwise_2_inner = VPUIP.NCEClusterTask {
          is_inplace = true, minimumHardwareExecutionCost = 10507 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
          input(%arg2 : !qTypeCMX3)
          weights(%arg3 : !qTypeCMX3)
          parent_input(%arg2 : !qTypeCMX3)
          parent_output(%arg4 : !qTypeCMX3)
          outputs(%arg4 : !qTypeCMX3) -> !qTypeCMX3
          variants : {
            DPUTask {
              cluster_id = 0 : i64, inEnd = [159, 53, 15], inStart = [0, 0, 0],
              mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [159, 53, 15], outStart = [0, 0, 0],
              pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            DPUTask {
              cluster_id = 1 : i64, inEnd = [159, 52, 15], inStart = [0, 0, 0],
              mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [159, 52, 15], outStart = [0, 0, 0],
              pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            DPUTask {
              cluster_id = 2 : i64, inEnd = [159, 52, 15], inStart = [0, 0, 0],
              mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [159, 52, 15], outStart = [0, 0, 0],
              pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
          } PPE : {
            PPETask {
              ppe = #VPU.PPEStub<>
              }
          }
    }

    return %eltwise_2 : !DistributedType5

  // CHECK: [[ELTWISE_1_INPUT_BUF:%.+]] = VPURT.AllocDistributed
  // CHECK-SAME: -> !VPUIP.DistributedBuffer<1x1x409600x1xf16

  // CHECK-NEXT: [[CREATED_SHAPE_CAST:%.+]] = VPUIP.ShapeCast {shape = [1, 16, 160, 160]}
  // CHECK-SAME: inputs([[ELTWISE_1_INPUT_BUF]] : !VPUIP.DistributedBuffer<1x1x409600x1xf16
  // CHECK-SAME: -> !VPUIP.DistributedBuffer<1x16x160x160xf16

  // CHECK-NEXT: [[CREATED_DISTRIBUTED_CAST:%.+]] = VPUIP.DistributedCast
  // CHECK-SAME: inputs([[CREATED_SHAPE_CAST]] : !VPUIP.DistributedBuffer<1x16x160x160xf16
  // CHECK-SAME: -> !VPUIP.DistributedBuffer<1x16x160x160xf16

  // CHECK-NEXT: [[ELTWISE_1_CONCAT:%.+]] = VPUIP.ConcatView
  // CHECK-NEXT: [[ELTWISE_1_SHAPE_CAST:%.+]] = VPUIP.ShapeCast
  // CHECK-NEXT: [[ELTWISE_1_DISTRIBUTED_CAST:%.+]] = VPUIP.DistributedCast

  // CHECK-NEXT: [[ELTWISE:%.+]] = VPUIP.NCEClusterTiling
  // CHECK-SAME: inputs([[IN3:%.+]] as %arg3: memref<1x16x160x160xf16, #NHWC, @CMX_NN>,
  // CHECK-SAME:        [[ELTWISE_1_DISTRIBUTED_CAST]] as %arg4: memref<1x16x160x160xf16, #NHWC, @CMX_NN>)
  // CHECK-SAME: outputs([[CREATED_DISTRIBUTED_CAST]] as %arg5: memref<1x16x160x160xf16, #NHWC, @CMX_NN>)
  // CHECK-SAME: -> !VPUIP.DistributedBuffer<1x16x160x160xf16

  // CHECK-NEXT: [[ELTWISE_INNER:%.+]] = VPUIP.NCEClusterTask
  // CHECK: return [[ELTWISE]] : !VPUIP.DistributedBuffer<1x16x160x160xf16

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qTypeCMX = memref<1x512x32x64xf16, #NHWC, @CMX_NN>

!DistributedType = !VPUIP.DistributedBuffer<
    1x64x512x32xf16,
    #NWCH, @CMX_NN, {
        mode = "SEGMENTED",
        num_tiles = [1, 1, 1, 3],
        num_clusters = 3 : i64,
        uniform_distributed_segments,
        compute_shapes = [[1, 64, 512, 11], [1, 64, 512, 11], [1, 64, 512, 10]],
        compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 11], [0, 0, 0, 22]],
        memory_shapes = [[1, 64, 512, 11], [1, 64, 512, 11], [1, 64, 512, 10]],
        memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 11], [0, 0, 0, 22]]
}>

!DistributedType1 = !VPUIP.DistributedBuffer<
    1x512x32x64xf16,
    #NHWC, @CMX_NN, {
        mode = "SEGMENTED",
        num_tiles = [1, 1, 3, 1],
        num_clusters = 3 : i64,
        uniform_distributed_segments,
        compute_shapes = [[1, 512, 11, 64], [1, 512, 11, 64], [1, 512, 10, 64]],
        compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0]],
        memory_shapes = [[1, 512, 11, 64], [1, 512, 11, 64], [1, 512, 10, 64]],
        memory_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0]]
}>

// CHECK-LABEL: NotCreateShapeCastOpWithLayoutChange
func.func @NotCreateShapeCastOpWithLayoutChange () -> (!DistributedType1) {
    %eltwise_in_0_cmx_buf = VPURT.AllocDistributed -> !DistributedType
    %eltwise_in_0 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NCHW} inputs(%eltwise_in_0_cmx_buf: !DistributedType)
            -> !DistributedType1
    %eltwise_in_1_cmx_buf = VPURT.AllocDistributed -> !DistributedType
    %eltwise_in_1 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NCHW} inputs(%eltwise_in_1_cmx_buf: !DistributedType)
            -> !DistributedType1
    %eltwise_out_cmx_buf = VPURT.AllocDistributed -> !DistributedType1
    %eltwise = VPUIP.NCEClusterTiling
      inputs(
        %eltwise_in_0 as %arg2: !qTypeCMX,
        %eltwise_in_1 as %arg3: !qTypeCMX)
      outputs(%eltwise_out_cmx_buf as %arg4: !qTypeCMX) -> !DistributedType1 {
        %eltwise_2_inner = VPUIP.NCEClusterTask {
          is_inplace = true, minimumHardwareExecutionCost = 10507 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
          input(%arg2 : !qTypeCMX)
          weights(%arg3 : !qTypeCMX)
          parent_input(%arg2 : !qTypeCMX)
          parent_output(%arg4 : !qTypeCMX)
          outputs(%arg4 : !qTypeCMX) -> !qTypeCMX
          variants : {
            DPUTask {
              cluster_id = 0 : i64, inEnd = [10, 63, 511], inStart = [0, 0, 0],
              mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [10, 63, 511], outStart = [0, 0, 0],
              pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            DPUTask {
              cluster_id = 1 : i64, inEnd = [10, 63, 511], inStart = [0, 0, 0],
              mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [10, 63, 511], outStart = [0, 0, 0],
              pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            DPUTask {
              cluster_id = 2 : i64, inEnd = [9, 63, 511], inStart = [0, 0, 0],
              mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [9, 63, 511], outStart = [0, 0, 0],
              pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
          } PPE : {
            PPETask {
              ppe = #VPU.PPEStub<>
              }
          }
    }

    return %eltwise : !DistributedType1

    // CHECK:       [[IN_BUF_0:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<1x64x512x32xf16, #NWCH, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 3], num_clusters = 3 : i64,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 64, 512, 11], [1, 64, 512, 11], [1, 64, 512, 10]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 11], [0, 0, 0, 22]]

    // CHECK-NEXT:  [[IN_0:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NCHW}
    // CHECK-SAME:      inputs([[IN_BUF_0]] : !VPUIP.DistributedBuffer<1x64x512x32xf16, #NWCH, @CMX_NN
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<1x512x32x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 512, 11, 64], [1, 512, 11, 64], [1, 512, 10, 64]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0]]

    // CHECK-NEXT:  [[IN_BUF_1:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<1x64x512x32xf16, #NWCH, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 3], num_clusters = 3 : i64,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 64, 512, 11], [1, 64, 512, 11], [1, 64, 512, 10]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 11], [0, 0, 0, 22]]

    // CHECK-NEXT:  [[OUT_BUF:%.+]] = VPUIP.ViewOp [[IN_BUF_1]]
    // CHECK-SAME:      !VPUIP.DistributedBuffer<1x64x512x32xf16, #NWCH, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 3], num_clusters = 3 : i64
    // CHECK-SAME:   to !VPUIP.DistributedBuffer<1x512x32x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64

    // CHECK-NEXT:  [[IN_1:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NCHW}
    // CHECK-SAME:      inputs([[IN_BUF_1]] : !VPUIP.DistributedBuffer<1x64x512x32xf16, #NWCH, @CMX_NN
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<1x512x32x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 512, 11, 64], [1, 512, 11, 64], [1, 512, 10, 64]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0]]

    // CHECK-NEXT:  [[NCE_CLUSTER_TILING:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[IN_0]] as {{[^:]+}}: memref<1x512x32x64xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:             [[IN_1]] as {{[^:]+}}: memref<1x512x32x64xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[OUT_BUF]] as {{[^:]+}}: memref<1x512x32x64xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<1x512x32x64xf16, #NHWC, @CMX_NN

    // CHECK: return [[NCE_CLUSTER_TILING]] : !VPUIP.DistributedBuffer<1x512x32x64xf16, #NHWC, @CMX_NN
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: CreateShapeCastOpSingleClusterCase
// CHECK-SAME: ([[IN1:%.+]]: memref<1x1x207360x1xf16, #NHWC, @CMX_NN>,
// CHECK-SAME:  [[IN2:%.+]]: memref<1x1x202240x1xf16, #NHWC, @CMX_NN>,
// CHECK-SAME:  [[IN3:%.+]]: memref<1x1x409600x1xf16, #NHWC, @CMX_NN>)
// CHECK-SAME: -> memref<1x16x160x160xf16, #NHWC, @CMX_NN> {
func.func @CreateShapeCastOpSingleClusterCase (%in1 : memref<1x1x207360x1xf16, #NHWC, @CMX_NN>, %in2 : memref<1x1x202240x1xf16, #NHWC, @CMX_NN>, %in3 : memref<1x1x409600x1xf16, #NHWC, @CMX_NN>) -> (memref<1x16x160x160xf16, #NHWC, @CMX_NN>) {
    %buf_0 = memref.alloc() : memref<1x1x409600x1xf16, #NHWC, @CMX_NN>
    %buf_1 = memref.alloc() : memref<1x16x160x160xf16, #NHWC, @CMX_NN>

    %concat = VPUIP.ConcatView
      inputs(%in1, %in2 : memref<1x1x207360x1xf16, #NHWC, @CMX_NN>, memref<1x1x202240x1xf16, #NHWC, @CMX_NN>)
      outputs(%buf_0 : memref<1x1x409600x1xf16, #NHWC, @CMX_NN>) -> memref<1x1x409600x1xf16, #NHWC, @CMX_NN>
    %shape_cast = VPUIP.ShapeCast {
      shape = [1, 16, 160, 160]}
      inputs(%concat : memref<1x1x409600x1xf16, #NHWC, @CMX_NN>) -> memref<1x16x160x160xf16, #NHWC, @CMX_NN>
    %eltwise = VPUIP.NCEClusterTask {
          is_inplace = true, minimumHardwareExecutionCost = 10507 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
          input(%in3 : memref<1x1x409600x1xf16, #NHWC, @CMX_NN>)
          weights(%shape_cast : memref<1x16x160x160xf16, #NHWC, @CMX_NN>)
          parent_input(%in3 : memref<1x1x409600x1xf16, #NHWC, @CMX_NN>)
          parent_output(%buf_1 : memref<1x16x160x160xf16, #NHWC, @CMX_NN>)
          outputs(%buf_1 : memref<1x16x160x160xf16, #NHWC, @CMX_NN>) -> memref<1x16x160x160xf16, #NHWC, @CMX_NN>
          variants : {
            DPUTask {
              mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [160, 160, 16], outStart = [0, 0, 0],
              pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
          } PPE : {
            PPETask {
              ppe = #VPU.PPEStub<>
              }
          }

    return %eltwise : memref<1x16x160x160xf16, #NHWC, @CMX_NN>

  // CHECK: [[BUF_0:%.+]] = memref.alloc() : memref<1x1x409600x1xf16, #NHWC, @CMX_NN>
  // CHECK-NOT: [[BUF_1:%.+]] = memref.alloc()

  // CHECK-NEXT: [[CREATED_SHAPE_CAST:%.+]] = VPUIP.ShapeCast
  // CHECK-SAME: {shape = [1, 16, 160, 160]}
  // CHECK-SAME: inputs([[BUF_0]] : memref<1x1x409600x1xf16, #NHWC, @CMX_NN>) -> memref<1x16x160x160xf16, #NHWC, @CMX_NN>

  // CHECK-NEXT: [[CONCAT:%.+]] = VPUIP.ConcatView
  // CHECK-SAME: outputs([[BUF_0]] : memref<1x1x409600x1xf16, #NHWC, @CMX_NN>) -> memref<1x1x409600x1xf16, #NHWC, @CMX_NN>
  // CHECK-NEXT: [[SHAPE_CAST:%.+]] = VPUIP.ShapeCast

  // CHECK-NEXT: [[ELTWISE:%.+]] =  VPUIP.NCEClusterTask
  // CHECK-SAME: input([[IN3]] : memref<1x1x409600x1xf16, #NHWC, @CMX_NN>)
  // CHECK-SAME: weights([[SHAPE_CAST]] : memref<1x16x160x160xf16, #NHWC, @CMX_NN>)
  // CHECK-SAME: parent_input([[IN3]] : memref<1x1x409600x1xf16, #NHWC, @CMX_NN>)
  // CHECK-SAME: parent_output([[CREATED_SHAPE_CAST]] : memref<1x16x160x160xf16, #NHWC, @CMX_NN>)
  // CHECK-SAME: outputs([[CREATED_SHAPE_CAST]] : memref<1x16x160x160xf16, #NHWC, @CMX_NN>) -> memref<1x16x160x160xf16, #NHWC, @CMX_NN> variants :

  // CHECK: return [[ELTWISE]] : memref<1x16x160x160xf16, #NHWC, @CMX_NN>

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
          is_inplace = true,
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
          PPETask {
            ppe = #VPU.PPEStub<>
            }
        }
    }

    return %eltwise : !DistributedType2

}
