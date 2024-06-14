//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --vpu-arch=%arch% %s | FileCheck %s --match-full-lines
// REQUIRES: arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @Test_1 {
    VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <128> : memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
    VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131200> : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>

    VPUIPDPU.DPUInvariant @DPUInvariant_0 {task_index = !VPURegMapped.Index<0:0:0>, taskLocation = @DeclareTaskBuffer_DPUInvariant_0, input = @DeclareBuffer_ActIn, output = @DeclareBuffer_ActOut, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>}
        DPUCfg : {
            ^bb0(%act_in: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>,
                 %act_out: memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]>,
                 %act_out_cast1: memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]>,
                 %act_out_cast2: memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]>,
                 %sparse_out: memref<1x16x64x64xi1, #NHWC, [@CMX_NN, 0]>):
            VPUIPDPU.IDUCfg {
                VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>)
                VPUIPDPU.IDUStorageElement se_size(32)
                VPUIPDPU.IDUKernel kernel_x(1) kernel_y(2)
                VPUIPDPU.IDUStride stride_x(0) stride_y(0)
                VPUIPDPU.IDUInputLayerCfg sparsity_pattern(7) {input_compressed}
                VPUIPDPU.IDUWorkloadCfg workload_type(MAXPOOL)
                VPUIPDPU.IDUWeights  wmode(f16)
                VPUIPDPU.IDUSESegment se_seg_size_0(16) se_seg_size_1(16) se_seg_size_2(16)
                VPUIPDPU.IDUSPSegment sp_seg_size_0(16) sp_seg_size_1(16) sp_seg_size_2(16)
            }
            VPUIPDPU.PPECfg {
                VPUIPDPU.PPEFpAddMultBypass bypass_mode(ON)
            }
            VPUIPDPU.ODUCfg {
                VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
                VPUIPDPU.ODUDataReuse activation_reuse(NTHW_8)
                VPUIPDPU.ODUPermuteData permute_mode(PERMUTE_ZXY)
                VPUIPDPU.ODUSwizzleData swizzle_key(SWIZZLE_KEY_1)
                VPUIPDPU.ODUSparsity %sparse_out: memref<1x16x64x64xi1, #NHWC, [@CMX_NN, 0]>
                VPUIPDPU.ODUOutActivations out_activations(%act_out: memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]>) data_type(ODU_DTYPE_FP16)
                VPUIPDPU.ODUCast cast_output(%act_out_cast1: memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]>)
                VPUIPDPU.ODUCast cast_output(%act_out_cast2: memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]>)
            }
        }
}

// CHECK:    VPUIPDPU.DPUInvariant @DPUInvariant_0 {input = @DeclareBuffer_ActIn, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, output = @DeclareBuffer_ActOut, taskLocation = @DeclareTaskBuffer_DPUInvariant_0, task_index = !VPURegMapped.Index<0:0:0>} DPUCfg : {
// CHECK:    ^bb0(%arg0: memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>, %arg1: memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]>, %arg2: memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]>, %arg3: memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]>, %arg4: memref<1x16x64x64xi1, #NHWC, [@CMX_NN, 0]>):
//CHECK:     VPUIPDPU.IDUCfg {
//CHECK-NEXT:       VPUIPDPU.IDUInActivations in_activations(%arg0 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>)
//CHECK-NEXT:       VPUIPDPU.IDUStorageElement se_size(32)
//CHECK-NEXT:       VPUIPDPU.IDUKernel kernel_x(1) kernel_y(2)
//CHECK-NEXT:       VPUIPDPU.IDUStride stride_x(0) stride_y(0)
//CHECK-NEXT:       VPUIPDPU.IDUInputLayerCfg sparsity_pattern(7) {input_compressed}
//CHECK-NEXT:       VPUIPDPU.IDUWorkloadCfg workload_type(MAXPOOL)
//CHECK-NEXT:       VPUIPDPU.IDUWeights wmode(f16)
//CHECK-NEXT:       VPUIPDPU.IDUSESegment se_seg_size_0(16) se_seg_size_1(16) se_seg_size_2(16)
//CHECK-NEXT:       VPUIPDPU.IDUSPSegment sp_seg_size_0(16) sp_seg_size_1(16) sp_seg_size_2(16)
//CHECK-NEXT:     }
