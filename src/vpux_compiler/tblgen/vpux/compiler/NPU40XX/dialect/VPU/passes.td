//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPU_ARCH_40XX_PASSES
#define VPUX_COMPILER_DIALECT_VPU_ARCH_40XX_PASSES

include "mlir/Pass/PassBase.td"

//
// FuseM2IOps
//

def FuseM2IOps : PassBase<"fuse-m2i-ops", "vpux::FunctionPass"> {
    let summary = "Fuse M2I ops to VPU::M2ITask";

    let constructor = "vpux::VPU::arch40xx::createFuseM2IOpsPass()";

    let dependentDialects = [
       "vpux::VPU::VPUDialect"
    ];
}

//
// ConvertM2IOps
//

def ConvertM2IOps : PassBase<"convert-m2i-ops", "vpux::FunctionPass"> {
    let summary = "Convert individual M2I ops to VPU::M2ITask";

    let constructor = "vpux::VPU::arch40xx::createConvertM2IOpsPass()";

    let dependentDialects = [
       "vpux::VPU::VPUDialect"
    ];
}

//
// CorrectNCEWorkloads
//

def CorrectNCEWorkloads : PassBase<"correct-NCE-workloads", "vpux::FunctionPass"> {
    let summary = "Correct NCE workloads if they do not fit requirements";

    let description = [{
        The pass adjusts workload size for NCEDepthConvolution, NCEMaxPool and NCEAveragePool,
        as well as for NCE operations that produce sparse activations.

        NCEDepthConvolutionOp, NCEMaxPoolOp and NCEAveragePoolOp require the number of channels to be 16, 32 or 64.
        If the number of channels does not match, workload is split.

        For depthwise ops, the number of channels will be adjusted if it can match the requirement of Small kernel
        Optimization for DW(L1OPT). The L1Opt requires:
        1. Kernel X = 3
        2. Stride X = 1
        3. Workload Channel Size = 32

        NCE operations with sparse outputs must have all variants with the same number of channels excluding the last one and the number
        of channels has to be a power of two. Additionally, if the NCE op shares a consumer with another NCE op
        (directly or indirectly), the number of channels of their variants must be aligned.
    }];

    let constructor = "vpux::VPU::arch40xx::createCorrectNCEWorkloadsPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// ComputeNCEInputWorkloads
//

def ComputeNCEInputWorkloads : PassBase<"compute-nce-input-workloads", "vpux::FunctionPass"> {
    let summary = "Compute input workloads start/end";

    let description = [{
        This pass will compute the input workloads start/end for each VPU.DPU.WorkloadOp, based on
        the output workloads' start/end.

        For arch 40XX, in cases where there are multiple consumers with different halo requirements
        from the parent op, we can select the largest halo size for the parent op, but we need to set
        the input workloads accordingly in the consumer ops.

        For example, an NCE op needs to have its input start at offset X on height to be able to compute
        its first output line. Due to kernel requirements, another NCE op that shares the same parent needs
        to have its input start at X - 2 height line. This situation can be resolved by having the previous
        NCE op put its output starting with line X - 2 and have the first NCE op set its inStart = 2.
    }];

    let constructor = "vpux::VPU::arch40xx::createComputeNCEInputWorkloadsPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// MoveConvertAroundViewLikeOps
//

def MoveConvertAroundViewLikeOps : PassBase<"move-convert-around-viewlike-ops", "vpux::FunctionPass"> {
    let summary = "Move ConvertOp around Viewlike ops";

    let description = [{
        This is the base pass which contains all rewriters for moving convertOp around in the IR
        MoveConvertAfterPermuteCast -> Moves ConvertOp after PermuteCast so we can fuse the CopyOp and ConvertDMAOp and avoid DDR DMAs
        MoveConvertAfterShapeCast -> Moves ConvertOp after ShapeCastOp so we can fuse the CopyOp and ConvertDMAOp and avoid DDR DMAs
        MoveConvertAffineReshape -> Moves ConvertOp before AffineReshape so we can wrap ConvertOp in NCEClusterTiling with all 4 Dims
    }];

    let constructor = "vpux::VPU::arch40xx::createMoveConvertAroundViewLikeOpsPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

#endif
