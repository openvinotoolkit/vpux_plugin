//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_IE_ARCH_30XX_PASSES
#define VPUX_COMPILER_DIALECT_IE_ARCH_30XX_PASSES

include "mlir/Pass/PassBase.td"

//
// ConvertTile2PerAxisTile
//

def ConvertTile2PerAxisTile : PassBase<"convert-tile-to-per-axis-tiles", "vpux::FunctionPass"> {
    let summary = "Convert tile op by multiple axes to multiple PerAxisTile operations";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        This pass replaces all `Tile` op with a set of `PerAxisTile` operations.
    }];

    let constructor = "vpux::IE::arch30xx::createConvertTile2PerAxisTilePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// InsertIdentityPoolBeforeOp
//

def InsertIdentityPoolBeforeOp : PassBase<"insert-identity-pool-before-op", "vpux::FunctionPass"> {
    let summary = "Insert Maxpool/AvgPool op before Activation ops and MemPermute ops";

    let description = [{
        The pass inserts MaxPool/AvgPool op before Activation ops and MemPermute ops.
        MaxPool/AvgPool will be lowered to a NCE op and the activation op will be fused into it.
        Supported activations: LeakyRelu, Clamp.
    }];

    let constructor = "vpux::IE::arch30xx::createInsertIdentityPoolBeforeOpPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// MapBilinearInterpolateOnDPUPass
//

def MapBilinearInterpolateOnDPUPass : PassBase<"map-bilinear-interpolate-on-dpu", "vpux::FunctionPass"> {
    let summary = "Convert bilinear interpolate op to strided concat, MaxPool and some depthwise convolution Ops";

    let description = [{
        Purpose:
        This pass replaces `Bilinear Interpolate` for which tiling is required to fit in CMX
        with sequences of operation that can be mapped on DPU and DMA.

        How it works:
        The supported interpolation axis currently supported are H and W.
        For each of these axis the scaling is happening individually, first perform vertical scaling and after perform horizontal scaling.
        On each axis the processing is split in three main regions BEGIN, MIDDLE and END.
        These three regions refers to slices from the output tensor and are influenced by the coordinate transformation mode attribute.
        * BEGIN - refers to the slice from output for which it is only needed to duplicate the first line/column from input
        * MIDDLE - refers to the slice from output where:
                    * for each output line/column from the output it is required to take two consecutive lines/colums from the input
                    * based on the coordinate transformation mode attribute compute the weight coefficients each of two lines/columns
                      has on theresulting output line/column
                    * each output line/column is computed with a GroupConvolution operation for which the weights are obtained by expanding
                     the weight coefficients of the input lines/columns
        * END - refers to the slice from output for which it is only needed to duplicate the last line/column from input
        ```
        Vertical scaling:                    Horizontal scaling
        ________________________       ____________________________
        |        BEGIN         |       |   |                  |   |
        |______________________|       |   |                  |   |
        |                      |       | B |        M         |   |
        |                      |       | E |        I         | E |
        |        MIDDLE        |       | G |        D         | N |
        |                      |       | I |        D         | D |
        |______________________|       | N |        L         |   |
        |         END          |       |   |        E         |   |
        |______________________|       |___|__________________|___|
        ```

        The rewrite implemented per each region is described below:
         BEGIN region:
        ```         Input
                      |
                    Slice
               first line/column
                |    ...    |
            Identity       Identity
            MaxPool        MaxPool

         MIDDLE region
                         Input
                  ---------|---------
                 |                   |
             Slice        ...       Slice
         two lines/colums       two lines/colums
               |                        |
           GroupConv               GroupConv
         one output line/colum   one output line/colum

         END region:
                    Input
                      |
                    Slice
               last line/column
                |    ...     |
            Identity       Identity
            MaxPool        MaxPool
        ```
        At the end the results of all the operation resulted are concatenated together on the scaling axis.

        In case the `interpolateAsSEOp` option is set to true, only cases that cannot be executed
        using the Storage Element hardware feature will be converted to concats.
    }];

    let constructor = "vpux::IE::arch30xx::createMapBilinearInterpolateOnDPUPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];

    let options = [
        Option<
            "interpolateAsSEOp", "interpolate-as-se-op",
            "bool", "false",
            "Flag which identifies whether an Interpolate operation can be executed using the Storage Element hardware feature"
        >
    ];
}

//
// OptimizeSliceExpand
//

def OptimizeSliceExpand : PassBase<"optimize-slice-expand", "vpux::FunctionPass"> {
    let summary = "Optimize patterns Slice->Expand and Slice->Implicit operations ->Expand";

    let description = [{
        The pass is a part of `buildHardwareModePipeline` pipeline.

        Optimize patterns Slice->Expand and Slice->Implicit operations ->Expand in order to avoid extra DMAs
    }];

    let constructor = "vpux::IE::arch30xx::createOptimizeSliceExpandPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ExpandActivationChannels
//

def ExpandActivationChannels : PassBase<"expand-activation-channels", "vpux::FunctionPass"> {
    let summary = "Align input tensors shape of DPU operation with hardware requirements";

    let description = [{
        The pass is a part of `buildHardwareModePipeline` pipeline.

        This pass processes operations, which can be compile as a DPU tasks and
            expands channels number to number divisible by 16 in case they doesn't satisfy hardware requirements
    }];

    let constructor = "vpux::IE::arch30xx::createExpandActivationChannelsPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];

    let options = [
        Option<
            "seOpsEnabled", "se-ops-enabled",
            "bool", "false",
            "Flag to identify whether operations that can be executed using the Storage Element hardware feature are enabled"
        >,
        Option<
            "seExperimentalOpsEnabled", "se-experimental-ops-enabled",
            "bool", "false",
            "This flag identifies operations that are still a work in progress and can be executed using the Storage Element hardware feature."
        >
    ];
}

//
// UnrollBatch
//

def UnrollBatch : PassBase<"unroll-batch", "vpux::FunctionPass"> {
    let summary = "Split inputs of NCE tasks when their batch size is greater than 1";

    let description = [{
        This pass splits inputs of NCE tasks by batch.

        For example:
        * `FullyConnected` input with 2x64 geometry will be split by two inputs with 1x64 dimensions.
        * `Convolution` input 3x16x32x64 will be split into three 1x16x32x64 inputs.
        Resulting tensors go to corresponding operations and the outputs are concatenated.
    }];

    let constructor = "vpux::IE::arch30xx::createUnrollBatchPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertToMixPrecision
//

def ConvertToMixedPrecision: PassBase<"convert-to-mixed-precision", "vpux::FunctionPass"> {
    let summary = "Convert DPU task without fake quantize behind to mixed-precision operation";

    let description = [{
        The pass is a part of `LowPrecision` pipeline.
        Converts DPU task to mixed-precision operation where there is no quantize operation for the output of a DPU task
    }];

    let constructor = "vpux::IE::arch30xx::createConvertToMixedPrecision()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

#endif
