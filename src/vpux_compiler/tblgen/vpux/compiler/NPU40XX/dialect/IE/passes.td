//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_IE_ARCH_40XX_PASSES
#define VPUX_COMPILER_DIALECT_IE_ARCH_40XX_PASSES

include "mlir/Pass/PassBase.td"

//
// MapBilinearInterpolateOnDPUPass
//

def MapBilinearInterpolateOnDPUPass : PassBase<"map-bilinear-interpolate-on-dpu", "vpux::FunctionPass"> {
    let summary = "Convert bilinear interpolate op to strided concat, AvgPool and some depthwise convolution Ops";

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
            AvgPool        AvgPool

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
            AvgPool        AvgPool
        ```
        At the end the results of all the operation resulted are concatenated together on the scaling axis.

        In case the `interpolateAsSEOp` option is set to true, only cases that cannot be executed
        using the Storage Element hardware feature will be converted to concats.
    }];

    let constructor = "vpux::IE::arch40xx::createMapBilinearInterpolateOnDPUPass()";

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

#endif
