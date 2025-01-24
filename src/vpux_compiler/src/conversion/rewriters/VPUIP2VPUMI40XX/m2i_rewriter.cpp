//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/rewriters/VPUIP2VPUMI40XX/m2i_rewriter.hpp"

#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"

namespace vpux::vpuip2vpumi40xx {

mlir::LogicalResult M2IRewriter::matchAndRewrite(VPUIP::M2ITaskOp origOp, OpAdaptor adaptor,
                                                 mlir::ConversionPatternRewriter& rewriter) const {
    auto ctx = origOp.getContext();
    const auto zeroIndex = VPURegMapped::IndexType::get(ctx, 0);

    auto doCscAttr = origOp.getDoCscAttr().getValue() ? mlir::UnitAttr::get(ctx) : nullptr;
    auto doNormAttr = origOp.getDoNormAttr().getValue() ? mlir::UnitAttr::get(ctx) : nullptr;

    // don't user replace method of rewriter as VPUIP::M2ITaskOp may have more than 1 output
    // in case if profiling enabled; in this case replace would fail as results counts don't
    // match - VPUMI40XX::M2I has single output
    rewriter.create<VPUMI40XX::M2IOp>(
            origOp.getLoc(), zeroIndex,
            nullptr,  // taskLocation
            nullptr,  // previousTask
            adaptor.getInput(), adaptor.getOutputBuff(), adaptor.getProfilingData(), doCscAttr, doNormAttr,
            adaptor.getInFmtAttr(), adaptor.getOutFmtAttr(), adaptor.getChromaInReverseChannelsAttr(),
            adaptor.getChromaOutReverseChannelsAttr(), adaptor.getLumaInReverseChannelsAttr(),
            adaptor.getLumaOutReverseChannelsAttr(), adaptor.getScaleFactorXAttr(), adaptor.getScaleFactorYAttr(),
            adaptor.getNormAttr(), adaptor.getTileOffsetXAttr(), adaptor.getTileOffsetYAttr(),
            adaptor.getProfilingMetadataAttr(), adaptor.getInterpAttr(),
            mlir::ValueRange(),                             // waitBarriers
            mlir::ValueRange(),                             // updateBarriers
            mlir::IntegerAttr::get(getUInt64Type(ctx), 0),  // startAfter
            mlir::IntegerAttr::get(getUInt64Type(ctx), 0),  // cleanAfter
            nullptr                                         // enqueueBarrier
    );

    rewriter.eraseOp(origOp);
    return mlir::success();
}

}  // namespace vpux::vpuip2vpumi40xx
