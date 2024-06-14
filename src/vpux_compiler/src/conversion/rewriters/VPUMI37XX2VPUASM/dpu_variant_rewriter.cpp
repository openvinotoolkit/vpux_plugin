//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/rewriters/VPUMI37XX2VPUASM/dpu_variant_rewriter.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

namespace vpux {
namespace vpumi37xx2vpuasm {

mlir::LogicalResult DPUVariantRewriter::symbolize(VPUMI37XX::DPUVariantOp op, SymbolMapper&,
                                                  mlir::ConversionPatternRewriter& rewriter) const {
    auto symName = findSym(op).getRootReference();
    auto taskLocation = findSym(op.getTaskLocation());

    auto invariant = findSym(op.getInvariant());
    auto taskIdx = mlir::TypeAttr::get(op.getType());

    rewriter.create<VPUASM::DPUVariantOp_37XX>(op.getLoc(), symName, taskIdx, taskLocation, invariant,
                                               op.getStartAttr(), op.getEndAttr(), op.getPadAttr(), op.getMpeModeAttr(),
                                               op.getClusterIdAttr());
    rewriter.eraseOp(op);

    return mlir::success();
}

}  // namespace vpumi37xx2vpuasm
}  // namespace vpux
