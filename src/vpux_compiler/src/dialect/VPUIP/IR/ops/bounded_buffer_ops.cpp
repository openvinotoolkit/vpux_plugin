//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"

using namespace vpux;

mlir::LogicalResult VPUIP::GroupBoundedBufferOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                                  std::optional<mlir::Location> optLoc,
                                                                  mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                                  mlir::OpaqueProperties, mlir::RegionRange /*ranges*/,
                                                                  SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPUIP::GroupBoundedBufferOpAdaptor op(operands, attrs);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }

    const auto dataTy = op.getData().getType();
    const auto shapeTy = op.getDynamicShape().getType();

    inferredReturnTypes.push_back(VPUIP::BoundedBufferType::get(dataTy, shapeTy));

    return mlir::success();
}

mlir::ValueRange VPUIP::GroupBoundedBufferOp::getViewSources() {
    return getOperands();
}

mlir::LogicalResult VPUIP::UngroupBoundedBufferOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                                    std::optional<mlir::Location> optLoc,
                                                                    mlir::ValueRange operands,
                                                                    mlir::DictionaryAttr attrs, mlir::OpaqueProperties,
                                                                    mlir::RegionRange /*ranges*/,
                                                                    SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPUIP::UngroupBoundedBufferOpAdaptor op(operands, attrs);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }

    const auto boundedBufferTy = op.getInput().getType().cast<VPUIP::BoundedBufferType>();
    inferredReturnTypes.push_back(boundedBufferTy.getData().cast<mlir::MemRefType>());
    inferredReturnTypes.push_back(boundedBufferTy.getDynamicShape().cast<mlir::MemRefType>());

    return mlir::success();
}

mlir::Value VPUIP::UngroupBoundedBufferOp::getViewSource(ptrdiff_t idx) {
    VPUX_THROW_UNLESS(idx == 0 || idx == 1,
                      "UngroupBoundedBufferOp should have one view source with two aliases, got {0} offset", idx);
    return getOperand();
}

namespace {
//
// RemoveGroupUngroup
//
class RemoveGroupUngroup final : public mlir::OpRewritePattern<VPUIP::GroupBoundedBufferOp> {
public:
    using mlir::OpRewritePattern<VPUIP::GroupBoundedBufferOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(VPUIP::GroupBoundedBufferOp op,
                                        mlir::PatternRewriter& /*rewriter*/) const override {
        auto hasNonUngroupBoundedBufferUsers = llvm::any_of(op.getOutput().getUsers(), [](mlir::Operation* userOp) {
            return !mlir::isa<VPUIP::UngroupBoundedBufferOp>(userOp);
        });
        if (hasNonUngroupBoundedBufferUsers) {
            return mlir::failure();
        }

        // The pass will remove Group/Ungroup pairs
        //
        //   [data] [shape]
        //      \     /
        //  GroupBoundedBuffer
        //         |
        // UngroupBoundedBuffer
        //      /     \.
        //   [data] [shape]

        const auto groupOperands = op.getOperands();
        for (auto* ungroupOp : op.getOutput().getUsers()) {
            for (const auto& ungroupResult : ungroupOp->getResults() | indexed) {
                const auto ungroupResultIndex = ungroupResult.index();
                VPUX_THROW_UNLESS(ungroupResultIndex < groupOperands.size(),
                                  "UngroupBoundBufferOp '{0}' has more results than GroupBoundedBufferOp '{1}'",
                                  op.getLoc(), ungroupOp->getLoc());

                ungroupResult.value().replaceAllUsesWith(groupOperands[ungroupResultIndex]);
            }
        }
        return mlir::success();
    }
};
}  // namespace

void VPUIP::GroupBoundedBufferOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                              mlir::MLIRContext* context) {
    patterns.add<RemoveGroupUngroup>(context);
}
