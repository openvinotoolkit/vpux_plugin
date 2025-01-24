//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_version_config.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux {

//
// EltwiseToNCE
//

template <class ConcreteOp>
class EltwiseToNCE final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    EltwiseToNCE<ConcreteOp>(mlir::MLIRContext* ctx, VPU::EltwiseType opType, Logger log)
            : mlir::OpRewritePattern<ConcreteOp>(ctx), _opType(opType), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    VPU::EltwiseType _opType;
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult EltwiseToNCE<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp->getLoc());

    auto ppeAttr = VPU::PpeVersionConfig::retrievePPEAttribute(origOp);

    auto nceOp = rewriter.create<VPU::NCEEltwiseOp>(origOp->getLoc(), origOp.getType(), origOp.getInput1(),
                                                    origOp.getInput2(),
                                                    VPU::EltwiseTypeAttr::get(this->getContext(), _opType), ppeAttr,
                                                    /*multi_cluster_strategyAttr=*/nullptr,
                                                    /*is_inplace=*/nullptr, origOp.getOutputChannelsAttr());
    rewriter.replaceOp(origOp, nceOp.getOutput());
    return mlir::success();
}

}  // namespace vpux
