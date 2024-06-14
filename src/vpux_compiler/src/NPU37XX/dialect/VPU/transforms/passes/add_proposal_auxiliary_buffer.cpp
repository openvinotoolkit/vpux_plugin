//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <openvino/op/op.hpp>
#include "vpux/compiler/NPU37XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

using namespace vpux;
using namespace VPU;

namespace {

class ProposalAuxiliaryBufferPass final : public mlir::OpRewritePattern<VPU::ProposalOp> {
public:
    ProposalAuxiliaryBufferPass(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::ProposalOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::ProposalOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

class AddProposalAuxiliaryBufferPass final :
        public VPU::arch37xx::AddProposalAuxiliaryBufferBase<AddProposalAuxiliaryBufferPass> {
public:
    explicit AddProposalAuxiliaryBufferPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

mlir::LogicalResult ProposalAuxiliaryBufferPass::matchAndRewrite(VPU::ProposalOp origOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    _log.trace("Found Proposal Operation '{0}'", origOp->getLoc());

    constexpr int32_t proposalBoxSize = 10;         // see: sw_runtime_kernels/kernels/src/proposal.cpp (proposalBox)
    constexpr int32_t anchorsBuffElementSize = 16;  // see: sw_runtime_kernels / kernels / src / proposal.cpp (anchors)
    const auto inType = origOp.getClassProbs().getType().cast<vpux::NDTypeInterface>();

    const auto inShape = inType.getShape().raw();
    // [ num_batches, 2 * K, H, W ]
    auto rank = inShape.size();

    VPUX_THROW_UNLESS(rank == 4, "Unsupported rank {0}", rank);
    const auto k = inShape[rank - 3] / 2;
    const auto h = inShape[rank - 2];
    const auto w = inShape[rank - 1];
    const auto numProposals = k * h * w;
    const auto auxiliaryBuffSize = alignValUp(numProposals * proposalBoxSize, static_cast<int64_t>(7)) +
                                   alignValUp(k * anchorsBuffElementSize, static_cast<int64_t>(7));
    std::vector<uint8_t> vals(auxiliaryBuffSize, 0.0f);

    const SmallVector<int64_t> shape({auxiliaryBuffSize});
    const auto auxiliaryType = mlir::RankedTensorType::get(shape, getUInt8Type(origOp.getContext()));
    const auto auxiliaryAttr = mlir::DenseElementsAttr::get(auxiliaryType, ArrayRef(vals));
    auto auxiliaryContentAttr = Const::ContentAttr::get(auxiliaryAttr);

    auto auxBuff = rewriter.create<Const::DeclareOp>(mlir::UnknownLoc::get(origOp.getContext()),
                                                     auxiliaryContentAttr.getType(), auxiliaryContentAttr);

    rewriter.replaceOpWithNewOp<VPU::ProposalOp>(origOp, origOp.getClassProbs(), origOp.getBboxDeltas(),
                                                 origOp.getImageShape(), auxBuff, origOp.getProposalAttrsAttr());

    return mlir::success();
}

//
// safeRunOnFunc
//

bool hasAuxiliaryInput(VPU::ProposalOp op) {
    return (op.getAuxiliary() != nullptr);
}

void AddProposalAuxiliaryBufferPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);

    target.addDynamicallyLegalOp<VPU::ProposalOp>(&hasAuxiliaryInput);
    target.addLegalOp<Const::DeclareOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ProposalAuxiliaryBufferPass>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        _log.debug("Failed to add auxiliary buffer for Proposal.");
        signalPassFailure();
    }
}

}  // namespace

//
// createAddProposalAuxiliaryBufferPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::arch37xx::createAddProposalAuxiliaryBufferPass(Logger log) {
    return std::make_unique<AddProposalAuxiliaryBufferPass>(log);
}
