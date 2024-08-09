//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPU/impl/convert_ops_to_dma_for_performant_execution_strategy.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIP/utils/convert_to_dma_utils.hpp"

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

namespace {
//
// MovetoDMAGather
//

class MovetoDMAGather final : public mlir::OpRewritePattern<VPU::GatherOp> {
public:
    MovetoDMAGather(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::GatherOp>(ctx), _log(log) {
        setDebugName("MovetoDMAGather");
    }

    mlir::LogicalResult matchAndRewrite(VPU::GatherOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

VPU::ReshapeOp reshapeIndices(const mlir::Value& input, const mlir::Value& indices, mlir::IntegerAttr axis,
                              mlir::PatternRewriter& rewriter, const mlir::Location& location) {
    const auto inputType = input.getType().cast<vpux::NDTypeInterface>();
    const auto inputShape = inputType.getShape();
    const auto indicesType = indices.getType().cast<vpux::NDTypeInterface>();
    const auto indicesShape = indicesType.getShape();

    const auto indicesCount = vpux::details::calcTotalShapeSize(indicesShape.raw());
    auto newShape = SmallVector<int64_t>(inputShape.size(), 1);
    newShape[axis.getInt()] = indicesCount;

    const auto newInputShapeAttr = getIntArrayAttr(input.getContext(), newShape);
    auto newOp = rewriter.create<VPU::ReshapeOp>(location, indices, nullptr, false, newInputShapeAttr);

    return newOp;
}

mlir::LogicalResult MovetoDMAGather::matchAndRewrite(VPU::GatherOp origOp, mlir::PatternRewriter& rewriter) const {
    auto reshapeOp = reshapeIndices(origOp->getOperand(0), origOp->getOperand(1), origOp.getAxisValueAttr(), rewriter,
                                    origOp->getLoc());

    auto requiredType64 = mlir::IntegerType::get(origOp->getContext(), 64);
    auto typeAttr = mlir::TypeAttr::get(requiredType64);
    auto convertOp = rewriter.create<VPU::ConvertOp>(origOp->getLoc(), reshapeOp.getResult(), typeAttr);
    const auto cmxMemSpace = IndexedSymbolAttr::get(origOp.getContext(), stringifyEnum(VPU::MemoryKind::CMX_NN), 0);
    const auto ddrMemSpace = IndexedSymbolAttr::get(origOp.getContext(), stringifyEnum(VPU::MemoryKind::DDR));
    mlir::Value indicesCMX = convertOp->getResult(0);
    const auto outConvertOpMemSpace = convertOp->getResult(0).getType().cast<NDTypeInterface>().getMemSpace();
    if (outConvertOpMemSpace != cmxMemSpace) {
        indicesCMX = rewriter.create<VPU::CopyOp>(origOp->getLoc(), convertOp->getResult(0), cmxMemSpace)->getResult(0);
    }
    auto gatherDMAOp =
            rewriter.create<VPU::GatherDMAOp>(origOp->getLoc(), origOp->getOperand(0), indicesCMX, origOp.getAxis(),
                                              origOp.getAxisValueAttr(), origOp.getBatchDims());

    auto copyOp = rewriter.create<VPU::CopyOp>(origOp->getLoc(), gatherDMAOp->getResult(0), ddrMemSpace);

    const auto copyOutputType = copyOp->getResult(0).getType().cast<NDTypeInterface>();

    auto copyResultType = mlir::RankedTensorType::get(copyOutputType.getShape().raw(), copyOutputType.getElementType());

    copyOp->getResult(0).setType(copyResultType);

    auto resultType = origOp->getResult(0).getType().cast<NDTypeInterface>();
    const auto newOutputShapeAttr = getIntArrayAttr(origOp.getContext(), resultType.getShape().toValues());

    auto reshapeOp2 = rewriter.replaceOpWithNewOp<VPU::ReshapeOp>(origOp, copyOp->getResult(0), nullptr, false,
                                                                  newOutputShapeAttr);

    const auto reshapeOutputType = reshapeOp2->getResult(0).getType().cast<NDTypeInterface>();

    auto reshapeResultType =
            mlir::RankedTensorType::get(reshapeOutputType.getShape().raw(), reshapeOutputType.getElementType());
    reshapeOp2->getResult(0).setType(reshapeResultType);

    return mlir::success();
}
}  // namespace

//
// ConvertOpToDMAForPerformantExecutionStrategy
//

void VPU::arch40xx::ConvertOpToDMAForPerformantExecutionStrategy::addPatterns(mlir::RewritePatternSet& patterns,
                                                                              Logger& log) const {
    auto ctx = patterns.getContext();
    patterns.insert<MovetoDMAGather>(ctx, log);
}

void VPU::arch40xx::ConvertOpToDMAForPerformantExecutionStrategy::markOpLegality(mlir::ConversionTarget& target,
                                                                                 Logger& log) const {
    target.addDynamicallyLegalOp<VPU::GatherOp>([&](VPU::GatherOp op) {
        if (!VPU::isLegalConvertToGatherDMA(op, /*isElementTile*/ false, /*isIndicesTile*/ false, log)) {
            return true;
        }
        return false;
    });

    target.addLegalOp<VPU::GatherDMAOp>();
    target.addLegalOp<VPU::ReshapeOp>();
    target.addLegalOp<VPU::CopyOp>();
    target.addLegalOp<VPU::ConvertOp>();
}
