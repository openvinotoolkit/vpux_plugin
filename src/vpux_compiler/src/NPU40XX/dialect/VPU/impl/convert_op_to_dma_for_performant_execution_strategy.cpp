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

    size_t indicesCount{0};
    for (size_t idx{0}; idx < indicesShape.size(); ++idx) {
        if (indicesShape[vpux::Dim(idx)] != 1) {
            indicesCount = indicesShape[vpux::Dim(idx)];
        }
    }
    if (indicesCount == 0) {
        indicesCount = 1;
    }
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
    auto resultType = origOp->getResult(0).getType().cast<NDTypeInterface>();
    const auto newOutputShapeAttr = getIntArrayAttr(origOp.getContext(), resultType.getShape().toValues());

    auto reshapeOp2 = rewriter.create<VPU::ReshapeOp>(origOp->getLoc(), gatherDMAOp->getResult(0), nullptr, false,
                                                      newOutputShapeAttr);
    auto copyOp = rewriter.replaceOpWithNewOp<VPU::CopyOp>(origOp, reshapeOp2->getResult(0), ddrMemSpace);
    auto copyResultType =
            mlir::RankedTensorType::get(copyOp->getResult(0).getType().cast<NDTypeInterface>().getShape().raw(),
                                        copyOp->getResult(0).getType().cast<NDTypeInterface>().getElementType());
    copyOp->getResult(0).setType(copyResultType);
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

bool isLegalConvertToGatherDMA(VPU::GatherOp op, vpux::Logger log) {
    log.trace("Got Gather Op at {0}.", op->getLoc());

    const auto outputType = op.getOutput().getType().cast<vpux::NDTypeInterface>();
    const auto indicesType = op.getIndices().getType().cast<vpux::NDTypeInterface>();
    const auto inputType = op.getInput().getType().cast<vpux::NDTypeInterface>();

    const auto outputSize = outputType.getTotalAllocSize();
    const auto indicesSize = indicesType.getTotalAllocSize();
    const auto cmxMemSize = VPU::getTotalCMXSize(op.getOperation());
    // Only DDR->CMX DMA gather has been implemented so far, not convert to GatherDMA when output can't fit on CMX
    // TODO: Support large output Gather layer with GatherDMA, see E*120478
    if (outputSize + indicesSize > cmxMemSize) {
        return false;
    }

    // For GatherDMA all dimensions before axis dimension must be 1
    size_t axis = op.getAxisValue().value();
    const auto inputShape = inputType.getShape();

    for (size_t idx = 0; idx < axis; ++idx) {
        if (inputShape[vpux::Dim(idx)] != 1) {
            return false;
        }
    }

    const size_t numberOfIndices = indicesType.getNumElements();
    if (numberOfIndices > VPUIP::arch40xx::DMA_MAX_INDICES_LIST_LENGTH) {
        return false;
    }
    const Bit elemOutSize = vpux::getElemTypeSize(outputType);
    const size_t dma_element_size =
            (outputType.getNumElements() / indicesType.getNumElements()) * elemOutSize.to<Byte>().count();
    if (dma_element_size > VPUIP::arch40xx::GATHER_DMA_MAX_ELEMENT_SIZE) {
        return false;
    }

    log.trace("GatherOp at {0} can be converted to GatherDMAOp.", op->getLoc());
    return true;
}

void VPU::arch40xx::ConvertOpToDMAForPerformantExecutionStrategy::markOpLegality(mlir::ConversionTarget& target,
                                                                                 Logger& log) const {
    target.addDynamicallyLegalOp<VPU::GatherOp>([&](VPU::GatherOp op) {
        if (!isLegalConvertToGatherDMA(op, log)) {
            return true;
        }
        return false;
    });

    target.addLegalOp<VPU::GatherDMAOp>();
    target.addLegalOp<VPU::ReshapeOp>();
    target.addLegalOp<VPU::CopyOp>();
    target.addLegalOp<VPU::ConvertOp>();
}
