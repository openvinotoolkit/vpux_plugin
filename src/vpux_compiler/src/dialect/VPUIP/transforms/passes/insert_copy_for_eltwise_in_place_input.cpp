//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

using namespace vpux;

namespace {
vpux::Strides getCompactStrides(NDTypeInterface type) {
    const Bit elemTypeSize = vpux::getElemTypeSize(type.getElementType());
    const auto dimsOrder = type.getDimsOrder();
    const auto memShape = dimsOrder.toMemoryOrder(Shape(type.getShape()));
    const auto memStrides = StrideReqs::compact(dimsOrder.numDims()).calcStrides(elemTypeSize, memShape);
    return dimsOrder.toLogicalOrder(memStrides);
}

//
// InsertCopyForEltwiseInPlaceInputPass
//

class InsertCopyForEltwiseInPlaceInputPass final :
        public VPUIP::InsertCopyForEltwiseInPlaceInputBase<InsertCopyForEltwiseInPlaceInputPass> {
public:
    explicit InsertCopyForEltwiseInPlaceInputPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    std::optional<mlir::Value> getInputToOverwrite(VPUIP::NCEClusterTaskOp op, AliasesInfo& aliasesInfo);
    void insertCopies(VPUIP::NCEClusterTaskOp eltwise, mlir::Value overwrittenInput);
    void insertCopiesClustered(VPUIP::NCEClusterTilingOp parentNCEClusterOp, mlir::Value overwrittenInput);
};

std::optional<mlir::Value> InsertCopyForEltwiseInPlaceInputPass::getInputToOverwrite(VPUIP::NCEClusterTaskOp op,
                                                                                     AliasesInfo& aliasesInfo) {
    if (op.getTaskType() != VPUIP::NCETaskType::ELTWISE) {
        return std::nullopt;
    }

    if (!op.getIsInplace().value_or(false)) {
        return std::nullopt;
    }

    _log.trace("Found in place Eltwise {0}", op.getLoc());

    auto clusterTilingParent = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(op->getParentOp());
    auto eltwiseInputs =
            (clusterTilingParent != nullptr) ? VPUIP::getLayerInputs(clusterTilingParent) : VPUIP::getLayerInputs(op);
    auto eltwiseOutput = (clusterTilingParent != nullptr) ? VPUIP::getLayerOutputs(clusterTilingParent)[0]
                                                          : VPUIP::getLayerOutputs(op)[0];

    auto outRootBuffers = aliasesInfo.getRoots(eltwiseOutput);
    VPUX_THROW_WHEN(outRootBuffers.size() != 1, "Too many root buffers for eltwise output = {0}", eltwiseOutput);

    auto outputRootBuff = *outRootBuffers.begin();

    auto overwrittenInput = llvm::find_if(eltwiseInputs, [&](mlir::Value input) -> bool {
        auto inRootBuffers = aliasesInfo.getRoots(input);
        VPUX_THROW_WHEN(inRootBuffers.size() != 1, "Too many root buffers for eltwise input = {0}", input);
        auto inputRootBuff = *inRootBuffers.begin();

        return inputRootBuff == outputRootBuff;
    });

    VPUX_THROW_WHEN(overwrittenInput == eltwiseInputs.end(), "In place eltwise has no overwritten input.");

    return *overwrittenInput;
}

void InsertCopyForEltwiseInPlaceInputPass::insertCopies(VPUIP::NCEClusterTaskOp eltwise, mlir::Value overwrittenInput) {
    mlir::OpBuilder builder(eltwise);
    auto inputType = overwrittenInput.getType().cast<vpux::NDTypeInterface>();
    // To DDR
    const auto compactStrides = getCompactStrides(inputType);
    auto newDDRType = inputType.changeMemSpace(VPU::MemoryKind::DDR).changeStrides(compactStrides);
    auto newAllocDDROp = builder.create<mlir::memref::AllocOp>(appendLoc(eltwise->getLoc(), "_elt_in_place_input_DDR"),
                                                               newDDRType.cast<mlir::MemRefType>());
    auto newCopyToDDR = builder.create<VPUIP::CopyOp>(appendLoc(eltwise->getLoc(), "_unique_consumer_spill"),
                                                      overwrittenInput, newAllocDDROp);

    // To CMX
    auto newAllocCMXOp = builder.create<mlir::memref::AllocOp>(appendLoc(eltwise->getLoc(), "_elt_in_place_input_CMX"),
                                                               inputType.cast<mlir::MemRefType>());
    auto newCopyToCMX = builder.create<VPUIP::CopyOp>(eltwise->getLoc(), newCopyToDDR->getResult(0), newAllocCMXOp);

    overwrittenInput.replaceUsesWithIf(newCopyToCMX->getResult(0), [&](mlir::OpOperand& opOperand) {
        return opOperand.getOwner() == eltwise;
    });

    auto eltwiseOutput = VPUIP::getLayerOutputs(eltwise)[0];
    eltwiseOutput.replaceUsesWithIf(newAllocCMXOp->getResult(0), [&](mlir::OpOperand& opOperand) {
        return opOperand.getOwner() == eltwise;
    });
}

void InsertCopyForEltwiseInPlaceInputPass::insertCopiesClustered(VPUIP::NCEClusterTilingOp nceClusterTiling,
                                                                 mlir::Value overwrittenInput) {
    VPUX_THROW_WHEN(!nceClusterTiling.getInnerTaskOpOfType<VPUIP::NCEClusterTaskOp>(),
                    "NCeClusterTIlingOp should have NCEClusterTask inner op, but it does not = {0}", nceClusterTiling);

    mlir::OpBuilder builder(nceClusterTiling);
    auto inDistributedType = overwrittenInput.getType().cast<VPUIP::DistributedBufferType>();
    auto compactInType = inDistributedType.getCompactType().cast<vpux::NDTypeInterface>();
    const auto copyOutBodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        builder.create<VPUIP::CopyOp>(loc, newOperands[0], newOperands[1]);
    };

    // To DDR
    const auto compactStrides = getCompactStrides(compactInType);
    auto newDDRType = compactInType.changeMemSpace(VPU::MemoryKind::DDR).changeStrides(compactStrides);
    auto newAllocDDROp = builder.create<mlir::memref::AllocOp>(
            appendLoc(nceClusterTiling->getLoc(), "_elt_in_place_input_DDR"), newDDRType.cast<mlir::MemRefType>());

    SmallVector<mlir::Value> ddrCopyOperands = {overwrittenInput, static_cast<mlir::Value>(newAllocDDROp)};
    auto newTillingCopyToDDR =
            builder.create<VPUIP::NCEClusterTilingOp>(appendLoc(nceClusterTiling->getLoc(), "_unique_consumer_spill"),
                                                      newDDRType, ddrCopyOperands, copyOutBodyBuilder);
    // To CMX
    auto newDistributedBuff = builder.create<VPURT::AllocDistributed>(
            appendLoc(nceClusterTiling->getLoc(), "_elt_in_place_input_CMX"), inDistributedType, nullptr, nullptr);
    SmallVector<mlir::Value> cmxCopyOperands = {newTillingCopyToDDR->getResult(0),
                                                static_cast<mlir::Value>(newDistributedBuff)};
    auto newTillingCopyToCMX = builder.create<VPUIP::NCEClusterTilingOp>(nceClusterTiling->getLoc(), inDistributedType,
                                                                         cmxCopyOperands, copyOutBodyBuilder);

    overwrittenInput.replaceUsesWithIf(newTillingCopyToCMX->getResult(0), [&](mlir::OpOperand& opOperand) {
        return opOperand.getOwner() == nceClusterTiling;
    });

    auto eltwiseOutput = VPUIP::getLayerOutputs(nceClusterTiling)[0];
    eltwiseOutput.replaceUsesWithIf(newDistributedBuff->getResult(0), [&](mlir::OpOperand& opOperand) {
        return opOperand.getOwner() == nceClusterTiling;
    });
}

void InsertCopyForEltwiseInPlaceInputPass::safeRunOnFunc() {
    auto& aliasesInfo = getAnalysis<AliasesInfo>();
    auto func = getOperation();

    func->walk([&](VPUIP::NCEClusterTaskOp op) {
        auto possibleInputToOverwrite = getInputToOverwrite(op, aliasesInfo);
        if (!possibleInputToOverwrite.has_value()) {
            return;
        }

        auto nestedLog = _log.nest();
        auto overwrittenInput = possibleInputToOverwrite.value();

        // Check that there are no other consumers up until the first producer compute op (NCEClusterTask/SWOp) or the
        // first Copy op
        if (VPUIP::isEltwiseTheOnlyConsumer(op, overwrittenInput, /*checkThroughCopyOps=*/false, nestedLog)) {
            nestedLog.trace("Input of in place Eltwise can be safely overwritten by the output. in = {0}",
                            overwrittenInput);
            return;
        }

        nestedLog.trace("Input buffer of in place Eltwise {0} has another consumer that might get overwritten by "
                        "Eltwise's output.",
                        op.getLoc());

        auto clusterTilingParent = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(op->getParentOp());
        if (clusterTilingParent == nullptr) {
            insertCopies(op, overwrittenInput);
        } else {
            insertCopiesClustered(clusterTilingParent, overwrittenInput);
        }

        nestedLog.trace("Inserted spilling copies.");
    });
}

}  // namespace

//
// createInsertCopyForEltwiseInPlaceInputPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createInsertCopyForEltwiseInPlaceInputPass(Logger log) {
    return std::make_unique<InsertCopyForEltwiseInPlaceInputPass>(log);
}
