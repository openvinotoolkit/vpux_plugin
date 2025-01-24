//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/barrier_rewriter.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

namespace vpux {
namespace vpumi40xx2vpuasm {

mlir::FailureOr<SymbolizationResult> BarrierRewriter::symbolize(VPUMI40XX::ConfigureBarrierOp op, SymbolMapper&,
                                                                mlir::ConversionPatternRewriter& rewriter) const {
    auto result = op.getResult();
    auto symName = findSym(result).getRootReference();
    auto taskIdx = mlir::TypeAttr::get(op.getType());

    mlir::Operation* newOperation = nullptr;
    if (_enablePWLM) {
        llvm::SmallVector<VPURegMapped::EnqueueOp> enqueueUsers;
        for (auto user : result.getUsers()) {
            auto enqu = mlir::dyn_cast<VPURegMapped::EnqueueOp>(user);
            if (enqu) {
                enqueueUsers.push_back(enqu);
            }
        }
        mlir::TypeAttr workItemIndex = nullptr;
        if (!enqueueUsers.empty()) {
            auto firstEnqu = std::min_element(enqueueUsers.begin(), enqueueUsers.end(),
                                              [](VPURegMapped::EnqueueOp lhs, VPURegMapped::EnqueueOp rhs) {
                                                  return lhs.getType().getValue() < rhs.getType().getValue();
                                              });

            workItemIndex = mlir::TypeAttr::get(firstEnqu->getType());
        }
        auto newOp = rewriter.create<VPUASM::ManagedBarrierOp>(
                op.getLoc(), symName, taskIdx, workItemIndex, rewriter.getUI32IntegerAttr(enqueueUsers.size()),
                op.getIdAttr(), op.getNextSameIdAttr(), op.getProducerCountAttr(), op.getConsumerCountAttr());
        newOperation = newOp.getOperation();
    } else {
        auto newOp = rewriter.create<VPUASM::ConfigureBarrierOp>(op.getLoc(), symName, taskIdx, nullptr, op.getIdAttr(),
                                                                 op.getNextSameIdAttr(), op.getProducerCountAttr(),
                                                                 op.getConsumerCountAttr());
        newOperation = newOp.getOperation();
    }

    rewriter.eraseOp(op);

    return SymbolizationResult(newOperation);
}

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux
