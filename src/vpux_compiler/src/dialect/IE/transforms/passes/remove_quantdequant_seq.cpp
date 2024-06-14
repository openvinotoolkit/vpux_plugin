//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {

//
// RemoveQuantDequantSeqPass
//

class RemoveQuantDequantSeqPass final : public IE::RemoveQuantDequantSeqBase<RemoveQuantDequantSeqPass> {
public:
    explicit RemoveQuantDequantSeqPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void RemoveQuantDequantSeqPass::safeRunOnFunc() {
    auto func = getOperation();
    // Remove remaining Quantize->Dequantize sequence to not perform explicit FakeQuantize.
    // This might have slight impact on accuracy but gives visible performance improvement
    // TODO: Evaluate possibility of replacing such sequence with ClampOp fused with DPU task
    func.walk([this](vpux::IE::QuantizeOp quantizeOp) {
        if (!quantizeOp->hasOneUse()) {
            return;
        }

        auto dequantizeOp = mlir::dyn_cast<vpux::IE::DequantizeOp>(*quantizeOp->getUsers().begin());
        if (dequantizeOp == nullptr) {
            SmallVector<mlir::Operation*> targetOps;
            mlir::Operation* operation = quantizeOp;
            _log.trace("Search target pattern for {0} at {1}", quantizeOp->getName(), quantizeOp->getLoc());
            while (operation && !operation->getUsers().empty()) {
                auto user = *(operation->getUsers().begin());

                if (!mlir::isa<IE::ElemTypeInfoOpInterface, IE::DequantizeOp>(user)) {
                    return;
                }

                if (mlir::isa<IE::ElemTypeInfoOpInterface>(user)) {
                    if (!user->hasOneUse()) {
                        return;
                    }
                    _log.trace("Push  ElemTypeInfoOpInterface {0} at {1}", user->getName(), user->getLoc());
                    targetOps.push_back(user);
                    operation = user;
                    continue;
                }

                if (mlir::isa<IE::DequantizeOp>(user)) {
                    _log.trace("Found dequantize user {0} at {1}, stop pattern searching", user->getName(),
                               user->getLoc());
                    dequantizeOp = mlir::dyn_cast<vpux::IE::DequantizeOp>(*user);
                    break;
                }
            }

            _log.trace("Capture the pattern for {0} at {1}", quantizeOp->getName(), quantizeOp->getLoc());

            //[Quantize]->[ElemTypeInfoOpInterface] ... ->[Dequantize] pattern is captured
            // Rewrite the sub-graph.
            targetOps.front()->getOpOperand(0).set(quantizeOp.getInput());
            for (auto op : targetOps) {
                inferReturnTypes(op, InferShapedTypeMode::ELEM_TYPE);
            }
            // Remove old Quantize & Dequantize ops.
            dequantizeOp.replaceAllUsesWith(targetOps.back());
            dequantizeOp.erase();
            quantizeOp.erase();
        } else {
            //[Quantize]->[Dequantize] pattern, remove it directly
            dequantizeOp.replaceAllUsesWith(quantizeOp.getInput());
        }
    });
}  // namespace

}  // namespace

//
// createRemoveQuantDequantSeqPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createRemoveQuantDequantSeqPass(Logger log) {
    return std::make_unique<RemoveQuantDequantSeqPass>(log);
}
