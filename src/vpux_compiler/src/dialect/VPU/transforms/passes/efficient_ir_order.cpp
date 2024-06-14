//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/utils/analysis.hpp"

#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;
using namespace VPU;

namespace {

//
// EfficientIROrderPass
//

class EfficientIROrderPass final : public EfficientIROrderBase<EfficientIROrderPass> {
public:
    explicit EfficientIROrderPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnModule
//

void EfficientIROrderPass::safeRunOnFunc() {
    auto func = getOperation();

    auto operationsInBlock = to_small_vector(func.getOps<VPU::NCEOpInterface>());

    // for NCE ops move it close to their users or parents:
    // for two-inputs operations place it right after the last parent in order not to
    // execute it later than both inputs are ready. Usually misalignment happens
    // when one of the input is executed too early
    //               CONV0
    //         |             |
    //       CONV1         CONV2
    //                       |
    //                     CONV3
    //         |             |
    //               ...
    //             ELTWISE
    // so, it would have been more beneficial to execute eltwise earlier just after convolutions
    // for other operations, place them close to users in order to execute them just before
    // they are needed

    for (auto origOp : operationsInBlock | reversed) {
        if (origOp->hasTrait<VPU::EltwiseOp>()) {
            SmallVector<mlir::Operation*> parents;
            for (auto operand : origOp->getOperands()) {
                if (auto parentOp = operand.getDefiningOp()) {
                    parents.push_back(parentOp);
                }
            }
            if (!parents.empty()) {
                llvm::sort(parents, [](auto* lhs, auto* rhs) {
                    return lhs->isBeforeInBlock(rhs);
                });
                origOp->moveAfter(parents.back());
            }

            continue;
        }

        auto* firstUser = getFirstUser(origOp->getResult(0));

        if (firstUser != nullptr) {
            origOp->moveBefore(firstUser);
        }
    }
}

}  // namespace

//
// createEfficientIROrderPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createEfficientIROrderPass(Logger log) {
    return std::make_unique<EfficientIROrderPass>(log);
}
