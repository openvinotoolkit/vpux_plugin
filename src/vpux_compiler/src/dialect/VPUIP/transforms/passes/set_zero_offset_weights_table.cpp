//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;
using namespace VPUIP;

namespace {

//
// SetZeroOffsetWeightsTablePass
//

class SetZeroOffsetWeightsTablePass final : public VPUIP::SetZeroOffsetWeightsTableBase<SetZeroOffsetWeightsTablePass> {
public:
    explicit SetZeroOffsetWeightsTablePass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void SetZeroOffsetWeightsTablePass::safeRunOnFunc() {
    auto func = getOperation();

    func.walk([&](VPUIP::NCEClusterTaskOp nceOp) {
        if (nceOp.getWeightTable() == nullptr) {
            return;
        }
        if (nceOp.getWeightsSparsityMap() != nullptr) {
            return;
        }
        _log.trace("Got '{0}' at '{1}'", nceOp->getName(), nceOp->getLoc());
        nceOp.setIsZeroOffsetWeightsTable(true);
    });
}

}  // namespace

//
// createSetZeroOffsetWeightsTablePass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createSetZeroOffsetWeightsTablePass(Logger log) {
    return std::make_unique<SetZeroOffsetWeightsTablePass>(log);
}
