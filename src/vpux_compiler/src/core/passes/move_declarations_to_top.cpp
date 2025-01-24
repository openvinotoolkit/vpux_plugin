//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/passes.hpp"

#include "vpux/compiler/core/ops_interfaces.hpp"

#include "vpux/utils/core/range.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>

#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"

using namespace vpux;

namespace {

//
// MoveDeclarationsToTopPass
//

class MoveDeclarationsToTopPass final : public MoveDeclarationsToTopBase<MoveDeclarationsToTopPass> {
public:
    explicit MoveDeclarationsToTopPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void MoveDeclarationsToTopPass::safeRunOnFunc() {
    auto func = getOperation();
    VPUIP::moveDeclarationsToTop(func);
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::createMoveDeclarationsToTopPass(Logger log) {
    return std::make_unique<MoveDeclarationsToTopPass>(log);
}
