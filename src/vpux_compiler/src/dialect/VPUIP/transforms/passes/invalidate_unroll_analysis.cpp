//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"

using namespace vpux;

namespace {

//
// UnrollDMAAnalysis
//

class InvalidateUnrollDMAAnalysisPass final :
        public VPUIP::InvalidateUnrollDMAAnalysisBase<InvalidateUnrollDMAAnalysisPass> {
public:
    InvalidateUnrollDMAAnalysisPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void InvalidateUnrollDMAAnalysisPass::safeRunOnFunc() {
    // Invalidates UnrollDMAAnalysis by not calling markAnalysesPreserved
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPUIP::createInvalidateUnrollDMAAnalysisPass(Logger log) {
    return std::make_unique<InvalidateUnrollDMAAnalysisPass>(log);
}
