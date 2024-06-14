//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/passes.hpp"

#include "vpux/compiler/utils/locations_verifier.hpp"

#include "vpux/utils/core/error.hpp"

using namespace vpux;

namespace {

//
// SetupLocationVerifierPass
//

class SetupLocationVerifierPass final : public SetupLocationVerifierBase<SetupLocationVerifierPass> {
public:
    SetupLocationVerifierPass(Logger log, LocationsVerifierMarker marker, LocationsVerificationMode mode)
            : _mode(mode), _marker(marker) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;
    void safeRunOnModule() final;

private:
    LocationsVerificationMode _mode;
    LocationsVerifierMarker _marker;
};

mlir::LogicalResult SetupLocationVerifierPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    if (!mode.hasValue()) {
        return mlir::success();
    }
    _mode = vpux::getLocationsVerificationMode(mode);
    return mlir::success();
}

void SetupLocationVerifierPass::safeRunOnModule() {
    auto moduleOp = getOperation();
    const auto currentMode = vpux::getLocationsVerificationMode(moduleOp);
    // Running full verification if verification was enabled before
    if (_marker == LocationsVerifierMarker::END && currentMode != LocationsVerificationMode::OFF) {
        const auto verificationResult = vpux::verifyLocationsUniquenessFull(moduleOp, getName());
        if (mlir::failed(verificationResult)) {
            signalPassFailure();
        }
    }
    vpux::setLocationsVerificationMode(moduleOp, _mode);
}

}  // namespace

//
// createSetupLocationVerifierPass
//

std::unique_ptr<mlir::Pass> vpux::createSetupLocationVerifierPass(Logger log) {
    return std::make_unique<SetupLocationVerifierPass>(log, LocationsVerifierMarker::BEGIN,
                                                       LocationsVerificationMode::OFF);
}

std::unique_ptr<mlir::Pass> vpux::createStartLocationVerifierPass(
        vpux::Logger log, const mlir::detail::PassOptions::Option<std::string>& locationsVerificationMode) {
    const auto mode = getLocationsVerificationMode(locationsVerificationMode);
    return std::make_unique<SetupLocationVerifierPass>(log, LocationsVerifierMarker::BEGIN, mode);
}

std::unique_ptr<mlir::Pass> vpux::createStopLocationVerifierPass(vpux::Logger log) {
    return std::make_unique<SetupLocationVerifierPass>(log, LocationsVerifierMarker::END,
                                                       LocationsVerificationMode::OFF);
}
