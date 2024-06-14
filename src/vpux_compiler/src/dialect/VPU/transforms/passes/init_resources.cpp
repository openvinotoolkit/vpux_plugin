//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"

using namespace vpux;

namespace {

//
// InitResourcesPass
//

class InitResourcesPass final : public VPU::InitResourcesBase<InitResourcesPass> {
public:
    InitResourcesPass() = default;
    InitResourcesPass(const VPU::InitCompilerOptions& initCompilerOptions, Logger log);

private:
    mlir::LogicalResult initializeOptions(StringRef options) final;
    void safeRunOnModule() final;

private:
    // Initialize fields from pass options
    void initializeFromOptions();

private:
    VPU::ArchKind _arch = VPU::ArchKind::UNKNOWN;
    VPU::CompilationMode _compilationMode = VPU::CompilationMode::DefaultHW;
    std::optional<int> _revisionID;
    std::optional<int> _numOfDPUGroups;
    std::optional<int> _numOfDMAPorts;
    std::optional<vpux::Byte> _availableCMXMemory;
    bool _allowCustomValues = false;
};

InitResourcesPass::InitResourcesPass(const VPU::InitCompilerOptions& initCompilerOptions, Logger log) {
    Base::initLogger(log, Base::getArgumentName());
    Base::copyOptionValuesFrom(initCompilerOptions);

    initializeFromOptions();
}

mlir::LogicalResult InitResourcesPass::initializeOptions(StringRef options) {
    if (mlir::failed(Base::initializeOptions(options))) {
        return mlir::failure();
    }

    initializeFromOptions();

    return mlir::success();
}

void InitResourcesPass::initializeFromOptions() {
    auto archStr = VPU::symbolizeEnum<VPU::ArchKind>(archOpt.getValue());
    if (!archStr.has_value()) {
        // TODO: Remove after #84053
        auto deprecatedArchKind = vpux::VPU::symbolizeDeprecatedArchKind(archOpt.getValue());
        VPUX_THROW_UNLESS(deprecatedArchKind.has_value(), "Unknown VPU architecture : '{0}'", archOpt.getValue());

        _arch = mapDeprecatedArchKind(deprecatedArchKind.value());
    } else {
        _arch = archStr.value();
    }

    auto compilationModeStr = VPU::symbolizeEnum<VPU::CompilationMode>(compilationModeOpt.getValue());
    VPUX_THROW_UNLESS(compilationModeStr.has_value(), "Unknown compilation mode: '{0}'", compilationModeOpt.getValue());
    _compilationMode = compilationModeStr.value();

    if (revisionIDOpt.hasValue()) {
        _revisionID = revisionIDOpt.getValue();
    }

    if (numberOfDPUGroupsOpt.hasValue()) {
        _numOfDPUGroups = numberOfDPUGroupsOpt.getValue();
    }

    if (numberOfDMAPortsOpt.hasValue()) {
        _numOfDMAPorts = numberOfDMAPortsOpt.getValue();
    }

    if (availableCMXMemoryOpt.hasValue()) {
        _availableCMXMemory = Byte(static_cast<double>(availableCMXMemoryOpt.getValue()));
    }

    if (allowCustomValues.hasValue()) {
        _allowCustomValues = allowCustomValues.getValue();
    }
}

void InitResourcesPass::safeRunOnModule() {
    auto module = getOperation();

    _log.trace("Set VPU architecture to {0}", _arch);
    VPU::setArch(module, _arch, _numOfDPUGroups, _numOfDMAPorts, _availableCMXMemory, _allowCustomValues);

    VPUX_THROW_WHEN(!_allowCustomValues && VPU::hasCompilationMode(module),
                    "CompilationMode is already defined, probably you run '--init-compiler' twice");
    if (!VPU::hasCompilationMode(module)) {
        _log.trace("Set compilation mode to {0}", _compilationMode);
        VPU::setCompilationMode(module, _compilationMode);
    }

    VPUX_THROW_WHEN(!_allowCustomValues && VPU::hasRevisionID(module),
                    "RevisionID is already defined, probably you run '--init-compiler' twice");
    if (!VPU::hasRevisionID(module)) {
        if (_revisionID.has_value()) {
            int revisionIDValue = _revisionID.value();
            std::optional<VPU::RevisionID> revID = VPU::symbolizeRevisionID(revisionIDValue);
            if (revID.has_value()) {
                _log.trace("Set RevisionID to {0}", revisionIDValue);
                VPU::setRevisionID(module, revID.value());
            } else {
                _log.trace("Set RevisionID to REVISION_NONE");
                VPU::setRevisionID(module, VPU::RevisionID::REVISION_NONE);
            }
        } else {
            _log.trace("Set RevisionID to REVISION_NONE");
            VPU::setRevisionID(module, VPU::RevisionID::REVISION_NONE);
        }
    }
}

}  // namespace

//
// createInitResourcesPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createInitResourcesPass() {
    return std::make_unique<InitResourcesPass>();
}

std::unique_ptr<mlir::Pass> vpux::VPU::createInitResourcesPass(const InitCompilerOptions& initCompilerOptions,
                                                               Logger log) {
    return std::make_unique<InitResourcesPass>(initCompilerOptions, log);
}
