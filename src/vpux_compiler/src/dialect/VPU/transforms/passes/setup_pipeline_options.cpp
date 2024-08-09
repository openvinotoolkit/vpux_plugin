//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/setup_pipeline_options_utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/utils/core/error.hpp"
using namespace vpux;

namespace {

//
// SetupPipelineOptionsPass
//

class SetupPipelineOptionsPass final : public VPU::SetupPipelineOptionsBase<SetupPipelineOptionsPass> {
public:
    SetupPipelineOptionsPass() = default;
    SetupPipelineOptionsPass(const VPU::InitCompilerOptions& initCompilerOptions, Logger log) {
        Base::initLogger(log, Base::getArgumentName());
        Base::copyOptionValuesFrom(initCompilerOptions);

        initializeFromOptions();
    }

private:
    mlir::LogicalResult initializeOptions(StringRef options) final;
    void safeRunOnModule() final;

private:
    // Initialize fields from pass options
    void initializeFromOptions();

private:
    bool _allowCustomValues = false;
};

mlir::LogicalResult SetupPipelineOptionsPass::initializeOptions(StringRef options) {
    if (mlir::failed(Base::initializeOptions(options))) {
        return mlir::failure();
    }

    initializeFromOptions();

    return mlir::success();
}

void SetupPipelineOptionsPass::initializeFromOptions() {
    if (allowCustomValues.hasValue()) {
        _allowCustomValues = allowCustomValues.getValue();
    }
}

void SetupPipelineOptionsPass::safeRunOnModule() {
    auto& ctx = getContext();
    auto moduleOp = getModuleOp(getOperation());

    const auto hasPipelineOptions = moduleOp.lookupSymbol<IE::PipelineOptionsOp>(VPU::PIPELINE_OPTIONS) != nullptr;
    VPUX_THROW_WHEN(!_allowCustomValues && hasPipelineOptions,
                    "PipelineOptions operation is already defined, probably you run '--init-compiler' twice");

    if (hasPipelineOptions) {
        return;
    }

    auto optionsBuilder = mlir::OpBuilder::atBlockBegin(moduleOp.getBody());
    auto pipelineOptionsOp =
            optionsBuilder.create<IE::PipelineOptionsOp>(mlir::UnknownLoc::get(&ctx), VPU::PIPELINE_OPTIONS);
    pipelineOptionsOp.getOptions().emplaceBlock();
}

}  // namespace

//
// createSetupPipelineOptionsPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createSetupPipelineOptionsPass() {
    return std::make_unique<SetupPipelineOptionsPass>();
}

std::unique_ptr<mlir::Pass> vpux::VPU::createSetupPipelineOptionsPass(
        const VPU::InitCompilerOptions& initCompilerOptions, Logger log) {
    return std::make_unique<SetupPipelineOptionsPass>(initCompilerOptions, log);
}
