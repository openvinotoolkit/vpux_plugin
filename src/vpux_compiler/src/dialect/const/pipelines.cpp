//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/const/passes.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

void Const::registerConstPipelines() {
    mlir::PassPipelineRegistration<>(
            "constant-folding-pipeline", "Constant folding pipeline", [](mlir::OpPassManager& pm) {
                pm.nest<mlir::func::FuncOp>().addNestedPass<Const::DeclareOp>(Const::createConstantFoldingPass());
            });
}
