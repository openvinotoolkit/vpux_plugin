#include "vpux/compiler/dialect/const/passes.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

void Const::registerConstPipelines() {
    mlir::PassPipelineRegistration<>("constant-folding-pipe", "Constant folding pipeline", [](mlir::OpPassManager& pm) {
        mlir::OpPassManager& constPm = pm.nest<mlir::func::FuncOp>().nest<Const::DeclareOp>();
        constPm.addPass(Const::createConstantFoldingPass());
    });
}