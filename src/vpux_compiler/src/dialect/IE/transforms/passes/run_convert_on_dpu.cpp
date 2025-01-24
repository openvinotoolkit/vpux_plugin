//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/core/type_interfaces.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/pooling_utils.hpp"

using namespace vpux;

//
// RunF16ToF32ConvertOnDPUPass
//

class RunF16ToF32ConvertOnDPUPass final : public IE::RunF16ToF32ConvertOnDPUBase<RunF16ToF32ConvertOnDPUPass> {
public:
    explicit RunF16ToF32ConvertOnDPUPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    void fuseWithParentDPUOp(IE::ConvertOp convert, mlir::Operation* parentOp);
};

void RunF16ToF32ConvertOnDPUPass::fuseWithParentDPUOp(IE::ConvertOp convert, mlir::Operation* parentOp) {
    _log.nest().debug("F16 -> F32 Convert will be fused with parent DPU op at loc {0}", parentOp->getLoc());

    auto parentOpOutputType = mlir::cast<NDTypeInterface>(parentOp->getResult(0).getType());
    auto outElemType = mlir::cast<NDTypeInterface>(convert.getOutput().getType()).getElementType();

    parentOp->getResult(0).setType(parentOpOutputType.changeElemType(outElemType));

    convert->replaceAllUsesWith(parentOp->getResults());
    convert->erase();
}

void RunF16ToF32ConvertOnDPUPass::safeRunOnFunc() {
    auto func = getOperation();

    auto nestedLog = _log.nest();
    SmallVector<IE::ConvertOp> f16Tof32Converts = {};
    for (auto convertOp : func.getOps<IE::ConvertOp>()) {
        _log.debug("Got '{0}' at '{1}'", convertOp->getName(), convertOp->getLoc());

        auto inputElemType = mlir::cast<NDTypeInterface>(convertOp.getInput().getType()).getElementType();
        auto outputElemType = mlir::cast<NDTypeInterface>(convertOp.getOutput().getType()).getElementType();

        if (!mlir::isa<mlir::Float16Type>(inputElemType) || !mlir::isa<mlir::Float32Type>(outputElemType)) {
            nestedLog.trace("Not a FP16 -> FP32 Convert.");
            continue;
        }

        auto parentOp = convertOp.getInput().getDefiningOp();
        if (parentOp == nullptr || !parentOp->hasOneUse()) {
            nestedLog.trace("No parent op or parent has more than one use.");
            continue;
        }

        auto parentInputType = mlir::cast<NDTypeInterface>(parentOp->getOperand(0).getType());
        if (!mlir::isa<mlir::FloatType>(parentInputType.getElementType())) {
            nestedLog.trace("Parent input is not a float type = {0}.", parentInputType.getElementType());
            continue;
        }

        if (mlir::failed(VPU::NCEInvariant::isSupported(parentOp))) {
            nestedLog.trace("Parent op of type {0} at loc {1} is not a supported DPU op.", parentOp->getName(),
                            parentOp->getLoc());
            continue;
        }

        if (mlir::isa<IE::MaxPoolOp>(parentOp)) {
            nestedLog.trace("Parent op of type {0} at loc {1} does not support FP32 output.", parentOp->getName(),
                            parentOp->getLoc());
            continue;
        }

        if (auto postOpIf = mlir::dyn_cast<IE::LayerWithPostOpInterface>(parentOp)) {
            if (postOpIf.getPostOp().has_value()) {
                auto postOp = postOpIf.getPostOp().value();
                if (postOp.getStringRef() == IE::ClampOp::getOperationName()) {
                    nestedLog.trace("Parent op of type {0} at loc {1} has Clamp post op.", parentOp->getName(),
                                    parentOp->getLoc());
                    continue;
                }
            }
        }

        f16Tof32Converts.emplace_back(convertOp);
    }

    for (auto eligibleConvert : llvm::make_early_inc_range(f16Tof32Converts)) {
        auto parentOp = eligibleConvert.getInput().getDefiningOp();
        fuseWithParentDPUOp(eligibleConvert, parentOp);
    }
}

//
// RunF16ToF32ConvertOnDPU
//

std::unique_ptr<mlir::Pass> vpux::IE::createRunF16ToF32ConvertOnDPUPass(Logger log) {
    return std::make_unique<RunF16ToF32ConvertOnDPUPass>(log);
}
