//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"

#include "vpux/compiler/core/aliases_info.hpp"

#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {

template <typename OpTy>
mlir::Value replaceCastWith(mlir::Operation* op, mlir::Value sourceRoot, mlir::Value inputValue) {
    mlir::OpBuilder builder(op);
    builder.setInsertionPoint(sourceRoot.getDefiningOp());
    auto newOperation = builder.create<OpTy>(op->getLoc(), sourceRoot.getType(), inputValue);
    return newOperation.getResult();
};

void fuseLastCopy(VPUIP::CopyOp copyOp, const AliasesInfo& aliasesInfo, Logger log) {
    log.trace("fuseLastCopy: Copy at {0}", copyOp->getLoc());
    auto nestedLogger = log.nest();

    auto inSourceMemory = copyOp.getInput().getType().cast<vpux::NDTypeInterface>().getMemoryKind();
    auto outSourceMemory = copyOp.getOutput().getType().cast<vpux::NDTypeInterface>().getMemoryKind();
    if (inSourceMemory != outSourceMemory) {
        nestedLogger.trace("Cannot match because the copy is not within the same memory space");
        return;
    }

    auto sourceOp = copyOp.getInput().getDefiningOp();
    if (sourceOp == nullptr) {
        nestedLogger.trace("Cannot match: copy input is a block argument");
        return;
    }

    const auto sourceRoots = aliasesInfo.getRoots(copyOp.getInput());
    if (sourceRoots.size() != 1) {
        nestedLogger.trace("Cannot match: expected single source root for {0} but got {1}", copyOp.getInput(),
                           sourceRoots.size());
        return;
    }

    const auto sourceRoot = *sourceRoots.begin();
    if (sourceRoot == nullptr || mlir::isa<mlir::BlockArgument>(sourceRoot)) {
        nestedLogger.trace("Cannot match: input is a block argument");
        return;
    }
    const auto sourceRootOp = sourceRoot.getDefiningOp();
    if (!isBufAllocOp(sourceRootOp)) {
        nestedLogger.trace("Cannot match: input isn't allocate op but '{0}'", sourceRootOp->getName());
        return;
    }

    auto allRootAliases = aliasesInfo.getAllAliases(sourceRoot);
    for (auto alias : allRootAliases) {
        for (auto userOp : alias.getUsers()) {
            if (auto copyUserOp = mlir::dyn_cast<VPUIP::CopyOp>(userOp)) {
                if (copyUserOp != copyOp && copyUserOp.getOutputBuff().isa<mlir::BlockArgument>()) {
                    nestedLogger.trace("Cannot fuse when there are multiple output copy operations");
                    return;
                }
            }
        }
    }

    VPUIP::ConcatViewOp concatViewOp;
    auto newBuffer = copyOp.getOutputBuff();
    auto newOutput = copyOp.getInput();

    if (sourceRoot.getType() != copyOp.getOutputBuff().getType()) {
        // check what operation changes the type
        auto typeCastOp = copyOp.getInput().getDefiningOp();

        if (typeCastOp == nullptr || std::distance(typeCastOp->getUsers().begin(), typeCastOp->getUsers().end()) != 1) {
            // skip if typeCastOp has multi users
            return;
        }

        if (mlir::isa<VPUIP::GenericReshapeOp, VPUIP::QuantizeCastOp>(typeCastOp)) {
            auto preOfTypeCastOp = typeCastOp->getOperand(0).getDefiningOp();
            while (mlir::isa<VPUIP::GenericReshapeOp, VPUIP::QuantizeCastOp, VPUIP::PermuteCastOp>(preOfTypeCastOp)) {
                if (!preOfTypeCastOp->hasOneUse()) {
                    return;
                }
                typeCastOp = preOfTypeCastOp;
                preOfTypeCastOp = preOfTypeCastOp->getOperand(0).getDefiningOp();
            }
            if (!mlir::isa<VPUIP::ConcatViewOp, VPUIP::CopyOp>(preOfTypeCastOp)) {
                nestedLogger.trace("Cannot match because of missed concat in case");
                return;
            }
            concatViewOp = mlir::dyn_cast<VPUIP::ConcatViewOp>(preOfTypeCastOp);
            if (concatViewOp && !concatViewOp.getOutput().hasOneUse()) {
                return;
            }
        }

        // we will make a OpTy(QuantizeCast/GenericReshape) over the output buffer and we will copy from CMX directly
        // to output buffer, and we will return the output buffer. After ConcatView and OpTy will be redundant.
        // from CMX -> CopyOp[DDR] -> (ConcatViewOp) -> OpTy -> CopyOp[block-arg] -> return CopyOp
        // Output of this step:
        //                        CMX -> CopyOp[OpTy] -> return block-arg
        //   block-arg -> OpTy /
        if (mlir::isa<VPUIP::GenericReshapeOp>(typeCastOp)) {
            newBuffer = replaceCastWith<VPUIP::GenericReshapeOp>(typeCastOp, sourceRoot, copyOp.getOutputBuff());
        } else if (mlir::isa<VPUIP::QuantizeCastOp>(typeCastOp)) {
            newBuffer = replaceCastWith<VPUIP::QuantizeCastOp>(typeCastOp, sourceRoot, copyOp.getOutputBuff());
        } else if (auto permuteCastOp = mlir::dyn_cast<VPUIP::PermuteCastOp>(typeCastOp)) {
            // do the permute in output
            mlir::OpBuilder builder(permuteCastOp);
            builder.setInsertionPoint(sourceRoot.getDefiningOp());

            auto newPermuteCast = builder.create<VPUIP::PermuteCastOp>(
                    permuteCastOp.getLoc(), sourceRoot.getType(), copyOp.getOutputBuff(),
                    permuteCastOp.getDstOrderAttr(), permuteCastOp.getMemPermAttr());

            newBuffer = newPermuteCast.getResult();
        } else {
            nestedLogger.trace("Cannot match because of missed concat in generic branch");
            return;
        }

        auto childTypeCast = *typeCastOp->getResult(0).getUsers().begin();
        if (mlir::isa<VPUIP::GenericReshapeOp, VPUIP::QuantizeCastOp, VPUIP::PermuteCastOp>(typeCastOp)) {
            childTypeCast->setOperand(0, newBuffer);
        }
        typeCastOp->replaceAllUsesWith(typeCastOp->getOperands());
        typeCastOp->erase();
        newOutput = copyOp.getOutputBuff();
    }

    // Function outputs have to be an alias of the output buffer
    log.trace("Root of the copy operation input {0}", sourceRoot);
    log.trace("Reassign outputs from {0} to {1}", sourceRoot, newBuffer);

    for (auto& use : llvm::make_early_inc_range(sourceRoot.getUses())) {
        log.nest().trace("Got user {0}", use.getOwner()->getName());
        log.nest().trace("Reassign {0} to {1}", use.get(), newBuffer);
        use.set(newBuffer);
    }

    copyOp.replaceAllUsesWith(newOutput);
    copyOp->erase();
    if (concatViewOp) {
        concatViewOp->erase();
    }

    if (sourceRootOp->use_empty()) {
        sourceRootOp->erase();
    }
}

//
// FuseLastCopy
//

class FuseLastCopyPass final : public VPUIP::FuseLastCopyBase<FuseLastCopyPass> {
public:
    explicit FuseLastCopyPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void FuseLastCopyPass::safeRunOnFunc() {
    auto func = getOperation();

    func->walk([&](VPUIP::CopyOp op) {
        if (!op.getOutputBuff().isa<mlir::BlockArgument>()) {
            return;
        }

        auto& aliasInfo = getAnalysis<AliasesInfo>();
        fuseLastCopy(op, aliasInfo, _log);
    });
}

}  // namespace

//
// createFuseLastCopyPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createFuseLastCopyPass(Logger log) {
    return std::make_unique<FuseLastCopyPass>(log);
}
