//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"

#include <npu_40xx_nnrt.hpp>

using namespace vpux;

namespace {
class UnGroupExecutionOpsPass : public VPUMI40XX::UnGroupExecutionOpsBase<UnGroupExecutionOpsPass> {
public:
    explicit UnGroupExecutionOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void UnGroupExecutionOpsPass::safeRunOnFunc() {
    auto netFunc = getOperation();

    for (auto groupOp : llvm::make_early_inc_range(netFunc.getOps<VPURegMapped::ExecutionGroupOp>())) {
        auto block = &groupOp.getTasks().front();
        auto yield = mlir::cast<VPURegMapped::GroupYieldOp>(block->getTerminator());

        for (auto [previousTask, blockArg] : llvm::zip_equal(groupOp.getPreviousTaskIdx(), block->getArguments())) {
            blockArg.replaceAllUsesWith(previousTask);
        }

        for (auto [startIdx, listHead, endIdx, listTail] : llvm::zip_equal(
                     groupOp.getStartIndexes(), yield.getListHeads(), groupOp.getEndIndexes(), yield.getListTails())) {
            startIdx.replaceAllUsesWith(listHead);
            endIdx.replaceAllUsesWith(listTail);
        }

        yield.getOperation()->erase();
        for (auto& op : llvm::make_early_inc_range(block->getOperations())) {
            op.moveBefore(groupOp.getOperation());
        }

        groupOp.getOperation()->erase();
    }
    return;
}

}  // namespace

//
// createUnGroupExecutionOpsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUMI40XX::createUnGroupExecutionOpsPass(Logger log) {
    return std::make_unique<UnGroupExecutionOpsPass>(log);
}
