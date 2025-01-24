//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"

using namespace vpux;

namespace {

class LinkAllOpsPass : public VPUMI40XX::LinkAllOpsBase<LinkAllOpsPass> {
public:
    explicit LinkAllOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void LinkAllOpsPass::safeRunOnFunc() {
    auto netFunc = getOperation();

    for (auto taskOp : netFunc.getOps<VPURegMapped::TaskOpInterface>()) {
        auto index = taskOp.getIndexType();

        if (taskOp.getTaskType() == VPURegMapped::TaskType::ActKernelInvocation) {
            // shave linked lists are disabled on non-WLM path
            // as they require FW changes that break compatiblity
            continue;
        }

        if ((index.getValue() != 0) && taskOp.supportsTaskLink()) {
            taskOp.linkToPreviousTask();
        }
    }

    return;
}
}  // namespace

//
// createLinkEnqueueTargetsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUMI40XX::createLinkAllOpsPass(Logger log) {
    return std::make_unique<LinkAllOpsPass>(log);
}
