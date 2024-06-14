#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"

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

        if ((index.getValue() != 0) && taskOp.supportsHardLink()) {
            taskOp.enableHardLink();
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
