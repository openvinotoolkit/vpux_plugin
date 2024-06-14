#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"

using namespace vpux;

namespace {

class LinkEnqueueTargetsPass : public VPUMI40XX::LinkEnqueueTargetsBase<LinkEnqueueTargetsPass> {
public:
    explicit LinkEnqueueTargetsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void LinkEnqueueTargetsPass::safeRunOnFunc() {
    auto netFunc = getOperation();

    for (auto enqueue : netFunc.getOps<VPURegMapped::EnqueueOp>()) {
        if (enqueue.getStart() == enqueue.getEnd()) {
            continue;
        }

        auto start = mlir::cast<VPURegMapped::TaskOpInterface>(enqueue.getStart().getDefiningOp());
        auto end = mlir::cast<VPURegMapped::TaskOpInterface>(enqueue.getEnd().getDefiningOp());

        if (!end.supportsHardLink()) {
            continue;
        }

        while (end != start) {
            end.enableHardLink();
            end = end.getPreviousTask();
        }

        // if we've hard-linked all n+1th tasks, then we only have to enqueue the first task
        enqueue.getEndMutable().assign(start.getResult());
    }

    return;
}
}  // namespace

//
// createLinkEnqueueTargetsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUMI40XX::createLinkEnqueueTargetsPass(Logger log) {
    return std::make_unique<LinkEnqueueTargetsPass>(log);
}
