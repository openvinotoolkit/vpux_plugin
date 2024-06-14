#include <mlir/Transforms/DialectConversion.h>
#include <vector>
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/utils/dma.hpp"

#include <npu_40xx_nnrt.hpp>

using namespace vpux;
using namespace npu40xx;

namespace {
// TODO: E111344
class NextSameIdAssignmentPass : public VPUMI40XX::NextSameIdAssignmentBase<NextSameIdAssignmentPass> {
public:
    explicit NextSameIdAssignmentPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void NextSameIdAssignmentPass::safeRunOnFunc() {
    auto funcOp = getOperation();
    mlir::MLIRContext* ctx = &(getContext());

    auto MAX_PID = VPUIP::getNumAvailableBarriers(funcOp);
    std::vector<std::list<size_t>> nextSameID(MAX_PID);
    std::vector<bool> touched(MAX_PID, false);

    for (auto op : funcOp.getOps<VPUMI40XX::ConfigureBarrierOp>()) {
        auto opIndexType = op.getOperation()->getResult(0).getType().cast<VPURegMapped::IndexType>();

        auto vid = opIndexType.getValue();

        // need to skip the first
        if (touched[op.getId()]) {
            nextSameID[op.getId()].push_back(vid);
        } else {
            touched[op.getId()] = true;
        }
    }

    for (auto op : funcOp.getOps<VPUMI40XX::ConfigureBarrierOp>()) {
        int64_t newNextSameID = -1;

        if (!nextSameID[op.getId()].empty()) {
            newNextSameID = nextSameID[op.getId()].front();
            nextSameID[op.getId()].pop_front();
        }

        auto newNextSameIDAttr =
                mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Signed), newNextSameID);

        op.setNextSameIdAttr(newNextSameIDAttr);
    }
}

}  // namespace

//
// createNextSameIdAssignmentPass
//

std::unique_ptr<mlir::Pass> vpux::VPUMI40XX::createNextSameIdAssignmentPass(Logger log) {
    return std::make_unique<NextSameIdAssignmentPass>(log);
}
