//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/ELF/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/ELF/passes.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

#include <vpux_elf/types/vpu_extensions.hpp>

using namespace vpux;

namespace {

class AddInnerSectionPaddingPass : public ELF::AddInnerSectionPaddingBase<AddInnerSectionPaddingPass> {
public:
    explicit AddInnerSectionPaddingPass(Logger log): _log(log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    Logger _log;
};

void AddInnerSectionPaddingPass::safeRunOnFunc() {
    auto netFunc = getOperation();
    auto moduleOp = netFunc.getOperation()->getParentOfType<mlir::ModuleOp>();
    VPUX_THROW_UNLESS(moduleOp, "The top-level module is missing");
    const auto arch = VPU::getArch(moduleOp);

    auto mainOps = to_small_vector(netFunc.getOps<ELF::MainOp>());
    VPUX_THROW_UNLESS(mainOps.size() == 1, "Expected exactly one ELF mainOp. Got {0}", mainOps.size());
    auto elfMain = mainOps[0];

    ELF::SymbolReferenceMap symRefMap(elfMain, true);

    for (auto section : elfMain.getOps<ELF::ElfSectionInterface>()) {
        auto block = section.getBlock();
        if (block->empty()) {
            continue;
        }
        auto builder = mlir::OpBuilder::atBlockEnd(block);
        size_t offsetTracker = 0;
        for (auto wrappableOp : block->getOps<ELF::WrappableOpInterface>()) {
            const auto wrappableOperation = wrappableOp.getOperation();
            const auto alignmentRequirement = wrappableOp.getAlignmentRequirements(arch);
            size_t paddingRequired = offsetTracker % alignmentRequirement;
            if (paddingRequired) {
                builder.setInsertionPoint(wrappableOperation);
                auto paddingSize = alignmentRequirement - paddingRequired;
                builder.template create<ELF::PadOp>(builder.getUnknownLoc(), paddingSize, nullptr);
                offsetTracker += paddingSize;
            }
            offsetTracker +=
                    mlir::cast<ELF::BinaryOpInterface>(wrappableOperation).getBinarySizeCached(symRefMap, arch);
        }
    }
}
}  // namespace

//
// createAddInnerSectionPadding
//

std::unique_ptr<mlir::Pass> vpux::ELF::createAddInnerSectionPaddingPass(Logger log) {
    return std::make_unique<AddInnerSectionPaddingPass>(log);
}
