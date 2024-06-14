//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/ELF/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/ELF/passes.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

#include <vpux_elf/types/vpu_extensions.hpp>

#include <mlir/IR/SymbolTable.h>

using namespace vpux;

namespace {

class SetOpOffsetsPass : public ELF::SetOpOffsetsBase<SetOpOffsetsPass> {
public:
    explicit SetOpOffsetsPass(Logger log): _log(log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    Logger _log;
};

void SetOpOffsetsPass::safeRunOnFunc() {
    auto netFunc = getOperation();
    mlir::MLIRContext* ctx = &getContext();

    auto mainOps = to_small_vector(netFunc.getOps<ELF::MainOp>());
    VPUX_THROW_UNLESS(mainOps.size() == 1, "Expected exactly one ELF mainOp. Got {0}", mainOps.size());
    auto elfMain = mainOps[0];

    ELF::SymbolReferenceMap symRefMap(elfMain, true);

    auto u64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);

    SmallVector<ELF::ElfSectionInterface> metadataSecVec;

    for (auto section : elfMain.getOps<ELF::ElfSectionInterface>()) {
        auto block = section.getBlock();

        if (section.getSectionType().value_or(ELF::SectionTypeAttr::SHT_NULL) ==
            ELF::SectionTypeAttr::VPU_SHT_CMX_METADATA) {
            metadataSecVec.push_back(section);
            continue;
        }

        uint64_t tracker = 0;
        for (auto& operation : block->getOperations()) {
            auto offsetAttr = mlir::IntegerAttr::get(u64Type, mlir::APInt(64, tracker, false));
            if (auto binaryOperation = mlir::dyn_cast_or_null<ELF::BinaryOpInterface>(&operation)) {
                tracker += binaryOperation.getBinarySizeCached(symRefMap);
            }

            // each wrappable operation is binary one, all operations in sections are wrappable
            // the reason why we need to check separately for binary and wrappable interfaces is PadOp
            // PadOp is binary operation which doesn't implement getSectionInteface method,
            // because it could be inserted into any section, after move-ops-into-section pass is executed
            // (as a part of add-inner-section-padding pass logic), however, it is a part of a section content
            // refactor E#85213
            if (auto wrappableOperation = mlir::dyn_cast_or_null<ELF::WrappableOpInterface>(&operation)) {
                wrappableOperation.setMemoryOffset(offsetAttr);
            }
        }
    }

    VPUX_THROW_WHEN(metadataSecVec.size() != 1, "Only 1 metadata section is permitted!");
    auto metadataSecBlock = metadataSecVec[0].getBlock();

    size_t offsetTracker = 0;
    size_t tileTracker = 0;
    for (auto taskBufferOp : metadataSecBlock->getOps<VPUASM::DeclareTaskBufferOp>()) {
        auto offsetAttr = mlir::IntegerAttr::get(u64Type, mlir::APInt(64, offsetTracker, false));
        if (tileTracker != taskBufferOp.getTileIndex()) {
            tileTracker = taskBufferOp.getTileIndex();
            offsetTracker = 0;
            offsetAttr = mlir::IntegerAttr::get(u64Type, mlir::APInt(64, offsetTracker, false));
            offsetTracker += taskBufferOp.getBinarySize();
        } else {
            offsetTracker += taskBufferOp.getBinarySize();
        }
        taskBufferOp.setMemoryOffset(offsetAttr);
    }
}

}  // namespace

//
// createAddELFSymbolTablePass
//

std::unique_ptr<mlir::Pass> vpux::ELF::createSetOpOffsetsPass(Logger log) {
    return std::make_unique<SetOpOffsetsPass>(log);
}
