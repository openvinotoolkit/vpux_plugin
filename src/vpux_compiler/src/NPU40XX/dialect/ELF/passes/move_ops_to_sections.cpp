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
//
// MoveOpsIntoSectionsPass
//

class MoveOpsIntoSectionsPass final : public ELF::MoveOpsIntoSectionsBase<MoveOpsIntoSectionsPass> {
public:
    explicit MoveOpsIntoSectionsPass(Logger log): _log(log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    ELF::MainOp buildElfMainOp(mlir::func::FuncOp netFunc);
    void safeRunOnFunc() final;
    Logger _log;
};

ELF::MainOp MoveOpsIntoSectionsPass::buildElfMainOp(mlir::func::FuncOp netFunc) {
    // create the main ELF op alongside the netFunc
    auto mainBuilder = mlir::OpBuilder(netFunc.getOperation());
    auto elf = mainBuilder.create<ELF::MainOp>(netFunc.getLoc(), "ELFMain");

    // take the body of the netFunc and put everything inside ELF Op, so we avoid clone of all OPS
    elf.getContent().takeBody(netFunc.getBody());
    auto netFuncBlock = netFunc.addEntryBlock();

    elf.getOperation()->moveBefore(netFuncBlock, netFuncBlock->end());

    // as we've moved the whole function we also moved the terminator
    auto terminator = elf.getContent().front().getTerminator();
    terminator->moveAfter(elf.getOperation());

    return elf;
}

void MoveOpsIntoSectionsPass::safeRunOnFunc() {
    auto netFunc = getOperation();

    // Need to define the DenseMapInfo
    std::unordered_map<ELF::SectionSignature, ELF::ElfSectionInterface> sectionMap;

    mlir::OpBuilder builderFunc(&(netFunc.getBody().front().back()));

    builderFunc.create<ELF::PerformanceMetricsOp>(builderFunc.getUnknownLoc());

    auto elf = buildElfMainOp(netFunc);

    mlir::SymbolTableCollection collection;
    auto symbolUserMap = mlir::SymbolUserMap(collection, elf);

    auto sectionBuilder = mlir::OpBuilder::atBlockEnd(&elf.getContent().front());

    auto createSection = [&sectionBuilder, &elf](ELF::SectionSignature signature, bool memFootprint, size_t opAling) {
        // Only enforce LCM alignment in case of sections that get allocated
        auto secAlignReq = ELF::bitEnumContainsAll(signature.getFlags(), ELF::SectionFlagsAttr::SHF_ALLOC)
                                   ? ELF::math::lcm(elf::VPU_SH_ADDR_ALIGN_FOR_VPU, opAling)
                                   : opAling;
        if (memFootprint) {
            auto sec = sectionBuilder.create<ELF::DataSectionOp>(elf.getLoc(),
                                                                 signature.getName(),  // llvm::StringRef secName
                                                                 secAlignReq,          // int64_t secAddrAlign
                                                                 signature.getType(),  // ELFVPUX40XX secType
                                                                 signature.getFlags()  // ELFVPUX40XX secFlags
            );

            return mlir::cast<ELF::ElfSectionInterface>(sec.getOperation());
        } else {
            auto sec = sectionBuilder.create<ELF::LogicalSectionOp>(elf.getLoc(),
                                                                    signature.getName(),  // llvm::StringRef secName
                                                                    secAlignReq,          // int64_t secAddrAlign
                                                                    signature.getType(),  // ELFVPUX40XX secType
                                                                    signature.getFlags()  // ELFVPUX40XX secFlags
            );

            return mlir::cast<ELF::ElfSectionInterface>(sec.getOperation());
        }
    };

    auto replaceRefsWithCandidate = [&symbolUserMap](mlir::Operation* op,
                                                     ELF::ElfSectionInterface secInterface) -> void {
        auto symbolOp = mlir::cast<mlir::SymbolOpInterface>(op);
        auto symbolRef = mlir::FlatSymbolRefAttr::get(symbolOp.getNameAttr());
        auto symContainer = mlir::cast<mlir::SymbolOpInterface>(secInterface.getOperation());

        auto newSymName = mlir::SymbolRefAttr::get(symContainer.getNameAttr(), {symbolRef});

        symbolUserMap.replaceAllUsesWith(symbolOp, newSymName);
    };

    for (auto& op : llvm::make_early_inc_range(elf.getOps())) {
        auto wrapOp = mlir::dyn_cast<ELF::WrappableOpInterface>(op);
        if (!wrapOp)
            continue;

        auto maybeSignature = wrapOp.getSectionSignature();
        if (!maybeSignature.has_value())
            continue;
        auto signature = *maybeSignature;

        auto sectionMapKey = sectionMap.find(signature);

        if (sectionMapKey != sectionMap.end()) {
            auto secInterface = sectionMapKey->second;

            replaceRefsWithCandidate(&op, secInterface);

            auto sectionBlock = secInterface.getBlock();
            op.moveAfter(&sectionBlock->back());

        } else {
            auto hasMemFootprint = wrapOp.hasMemoryFootprint();

            // based on tblgen, any WrappableOp is also a BinaryOp
            auto binOp = mlir::cast<ELF::WrappableOpInterface>(op);
            auto opAddrAling = binOp.getAlignmentRequirements();

            auto secInterface = createSection(signature, hasMemFootprint, opAddrAling);

            replaceRefsWithCandidate(&op, secInterface);

            mlir::Block* sectionBlock = secInterface.getBlock();
            op.moveBefore(sectionBlock, sectionBlock->end());

            sectionMap[signature] = secInterface;
        }
    }

    return;
}

}  // namespace

//
// createConvertVPUASM2ELFPass
//

std::unique_ptr<mlir::Pass> vpux::ELF::createMoveOpsIntoSectionsPass(Logger log) {
    return std::make_unique<MoveOpsIntoSectionsPass>(log);
}
