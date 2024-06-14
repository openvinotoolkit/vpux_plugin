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

class AddElfSymbolTablePass : public ELF::AddELFSymbolTableBase<AddElfSymbolTablePass> {
public:
    explicit AddElfSymbolTablePass(Logger log): _log(log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    Logger _log;
};

void AddElfSymbolTablePass::safeRunOnFunc() {
    mlir::MLIRContext* ctx = &getContext();
    auto netFunc = getOperation();

    auto mainOps = to_small_vector(netFunc.getOps<ELF::MainOp>());
    VPUX_THROW_UNLESS(mainOps.size() == 1, "Expected exactly one ELF mainOp. Got {0}", mainOps.size());
    auto elfMain = mainOps[0];

    auto sectionBuilder = mlir::OpBuilder::atBlockEnd(&elfMain.getContent().front());
    auto symTab = sectionBuilder.create<ELF::CreateSymbolTableSectionOp>(elfMain.getLoc(), "symtab",
                                                                         ELF::SectionFlagsAttr::SHF_NONE);

    auto symbolBuilder = mlir::OpBuilder::atBlockBegin(symTab.getBlock());

    auto validSection = [](mlir::Operation* op) -> bool {
        return mlir::isa<ELF::DataSectionOp>(op) || mlir::isa<ELF::LogicalSectionOp>(op);
    };

    for (auto section : elfMain.getOps<ELF::ElfSectionInterface>()) {
        if (!validSection(section.getOperation()))
            continue;

        auto sectionSymbol = mlir::cast<mlir::SymbolOpInterface>(section.getOperation());
        auto sectionName = sectionSymbol.getNameAttr();
        auto sectionRef = mlir::FlatSymbolRefAttr::get(sectionName);

        auto symbolOpName = mlir::StringAttr::get(ctx, ELF::SymbolOp::getDefaultNamePrefix() + sectionName.strref());

        symbolBuilder.create<ELF::SymbolOp>(section.getLoc(), symbolOpName, sectionRef, ELF::SymbolType::STT_SECTION);
    }

    return;
}
}  // namespace

//
// createAddELFSymbolTablePass
//

std::unique_ptr<mlir::Pass> vpux::ELF::createAddELFSymbolTablePass(Logger log) {
    return std::make_unique<AddElfSymbolTablePass>(log);
}
