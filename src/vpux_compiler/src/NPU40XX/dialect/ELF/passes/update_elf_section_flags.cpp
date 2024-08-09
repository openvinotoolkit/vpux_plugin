//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/NPU40XX/dialect/ELF/passes.hpp"

using namespace vpux;

namespace {

class UpdateELFSectionFlagsPass final : public ELF::UpdateELFSectionFlagsBase<UpdateELFSectionFlagsPass> {
public:
    explicit UpdateELFSectionFlagsPass(Logger log): _log(log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    template <typename OpTy>
    void runOnSectionOps(ELF::MainOp& elfMain, mlir::SymbolUserMap& symbolUserMap) {
        for (auto sectionOp : elfMain.getOps<OpTy>()) {
            auto currFlagsAttrVal = sectionOp.getSecFlags();

            for (auto sectionOpMember : sectionOp.template getOps<ELF::WrappableOpInterface>()) {
                currFlagsAttrVal = currFlagsAttrVal | sectionOpMember.getAccessingProcs(symbolUserMap);
            }

            sectionOp.setSecFlagsAttr(ELF::SectionFlagsAttrAttr::get(sectionOp.getContext(), currFlagsAttrVal));
        }
    }

    void safeRunOnModule() final {
        mlir::ModuleOp moduleOp = getOperation();

        IE::CNNNetworkOp cnnOp;
        mlir::func::FuncOp funcOp;
        IE::CNNNetworkOp::getFromModule(moduleOp, cnnOp, funcOp);

        auto mainOps = to_small_vector(funcOp.getOps<ELF::MainOp>());
        VPUX_THROW_UNLESS(mainOps.size() == 1, "Expected exactly one ELF mainOp. Got {0}", mainOps.size());
        auto elfMain = mainOps[0];

        mlir::SymbolTableCollection collection;
        auto symbolUserMap = mlir::SymbolUserMap(collection, elfMain);

        runOnSectionOps<ELF::DataSectionOp>(elfMain, symbolUserMap);
        runOnSectionOps<ELF::LogicalSectionOp>(elfMain, symbolUserMap);
    };

    Logger _log;
};

}  // namespace

std::unique_ptr<mlir::Pass> vpux::ELF::createUpdateELFSectionFlagsPass(Logger log) {
    return std::make_unique<UpdateELFSectionFlagsPass>(log);
}
