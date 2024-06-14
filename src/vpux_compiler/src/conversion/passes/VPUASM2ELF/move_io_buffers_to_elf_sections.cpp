
#include "vpux/compiler/NPU40XX/dialect/ELF/ops.hpp"
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

using namespace vpux;

class MoveIOBuffersToElfSectionsPass : public MoveIOBuffersToElfSectionsBase<MoveIOBuffersToElfSectionsPass> {
public:
    MoveIOBuffersToElfSectionsPass(Logger log): _log(log) {
        Base::initLogger(log, Base::getArgumentName());
    }

    void safeRunOnModule() override;

private:
    template <typename Functor = vpux::FuncRef<void(VPUASM::DeclareBufferOp)>>
    void walkBufferOps(ELF::MainOp elfMain, vpux::VPURT::BufferSection locale, uint64_t ioIdx, Functor functor) {
        for (auto op : llvm::make_early_inc_range(elfMain.getOps<VPUASM::DeclareBufferOp>())) {
            auto buffLocation = op.getBufferType().getLocation();
            if (buffLocation.getSection() == locale && buffLocation.getSectionIndex() == ioIdx) {
                functor(op);
            }
        }
    }

    Logger _log;
};

void MoveIOBuffersToElfSectionsPass::safeRunOnModule() {
    auto moduleOp = getOperation();
    mlir::MLIRContext* ctx = &getContext();

    mlir::func::FuncOp netFunc;
    IE::CNNNetworkOp netInfo;
    IE::CNNNetworkOp::getFromModule(moduleOp, netInfo, netFunc);

    auto mainOps = to_small_vector(netFunc.getOps<ELF::MainOp>());
    VPUX_THROW_UNLESS(mainOps.size() == 1, "Expected exactly one ELF mainOp. Got {0}", mainOps.size());
    auto elfMain = mainOps[0];

    auto symTabBuilder = mlir::OpBuilder::atBlockEnd(elfMain.getBody());
    auto sectionBuilder = mlir::OpBuilder::atBlockBegin(elfMain.getBody());

    auto replaceRefsWithCandidate = [&elfMain](mlir::Operation* op, ELF::ElfSectionInterface secInterface) -> void {
        auto symbolOp = mlir::cast<mlir::SymbolOpInterface>(op);
        auto symbolRef = mlir::FlatSymbolRefAttr::get(symbolOp.getNameAttr());
        auto symContainer = mlir::cast<mlir::SymbolOpInterface>(secInterface.getOperation());

        auto newSymName = mlir::SymbolRefAttr::get(symContainer.getNameAttr(), {symbolRef});

        for (auto section : elfMain.getOps<ELF::ElfSectionInterface>()) {
            auto resplaced = mlir::SymbolTable::replaceAllSymbolUses(symbolOp, newSymName, section);
            VPUX_THROW_UNLESS(resplaced.succeeded(), "Failed to replace symbol uses for {0}", op);
        }
    };

    auto moveIOToSection = [&](VPURT::BufferSection locale, ELF::SectionFlagsAttr ioFlag, llvm::StringRef baseName,
                               mlir::Block* ioBlock) {
        auto ioOps = to_small_vector(ioBlock->getOps<VPUASM::DeclareBufferOp>());

        // pre-create the symTab for this locale
        auto symTabName = mlir::StringAttr::get(ctx, "symtab." + baseName);
        auto ioFlagAttr = ELF::SectionFlagsAttrAttr::get(ctx, ioFlag);

        auto symTab = symTabBuilder.create<ELF::CreateSymbolTableSectionOp>(elfMain.getLoc(), symTabName, ioFlagAttr);
        auto& symTabBlock = symTab.getSymbols().emplaceBlock();
        auto symbolBuilder = mlir::OpBuilder::atBlockBegin(&symTabBlock);

        // TODO: E#59169 Optimize logic so we don't walk over the whole IR in each loop iteration
        for (auto ioOp : ioOps) {
            auto ioLocation = ioOp.getBufferType().getLocation();
            auto ioIdx = ioLocation.getSectionIndex();

            auto sectionNameAttr = mlir::StringAttr::get(ctx, baseName + std::to_string(ioIdx));

            auto sec = sectionBuilder.create<ELF::LogicalSectionOp>(elfMain.getLoc(), sectionNameAttr, 1, /*addrAling*/
                                                                    ELF::SectionTypeAttr::SHT_NOBITS, ioFlag);

            auto& sectionBlock = sec.getContent().emplaceBlock();

            walkBufferOps(elfMain, locale, ioIdx, [&](VPUASM::DeclareBufferOp buffer) {
                replaceRefsWithCandidate(buffer.getOperation(), sec);

                buffer.getOperation()->moveBefore(&sectionBlock, sectionBlock.end());
            });

            // setting since moved operations from the ELFMain's block (above), we've potentially moved the
            // isertionPoint of the sectionBuilder (since insertionPoint is tied to Operation position). Re-setting
            // after the current section  here.
            sectionBuilder.setInsertionPointAfter(sec.getOperation());

            // create the symbolOp for this section
            auto sectionNameRef = mlir::FlatSymbolRefAttr::get(sectionNameAttr);
            auto symSize = ioOp.getBufferType().getMemref().cast<vpux::NDTypeInterface>().getTotalAllocSize().count();
            auto symbolOpName =
                    mlir::StringAttr::get(ctx, ELF::SymbolOp::getDefaultNamePrefix() + sectionNameAttr.strref());
            symbolBuilder.create<ELF::SymbolOp>(sec.getLoc(), symbolOpName, sectionNameRef,
                                                ELF::SymbolType::STT_SECTION, static_cast<uint64_t>(symSize));
        }
    };

    auto ioBindings = VPUASM::IOBindingsOp::getFromModule(moduleOp);

    moveIOToSection(VPURT::BufferSection::NetworkInput, ELF::SectionFlagsAttr::VPU_SHF_USERINPUT, "io.NetworkInput",
                    &ioBindings.getInputDeclarations().front());
    moveIOToSection(VPURT::BufferSection::NetworkOutput, ELF::SectionFlagsAttr::VPU_SHF_USEROUTPUT, "io.NetworkOutput",
                    &ioBindings.getOutputDeclarations().front());
    moveIOToSection(VPURT::BufferSection::ProfilingOutput, ELF::SectionFlagsAttr::VPU_SHF_PROFOUTPUT,
                    "io.ProfilingOutput", &ioBindings.getProfilingBuffDeclarations().front());
}

//
// createMoveIOBuffersToSectionsPass
//

std::unique_ptr<mlir::Pass> vpux::createMoveIOBuffersToSectionsPass(Logger log) {
    return std::make_unique<MoveIOBuffersToElfSectionsPass>(log);
}
