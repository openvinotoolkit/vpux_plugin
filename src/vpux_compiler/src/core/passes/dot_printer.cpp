//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/utils/dot_printer.hpp"
#include "vpux/compiler/core/passes.hpp"
#include "vpux/compiler/utils/dot_graph_writer.hpp"
#include "vpux/compiler/utils/logging.hpp"

#include <iostream>

using namespace vpux;

namespace {

class PrintDotPass final : public PrintDotBase<PrintDotPass> {
public:
    PrintDotPass(StringRef fileName, bool printConst, bool printDeclarations, StringRef passName)
            : fileName(fileName), printConstOp(printConst), printDeclarationsOp(printDeclarations), passName(passName) {
    }

    std::string getOpName(mlir::Operation& op) {
        auto symbolAttr = op.getAttrOfType<mlir::StringAttr>(mlir::SymbolTable::getSymbolAttrName());
        if (symbolAttr)
            return std::string(symbolAttr.getValue());
        return op.getName().getStringRef().data();
    }

    void processModule(mlir::ModuleOp module);

    std::string getPass() {
        return passName;
    }

    mlir::LogicalResult initializeOptions(StringRef options) final;

private:
    void safeRunOnModule() final;

    bool initialized = false;
    std::string fileName;
    bool printConstOp = false;
    bool printDeclarationsOp = false;
    std::string passName;
};

class PassDotGraph final : public mlir::PassInstrumentation {
public:
    explicit PassDotGraph() {
    }

    void runAfterPass(mlir::Pass* pass, mlir::Operation* op) {
        for (auto& printDotPass : printDotPasses) {
            if (printDotPass->getPass() == pass->getName()) {
                llvm::errs() << " Generating Dot after " << pass->getName() << "\n";
                if (auto module = mlir::dyn_cast<mlir::ModuleOp>(op)) {
                    printDotPass->processModule(module);
                } else if (auto func = mlir::dyn_cast<mlir::FuncOp>(op)) {
                    auto module = func->getParentOfType<mlir::ModuleOp>();
                    printDotPass->processModule(module);
                }
            }
        }
    }

    static std::vector<std::unique_ptr<PrintDotPass>> printDotPasses;

private:
};

// Print all the ops in a module.
void PrintDotPass::processModule(mlir::ModuleOp module) {
    for (mlir::Operation& op : module) {
        if (mlir::isa<mlir::FuncOp>(op)) {
            for (mlir::Region& region : op.getRegions()) {
                for (auto indexed_block : llvm::enumerate(region)) {
                    vpux::WriteGraph(fileName, &indexed_block.value(), printConstOp, printDeclarationsOp);
                }
            }
        }
    }
}

void PrintDotPass::safeRunOnModule() {
    if (afterPass.empty())
        processModule(getOperation());
}

mlir::LogicalResult PrintDotPass::initializeOptions(StringRef options) {
    // Skip default initialiation in case in already been initialized externally
    // See VPUX_DEVELOPER_BUILD enviroment variable
    if (initialized) {
        return mlir::success();
    }
    initialized = true;

    if (mlir::failed(Base::initializeOptions(options))) {
        return mlir::failure();
    }

    if (declareOp.hasValue())
        printDeclarationsOp = declareOp.getValue();
    if (constOp.hasValue())
        printConstOp = constOp.getValue();
    if (outputFile.hasValue())
        fileName = outputFile.getValue();
    if (afterPass.hasValue()) {
        PassDotGraph::printDotPasses.push_back(
                std::make_unique<PrintDotPass>(fileName, printConstOp, printDeclarationsOp, afterPass.getValue()));
    }

    return mlir::success();
}

}  // namespace

std::vector<std::unique_ptr<PrintDotPass>> PassDotGraph::printDotPasses;

void vpux::addDotPrinter(mlir::PassManager& pm) {
    pm.addInstrumentation(std::make_unique<PassDotGraph>());
}

void vpux::addDotPrinterFromEnvVar(mlir::PassManager& pm, StringRef options) {
    std::stringstream ss(options.data());
    std::string split;
    while (getline(ss, split, ',')) {
        std::unique_ptr<PrintDotPass> printDotPass = std::make_unique<PrintDotPass>("", "", "", "");
        VPUX_THROW_UNLESS(!mlir::failed(printDotPass->initializeOptions(split)), "Failed to initialize options");
        pm.addPass(std::move(printDotPass));
    }
}

std::unique_ptr<mlir::Pass> vpux::createPrintDot(StringRef fileName, bool printConst, bool printDeclarations) {
    VPUX_THROW_UNLESS(!fileName.size(), "Empty output file name");
    return std::make_unique<PrintDotPass>(fileName, printConst, printDeclarations, "");
}
