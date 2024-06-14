//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/ELF/export.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/export.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/import.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/export.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/import.hpp"
#include "vpux/compiler/frontend/IE.hpp"
#include "vpux/compiler/init.hpp"
#include "vpux/compiler/interfaces_registry.hpp"
#include "vpux/compiler/tools/options.hpp"
#include "vpux/hwtest/hwtest.hpp"

#include "vpux/utils/core/format.hpp"

// TODO: E66812, it should be sufficient to have warnings disabled for 3-rd parties
// in CMake but it does not work for early versions of MSVC 2019
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4146)
#endif
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>

#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>

#include <mlir/IR/Dialect.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>
#include <mlir/Tools/mlir-translate/MlirTranslateMain.h>
#include <mlir/Tools/mlir-translate/Translation.h>

#include <openvino/runtime/core.hpp>
#include <openvino/runtime/runtime.hpp>

#include <cstdlib>
#include <fstream>
#include <iostream>

#include <sstream>
#include <string>

using namespace vpux;

namespace {

llvm::cl::opt<bool> vpuxProfiling("vpux-profiling", llvm::cl::desc("Add profilingOutput region to the imported IR"),
                                  llvm::cl::init(false));

llvm::cl::opt<std::string> vpuxBounds("set-upper-bounds",
                                      llvm::cl::desc("Set upper bounds for dynamic shapes. E.g: --set-upper-bounds=\"1 "
                                                     "18 3 3\", no commas between bounds values"),
                                      llvm::cl::init(""));

llvm::cl::opt<bool> enableDummyOpReplacement{"dummy-op-replacement",
                                             llvm::cl::desc("Replace unsupported SW Kernel ops with Dummy ones"),
                                             llvm::cl::init(false)};

llvm::cl::opt<bool> dynamicShapeToStatic{
        "dynamic-shape-to-static",
        llvm::cl::desc("If set, we immediately apply the bounds so that we "
                       "have a static shape for further work. "
                       "If not, we store the related information in TensorAttr and the IE representation looks "
                       "like this: tensor<1x?x3xf32, {bounds = [1, 18, 3], ..}>."),
        llvm::cl::init(false)};

enum class NetworkIOType { INPUT, OUTPUT };

//
// import-IE
//

mlir::OwningOpRef<mlir::ModuleOp> importIE(llvm::SourceMgr& sourceMgr, mlir::MLIRContext* ctx) {
    if (sourceMgr.getNumBuffers() != 1) {
        printTo(llvm::errs(),
                "Invalid source file for IE IR, it has unsupported number of "
                "buffers {0}",
                sourceMgr.getNumBuffers());
        return nullptr;
    }

    const auto netFileName = sourceMgr.getMemoryBuffer(1)->getBufferIdentifier();
    if (netFileName.empty()) {
        printTo(llvm::errs(), "Invalid source file for IE IR, not a file");
        return nullptr;
    }

    ov::Core core;
    std::shared_ptr<ov::Model> model;

    try {
        model = core.read_model(netFileName.str());
    } catch (const std::exception& ex) {
        printTo(llvm::errs(), "Failed to read model {0} : {1}", netFileName, ex.what());
        return nullptr;
    }

    if (!vpuxBounds.empty()) {
        std::stringstream stream(vpuxBounds);
        std::vector<int> boundsFromAnOption;
        int eachBoundValue = 0;
        while (stream >> eachBoundValue) {
            boundsFromAnOption.push_back(eachBoundValue);
        }

        std::map<ov::Output<ov::Node>, ov::PartialShape> newShapes;
        for (const ov::Output<ov::Node>& input : model->inputs()) {
            ov::PartialShape shape = input.get_partial_shape();
            for (const auto& dim : shape | indexed) {
                if (dim.value().is_dynamic()) {
                    dim.value() = ov::Dimension(0, boundsFromAnOption[dim.index()]);
                }
            }
            newShapes[input] = shape;
        }
        model->reshape(newShapes);
    }

    mlir::OwningOpRef<mlir::ModuleOp> module;

    try {
        mlir::DefaultTimingManager tm;
        auto rootTiming = tm.getRootScope();

        // Do NOT share OV constants with MLIR here: the OV model is created
        // locally and deleted at the end of the function scope, while MLIR
        // module would still be used in the outer scope. deep-copying the
        // constants in MLIR protects the code from use-after-free errors.
        constexpr bool useSharedConstants = false;

        // For VPUX37XX and VPUX40XX the graph transformations are different compared to the rest of the platforms
        // because scales do not need to be aligned. Running with VPU::ArchKind::UNKNOWN will align scales, which
        // can result in an accuracy drop for VPUX37XX and VPUX40XX.
        module = IE::importNetwork(ctx, model, useSharedConstants, rootTiming, vpuxProfiling, enableDummyOpReplacement,
                                   dynamicShapeToStatic, VPU::ArchKind::UNKNOWN);
    } catch (const std::exception& ex) {
        printTo(llvm::errs(), "Failed to translate IE IR {0} to MLIR : {1}", netFileName, ex.what());
        return nullptr;
    }

    return module;
}

//
// import-VPUIP
//

mlir::OwningOpRef<mlir::ModuleOp> importVPUIP(llvm::SourceMgr& sourceMgr, mlir::MLIRContext* ctx) {
    if (sourceMgr.getNumBuffers() != 1) {
        printTo(llvm::errs(),
                "Invalid source file for blob, it has unsupported number of "
                "buffers {0}",
                sourceMgr.getNumBuffers());
        return nullptr;
    }

    const auto blobFileName = sourceMgr.getMemoryBuffer(1)->getBufferIdentifier();
    if (blobFileName.empty()) {
        printTo(llvm::errs(), "Invalid source file for blob, not a file");
        return nullptr;
    }

    mlir::OwningOpRef<mlir::ModuleOp> module;
    std::ifstream blobStream(blobFileName.str(), std::ios::binary);
    auto blob = std::vector<char>(std::istreambuf_iterator<char>(blobStream), std::istreambuf_iterator<char>());

    try {
        module = VPUIP::importBlob(ctx, blob);
    } catch (const std::exception& ex) {
        printTo(llvm::errs(), "Failed to translate blob {0} to MLIR : {1}", blobFileName, ex.what());
        return nullptr;
    }

    return module;
}

//
// import-ELF
//

mlir::OwningOpRef<mlir::ModuleOp> importELF(llvm::SourceMgr& sourceMgr, mlir::MLIRContext* ctx) {
    if (sourceMgr.getNumBuffers() != 1) {
        printTo(llvm::errs(),
                "Invalid source file for elf, it has unsupported number of "
                "buffers {0}",
                sourceMgr.getNumBuffers());
        return nullptr;
    }

    const auto elfFileName = sourceMgr.getMemoryBuffer(1)->getBufferIdentifier();
    if (elfFileName.empty()) {
        printTo(llvm::errs(), "Invalid source file for elf, not a file");
        return nullptr;
    }

    mlir::OwningOpRef<mlir::ModuleOp> module;

    try {
        module = ELFNPU37XX::importELF(ctx, elfFileName.str());
    } catch (const std::exception& ex) {
        printTo(llvm::errs(), "Failed to translate elf {0} to MLIR : {1}", elfFileName, ex.what());
        return nullptr;
    }

    return module;
}

//
// export-VPUIP
//

mlir::LogicalResult exportVPUIP(mlir::ModuleOp module, llvm::raw_ostream& output) {
    mlir::DefaultTimingManager tm;
    auto rootTiming = tm.getRootScope();
    const std::vector<std::shared_ptr<const ov::Node>> params;
    const std::vector<std::shared_ptr<const ov::Node>> results;
    const auto buf = VPUIP::exportToBlob(module, rootTiming, params, results);
    output.write(reinterpret_cast<const char*>(buf.data()), buf.size());
    return mlir::success();
}

mlir::LogicalResult exportELF(mlir::ModuleOp module, llvm::raw_ostream& output) {
    mlir::DefaultTimingManager tm;

    auto arch = VPU::getArch(module.getOperation());

    if (arch == VPU::ArchKind::NPU37XX) {
        const auto buf = ELFNPU37XX::exportToELF(module);
        output.write(reinterpret_cast<const char*>(buf.data()), buf.size());
    } else if (arch == VPU::ArchKind::NPU40XX) {
        const auto buf = ELF::exportToELF(module);
        output.write(reinterpret_cast<const char*>(buf.data()), buf.size());
    } else {
        VPUX_THROW("ELF Flow not supported for ARCH {0}", VPU::stringifyArchKind(arch));
    }

    return mlir::success();
}

//
// export-LLVMIR
//

int dumpLLVMIR(mlir::ModuleOp module, llvm::raw_ostream& output) {
    Logger log("dumpLLVMIR", vpux::LogLevel::Info);

    // Register the translation to LLVM IR with the MLIR context.
    mlir::registerLLVMDialectTranslation(*module->getContext());

    // Convert the module to LLVM IR in a new LLVM IR context.
    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
    if (!llvmModule) {
        log.error("Failed to emit LLVM IR\n");
        return -1;
    }

    // Initialize LLVM targets.
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    mlir::ExecutionEngine::setupTargetTripleAndDataLayout(llvmModule.get(), nullptr);

    /// Optionally run an optimization pipeline over the llvm module.
    auto optPipeline = mlir::makeOptimizingTransformer(
            /*optLevel=*/0,  // enableOpt ? 3 : 0, // Note: seems small diffs on my tests
            /*sizeLevel=*/0,
            /*targetMachine=*/nullptr);
    if (auto err = optPipeline(llvmModule.get())) {
        log.error("Failed to optimize LLVM IR {0}\n", err);
        return -1;
    }

    output << "dumpLLVMIR() for output:\n" << *llvmModule << "\n";

    return 0;
}

mlir::LogicalResult exportLLVMIR(mlir::ModuleOp module, llvm::raw_ostream& output) {
    dumpLLVMIR(module, output);

    return mlir::success();
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        // TODO(E#84874):
        // currently, arch is used for both import and export
        // however, it would be a better option to extract arch info from module for the export
        const auto arch = vpux::parseArchKind(argc, argv);
        auto dialectRegistration = [&](mlir::DialectRegistry& registry) {
            registerDialects(registry);
            vpux::registerCommonInterfaces(registry, /*enableDummyOp*/ true);

            auto interfacesRegistry = vpux::createInterfacesRegistry(arch);
            interfacesRegistry->registerInterfaces(registry);
        };
        mlir::TranslateToMLIRRegistration("import-IE", "Translate OV IR to IE dialect", importIE, dialectRegistration);
        mlir::TranslateToMLIRRegistration("import-HWTEST", "Translate PSS test case described by config.json to blob",
                                          importHWTEST, dialectRegistration);
        mlir::TranslateToMLIRRegistration("import-VPUIP", "Translate blob to VPUIP dialect", importVPUIP,
                                          dialectRegistration);
        mlir::TranslateToMLIRRegistration("import-ELF", "Translate blob to ELF dialect", importELF,
                                          dialectRegistration);

        mlir::TranslateFromMLIRRegistration("export-VPUIP", "Translate VPUIP dialect to blob", exportVPUIP,
                                            dialectRegistration);
        mlir::TranslateFromMLIRRegistration("export-ELF", "Translate ELF dialect to blob", exportELF,
                                            dialectRegistration);
        mlir::TranslateFromMLIRRegistration("export-LLVMIR", "Translate LLVMIR dialect to blob", exportLLVMIR,
                                            dialectRegistration);

        return mlir::asMainReturnCode(mlir::mlirTranslateMain(argc, argv, "NPU Translation Testing Tool"));
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
