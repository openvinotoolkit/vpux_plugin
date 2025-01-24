//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/llvm_to_binary.hpp"
#include "vpux/compiler/conversion.hpp"

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Program.h>

using namespace vpux;

namespace {
std::string getMoviToolsArchArgument(VPU::ArchKind arch) {
    switch (arch) {
    case VPU::ArchKind::NPU37XX:
        return "3720xx";
    case VPU::ArchKind::NPU40XX:
        return "4000xx";
    default:
        VPUX_THROW("Invalid ArchKind for MoviTools usage");
    }
}

std::string getMoviLDArchPath(VPU::ArchKind arch) {
    switch (arch) {
    case VPU::ArchKind::NPU37XX:
        return "37xxxx";
    case VPU::ArchKind::NPU40XX:
        return "40xxxx";
    default:
        VPUX_THROW("Invalid ArchKind for Movi LLD path resolution");
    }
}
}  // namespace

void vpux::translateToLLVMIR(mlir::ModuleOp moduleOp, mlir::SymbolRefAttr swKernelSymbol, vpux::Logger log) {
    auto llvmFuncOp = moduleOp.lookupSymbol<mlir::LLVM::LLVMFuncOp>(swKernelSymbol);

    VPUX_THROW_UNLESS(llvmFuncOp != nullptr, "llvmFuncOp should be valid");

    // We create a temporary module in which we clone the llvmFuncOp and then translate it
    // to LLVM IR, and write it to disk.
    auto moduleBuilder = mlir::OpBuilder::atBlockBegin(moduleOp.getBody());
    auto tmpModuleOp = moduleBuilder.create<mlir::ModuleOp>(moduleOp.getLoc(), llvm::StringRef("TempModule"));

    tmpModuleOp.getBody()->push_back(llvmFuncOp.clone());

    // Translate the LLVM dialect module to the LLVM IR module. The translation
    // is inspired from MLIR Toy example chapter 6 (https://mlir.llvm.org/docs/Tutorials/Toy/Ch-6/).
    // Note: mlir::registerBuiltinDialectTranslation() and
    // mlir::registerLLVMDialectTranslation() are called in init.cpp,
    // in function vpux::registerCommonInterfaces().

    // Convert the module to LLVM IR in a new LLVM IR context.
    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(tmpModuleOp, llvmContext);
    if (!llvmModule) {
        log.error("Failed to emit LLVM IR\n");
        return;
    }

    tmpModuleOp.erase();

    // We write llvmModule to file sw_layer.ll.
    std::error_code llFileEC;
    llvm::raw_fd_ostream llFile("sw_layer.ll", llFileEC);
    llFile << *llvmModule;
}

void vpux::lowerLLVMToBinary(mlir::ModuleOp moduleOp, mlir::SymbolRefAttr swKernelSymbol) {
    auto llvmFuncOp = moduleOp.lookupSymbol<mlir::LLVM::LLVMFuncOp>(swKernelSymbol);
    VPUX_THROW_UNLESS(llvmFuncOp != nullptr, "llvmFuncOp should be valid");

    const auto arch = VPU::getArch(moduleOp);
    VPUX_THROW_UNLESS(arch != VPU::ArchKind::UNKNOWN, "Could not identify arch");

    auto archArgument = getMoviToolsArchArgument(arch);

    auto llvmFuncOpNameStr = llvmFuncOp.getName().str();

    llvm::SmallVector<std::optional<StringRef>> redirects = {
            std::nullopt,  // stdin(0)
            std::nullopt,  // stdout(1)
            std::nullopt   // stderr(2)
    };

    std::string errMsg;

    // We replace, if any, in file sw_layer.ll, string "memory(none)" with "readnone".
    //   (our MLIR is based on LLVM 16, while moviCompile uses LLVM 15 - see changes
    //   to the LLVM IR here:
    //   https://releases.llvm.org/16.0.0/docs/ReleaseNotes.html#changes-to-the-llvm-ir).
    //  (ExecuteAndWait inspired from src/vpux_imd_backend/src/executor.cpp)
    llvm::StringRef prgSed = "/usr/bin/sed";
    llvm::SmallVector<llvm::StringRef> runArgsSed = {
            prgSed, "-e", "s/memory(none)/readnone/", "-e", "s/memory(only)/readonly/", "sw_layer.ll", "-i.bak"};

    const auto procErrSed = llvm::sys::ExecuteAndWait(prgSed, runArgsSed, /*Env=*/std::nullopt, redirects,
                                                      /*SecondsToWait*/ 100, /*MemoryLimit=*/0, &errMsg);
    VPUX_THROW_UNLESS(procErrSed == 0, "Call to sed failed");

    // We compile with moviCompile the sw_layer.ll to sw_layer.s (SHAVE assembly).
    auto mvCompileEnvVar = std::getenv("MV_COMPILE_DIR");
    auto mvToolsEnvVar = std::getenv("MV_TOOLS_DIR");
    auto mvToolsVersionEnvVar = std::getenv("MV_TOOLS_VERSION");

    auto mvToolsDirStrWoNull = std::string(mvToolsEnvVar == NULL ? "" : mvToolsEnvVar);
    auto mvToolsVersionStrWoNull = std::string(mvToolsVersionEnvVar == NULL ? "" : mvToolsVersionEnvVar);
    auto mvToolsPathCompleteStr = mvToolsDirStrWoNull + "/" + mvToolsVersionStrWoNull;

    // if MV_COMPILE_DIR is not defined, fallback to mvToolsDir composed by MV_TOOLS_DIR + MV_TOOLS_VERSION
    auto prgMCStr = std::string(mvCompileEnvVar == NULL ? mvToolsPathCompleteStr : mvCompileEnvVar) +
                    "/linux64/bin/moviCompile";
    llvm::StringRef prgMC = prgMCStr;
    llvm::SmallVector<llvm::StringRef> runArgsMC = {prgMC,
                                                    std::string("-mcpu=") + archArgument,
                                                    "-S",
                                                    "-o",
                                                    "sw_layer.s",
                                                    "-x",
                                                    "ir",
                                                    "-mllvm",
                                                    "-opaque-pointers",
                                                    "sw_layer.ll"};

    const auto procErrMC = llvm::sys::ExecuteAndWait(prgMC, runArgsMC, /*Env=*/std::nullopt, redirects,
                                                     /*SecondsToWait*/ 100, /*MemoryLimit=*/0, &errMsg);
    VPUX_THROW_UNLESS(procErrMC == 0, "Call to moviCompile failed");

    // We run moviAsm from MoviTools to obtain from sw_layer.s a file sw_layer.o.
    std::string prgAsmStr = mvToolsPathCompleteStr + "/linux64/bin/moviAsm";
    llvm::StringRef prgAsm = prgAsmStr;
    //
    llvm::SmallVector<llvm::StringRef> runArgsAsm = {prgAsm, "sw_layer.s", "--cv", archArgument, "--noSPrefixing"};

    const auto procErrAsm = llvm::sys::ExecuteAndWait(prgAsm, runArgsAsm, /*Env=*/std::nullopt, redirects,
                                                      /*SecondsToWait*/ 100, /*MemoryLimit=*/0, &errMsg);
    VPUX_THROW_UNLESS(procErrAsm == 0, "Call to moviAsm failed");

    std::string elfPathFileNameStr = llvmFuncOpNameStr + "/a.out";

    // We create folder (e.g. generated_Cos0)
    llvm::sys::fs::create_directory(llvmFuncOpNameStr);

    // We run the linker to obtain the ELF file a.out from sw_layers.o
    //   (we include 4 libraries as dependencies to link the
    //   external __coss function, which returns cos applied on
    //   the float input value, for which it does check if it
    //   is in range 0..pi, etc)

    auto moviLibArchPath = getMoviLDArchPath(arch);

    std::string prgLdStr = mvToolsPathCompleteStr + "/linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-ld";
    std::string mLibMStr = mvToolsPathCompleteStr + "/common/moviCompile/lib/" + moviLibArchPath + "/mlibm.a";
    std::string mLibCrtStr = mvToolsPathCompleteStr + "/common/moviCompile/lib/" + moviLibArchPath + "/mlibcrt.a";
    std::string mLibCLGPLStr = mvToolsPathCompleteStr + "/common/moviCompile/lib/" + moviLibArchPath + "/mlibc_lgpl.a";
    std::string mLibCStr = mvToolsPathCompleteStr + "/common/moviCompile/lib/" + moviLibArchPath + "/mlibc.a";
    llvm::StringRef prgLd = prgLdStr;
    auto envVar = std::getenv("VPUX_PLUGIN_DIR");
    std::string scriptStr = std::string("--script=") + std::string(envVar == NULL ? "" : envVar) +
                            "/sw_runtime_kernels/kernels/prebuild/shave_kernel.ld";

    llvm::SmallVector<llvm::StringRef> runArgsLd = {prgLd,
                                                    llvm::StringRef(scriptStr),
                                                    "-entry",
                                                    llvmFuncOpNameStr,
                                                    "--strip-debug",
                                                    "--discard-all",
                                                    "-zmax-page-size=16",
                                                    "-EL",
                                                    "-O9",
                                                    "--gc-sections",
                                                    "sw_layer.o",
                                                    "--start-group",
                                                    llvm::StringRef(mLibMStr),
                                                    llvm::StringRef(mLibCrtStr),
                                                    llvm::StringRef(mLibCLGPLStr),
                                                    llvm::StringRef(mLibCStr),
                                                    "--end-group",
                                                    "--output",
                                                    llvm::StringRef(elfPathFileNameStr)};

    const auto procErrLd = llvm::sys::ExecuteAndWait(prgLd, runArgsLd, /*Env=*/std::nullopt, redirects,
                                                     /*SecondsToWait*/ 100, /*MemoryLimit=*/0, &errMsg);
    VPUX_THROW_UNLESS(procErrLd == 0, "Call to sparc-myriad-rtems-ld failed");

    // We create file FileList.in containing each ELF file and
    //   folder (name, more exactly key in ShaveBinaryResources dictionary),
    //   associated to the current sw layer.
    std::ofstream fOut("FileList.in", std::ios::app);
    if (fOut.is_open()) {  // Make sure file opened before writing
        if (!(fOut << llvmFuncOpNameStr + "/a.out\n")) {
            llvm::errs() << "Write to FileList.in failed.\n";
        }

        if (!(fOut << llvmFuncOpNameStr + "\n")) {
            llvm::errs() << "Write to FileList.in failed.\n";
        }
    } else {
        llvm::errs() << "Cannot open file FileList.in.\n";
    }
}
