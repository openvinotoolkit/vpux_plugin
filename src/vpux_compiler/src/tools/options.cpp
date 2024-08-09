//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/tools/options.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/utils/core/error.hpp"

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"

#include <llvm/Support/CommandLine.h>
#include <mlir/Pass/PassOptions.h>

using namespace vpux;

//
// parseArchKind
//

vpux::VPU::ArchKind vpux::parseArchKind(int argc, char* argv[], StringRef helpHeader) {
    static llvm::cl::OptionCategory vpuxOptOptions("NPU Options");

    // Please use this option to test pipelines only (DefaultHW, ReferenceSW, etc.)
    // This option allows us to avoid ambiguity here when the parameters contradict each other:
    // "vpux-opt --init-compiler="vpu-arch=VPUX37XX compilation-mode=ReferenceSW" --default-hw-mode"
    // Instead you should only pass arch version:
    // "vpux-opt --vpu-arch=VPUX37XX --default-hw-mode"
    static llvm::cl::opt<std::string> archOpt("vpu-arch", llvm::cl::desc("VPU architecture to compile for"),
                                              llvm::cl::init(""), llvm::cl::cat(vpuxOptOptions));

    // To test the appropriate passes and pipelines depending on the type of device, we need to get a value for arch
    // kind. Most of lit-tests use command line that contains following option: --init-compiler="vpu-arch=VPUX37XX" To
    // avoid duplicating command line options like this: "vpux-opt --vpu-arch=VPUX37XX
    // --init-compiler="vpu-arch=VPUX37XX".." let's parse init-compiler option to get arch kind. There is still code
    // duplication, but in one place that does not affect developers:

    static llvm::cl::opt<std::string> initCompiler("init-compiler",
                                                   llvm::cl::desc("Duplicated option for VPU::InitCompilerPass"),
                                                   llvm::cl::init(""), llvm::cl::cat(vpuxOptOptions));

    // skip unknown options here, like split-input-file, show-dialects, etc.
    llvm::cl::list<std::string> sink(llvm::cl::Sink);

    const auto toolName = helpHeader.empty() ? "Register passes and pipelines depending on device type" : helpHeader;
    llvm::cl::ParseCommandLineOptions(argc, argv, toolName, nullptr, nullptr,
                                      /*LongOptionsUseDoubleDash=*/true);

    VPUX_THROW_WHEN(!archOpt.empty() && !initCompiler.empty(),
                    "ArchKind value is ambiguous. Only one option can be used at a time, either \"vpu-arch\" or "
                    "\"init-compiler\"");
    VPUX_THROW_WHEN(archOpt.empty() && initCompiler.empty(), "Can't get ArchKind value");

    const auto getArchFromString = [](vpux::StringRef archOptStr) {
        auto archKind = vpux::VPU::symbolizeEnum<vpux::VPU::ArchKind>(archOptStr);
        VPUX_THROW_UNLESS(archKind.has_value(), "Unknown VPU architecture : '{0}'", archOpt.getValue());

        return archKind.value();
    };

    auto arch = vpux::VPU::ArchKind::UNKNOWN;
    if (!archOpt.empty()) {
        arch = getArchFromString(archOpt);
    } else {
        const auto options = VPU::InitCompilerOptions::createFromString(initCompiler);
        arch = getArchFromString(options->arch);
    }

    initCompiler.removeArgument();
    sink.removeArgument();
    llvm::cl::ResetAllOptionOccurrences();

    return arch;
}
