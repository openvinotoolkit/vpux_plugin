//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/dialect/ELF/passes.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/core/passes.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/passes.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUASM/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/passes.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURegMapped/passes.hpp"
#include "vpux/compiler/dialect/const/passes.hpp"
#include "vpux/compiler/init.hpp"
#include "vpux/compiler/interfaces_registry.hpp"
#include "vpux/compiler/passes_register.hpp"
#include "vpux/compiler/pipelines_register.hpp"
#include "vpux/compiler/tools/options.hpp"

#include "vpux/utils/core/error.hpp"

#include <mlir/Dialect/Func/Transforms/Passes.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>
#include <mlir/Transforms/Passes.h>

#include <cstdlib>
#include <iostream>

int main(int argc, char* argv[]) {
    try {
        // TODO: need to rework this unconditional replacement for dummy ops
        // there is an option for vpux-translate we can do it in the same way
        // Ticket: E#50937
        auto registry = vpux::createDialectRegistry(vpux::DummyOpMode::ENABLED);

        const auto hwSpecificRegistration = [&](vpux::StringRef helpHeader) {
            const auto archKind = vpux::parseArchKind(argc, argv, helpHeader);

            const auto pipelineRegistry = vpux::createPipelineRegistry(archKind);
            pipelineRegistry->registerPipelines();

            const auto passesRegistry = vpux::createPassesRegistry(archKind);
            passesRegistry->registerPasses();

            auto interfacesRegistry = vpux::createInterfacesRegistry(archKind);
            interfacesRegistry->registerInterfaces(registry);
        };

        vpux::registerCorePasses();
        vpux::Const::registerConstPasses();
        vpux::IE::registerIEPasses();
        vpux::IE::registerIEPipelines();
        vpux::VPU::registerVPUPasses();
        vpux::VPU::registerVPUPipelines();
        vpux::VPUIP::registerVPUIPPasses();
        vpux::VPUIP::registerVPUIPPipelines();
        vpux::VPURT::registerVPURTPipelines();
        vpux::VPURT::registerVPURTPasses();
        vpux::ELFNPU37XX::registerELFNPU37XXPasses();
        vpux::ELF::registerELFPasses();
        vpux::VPUMI37XX::registerVPUMI37XXPasses();
        vpux::VPUMI40XX::registerVPUMI40XXPasses();
        vpux::VPUASM::registerVPUASMPasses();
        vpux::VPUIPDPU::registerVPUIPDPUPasses();
        vpux::registerConversionPasses();
        vpux::registerConversionPipelines();

        mlir::registerTransformsPasses();
        mlir::func::registerFuncPasses();

        vpux::Const::registerConstPipelines();

        return mlir::asMainReturnCode(
                mlir::MlirOptMain(argc, argv, "NPU Optimizer Testing Tool", registry, hwSpecificRegistration));
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
