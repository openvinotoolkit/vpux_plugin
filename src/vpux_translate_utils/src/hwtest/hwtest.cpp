//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/hwtest/hwtest.hpp"
#include <llvm/Support/ToolOutputFile.h>
#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Support/DebugStringHelper.h>
#include <mlir/Support/FileUtilities.h>
#include "intel_npu/config/compiler.hpp"

#include "vpux/compiler/NPU37XX/conversion.hpp"
#include "vpux/compiler/NPU40XX/conversion.hpp"
#include "vpux/compiler/NPU40XX/pipelines.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/init.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/utils/core/error.hpp"

#include "vpux/compiler/NPU40XX/dialect/ELF/export.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/export.hpp"

#include "vpux/compiler/dialect/VPUMI37XX/dialect.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/dialect.hpp"
namespace vpux {

mlir::OwningOpRef<mlir::ModuleOp> importHWTEST(llvm::StringRef sourceJson, mlir::MLIRContext* ctx) {
    ctx->loadDialect<VPUIP::VPUIPDialect>();
    ctx->loadDialect<VPURT::VPURTDialect>();
    ctx->loadDialect<NPUReg40XX::NPUReg40XXDialect>();
    ctx->loadDialect<VPUMI37XX::VPUMI37XXDialect>();
    ctx->loadDialect<VPUMI40XX::VPUMI40XXDialect>();

    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(ctx), StringRef("mainModule"));
    auto log = Logger{"vpux-hwtest", LogLevel::Trace};
    auto builder = mlir::OpBuilder(module.getBodyRegion());

    nb::TestCaseJsonDescriptor jsonDesc(sourceJson);

    // TODO:
    // This will be handled later based on op type in config json
    auto opType = jsonDesc.getCaseStr();

    auto mainOpJsonDesc = jsonDesc;
    if (jsonDesc.getCaseType() == nb::CaseType::RaceCondition) {
        auto underlyingOp = jsonDesc.getUnderlyingOp();
        VPUX_THROW_WHEN(underlyingOp == nullptr, "underlyingOp is nullptr for CaseType::RaceCondition");
        mainOpJsonDesc = *underlyingOp;
    }

    const SmallVector<nb::InputLayer> inputList = mainOpJsonDesc.getInputLayerList();
    auto outputs = mainOpJsonDesc.getOutputLayers();

    SmallVector<mlir::Type> input_types;
    for (std::size_t idx = 0; idx < inputList.size(); idx++) {
        input_types.push_back(hwtest::parseInputType(builder, inputList[idx]));
    }

    mlir::Type output_type = hwtest::parseOutputType(builder, outputs.front());

    const SmallVector<nb::WeightLayer> weightList = mainOpJsonDesc.getWeightLayers();

    SmallVector<mlir::Type> weightTypes;
    for (std::size_t idx = 0; idx < weightList.size(); idx++) {
        weightTypes.push_back(hwtest::parseWeightsType(builder, weightList[idx]));
    }

    switch (jsonDesc.getCaseType()) {
    case nb::CaseType::DMA: {
        if (jsonDesc.getDMAparams().dstLocations.size() == 1) {
            hwtest::buildDMA(jsonDesc, module, builder, log, input_types.front(), output_type);
        } else {
            hwtest::buildDMABroadcast(jsonDesc, module, builder, log, input_types.front(), output_type);
        }
        break;
    }
    case nb::CaseType::DMACompressActDense: {
        hwtest::buildDMACompressActDense(jsonDesc, module, builder, log, input_types.front(), output_type);
        break;
    }
    case nb::CaseType::DMACompressActSparse: {
        hwtest::buildDMACompressActSparse(jsonDesc, module, builder, log, input_types.front(), output_type);
        break;
    }
    case nb::CaseType::ReduceSumSquare:
    case nb::CaseType::ReduceMean:
    case nb::CaseType::ReduceOut: {
        hwtest::buildReductionTest(jsonDesc, module, builder, log, input_types.front(), output_type);
        break;
    }
    case nb::CaseType::GatherDMA: {
        hwtest::buildGatherDMA(jsonDesc, module, builder, log, input_types.front(), output_type);
        break;
    }
    case nb::CaseType::ZMajorConvolution: {
        const auto weightInChannels = weightList.front().shape[1];

        if (weightInChannels > 8 * 1024) {
            hwtest::buildContinuedConv(jsonDesc, module, builder, log, input_types.front(), weightTypes.front(),
                                       output_type);
        } else {
            hwtest::buildSimpleZMajorConv(jsonDesc, module, builder, log, input_types.front(), weightTypes.front(),
                                          output_type);
        }
        break;
    }
    case nb::CaseType::SparseZMajorConvolution: {
        hwtest::buildSparseZMajorConv(jsonDesc, module, builder, log, input_types.front(), weightTypes.front(),
                                      output_type);
        break;
    }
    case nb::CaseType::DepthWiseConv: {
        hwtest::buildDWConv(jsonDesc, module, builder, log, input_types.front(), weightTypes.front(), output_type);
        break;
    }
    case nb::CaseType::DoubleZMajorConvolution: {
        hwtest::buildDoubleConv(jsonDesc, module, builder, log, input_types.front(), weightTypes.front(), output_type);
        break;
    }
    case nb::CaseType::EltwiseDense: {
        hwtest::buildEltwise(jsonDesc, module, builder, log, input_types.front(), weightTypes.front(), output_type);
        break;
    }
    case nb::CaseType::EltwiseMultDW: {
        hwtest::buildEltwiseMultWithDwConv(jsonDesc, module, builder, log, input_types.front(), weightTypes.front(),
                                           output_type);
        break;
    }
    case nb::CaseType::EltwiseSparse: {
        hwtest::buildEltwiseSparse(jsonDesc, module, builder, log, input_types.front(), weightTypes.front(),
                                   output_type);
        break;
    }
    case nb::CaseType::MaxPool: {
        hwtest::buildMaxPool(jsonDesc, module, builder, log, input_types.front(), output_type);
        break;
    }
    case nb::CaseType::AvgPool: {
        hwtest::buildAvgpool(jsonDesc, module, builder, log, input_types.front(), output_type);
        break;
    }
    case nb::CaseType::DifferentClustersDPU: {
        hwtest::buildDifferentClustersDPUTest(jsonDesc, module, builder, log, input_types.front(), weightTypes.front(),
                                              output_type);
        break;
    }
    case nb::CaseType::MultiClustersDPU: {
        hwtest::buildMultiClustersDPUTest(jsonDesc, module, builder, log, input_types.front(), weightTypes.front(),
                                          output_type);
        break;
    }
    case nb::CaseType::HaloMultiClustering: {
        hwtest::buildHaloMultiClusteringTest(jsonDesc, module, builder, log, input_types.front(), weightTypes.front(),
                                             output_type);
        break;
    }
    case nb::CaseType::ActShave: {
        if (jsonDesc.getActShaveBroadcastingParams().dstLocations.size() == 1) {
            hwtest::buildActShave(jsonDesc, module, builder, log, input_types, output_type);
        } else {
            hwtest::buildActShaveBroadcast(jsonDesc, module, builder, log, input_types, output_type);
        }
        break;
    }
    case nb::CaseType::ReadAfterWriteDPUDMA: {
        hwtest::buildReadAfterWriteDPUDMATest(jsonDesc, module, builder, log, input_types.front(), weightTypes.front(),
                                              output_type);
        break;
    }
    case nb::CaseType::ReadAfterWriteDMADPU: {
        hwtest::buildReadAfterWriteDMADPUTest(jsonDesc, module, builder, log, input_types.front(), weightTypes.front(),
                                              output_type);
        break;
    }
    case nb::CaseType::ReadAfterWriteACTDMA: {
        hwtest::buildReadAfterWriteACTDMATest(jsonDesc, module, builder, log, input_types.front(), output_type);
        break;
    }
    case nb::CaseType::ReadAfterWriteDMAACT: {
        hwtest::buildReadAfterWriteDMAACTTest(jsonDesc, module, builder, log, input_types.front(), output_type);
        break;
    }
    case nb::CaseType::ReadAfterWriteDPUACT: {
        hwtest::buildReadAfterWriteDPUACTTest(jsonDesc, module, builder, log, input_types.front(), weightTypes.front(),
                                              output_type);
        break;
    }
    case nb::CaseType::ReadAfterWriteACTDPU: {
        hwtest::buildReadAfterWriteACTDPUTest(jsonDesc, module, builder, log, input_types.front(), weightTypes.front(),
                                              output_type);
        break;
    }
    case nb::CaseType::RaceConditionDMA: {
        hwtest::buildRaceConditionDMATest(jsonDesc, module, builder, log, input_types.front(), output_type);
        break;
    }
    case nb::CaseType::RaceConditionDPU: {
        hwtest::buildRaceConditionDPUTest(jsonDesc, module, builder, log, input_types.front(), weightTypes.front(),
                                          output_type);
        break;
    }
    case nb::CaseType::RaceConditionDPUDMA: {
        hwtest::buildRaceConditionDPUDMATest(jsonDesc, module, builder, log, input_types.front(), weightTypes.front(),
                                             output_type);
        break;
    }
    case nb::CaseType::RaceConditionDPUDMAACT: {
        hwtest::buildRaceConditionDPUDMAACTTest(jsonDesc, module, builder, log, input_types.front(),
                                                weightTypes.front(), output_type);
        break;
    }
    case nb::CaseType::RaceConditionDPUACT: {
        hwtest::buildRaceConditionDPUACTTest(jsonDesc, module, builder, log, input_types.front(), weightTypes.front(),
                                             output_type);
        break;
    }
    case nb::CaseType::RaceCondition: {
        hwtest::buildRaceConditionTest(jsonDesc, module, builder, log, input_types.front(), output_type);
        break;
    }
    case nb::CaseType::M2iTask: {
        hwtest::buildM2iTest(jsonDesc, module, builder, log, input_types.front(), output_type);
        break;
    }
    case nb::CaseType::StorageElementTableDPU: {
        hwtest::buildSETableTest(jsonDesc, module, builder, log, input_types.front(), weightTypes.front(), output_type);
        break;
    }
    case nb::CaseType::DualChannelDMA: {
        hwtest::buildDualChannelDMATest(jsonDesc, module, builder, log, input_types.front(), output_type);
        break;
    }
    case nb::CaseType::GenerateScaleTable: {
        hwtest::buildGenerateScaleTableTest(jsonDesc, module, builder, log, input_types.front(), weightTypes.front(),
                                            output_type);
        break;
    }
    default:
        VPUX_THROW("Unknown type: {0}", opType);
        break;
    };

    // We need to add DMA profiling memory IE.MemoryResource
    if (jsonDesc.getArchitecture() == vpux::VPU::ArchKind::NPU40XX) {
        auto memSpaceAttr = mlir::SymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));
        auto op = IE::setDmaProfilingReservedMemory(module, memSpaceAttr, VPUIP::HW_DMA_PROFILING_SIZE_BYTES_40XX);
        const auto DMA_HWP_SCRATCH_BUFFER_OFFSET =
                VPU::getTotalCMXSize(module).count() - VPUIP::HW_DMA_PROFILING_SIZE_BYTES_40XX;
        op.setOffset(DMA_HWP_SCRATCH_BUFFER_OFFSET);
    }

    // llvm::dbgs() << "Current module: " << mlir::debugString(module);

    VPUX_THROW_UNLESS(mlir::succeeded(mlir::verify(module)), "Failed to create a valid MLIR module for the IR model");

    mlir::PassManager pm(module->getName(), mlir::OpPassManager::Nesting::Implicit);

    auto getLoweringPipeline = [&jsonDesc](vpux::VPU::ArchKind arch, nb::ProfilingParams /*profParams*/,
                                           mlir::OpPassManager& pm, Logger log) {
        switch (arch) {
        case vpux::VPU::ArchKind::NPU40XX: {
            auto backendCompilationOptions40XX = BackendCompilationOptions40XX();
            if (jsonDesc.getCaseType() == nb::CaseType::DMA && jsonDesc.getDMAparams().testMemSideCache == true &&
                jsonDesc.getDMAparams().cacheEnabled == false) {
                backendCompilationOptions40XX.enableMemorySideCache = false;
            }

            backendCompilationOptions40XX.enablePartialWorkloadManagement = jsonDesc.getWLMParams().isWLMPartialEnabled;
            return vpux::arch40xx::buildLowerVPUIP2ELFPipeline(pm, backendCompilationOptions40XX, log);
        }
        default:
            return vpux::arch37xx::buildLowerVPUIP2ELFPipeline(pm, log);
        }
    };

    getLoweringPipeline(jsonDesc.getArchitecture(), jsonDesc.getProfilingParams(), pm, log);

    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Failed to lower test model to ELF");

    return module;
}

}  // namespace vpux
