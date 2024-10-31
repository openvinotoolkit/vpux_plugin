//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <climits>
#include <numeric>

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/Support/DebugStringHelper.h>

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/nce_sparsity_converters.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/platform_resources.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/numeric.hpp"

namespace vpux {
namespace hwtest {

void buildMaxPool(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                  Logger& log, mlir::Type inputType, mlir::Type outputType) {
    auto* ctx = builder.getContext();

    const auto arch = testDesc.getArchitecture();
    auto profilingParams = testDesc.getProfilingParams();
    auto input = testDesc.getInputLayerList().front();
    auto poolOp = testDesc.getPoolLayer();
    auto output = testDesc.getOutputLayers().front();

    SmallVector<int64_t> inputShape(input.shape.begin(), input.shape.end());
    SmallVector<int64_t> outputShape(output.shape.begin(), output.shape.end());

    // set runtime resources
    std::optional<vpux::Byte> availableCMXMemory = std::nullopt;

    mlir::PassManager pmBuilderInit(module->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(arch, VPU::CompilationMode::DefaultHW);
    initCompilerOptions.numberOfDPUGroups = 1;
    initCompilerOptions.numberOfDMAPorts = 1;
    initCompilerOptions.setAvailableCMXMemory(availableCMXMemory);
    VPU::buildInitCompilerPipeline(pmBuilderInit, initCompilerOptions, log);

    VPUX_THROW_UNLESS(mlir::succeeded(pmBuilderInit.run(module)), "Init compilation failed");

    // Allocate a buffer to store N DMA HW profiling entries
    const size_t HWP_DMA_BUFFER_SIZE = 4;

    int64_t dmaProfilingBufferSizeBytes = 0;
    int64_t dpuProfilingBufferSizeBytes = 0;
    int64_t workpointProfilingBufferSizeBytes = 0;
    int64_t totalProfilingBufferSizeBytes = 0;
    if (profilingParams.dpuProfilingEnabled) {
        dpuProfilingBufferSizeBytes = HWP_DPU_BYTES_PER_ENTRY;
        totalProfilingBufferSizeBytes += dpuProfilingBufferSizeBytes;
    }
    if (profilingParams.dmaProfilingEnabled) {
        totalProfilingBufferSizeBytes =
                vpux::alignValUp(totalProfilingBufferSizeBytes, Byte(HWP_DMA_BYTES_PER_ENTRY).count());
        dmaProfilingBufferSizeBytes = HWP_DMA_BYTES_PER_ENTRY * (HWP_DMA_BUFFER_SIZE + 1);
        totalProfilingBufferSizeBytes += dmaProfilingBufferSizeBytes;
    }
    if (profilingParams.workpointEnabled) {
        totalProfilingBufferSizeBytes =
                vpux::alignValUp(totalProfilingBufferSizeBytes, Byte(HWP_PLL_WORKPOINT_BYTES_PER_ENTRY).count());
        workpointProfilingBufferSizeBytes = HWP_PLL_WORKPOINT_BYTES_PER_ENTRY;
        totalProfilingBufferSizeBytes += workpointProfilingBufferSizeBytes;
    }

    SmallVector<int64_t> dpuProfShapeUI64{dpuProfilingBufferSizeBytes / 8};
    SmallVector<int64_t> dmaProfShapeUI64{dmaProfilingBufferSizeBytes / 8};
    SmallVector<int64_t> workpointProfShapeUI64{workpointProfilingBufferSizeBytes / 8};

    SmallVector<int64_t> profilingOutputShapeUI64{totalProfilingBufferSizeBytes / 8};

    VPUX_THROW_UNLESS(inputShape.size() >= 4, "buildMaxPool: Input rank is less than 4");
    VPUX_THROW_UNLESS(outputShape.size() >= 4, "buildMaxPool: Output rank is less than 4");

    std::vector<int64_t> filterSize{poolOp.kernel_shape.at(0), poolOp.kernel_shape.at(1)};
    std::vector<int64_t> strideVec(poolOp.stride.begin(), poolOp.stride.end());
    std::vector<int64_t> paddingVec = convertNBPadtoNCETaskPad(poolOp.pad);

    auto inputTotalSize = totalTensorSize(inputShape, inputType);
    auto outputTotalSize = totalTensorSize(outputShape, outputType);
    auto profOutputTotalSize = totalTensorSize(profilingOutputShapeUI64, getUInt64Type(ctx));

    const auto OUTPUT_CMX_OFFSET = 0;
    const auto INPUT0_CMX_OFFSET = OUTPUT_CMX_OFFSET + outputTotalSize;
    const auto PROF_OUTPUT_CMX_OFFSET = INPUT0_CMX_OFFSET + inputTotalSize;
    const auto WEIGHTSTABLE_CMX_OFFSET = PROF_OUTPUT_CMX_OFFSET + profOutputTotalSize;

    auto inputParamType = getMemRefType(VPURT::BufferSection::NetworkInput, inputShape, inputType, DimsOrder::NHWC);
    auto outputParamType = getMemRefType(VPURT::BufferSection::NetworkOutput, outputShape, outputType, DimsOrder::NHWC);
    auto profOutputParamType = getMemRefType(VPURT::BufferSection::ProfilingOutput, profilingOutputShapeUI64,
                                             getUInt64Type(ctx), DimsOrder::C);
    int32_t dmaHwpId = 0;

    SmallVector<mlir::Type> inputTypes{inputParamType, outputParamType};
    SmallVector<mlir::Type> outputTypes{outputParamType};
    if (profilingParams.profilingEnabled()) {
        inputTypes.push_back(profOutputParamType);
        outputTypes.push_back(profOutputParamType);
    }

    const auto funcType = builder.getFunctionType(ArrayRef(inputTypes), ArrayRef(outputTypes));

    auto func = builder.create<mlir::func::FuncOp>(
            builder.getUnknownLoc(), printToString("maxpool_{0}_{1}", inputType, outputType), funcType,
            builder.getStringAttr("private"), /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr);

    auto funcBuilder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), builder.getListener());

    // Build VPUIP ops
    auto funcInput0 = func.getArgument(0);
    auto funcOutput = func.getArgument(1);
    auto funcProfOutput = profilingParams.profilingEnabled() ? func.getArgument(2) : nullptr;

    // input - output cmx tensors
    auto input0CmxType = getMemRefType(VPURT::BufferSection::CMX_NN, 0, inputShape, inputType, DimsOrder::NHWC);
    auto input0Cmx =
            createDeclareTensorOp(funcBuilder, input0CmxType, VPURT::BufferSection::CMX_NN, 0, INPUT0_CMX_OFFSET);

    auto output0CmxType = getMemRefType(VPURT::BufferSection::CMX_NN, 0, outputShape, outputType, DimsOrder::NHWC);
    auto output0Cmx =
            createDeclareTensorOp(funcBuilder, output0CmxType, VPURT::BufferSection::CMX_NN, 0, OUTPUT_CMX_OFFSET);

    auto parentInput0Cmx =
            createDeclareTensorOp(funcBuilder, input0CmxType, VPURT::BufferSection::CMX_NN, 0, INPUT0_CMX_OFFSET);
    auto parentOutput0Cmx =
            createDeclareTensorOp(funcBuilder, output0CmxType, VPURT::BufferSection::CMX_NN, 0, OUTPUT_CMX_OFFSET);

    VPURT::DeclareBufferOp dpuProfOutput0Cmx;
    VPURT::DeclareBufferOp dpuProfOutput0Ddr;
    VPURT::DeclareBufferOp dmaProfBufferDdr;
    VPURT::DeclareBufferOp dmaProfOutput0Ddr;

    if (profilingParams.profilingEnabled()) {
        size_t offset = 0;
        if (profilingParams.dpuProfilingEnabled) {
            auto dpuProfOutput0CmxType =
                    getMemRefType(VPURT::BufferSection::CMX_NN, 0, dpuProfShapeUI64, getUInt64Type(ctx), DimsOrder::C);
            dpuProfOutput0Cmx = createDeclareTensorOp(funcBuilder, dpuProfOutput0CmxType, VPURT::BufferSection::CMX_NN,
                                                      0, PROF_OUTPUT_CMX_OFFSET);
            dpuProfOutput0Ddr = createDeclareTensorOp(funcBuilder,
                                                      getMemRefType(VPURT::BufferSection::ProfilingOutput,
                                                                    dpuProfShapeUI64, getUInt64Type(ctx), DimsOrder::C),
                                                      VPURT::BufferSection::ProfilingOutput, 0, offset);
            offset += dpuProfilingBufferSizeBytes;
        }
        if (profilingParams.dmaProfilingEnabled) {
            offset = vpux::alignValUp(offset, HWP_DMA_BYTES_PER_ENTRY);
            dmaProfBufferDdr = createDeclareTensorOp(
                    funcBuilder,
                    getMemRefType(VPURT::BufferSection::DDR, dmaProfShapeUI64, getUInt64Type(ctx), DimsOrder::C),
                    VPURT::BufferSection::DDR, 0);
            dmaProfOutput0Ddr = createDeclareTensorOp(funcBuilder,
                                                      getMemRefType(VPURT::BufferSection::ProfilingOutput,
                                                                    dmaProfShapeUI64, getUInt64Type(ctx), DimsOrder::C),
                                                      VPURT::BufferSection::ProfilingOutput, 0, offset);
            offset += dmaProfilingBufferSizeBytes;
        }
        if (profilingParams.workpointEnabled) {
            offset = vpux::alignValUp(offset, HWP_PLL_WORKPOINT_BYTES_PER_ENTRY);
            createDeclareTensorOp(funcBuilder,
                                  getMemRefType(VPURT::BufferSection::ProfilingOutput, workpointProfShapeUI64,
                                                getUInt64Type(ctx), DimsOrder::C),
                                  VPURT::BufferSection::ProfilingOutput, 0, offset);
            offset += workpointProfilingBufferSizeBytes;
        }
    }

    auto [waitWLMBarrier, freeBarrierId] =
            insertWLMStartSequence(funcBuilder, testDesc.getWLMParams().isWLMPartialEnabled);

    // barrier config
    auto barrier0 = funcBuilder.create<VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), freeBarrierId++);
    auto barrier1 = funcBuilder.create<VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), freeBarrierId++);
    // finalBarrier passed as production barrier to last DMA task
    auto finalBarrier = funcBuilder.create<vpux::VPURT::ConfigureBarrierOp>(
            builder.getUnknownLoc(), freeBarrierId++, testDesc.getWLMParams().isWLMPartialEnabled);

    // DMA input-->cmx
    auto nndmaOp = VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
            funcBuilder, waitWLMBarrier, barrier0.getBarrier(),
            mlir::NameLoc::get(mlir::StringAttr::get(ctx, "maxpool?t_MaxPool/cluster_0")), funcInput0,
            input0Cmx.getOperation()->getResult(0), 0);

    if (profilingParams.dmaProfilingEnabled) {
        dmaHwpId++;
        nndmaOp.setDmaHwpIdAttr(
                mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Signed), dmaHwpId));
        nndmaOp.setProfilingMetadataAttr(
                VPUIP::DmaProfilingMetadataAttr::get(ctx, getIntAttr(ctx, dmaHwpId), /*profBegin=*/nullptr));
    }

    mlir::Value wtTblValue;
    if (outputType.dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>()) {
        // weights table ddr tensor
        SmallVector<int64_t> wtTblDataShape{output.shape[1], 1, 1, 4};
        auto wtTblDataDdrType = getMemRefType(VPURT::BufferSection::DDR, wtTblDataShape,
                                              builder.getIntegerType(32, true), DimsOrder::NHWC);
        const auto wtTblDataDdrValueType =
                mlir::RankedTensorType::get(wtTblDataShape, builder.getIntegerType(32, /*isSigned=*/true));

        const auto ppeConverter = VPU::NCESparsity::getPPEConverterCb(arch);
        const auto biasConverter = VPU::NCESparsity::getBiasConverterCb(arch);
        const std::vector<int32_t> wtTblDataValuesVec = VPU::NCESparsity::getWeightsTable(
                inputType, outputType, /*weightsPtrs*/ std::nullopt, static_cast<int32_t>(0), 0,
                static_cast<int32_t>(0), ppeConverter, biasConverter, output.shape[1]);

        auto wtTblDataValues = ArrayRef<int32_t>(wtTblDataValuesVec);
        auto wtTblDataVals = mlir::DenseElementsAttr::get(wtTblDataDdrValueType, wtTblDataValues);
        auto wtTblDataDdr = funcBuilder.create<Const::DeclareOp>(
                builder.getUnknownLoc(), wtTblDataDdrType,
                Const::ContentAttr::transform(wtTblDataVals).reorder(DimsOrder::NHWC).get());

        // weights table cmx tensor

        auto wtTblCmxType = getMemRefType(VPURT::BufferSection::CMX_NN, 0, wtTblDataShape,
                                          builder.getIntegerType(32, true), DimsOrder::NHWC);
        auto wtTblCmx = createDeclareTensorOp(funcBuilder, wtTblCmxType, VPURT::BufferSection::CMX_NN, 0,
                                              WEIGHTSTABLE_CMX_OFFSET);
        wtTblValue = wtTblCmx.getOperation()->getResult(0);

        // weights table dma ddr->cmx
        auto weightsTableDma = VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
                funcBuilder, mlir::ValueRange(), mlir::ValueRange(barrier0.getBarrier()),
                mlir::NameLoc::get(mlir::StringAttr::get(ctx, "maxpool?t_MaxPool/cluster_0")),
                wtTblDataDdr.getOperation()->getResult(0), wtTblCmx.getOperation()->getResult(0), 0);

        if (profilingParams.dmaProfilingEnabled) {
            dmaHwpId++;
            weightsTableDma.setDmaHwpIdAttr(
                    mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Signed), dmaHwpId));
            weightsTableDma.setProfilingMetadataAttr(
                    VPUIP::DmaProfilingMetadataAttr::get(ctx, getIntAttr(ctx, dmaHwpId), /*profBegin=*/nullptr));
        }
    }

    // NCE Task
    auto filtersize = getIntArrayAttr(builder, filterSize);
    auto strides = getIntArrayAttr(builder, strideVec);
    auto kernelPadding = VPU::getPaddingAttr(ctx, paddingVec[PAD_NCETASK_LEFT], paddingVec[PAD_NCETASK_RIGHT],
                                             paddingVec[PAD_NCETASK_TOP], paddingVec[PAD_NCETASK_BOTTOM]);

    const auto taskName = "maxpool?t_MaxPool/cluster_0";

    auto nceTask = VPURT::wrapIntoTaskOp<VPUIP::NCEClusterTaskOp>(
            funcBuilder, mlir::ValueRange(barrier0.getBarrier()), mlir::ValueRange(barrier1.getBarrier()),
            mlir::NameLoc::get(mlir::StringAttr::get(ctx, taskName)), input0Cmx.getOperation()->getResult(0),
            mlir::Value(), wtTblValue,
            /*instruction_table_list*/ nullptr, /*spr_lookup_table*/ nullptr,
            parentInput0Cmx.getOperation()->getResult(0), parentOutput0Cmx.getOperation()->getResult(0),
            output0Cmx.getOperation()->getResult(0),
            profilingParams.dpuProfilingEnabled ? dpuProfOutput0Cmx.getOperation()->getResult(0) : nullptr,
            VPUIP::NCETaskType::MAXPOOL, filtersize, strides, kernelPadding,
            /*is_continued*/ nullptr, /*sp_pattern*/ nullptr, /*is_segmented*/ nullptr,
            /*out_channel_offset*/ nullptr, /*input_channels_compression*/ nullptr, /*is_superdense*/ nullptr,
            /*is_inplace*/ nullptr, /*input_se_size*/ nullptr, /*output_se_size*/ nullptr,
            /*is_permute_quantize*/ nullptr, /*is_small_kernel_optimized*/ nullptr);
    if (profilingParams.dpuProfilingEnabled) {
        auto profAttr = VPUIP::DpuProfilingMetadataAttr::get(
                ctx, /*bufferId*/ getIntAttr(ctx, 0),
                /*taskId*/ getIntAttr(ctx, 1), /*maxVariants*/ getIntAttr(ctx, 1),
                /*numVariants*/ getIntAttr(ctx, 1), /*clusterId*/ getIntAttr(ctx, 0));
        nceTask.setProfilingMetadataAttr(profAttr);
    }

    int64_t clampLow = std::numeric_limits<int32_t>::min();
    int64_t clampHigh = std::numeric_limits<int32_t>::max();
    if (auto outElemQType = outputType.template dyn_cast<mlir::quant::QuantizedType>()) {
        const auto zps = extractScalesAndZeroPoints(outputType).second;
        clampLow = outElemQType.getStorageTypeMin() - zps.front();
        clampHigh = outElemQType.getStorageTypeMax() - zps.front();
    }
    int64_t bypassMult = 1;
    int64_t bypassShift = 0;
    auto ppeAttr = VPU::PPEIntAttr::get(
            ctx, VPU::PPEModeAttr::get(ctx, VPU::PPEMode::NOOP), vpux::getIntAttr(ctx, clampLow),
            vpux::getIntAttr(ctx, clampHigh), vpux::getIntAttr(ctx, bypassMult), vpux::getIntAttr(ctx, bypassShift),
            /* quantScale = */ nullptr, /* quantMult = */ nullptr, /* quantShift = */ nullptr,
            /* quantPostShift = */ nullptr, /* in1QuantMult = */ nullptr,
            /* in2QuantMult = */ nullptr,
            /* fpPreluAlpha = */ nullptr);
    nceTask.addPPETask(funcBuilder, ppeAttr);

    // Create DPU task for NCE task
    auto variantbuilder = mlir::OpBuilder::atBlockBegin(&nceTask.getVariants().front(), builder.getListener());
    auto dpuTask = createDPUTaskOp(funcBuilder, variantbuilder, outputShape, inputShape, paddingVec,
                                   VPU::MPEMode::CUBOID_16x16, /* clusterId */ 0);

    // copy NCE task output to function output
    auto outputDma = VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
            funcBuilder, mlir::ValueRange(barrier1.getBarrier()), mlir::ValueRange(finalBarrier.getBarrier()),
            mlir::NameLoc::get(mlir::StringAttr::get(ctx, "maxpool?t_MaxPool/cluster_0")),
            output0Cmx.getOperation()->getResult(0), funcOutput, 0);

    if (profilingParams.dmaProfilingEnabled) {
        dmaHwpId++;
        outputDma.setDmaHwpIdAttr(
                mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Signed), dmaHwpId));
        outputDma.setProfilingMetadataAttr(
                VPUIP::DmaProfilingMetadataAttr::get(ctx, getIntAttr(ctx, dmaHwpId), /*profBegin=*/nullptr));
    }

    if (profilingParams.dpuProfilingEnabled) {
        // align DPU HWP buffer to HWP_DPU_BYTES_PER_ENTRY bytes
        dpuTask.setWorkloadIdAttr(vpux::getIntAttr(ctx, PROF_OUTPUT_CMX_OFFSET / HWP_DPU_BYTES_PER_ENTRY));

        // copy NCE task profiling data into DDR
        VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcBuilder, mlir::ValueRange(barrier1.getBarrier()),
                                              mlir::ValueRange(finalBarrier.getBarrier()), builder.getUnknownLoc(),
                                              dpuProfOutput0Cmx.getOperation()->getResult(0),
                                              dpuProfOutput0Ddr.getOperation()->getResult(0), 0);
    }

    if (profilingParams.dmaProfilingEnabled) {
        // copy DMA profiling data into DDR
        VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcBuilder, mlir::ValueRange(barrier1.getBarrier()),
                                              mlir::ValueRange(finalBarrier.getBarrier()), builder.getUnknownLoc(),
                                              dmaProfBufferDdr.getOperation()->getResult(0),
                                              dmaProfOutput0Ddr.getOperation()->getResult(0), 0);
    }

    // create ReturnOp
    mlir::SmallVector<mlir::Value> funcOutputs;
    funcOutputs.push_back(funcOutput);
    if (profilingParams.profilingEnabled()) {
        funcOutputs.push_back(funcProfOutput);
    }
    funcBuilder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), funcOutputs);

    // IE.CNNNetwork
    mlir::SmallVector<ProfilingDataSection> profilingDataSections;
    size_t offset = 0;
    if (profilingParams.dpuProfilingEnabled) {
        profilingDataSections.push_back({HWP_DPU_SECTION_EXEC_TYPE, offset, dpuProfilingBufferSizeBytes});
        offset += dpuProfilingBufferSizeBytes;
    }
    if (profilingParams.dmaProfilingEnabled) {
        VPUX_THROW_WHEN(static_cast<size_t>(dmaHwpId) > HWP_DMA_BUFFER_SIZE,
                        "Only {0} entries are reserved for DMA profiling (while {1} "
                        "are profiled), HWP_DMA_BUFFER_SIZE needs to be increased",
                        HWP_DMA_BUFFER_SIZE, dmaHwpId);
        offset = vpux::alignValUp(offset, HWP_DMA_BYTES_PER_ENTRY);
        profilingDataSections.push_back({HWP_DMA_SECTION_EXEC_TYPE, offset, dmaProfilingBufferSizeBytes});
        offset += dmaProfilingBufferSizeBytes;
    }
    if (profilingParams.workpointEnabled) {
        offset = vpux::alignValUp(offset, HWP_DMA_BYTES_PER_ENTRY);
        profilingDataSections.push_back({HWP_WORKPOINT_SECTION_EXEC_TYPE, offset, workpointProfilingBufferSizeBytes});
        offset += workpointProfilingBufferSizeBytes;
    }
    if (profilingParams.profilingEnabled()) {
        auto memSpaceAttr = mlir::SymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));
        IE::setDmaProfilingReservedMemory(module, memSpaceAttr, HWP_DMA_PROFILING_MAX_BUFFER_SIZE);
        IE::setDmaProfilingReservedMemory(module, mlir::SymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::DDR)),
                                          HWP_DMA_ID_LIMIT * HWP_DMA_BYTES_PER_ENTRY);
        // set offset for reserved DDR memory manually to 0 so that SetupProfilingVPUMI40XX pass works
        IE::MemoryResourceOp ddrReservedMem = IE::getDmaProfilingReservedMemory(module, VPU::MemoryKind::DDR);
        ddrReservedMem.setOffsetAttr(getIntAttr(ctx, 0));
    }
    buildCNNOp(builder, func.getName(), {getTensorType(ShapeRef(inputShape), inputType, DimsOrder::NHWC, nullptr)},
               {getTensorType(ShapeRef(outputShape), outputType, DimsOrder::NHWC, nullptr)}, profilingDataSections);
}

}  // namespace hwtest
}  // namespace vpux
