//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/graph-schema/export.hpp"

#include "vpux/compiler/compiler_version.hpp"
#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/core/profiling_metadata.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/generated/schema/gf_version.h"
#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_writer.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/schema.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"

#include "vpux/compiler/dialect/VPU/utils/performance_metrics.hpp"
#include "vpux/compiler/utils/loop.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/string_ref.hpp"
#include "vpux/utils/profiling/metadata.hpp"

#include <llvm/ADT/DenseMap.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>

#include <transformations/utils/utils.hpp>

#include <unordered_map>

using namespace vpux;

namespace {

flatbuffers::Offset<MVCNN::Version> createVersion(VPUIP::BlobWriter& writer, Logger log) {
    log.info("Blob version: majorV={0}, minorV={1}, patch={2}, hash={3}, context={4}", MVCNN_VERSION_MAJOR,
             MVCNN_VERSION_MINOR, MVCNN_VERSION_PATCH, VPUX_COMPILER_VERSION, "NPU Compiler");

    const auto serializedHash = writer.createString(VPUX_COMPILER_VERSION);
    const auto serializedContext = writer.createString("NPU Compiler");

    MVCNN::VersionBuilder builder(writer);
    builder.add_majorV(checked_cast<uint32_t>(MVCNN_VERSION_MAJOR));
    builder.add_minorV(checked_cast<uint32_t>(MVCNN_VERSION_MINOR));
    builder.add_patchV(checked_cast<uint32_t>(MVCNN_VERSION_PATCH));
    builder.add_hash(serializedHash);
    builder.add_context(serializedContext);
    return builder.Finish();
}

flatbuffers::Offset<MVCNN::Resources> createResources(VPUIP::BlobWriter& writer, mlir::ModuleOp module) {
    const EnumSet<VPU::ExecutorKind> supportedProcessors{
            VPU::ExecutorKind::SHAVE_UPA,  //
            VPU::ExecutorKind::SHAVE_NN,   //
            VPU::ExecutorKind::DPU         //
    };

    const auto usedMemory = writer.createVector(IE::getUsedMemory(module) | transformed([&](IE::MemoryResourceOp res) {
                                                    return createMemoryMapping(writer, res);
                                                }));

    SmallVector<flatbuffers::Offset<MVCNN::ProcessorMapping>> executorsOffsets;
    SmallVector<flatbuffers::Offset<MVCNN::ProcessorMapping>> processorVec;
    module.walk([&](IE::TileResourceOp res) {
        executorsOffsets.push_back(VPUIP::createProcessorMapping(writer, res, module));
        if (res.hasProcessorFrequency()) {
            processorVec.push_back(VPUIP::createProcessorFreqMapping(writer, res));
        }
    });
    module.walk([&](IE::ExecutorResourceOp res) {
        const auto execKind = VPU::getKindValue<VPU::ExecutorKind>(res);
        if (supportedProcessors.count(execKind) != 0) {
            executorsOffsets.push_back(createProcessorMapping(writer, res, module));
        }
    });
    const auto executors = writer.createVector(executorsOffsets);
    const auto processorFrequency = writer.createVector(processorVec);

    SmallVector<flatbuffers::Offset<MVCNN::MemoryRelationshipMapping>> memoryVec;
    SmallVector<IE::MemoryResourceOp> memoryTypes;
    for (auto src : module.getOps<IE::MemoryResourceOp>()) {
        if (src->hasAttr(VPU::getMemoryBandwidthAttrName())) {
            memoryTypes.push_back(src);
        }
    }

    double DMA_BANDWIDTH = vpux::VPU::getDmaBandwidthGBps(module);
    for (auto src : memoryTypes) {
        for (auto dst : memoryTypes) {
            // TODO E#20897: update calculations with the below factors:
            // auto memoryBandwidth = VPUIP::getMemoryBandwidth(src);
            // auto memoryDerateFactor = VPUIP::getMemoryDerateFactor(src);
            if (src != dst) {
                memoryVec.push_back(createBandwidthMapping(writer, src, dst, DMA_BANDWIDTH));
            }
        }
    }

    const auto memoryBandwidthVec = writer.createVector(memoryVec);

    MVCNN::ResourcesBuilder builder(writer);
    builder.add_processor_allocation(executors);
    builder.add_memory_sizes(usedMemory);
    builder.add_processor_frequencies(processorFrequency);
    builder.add_memory_bandwidth(memoryBandwidthVec);
    return builder.Finish();
}

flatbuffers::Offset<MVCNN::PerformanceMetrics> createPerformanceMetrics(VPUIP::BlobWriter& writer,
                                                                        mlir::ModuleOp module) {
    // same range for each entry
    auto byBWScales = VPU::getBWScales();
    auto numEntries = VPU::getNumEntries();

    SmallVector<flatbuffers::Offset<MVCNN::ScalabilityByBandwidth>> scaleByFreq;
    SmallVector<flatbuffers::Offset<MVCNN::InferenceTimingByBandwidth>> inferenceTimingByFreq;
    scaleByFreq.reserve(numEntries);
    inferenceTimingByFreq.reserve(numEntries);

    auto byBWTicks = VPU::getBWTicks(module);
    for (uint32_t i = 0; i < numEntries; ++i) {
        scaleByFreq.push_back(MVCNN::CreateScalabilityByBandwidth(writer, writer.createVector(byBWScales)));
        inferenceTimingByFreq.push_back(
                MVCNN::CreateInferenceTimingByBandwidth(writer, writer.createVector(byBWTicks[i])));
    }

    auto arch = vpux::VPU::getArch(module);
    auto freqBase = VPU::getFreqBase(arch);
    auto freqStep = VPU::getFreqStep(arch);
    return MVCNN::CreatePerformanceMetrics(
            writer, freqBase, freqStep, VPU::getBWBase(), VPU::getBWStep(),
            MVCNN::CreateScalability(writer, writer.createVector(scaleByFreq)),
            MVCNN::CreateInferenceTiming(writer, writer.createVector(inferenceTimingByFreq)));
}

flatbuffers::Offset<MVCNN::ActKernelRuntime> createActKernelRuntime(VPUIP::BlobWriter& writer, mlir::ModuleOp module,
                                                                    mlir::func::FuncOp netFunc, Logger log) {
    // only SwKernelOp operations can generate kernelData
    auto graphHasKernels = false;
    netFunc.walk([&](VPURT::TaskOp taskOp) {
        if (taskOp.getExecutorKind() == VPU::ExecutorKind::SHAVE_ACT) {
            graphHasKernels = true;
        }
    });
    if (!graphHasKernels) {
        return {};
    }

    // TODO: always extract num shaves info from VPURT::SW.Runtime, which can be extracted from module
    auto maxShaves = 4;
    if (vpux::VPU::getArch(module) == vpux::VPU::ArchKind::NPU40XX) {
        maxShaves = 12;
    }
    constexpr auto defaultStackSize = Byte(4_KB).count();
    constexpr auto alignmentReq = Byte(1_KB).count();

    auto swRuntimeOps = module.getOps<VPURT::SWRunTimeOp>();
    auto runtimeKernelDeclared = std::distance(swRuntimeOps.begin(), swRuntimeOps.end());
    VPUX_THROW_UNLESS(runtimeKernelDeclared <= 1, "if runtime kernel is present it should be unique, but found {0}",
                      runtimeKernelDeclared);

    flatbuffers::Offset<MVCNN::ActKernel> runtimeKernel;
    SmallVector<uint32_t> runtimeStacks(maxShaves, defaultStackSize);

    if (runtimeKernelDeclared) {
        auto swRuntimeOp = *swRuntimeOps.begin();
        runtimeKernel = writer.createRuntimeKernelTask(module, swRuntimeOp);
        runtimeStacks = parseIntArrayAttr<uint32_t>(swRuntimeOp.getStacks());
    }

    SmallVector<flatbuffers::Offset<MVCNN::KernelDataReference>> stacks(maxShaves);

    const auto maxStackIt = std::max_element(runtimeStacks.begin(), runtimeStacks.end());
    const std::vector<uint8_t> shave_stack_data_max(*maxStackIt);

    for (auto shvInd : irange(maxShaves)) {
        const auto shaveStackData = ArrayRef(shave_stack_data_max).take_front(runtimeStacks[shvInd]);

        log.trace("act-shave {0}_stack size is {1}", shvInd, shaveStackData.size());

        const auto dataName = printToString("actSHAVE{0}_stack", shvInd);
        stacks[shvInd] = writer.createKernelDataRef(dataName, 0, shaveStackData.size(), shaveStackData);
    }

    const auto stackBuffers = writer.createVector(stacks);

    VPUIP::BlobWriter::KernelDataRef scratchBuffer;
    if (!runtimeKernelDeclared) {
        constexpr auto scratchBufferSize = Byte(64_KB).count();

        const std::vector<uint8_t> scratchBufferData(alignmentReq + scratchBufferSize);
        constexpr uint64_t reservedOffset = 1;

        scratchBuffer =
                writer.createKernelDataRef("scratch_buffer", reservedOffset, scratchBufferSize, scratchBufferData);
    }

    SmallVector<int8_t> counters = {
            MVCNN::ActKernelPerfCounter_STALL_CYCLE_CNT_EN, MVCNN::ActKernelPerfCounter_EXEC_INST_CNT_EN,
            MVCNN::ActKernelPerfCounter_CLK_CYCLE_CNT_EN, MVCNN::ActKernelPerfCounter_BRANCH_TAKEN_CNT_EN};
    SmallVector<int8_t> stallFilters = {
            MVCNN::ActKernelPerfStallFilter_LSU0_DATA_WAIT, MVCNN::ActKernelPerfStallFilter_LSU1_DATA_WAIT,
            MVCNN::ActKernelPerfStallFilter_LSU0_ACCESS, MVCNN::ActKernelPerfStallFilter_LSU1_ACCESS};

    const auto flatCounters = writer.createVector(counters);
    const auto flatStallFilters = writer.createVector(stallFilters);
    MVCNN::PerfPayloadConfigBuilder perfConfigBuilder(writer);
    perfConfigBuilder.add_counters(flatCounters);
    perfConfigBuilder.add_stallFilters(flatStallFilters);
    perfConfigBuilder.add_frcDuration(true);
    perfConfigBuilder.add_frcTimestamp(true);
    auto perfConfig = perfConfigBuilder.Finish();

    MVCNN::ActKernelRuntimeBuilder builder(writer);
    builder.add_shaveStacks(stackBuffers);
    if (runtimeKernelDeclared) {
        builder.add_kernel(runtimeKernel);
    } else {
        builder.add_codeScratchBuffer(scratchBuffer);
    }
    builder.add_perfConfig(perfConfig);

    return builder.Finish();
}

VPUIP::BlobWriter::TensorReference createDmaHwpBase(VPUIP::BlobWriter& writer, mlir::ModuleOp& module) {
    auto maybeDmaSection = vpux::getProfilingSection(module, profiling::ExecutorType::DMA_HW);
    VPUX_THROW_UNLESS(maybeDmaSection.has_value(), "Cannot find DMAHW section in profiling output");
    auto dmaSection = maybeDmaSection.value();

    VPUX_THROW_UNLESS((dmaSection.getOffset() % VPUIP::HW_DMA_PROFILING_SIZE_BYTES_40XX) == 0, "Unaligned HWP base");
    VPUX_THROW_UNLESS((dmaSection.getSize() % VPUIP::HW_DMA_PROFILING_SIZE_BYTES_40XX) == 0, "Bad DMA section size");

    auto ctx = module->getContext();
    const auto outputType = getMemRefType({dmaSection.getSize() / 4}, getUInt32Type(ctx), DimsOrder::C,
                                          VPURT::BufferSection::ProfilingOutput)
                                    .cast<vpux::NDTypeInterface>();

    return writer.createTensorRef("dma_hwp_base", outputType, VPURT::BufferSection::ProfilingOutput, 0,
                                  dmaSection.getOffset());
}

VPUIP::BlobWriter::TensorReference createWorkpointSectionOutput(VPUIP::BlobWriter& writer, mlir::ModuleOp& module) {
    if (VPU::getArch(module) != VPU::ArchKind::NPU40XX) {
        return {};
    }

    auto ctx = module->getContext();
    auto maybeCaptureSection = vpux::getProfilingSection(module, profiling::ExecutorType::WORKPOINT);
    if (!maybeCaptureSection.has_value()) {
        return {};
    }

    auto captureSection = maybeCaptureSection.value();
    unsigned pllSizeBytes = checked_cast<unsigned>(captureSection.getSize());
    VPUX_THROW_UNLESS(pllSizeBytes == profiling::WORKPOINT_BUFFER_SIZE, "Bad PLL section size: {0}", pllSizeBytes);

    const auto outputType =
            getMemRefType({pllSizeBytes / 4}, getUInt32Type(ctx), DimsOrder::C, VPURT::BufferSection::ProfilingOutput)
                    .cast<vpux::NDTypeInterface>();

    return writer.createTensorRef("workpoint_output", outputType, VPURT::BufferSection::ProfilingOutput, 0,
                                  captureSection.getOffset());
}

VPUIP::BlobWriter::TensorReference createDmaHwpScratchBuffer(VPUIP::BlobWriter& writer, mlir::ModuleOp& module) {
    auto ctx = module->getContext();
    const auto outputType = getMemRefType({VPUIP::HW_DMA_PROFILING_SIZE_BYTES_40XX / 4}, getUInt32Type(ctx),
                                          DimsOrder::C, VPU::MemoryKind::CMX_NN)
                                    .cast<vpux::NDTypeInterface>();
    int64_t dmaProfMemOffset = 0;
    if (auto dmaProfMem = IE::getDmaProfilingReservedMemory(module, VPU::MemoryKind::CMX_NN)) {
        VPUX_THROW_UNLESS(dmaProfMem.getOffset().has_value(), "No offset setting provided");
        dmaProfMemOffset = dmaProfMem.getOffset().value();
    }

    return writer.createTensorRef("dma_hwp_scratch", outputType, VPURT::BufferSection::CMX_NN, 0, dmaProfMemOffset);
}

flatbuffers::Offset<MVCNN::SummaryHeader> createSummaryHeader(
        VPUIP::BlobWriter& writer, mlir::ModuleOp module, IE::CNNNetworkOp netOp, mlir::func::FuncOp netFunc,
        bool withDynamicBarriers, mlir::TimingScope& rootTiming,
        const std::vector<std::shared_ptr<const ov::Node>>& parameters,
        const std::vector<std::shared_ptr<const ov::Node>>& results, Logger log) {
    auto scopeTiming = rootTiming.nest("Create summary header");

    const auto allTasks = netFunc.getOps<VPURT::TaskOp>();
    const auto allBarriers = netFunc.getOps<VPURT::ConfigureBarrierOp>();
    const auto taskCount =
            std::distance(allTasks.begin(), allTasks.end()) + std::distance(allBarriers.begin(), allBarriers.end());

    auto inputsInfo = netOp.getInputsDataInfo();
    auto outputsInfo = netOp.getOutputsDataInfo();
    auto profilingOutputsInfo = netOp.getProfilingOutputsDataInfo();

    SmallVector<VPUIP::BlobWriter::TensorReference> graphInputs, userInputs;
    graphInputs.reserve(inputsInfo.size());
    userInputs.reserve(inputsInfo.size());

    for (const auto& p : inputsInfo | indexed) {
        const auto ind = checked_cast<uint32_t>(p.index());
        auto userInfo = p.value();
        const auto val = netFunc.getArgument(ind);

        const auto userType = userInfo.getUserType().cast<vpux::NDTypeInterface>();

        graphInputs.push_back(
                writer.createTensorRef(val, userInfo.getName(), VPURT::BufferSection::NetworkInput, ind, 0));

        userInputs.push_back(
                writer.createTensorRef(userInfo.getName(), userType, VPURT::BufferSection::NetworkInput, ind, 0));
    }

    SmallVector<VPUIP::BlobWriter::TensorReference> graphOutputs, graphProfilingOutputs, userOutputs;
    graphOutputs.reserve(outputsInfo.size());
    userOutputs.reserve(outputsInfo.size());
    graphProfilingOutputs.reserve(profilingOutputsInfo.size());

    for (const auto& p : outputsInfo | indexed) {
        const auto ind = p.index();
        const auto funcArgInd = inputsInfo.size() + ind;

        auto userInfo = p.value();
        const auto val = netFunc.getArgument(checked_cast<uint32_t>(funcArgInd));

        const auto userType = userInfo.getUserType().cast<vpux::NDTypeInterface>();

        graphOutputs.push_back(
                writer.createTensorRef(val, userInfo.getName(), VPURT::BufferSection::NetworkOutput, ind, 0));

        userOutputs.push_back(
                writer.createTensorRef(userInfo.getName(), userType, VPURT::BufferSection::NetworkOutput, ind, 0));
    }

    if (vpux::isProfilingEnabled(netOp)) {
        auto profilingOutputInfo = profilingOutputsInfo.front();
        const auto profilingOutputsName = profilingOutputInfo.getName().str();

        const auto funcArgInd = inputsInfo.size() + outputsInfo.size();
        const auto val = netFunc.getArgument(checked_cast<uint32_t>(funcArgInd));

        graphProfilingOutputs.push_back(
                writer.createTensorRef(val, profilingOutputsName, VPURT::BufferSection::ProfilingOutput, 0, 0));
    }

    auto createOVNodes = [&](const std::vector<std::shared_ptr<const ov::Node>>& nodes, const bool isResult) {
        SmallVector<VPUIP::BlobWriter::OVNodes> ovNodes;
        ovNodes.reserve(nodes.size());

        for (const auto& node : nodes) {
            VPUX_THROW_WHEN(node == nullptr, "Null OV node");
            const auto nodeFriendlyName = writer.createString(node->get_friendly_name());
            const auto nodeElementType = VPUIP::mapElementType.at(node->get_element_type());
            const auto nodeShape = writer.createVector(node->get_output_partial_shape(0).get_shape());
            const auto tmpTensorNames = node->get_output_tensor(0).get_names();
            SmallVector<VPUIP::BlobWriter::String> auxTensorNames;
            for (const auto& tensorName : tmpTensorNames) {
                auxTensorNames.push_back(writer.createString(tensorName));
            }
            const auto nodeTensorNames = writer.createVector(auxTensorNames);
            const auto tmpInputName =
                    ov::op::util::create_ie_output_name(isResult ? node->input_value(0) : node->output(0));
            const auto nodeInputName = writer.createString(tmpInputName);
            ovNodes.push_back(MVCNN::CreateOVNode(writer, nodeFriendlyName, nodeElementType, nodeShape, nodeTensorNames,
                                                  nodeInputName));
        }

        return ovNodes;
    };

    const auto ovParam = createOVNodes(parameters, false);
    const auto ovRes = createOVNodes(results, true);

    SmallVector<int8_t> options;
    if (withDynamicBarriers) {
        options.push_back(static_cast<int8_t>(MVCNN::ExecutionFlag_DynamicBarriers));
    }
    const auto serializedOptions = writer.createVector(options);

    const auto serializedVersion = createVersion(writer, log);
    const auto serializedName = writer.createString(module.getName().value_or("network"));
    const auto serializedGraphInputs = writer.createVector(graphInputs);
    const auto serializedUserInputs = writer.createVector(userInputs);
    const auto serializedGraphOutputs = writer.createVector(graphOutputs);
    const auto serializedGraphProfilingOutputs = writer.createVector(graphProfilingOutputs);
    const auto serializedUserOutputs = writer.createVector(userOutputs);
    const auto serializedResources = createResources(writer, module);
    const auto serializedParameters = writer.createVector(ovParam);
    const auto serializedResults = writer.createVector(ovRes);
    const auto serializedActKernelsRuntime = createActKernelRuntime(writer, module, netFunc, log);
    const auto serializedPerformanceMetrics = createPerformanceMetrics(writer, module);
    VPUIP::BlobWriter::TensorReference serializedDmaHwpBase;
    const bool isDmaHwpUsed = isDmaHwpUsedInVPURT(module);
    if (isDmaHwpUsed) {
        serializedDmaHwpBase = createDmaHwpBase(writer, module);
    } else if (VPU::getArch(module) == VPU::ArchKind::NPU40XX) {
        // Scratch DMA HWP buffer is required in VPUX40xx series
        serializedDmaHwpBase = createDmaHwpScratchBuffer(writer, module);
    }
    VPUIP::BlobWriter::TensorReference serializedWorkpointSectionOutput = createWorkpointSectionOutput(writer, module);

    MVCNN::SummaryHeaderBuilder builder(writer);
    builder.add_version(serializedVersion);
    builder.add_identifier(serializedName);
    builder.add_net_input(serializedGraphInputs);
    builder.add_net_output(serializedGraphOutputs);
    builder.add_profiling_output(serializedGraphProfilingOutputs);
    builder.add_task_count(checked_cast<uint32_t>(taskCount));
    builder.add_options(serializedOptions);
    builder.add_resources(serializedResources);
    builder.add_in_tensor_desc(serializedUserInputs);
    builder.add_out_tensor_desc(serializedUserOutputs);
    builder.add_ov_parameters(serializedParameters);
    builder.add_ov_results(serializedResults);
    builder.add_device(VPUIP::mapTargetDevice(VPU::getArch(module)));
    builder.add_device_revision(VPUIP::mapTargetDeviceRevision(VPU::getArch(module)));
    builder.add_act_kernel_runtime(serializedActKernelsRuntime);
    builder.add_performance_metrics(serializedPerformanceMetrics);
    builder.add_profiling_data_mode(VPUIP::mapProfilingMode(VPU::getArch(module)));
    builder.add_dma_hwp_enabled(isDmaHwpUsed);
    builder.add_dma_hwp_base(serializedDmaHwpBase);
    builder.add_workpoint_output(serializedWorkpointSectionOutput);
    return builder.Finish();
}

void serializeTensorDecls(VPUIP::BlobWriter& writer, mlir::func::FuncOp netFunc, mlir::TimingScope& rootTiming) {
    auto scopeTiming = rootTiming.nest("Serialize tensor declarations");

    llvm::DenseMap<mlir::Operation*, std::tuple<std::optional<int64_t>, std::optional<int64_t>, std::optional<int64_t>>>
            sparsityInfo;
    const auto findSparsityInfo = [&](mlir::Value data, mlir::Value sparsityMap, mlir::Value storageElementTable,
                                      std::optional<int64_t> seSize = std::nullopt) {
        if (data == nullptr) {
            return;
        }
        auto dataBuffer = data.getDefiningOp<VPURT::DeclareBufferOp>();

        std::optional<int64_t> sparsityMapOffset = std::nullopt;
        std::optional<int64_t> storageElementOffset = std::nullopt;
        if (sparsityMap != nullptr) {
            auto buffer = sparsityMap.getDefiningOp<VPURT::DeclareBufferOp>();
            sparsityMapOffset = buffer.getByteOffset();
        }
        if (storageElementTable != nullptr) {
            auto buffer = storageElementTable.getDefiningOp<VPURT::DeclareBufferOp>();
            storageElementOffset = buffer.getByteOffset();
        }

        if (sparsityMapOffset.has_value() || storageElementOffset.has_value()) {
            if (sparsityInfo.find(dataBuffer) != sparsityInfo.end()) {
                return;
            }
            const auto info = std::make_tuple(sparsityMapOffset, storageElementOffset, seSize);
            sparsityInfo.insert(std::make_pair(dataBuffer.getOperation(), info));
        }
    };

    netFunc.walk([&](VPUIP::NCEClusterTaskOp nceTask) {
        findSparsityInfo(nceTask.getInput(), nceTask.getInputSparsityMap(), nceTask.getInputStorageElementTable(),
                         nceTask.getInputSeSize());
        findSparsityInfo(nceTask.getWeights(), nceTask.getWeightsSparsityMap(), nullptr);
        findSparsityInfo(nceTask.getParentInput(), nceTask.getParentInputSparsityMap(),
                         nceTask.getParentInputStorageElementTable(), nceTask.getInputSeSize());
        findSparsityInfo(nceTask.getParentOutput(), nceTask.getParentOutputSparsityMap(), nullptr,
                         nceTask.getOutputSeSize());
        findSparsityInfo(nceTask.getOutputBuff(), nceTask.getOutputSparsityMapBuff(), nullptr,
                         nceTask.getOutputSeSize());
    });

    size_t tempTensorInd = 0;
    const auto createTensorRef = [&](VPURT::DeclareBufferOp bufOp,
                                     const std::optional<int64_t> sparsityMapOffset = std::nullopt,
                                     const std::optional<int64_t> storageElementOffset = std::nullopt,
                                     const std::optional<int64_t> storageElementSize = std::nullopt) {
        auto sectionIndex = bufOp.getNonEmptySectionIndex();
        std::optional<uint64_t> descriptor;
        if (auto value = bufOp->getAttr("DescriptorHandle").dyn_cast_or_null<mlir::IntegerAttr>()) {
            descriptor = value.getUInt();
        }
        writer.createTensorRef(bufOp.getBuffer(), printToString("temp-{0}", tempTensorInd), bufOp.getSection(),
                               sectionIndex, bufOp.getByteOffset(), sparsityMapOffset, storageElementOffset,
                               storageElementSize, bufOp.getSwizzlingKey(), descriptor);
        tempTensorInd++;
    };

    netFunc.walk([&](VPURT::DeclareBufferOp bufOp) {
        std::optional<int64_t> sparsityMapOffset = std::nullopt;
        std::optional<int64_t> storageElementOffset = std::nullopt;
        std::optional<int64_t> storageElementSize = std::nullopt;
        if (sparsityInfo.find(bufOp.getOperation()) != sparsityInfo.end()) {
            std::tie(sparsityMapOffset, storageElementOffset, storageElementSize) = sparsityInfo[bufOp.getOperation()];
        }

        createTensorRef(bufOp, sparsityMapOffset, storageElementOffset, storageElementSize);
    });
}

void extendBinaryDataWithProfilingSchema(VPUIP::BlobWriter& writer, IE::CNNNetworkOp netOp, mlir::func::FuncOp netFunc,
                                         SmallVector<VPUIP::BlobWriter::BinaryData>& binaryData, Logger log) {
    log.trace("Got profiling metadata");
    auto profilingData = buildProfilingMetadataBuffer(netOp, netFunc, log);
    const auto dataSize = profilingData.size();

    // Binary data is stored as vector of uint64_t, while FB return array of uint8_t. Align buffer to be compatible with
    // FB schema
    const auto alignedDstBufferSize = alignValUp(dataSize, sizeof(uint64_t)) / sizeof(uint64_t);
    std::vector<uint64_t> alignedBuffer(alignedDstBufferSize, 0);
    std::memcpy(alignedBuffer.data(), profilingData.data(), dataSize);
    binaryData.push_back(writer.createBinaryData(alignedBuffer, dataSize));
}

SmallVector<VPUIP::BlobWriter::BinaryData> serializeBinaryData(VPUIP::BlobWriter& writer, mlir::func::FuncOp netFunc,
                                                               IE::CNNNetworkOp netOp, mlir::TimingScope& rootTiming,
                                                               Logger log) {
    auto scopeTiming = rootTiming.nest("Serialize binary data");

    auto constOps = to_small_vector(netFunc.getOps<Const::DeclareOp>());
    size_t opsToSerializeCount = constOps.size();
    if (vpux::isProfilingEnabled(netOp)) {
        ++opsToSerializeCount;
    }

    SmallVector<std::vector<uint64_t>> bufs(opsToSerializeCount);

    loop_1d(LoopExecPolicy::Parallel, netFunc.getContext(), checked_cast<int64_t>(constOps.size()), [&](int64_t ind) {
        const auto attr = constOps[static_cast<size_t>(ind)].getContentAttr();

        const auto type = attr.getType();
        const auto content = attr.fold();

        const auto totalByteSize = type.cast<vpux::NDTypeInterface>().getTotalAllocSize();
        bufs[static_cast<size_t>(ind)].resize(
                alignValUp(static_cast<size_t>(totalByteSize.count()), sizeof(uint64_t)) / sizeof(uint64_t), 0);

        const auto buf =
                MutableArrayRef(reinterpret_cast<char*>(bufs[static_cast<size_t>(ind)].data()), totalByteSize.count());
        content.copyTo(buf);
    });

    SmallVector<VPUIP::BlobWriter::BinaryData> binaryData(constOps.size());

    for (auto constTensorInd : irange(constOps.size())) {
        auto constOp = constOps[constTensorInd];
        const auto& content = bufs[constTensorInd];

        log.trace("Got constant at '{0}' with type '{1}'", constOp->getLoc(), constOp.getType());

        binaryData[constTensorInd] = writer.createBinaryData(content, constOp.getType().cast<vpux::NDTypeInterface>());

        writer.createTensorRef(constOp.getOutput(), printToString("constant-{0}", constTensorInd),
                               VPURT::BufferSection::Constant, checked_cast<uint32_t>(constTensorInd), 0);
    }

    return binaryData;
}

SmallVector<VPUIP::BlobWriter::KernelData> serializeKernelData(VPUIP::BlobWriter& writer, mlir::func::FuncOp,
                                                               mlir::TimingScope&, Logger) {
    SmallVector<VPUIP::BlobWriter::KernelData> vec;
    for (auto&& e : writer.getKernelData()) {
        vec.push_back(e.second.data);
    }
    return vec;
}

SmallVector<VPUIP::BlobWriter::Barrier> serializeVirtBarriers(VPUIP::BlobWriter& writer, mlir::func::FuncOp netFunc,
                                                              bool withDynamicBarriers, mlir::TimingScope& rootTiming,
                                                              Logger log) {
    auto scopeTiming = rootTiming.nest("Serialize virtual barriers");

    SmallVector<VPUIP::BlobWriter::Barrier> virtBarriers;

    netFunc.walk([&](VPURT::DeclareVirtualBarrierOp barrierOp) {
        log.trace("Got virtual varrier at '{0}'", barrierOp->getLoc());

        VPUX_THROW_UNLESS(withDynamicBarriers, "Compiler was not configured for virtual barriers usage");

        const auto virtBarrier = writer.createBarrier(barrierOp.getBarrier());
        virtBarriers.push_back(virtBarrier);
    });

    return virtBarriers;
}

SmallVector<VPUIP::BlobWriter::TaskList> serializeTaskLists(VPUIP::BlobWriter& writer, mlir::func::FuncOp netFunc,
                                                            mlir::TimingScope& rootTiming, Logger log) {
    auto scopeTiming = rootTiming.nest("Serialize task lists");

    using TaskList = SmallVector<VPUIP::BlobWriter::Task>;
    using TaskListMap = std::map<VPU::ExecutorKind, TaskList>;

    TaskList barriersList;
    netFunc.walk([&](VPURT::ConfigureBarrierOp barrierOp) {
        log.trace("Got '{0}' at '{1}'", barrierOp->getName(), barrierOp->getLoc());
        barriersList.push_back(writer.createTask(barrierOp));
    });

    TaskListMap tasksMap;
    netFunc.walk([&](VPURT::TaskOp taskOp) {
        log.trace("Got '{0}' Task '{1}' at '{2}'", taskOp.getExecutorKind(), taskOp->getName(), taskOp->getLoc());
        tasksMap[taskOp.getExecutorKind()].push_back(writer.createTask(taskOp));
    });

    SmallVector<VPUIP::BlobWriter::TaskList> taskLists;
    taskLists.reserve(tasksMap.size() + 1);

    const auto serializeTaskList = [&](const TaskList& taskList) {
        const auto serializedTaskList = writer.createVector(taskList);

        MVCNN::TaskListBuilder builder(writer);
        builder.add_content(serializedTaskList);
        taskLists.push_back(builder.Finish());
    };

    log.trace("Serialize barriers list");
    serializeTaskList(barriersList);

    for (const auto& taskList : tasksMap) {
        log.trace("Serialize tasks list '{0}'", taskList.first);
        serializeTaskList(taskList.second);
    }

    return taskLists;
}

flatbuffers::Offset<MVCNN::GraphFile> createGraphFile(VPUIP::BlobWriter& writer,
                                                      flatbuffers::Offset<MVCNN::SummaryHeader> header,
                                                      ArrayRef<VPUIP::BlobWriter::TaskList> taskLists,
                                                      ArrayRef<VPUIP::BlobWriter::BinaryData> binaryData,
                                                      ArrayRef<VPUIP::BlobWriter::KernelData> kernelData,
                                                      ArrayRef<VPUIP::BlobWriter::Barrier> virtBarriers,
                                                      mlir::TimingScope& rootTiming) {
    auto scopeTiming = rootTiming.nest("Create graph file");

    const auto serializedTaskLists = writer.createVector(taskLists);
    const auto serializedBinaryData = writer.createVector(binaryData);
    const auto barrierTable = writer.createVector(virtBarriers);
    const auto serializedKernelData = writer.createVector(kernelData);

    MVCNN::GraphFileBuilder graphBuilder(writer);
    graphBuilder.add_header(header);
    graphBuilder.add_task_lists(serializedTaskLists);
    graphBuilder.add_binary_data(serializedBinaryData);
    graphBuilder.add_barrier_table(barrierTable);
    graphBuilder.add_kernel_data(serializedKernelData);

    return graphBuilder.Finish();
}

}  // namespace

flatbuffers::DetachedBuffer vpux::VPUIP::exportToBlob(mlir::ModuleOp module, mlir::TimingScope& rootTiming,
                                                      const std::vector<std::shared_ptr<const ov::Node>>& parameters,
                                                      const std::vector<std::shared_ptr<const ov::Node>>& results,
                                                      Logger log) {
    log.setName("VPUIP::BackEnd");

    log.trace("Extract 'IE.{0}' from Module", IE::CNNNetworkOp::getOperationName());
    IE::CNNNetworkOp netOp;
    mlir::func::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netOp, netFunc);

    VPUIP::BlobWriter writer(log.nest(), VPU::getArch(module));

    const auto withDynamicBarriers = !netFunc.getOps<VPURT::DeclareVirtualBarrierOp>().empty();

    const auto header = createSummaryHeader(writer, module, netOp, netFunc, withDynamicBarriers, rootTiming, parameters,
                                            results, log);

    serializeTensorDecls(writer, netFunc, rootTiming);
    auto binaryData = serializeBinaryData(writer, netFunc, netOp, rootTiming, log);
    if (vpux::isProfilingEnabled(netOp)) {
        extendBinaryDataWithProfilingSchema(writer, netOp, netFunc, binaryData, log);
    }

    const auto virtBarriers = serializeVirtBarriers(writer, netFunc, withDynamicBarriers, rootTiming, log);
    const auto taskLists = serializeTaskLists(writer, netFunc, rootTiming, log);
    const auto kernelData = serializeKernelData(writer, netFunc, rootTiming, log);
    const auto graphFile = createGraphFile(writer, header, taskLists, binaryData, kernelData, virtBarriers, rootTiming);

    auto finalTiming = rootTiming.nest("Finalize serialized graph");
    writer.impl().Finish(graphFile, "BLOB");
    auto detached = writer.impl().Release();

    auto serializedGraphFile = MVCNN::GetGraphFile(detached.data());

    const uint64_t reserved_offset = 1;
    std::unordered_set<uint32_t> kernelDataAligned;

    // align KernelData section referenced by given KernelDataReference
    // returns moved offset
    auto alignKernelDataSection = [&](const MVCNN::KernelDataReference* section, auto sectionLogical) {
        //  current align requirements is 1KB, for .text, .data, .scratch
        constexpr auto alignmentReq = Byte(1_KB).count();

        auto section_data = serializedGraphFile->kernel_data()->Get(section->locale_offset())->data();

        // checking that current description designates aligned section -
        // TODO: implement cases where offset is 1 actually fixes alignment
        auto section_data_plus_offset = section_data->Data() + section->data_offset() - detached.data();

        if (!(section_data_plus_offset % alignmentReq)) {
            VPUX_THROW_UNLESS(section->data_offset() != reserved_offset,
                              "kernelDataReference: {0} {1}, offset from blob start: {2}, already aligned with "
                              "reserved data_offset={3}",
                              section->name()->c_str(), sectionLogical, section_data_plus_offset, reserved_offset);
            return static_cast<ptrdiff_t>(0);
        }
        ptrdiff_t offset = section_data->Data() - detached.data();
        log.trace("offset to kernel {0} {1} in Finished FBB is {2}", section->name()->c_str(), sectionLogical, offset);

        auto aligned_offset = llvm::alignTo(offset, alignmentReq);
        offset = aligned_offset - offset;

        // check whether given kernelData element already aligned
        if (kernelDataAligned.find(section->locale_offset()) == kernelDataAligned.end()) {
            // if there is no data - do not move
            if (section_data->size() == 0) {
                return static_cast<ptrdiff_t>(0);
            }
            // check whether we do have a room for alignment
            VPUX_THROW_UNLESS(section_data->size() > alignmentReq,
                              "cannot align section: {0} {1},  space(alignment + size): {2} alignment: {3}",
                              section->name()->c_str(), sectionLogical, section_data->size(), alignmentReq);

            auto moveSize = section_data->size() - alignmentReq;
            VPUX_THROW_UNLESS(section_data->size() > moveSize + offset,
                              "cannot align section: {0} {1}, no room for moving space={2} offset={3} size={4}",
                              section->name()->c_str(), sectionLogical, section_data->size(), offset, moveSize);

            log.trace("move kernel {0} {1} by {2} bytes to be {3}", section->name()->c_str(), sectionLogical, offset,
                      aligned_offset);

            memmove(const_cast<uint8_t*>(section_data->Data() + offset), section_data->Data(),
                    section_data->size() - alignmentReq);

            // clear beginning
            memset(const_cast<uint8_t*>(section_data->Data()), 0, offset);
            // marking this kernel data content already aligned
            kernelDataAligned.insert(section->locale_offset());
        }

        return offset;
    };

    auto alignReferenceSection = [&](const MVCNN::KernelDataReference* section, uint64_t offset) {
        if (section->data_offset() != reserved_offset)
            return;
        // correcting data offset for section in schema
        auto table = reinterpret_cast<flatbuffers::Table*>(const_cast<MVCNN::KernelDataReference*>(section));

        // updating offset pointer
        // TODO: why we add here?
        table->SetField(MVCNN::KernelDataReference::VT_DATA_OFFSET,
                        checked_cast<uint32_t>(section->data_offset() + offset - reserved_offset), 0u);
    };

    auto alignSection = [&](const MVCNN::KernelDataReference* section, auto sectionLogical) {
        auto offset = alignKernelDataSection(section, sectionLogical);
        alignReferenceSection(section, offset);
    };

    // locating act-kernel
    for (auto&& task_list : *serializedGraphFile->task_lists()) {
        for (auto&& task : *task_list->content()) {
            if (auto actKernelTask = task->task_as_ActKernelTask()) {
                auto kernelTextSection = actKernelTask->kernel()->kernelText();
                alignSection(kernelTextSection, ".text");

                // Invocations aligning
                auto invocations = actKernelTask->invocations();
                for (auto&& invocation : *invocations) {
                    auto dataOffset = alignKernelDataSection(invocation->dataSection(), ".data");
                    alignReferenceSection(invocation->dataSection(), dataOffset);

                    // TODO: need a test of serialisation with non zero data section that clearly show problem
                    // data section and invocation args are packed in single locale buffer one by one
                    // [Track number: E#32997]
                    auto args_offset = dataOffset + invocation->dataSection()->referenced_data_size();
                    alignReferenceSection(invocation->invocationArgs(), args_offset);
                }
            }
        }
    }

    if (serializedGraphFile->header()->act_kernel_runtime()) {
        // scratchBuffer aligning
        if (auto scratchBuffer = serializedGraphFile->header()->act_kernel_runtime()->codeScratchBuffer()) {
            alignSection(scratchBuffer, ".scratchBuffer");
        }
        // management kernel aligning if present
        if (auto managementKernel = serializedGraphFile->header()->act_kernel_runtime()->kernel()) {
            alignSection(managementKernel->kernelText(), ".runtime.text");
            alignSection(managementKernel->globalArgs(), ".runtime.data");
        }
    }

    return detached;
}
