//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/passes/VPU2VPUIP/bufferize_vpu_nce_ops_interface.hpp"
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/conversion/passes/VPU2VPUIP/bufferizable_ops_interface.hpp"

#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"

#include "vpux/compiler/utils/allocate_buffers.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/custom_pwl_table.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

namespace {

void addPPETask(const Logger& log, mlir::OpBuilder& builder, VPUIP::NCEClusterTaskOp& nceOp, VPU::PPETaskAttr ppeAttr) {
    log.nest().trace("Adding PPE task '{0}'", ppeAttr);

    const auto input1MultList =
            ppeAttr.getIn1QuantMult() != nullptr
                    ? builder.getI64ArrayAttr(ArrayRef(parseIntArrayAttr<int64_t>(ppeAttr.getIn1QuantMult())))
                    : nullptr;
    const auto input2MultList =
            ppeAttr.getIn2QuantMult() != nullptr
                    ? builder.getI64ArrayAttr(ArrayRef(parseIntArrayAttr<int64_t>(ppeAttr.getIn2QuantMult())))
                    : nullptr;
    const auto outputMultList =
            ppeAttr.getQuantMult() != nullptr
                    ? builder.getI64ArrayAttr(ArrayRef(parseIntArrayAttr<int64_t>(ppeAttr.getQuantMult())))
                    : nullptr;
    const auto shiftList =
            ppeAttr.getQuantShift() != nullptr
                    ? builder.getI64ArrayAttr(ArrayRef(parseIntArrayAttr<int64_t>(ppeAttr.getQuantShift())))
                    : nullptr;
    nceOp.addPPETask(builder, ppeAttr.getMode(), ppeAttr.getClampLow(), ppeAttr.getClampHigh(), ppeAttr.getLreluMult(),
                     ppeAttr.getLreluShift(), outputMultList, shiftList, ppeAttr.getQuantPostShift(),
                     ppeAttr.getQuantScale(), input1MultList, input2MultList, ppeAttr.getFpPreluAlpha(),
                     /*ppe_fp_scale=*/nullptr,
                     /*ppe_fp_bias=*/nullptr);
}

void addDPUTasks(const Logger& log, VPUIP::NCEClusterTaskOp nceOp, mlir::OpBuilder& rewriter, mlir::Region& workloads,
                 bool isNCEPermute) {
    log.nest().trace("Adding DPU tasks");

    for (auto dpuTaskOp : workloads.getOps<VPU::DPUWorkloadOp>()) {
        SmallVector<int64_t> ends;
        const auto offsets = parseIntArrayAttr<int64_t>(dpuTaskOp.getOutOffsets());
        const auto sizes = parseIntArrayAttr<int64_t>(dpuTaskOp.getOutSizes());
        ends.reserve(sizes.size());

        llvm::transform(llvm::seq<size_t>(0, sizes.size()), std::back_inserter(ends), [&](size_t index) {
            return offsets[index] + sizes[index] - 1;
        });

        mlir::ArrayAttr inStartAttr = nullptr;
        mlir::ArrayAttr inEndAttr = nullptr;
        const auto isGroupedMatMul = offsets.size() == DimsGroups5D::Act::numDims;
        // Update workloads padding, offsets and sizes
        // after reshape and layout changes.
        if (isNCEPermute) {
            // Reshape Offsets and Sizes from CHW to HCW layout
            const SmallVector<int64_t> outDpuStart{offsets[Dims4D::Act::H.ind()], offsets[Dims4D::Act::C.ind()],
                                                   offsets[Dims4D::Act::W.ind()]};
            const SmallVector<int64_t> outDpuEnds{ends[Dims4D::Act::H.ind()], ends[Dims4D::Act::C.ind()],
                                                  ends[Dims4D::Act::W.ind()]};
            if (dpuTaskOp.getInOffsetsAttr() != nullptr && dpuTaskOp.getInSizesAttr() != nullptr) {
                const auto inOffset = parseIntArrayAttr<int64_t>(dpuTaskOp.getInOffsetsAttr());
                const auto inSizes = parseIntArrayAttr<int64_t>(dpuTaskOp.getInSizesAttr());
                const SmallVector<int64_t> inDpuStart{inOffset[Dims4D::Act::H.ind()], inOffset[Dims4D::Act::C.ind()],
                                                      inOffset[Dims4D::Act::W.ind()]};
                const SmallVector<int64_t> inDpuEnds{
                        inOffset[Dims4D::Act::H.ind()] + inSizes[Dims4D::Act::H.ind()] - 1,
                        inOffset[Dims4D::Act::C.ind()] + inSizes[Dims4D::Act::C.ind()] - 1,
                        inOffset[Dims4D::Act::W.ind()] + inSizes[Dims4D::Act::W.ind()] - 1};

                inStartAttr = getIntArrayAttr(rewriter, inDpuStart);
                inEndAttr = getIntArrayAttr(rewriter, inDpuEnds);
            }
            nceOp.addDPUTask(rewriter, getIntArrayAttr(rewriter, outDpuStart), getIntArrayAttr(rewriter, outDpuEnds),
                             inStartAttr, inEndAttr, dpuTaskOp.getPadAttr(), dpuTaskOp.getMpeMode(),
                             dpuTaskOp.getClusterIdAttr());
        } else if (isGroupedMatMul) {
            // This part is for grouped Matmul which has 5D input/output
            // Logic is same only dimensions are adjusted for 5D
            const auto dimC = DimsGroups5D::Act::C;
            const auto dimH = DimsGroups5D::Act::H;
            const auto dimW = DimsGroups5D::Act::W;
            const SmallVector<int64_t> outDpuStart{offsets[dimW.ind()], offsets[dimH.ind()], offsets[dimC.ind()]};
            const SmallVector<int64_t> outDpuEnds{ends[dimW.ind()], ends[dimH.ind()], ends[dimC.ind()]};

            if (dpuTaskOp.getInOffsetsAttr() != nullptr && dpuTaskOp.getInSizesAttr() != nullptr) {
                const auto inOffset = parseIntArrayAttr<int64_t>(dpuTaskOp.getInOffsetsAttr());
                const auto inSizes = parseIntArrayAttr<int64_t>(dpuTaskOp.getInSizesAttr());

                const SmallVector<int64_t> inDpuStart{inOffset[dimW.ind()], inOffset[dimH.ind()], inOffset[dimC.ind()]};
                const SmallVector<int64_t> inDpuEnds{inOffset[dimW.ind()] + inSizes[dimW.ind()] - 1,
                                                     inOffset[dimH.ind()] + inSizes[dimH.ind()] - 1,
                                                     inOffset[dimC.ind()] + inSizes[dimC.ind()] - 1};

                inStartAttr = getIntArrayAttr(rewriter, inDpuStart);
                inEndAttr = getIntArrayAttr(rewriter, inDpuEnds);
            }

            nceOp.addDPUTask(rewriter, getIntArrayAttr(rewriter, outDpuStart), getIntArrayAttr(rewriter, outDpuEnds),
                             inStartAttr, inEndAttr, dpuTaskOp.getPad(), dpuTaskOp.getMpeMode(),
                             dpuTaskOp.getClusterIdAttr());
        } else {
            // as soon as we need workload_x, workload_y, workload_z coords
            const SmallVector<int64_t> outDpuStart{offsets[Dims4D::Act::W.ind()], offsets[Dims4D::Act::H.ind()],
                                                   offsets[Dims4D::Act::C.ind()]};
            const SmallVector<int64_t> outDpuEnds{ends[Dims4D::Act::W.ind()], ends[Dims4D::Act::H.ind()],
                                                  ends[Dims4D::Act::C.ind()]};

            if (dpuTaskOp.getInOffsetsAttr() != nullptr && dpuTaskOp.getInSizesAttr() != nullptr) {
                const auto inOffset = parseIntArrayAttr<int64_t>(dpuTaskOp.getInOffsetsAttr());
                const auto inSizes = parseIntArrayAttr<int64_t>(dpuTaskOp.getInSizesAttr());

                const SmallVector<int64_t> inDpuStart{inOffset[Dims4D::Act::W.ind()], inOffset[Dims4D::Act::H.ind()],
                                                      inOffset[Dims4D::Act::C.ind()]};
                const SmallVector<int64_t> inDpuEnds{
                        inOffset[Dims4D::Act::W.ind()] + inSizes[Dims4D::Act::W.ind()] - 1,
                        inOffset[Dims4D::Act::H.ind()] + inSizes[Dims4D::Act::H.ind()] - 1,
                        inOffset[Dims4D::Act::C.ind()] + inSizes[Dims4D::Act::C.ind()] - 1};

                inStartAttr = getIntArrayAttr(rewriter, inDpuStart);
                inEndAttr = getIntArrayAttr(rewriter, inDpuEnds);
            }

            nceOp.addDPUTask(rewriter, getIntArrayAttr(rewriter, outDpuStart), getIntArrayAttr(rewriter, outDpuEnds),
                             inStartAttr, inEndAttr, dpuTaskOp.getPad(), dpuTaskOp.getMpeMode(),
                             dpuTaskOp.getClusterIdAttr());
        }
    }
}

//
// Create VPUIP.NCEClusterTask and ensure sparse types interact with the operation as individual buffers
//

mlir::Value createNCEClusterTask(mlir::OpBuilder& rewriter, mlir::Location loc, mlir::Value input, mlir::Value weights,
                                 mlir::Value weightsTable, mlir::Value instructionListTable,
                                 mlir::Value activationWindow, ArrayRef<mlir::Value> outputBuffs,
                                 vpux::VPUIP::NCETaskType taskType, mlir::ArrayAttr kernelSizeAttr,
                                 mlir::ArrayAttr kernelStridesAttr, vpux::VPU::PaddingAttr kernelPaddingAttr,
                                 mlir::IntegerAttr activationWindowChannelLengthAttr, mlir::Region& workloads,
                                 mlir::UnitAttr isSuperdenseAttr = nullptr, VPU::PPETaskAttr ppeAttr = nullptr,
                                 mlir::Attribute dpuCostAttr = nullptr, mlir::BoolAttr isInplace = nullptr,
                                 mlir::UnitAttr isPermuteQuantize = nullptr, mlir::IntegerAttr cmSpPattern = nullptr,
                                 mlir::UnitAttr inputChannelsCompression = nullptr, bool isNCEPermute = false,
                                 mlir::UnitAttr smallKernelOptimization = nullptr, Logger log = Logger::global()) {
    const auto getIndividualBuffers = [&](mlir::Value value) {
        mlir::Value data = value;
        mlir::Value sparsityMap = nullptr;
        mlir::Value seTable = nullptr;
        if (value != nullptr && value.getType().isa<VPUIP::SparseBufferType>()) {
            auto ungroupedOp = rewriter.create<VPUIP::UngroupSparseBufferOp>(loc, value);
            data = ungroupedOp.getData();
            sparsityMap = ungroupedOp.getSparsityMap();
            seTable = ungroupedOp.getStorageElementTable();
        }
        return std::make_tuple(data, sparsityMap, seTable);
    };

    mlir::Value inputData, inputSparsityMap, inputSETable;
    std::tie(inputData, inputSparsityMap, inputSETable) = getIndividualBuffers(input);

    mlir::Value weightsData, weightsSparsityMap;
    std::tie(weightsData, weightsSparsityMap, std::ignore) = getIndividualBuffers(weights);

    mlir::Value outputBuffData = outputBuffs[0];
    mlir::Value outputBuffSparsityMap = (outputBuffs.size() > 1) ? outputBuffs[1] : nullptr;

    auto nceClusterTask = rewriter.create<VPUIP::NCEClusterTaskOp>(
            loc, inputData, inputSparsityMap, inputSETable, weightsData, weightsSparsityMap, weightsTable,
            instructionListTable, /*sprLookupTable=*/nullptr, activationWindow, inputData, inputSparsityMap,
            inputSETable, outputBuffData, outputBuffSparsityMap, outputBuffData, outputBuffSparsityMap, nullptr,
            taskType, kernelSizeAttr, kernelStridesAttr, kernelPaddingAttr, activationWindowChannelLengthAttr,
            /*is_continued=*/nullptr, cmSpPattern,
            /*is_segmented=*/nullptr,
            /*out_channel_offset=*/nullptr, inputChannelsCompression, isSuperdenseAttr, isInplace,
            /*input_se_size=*/nullptr,
            /*output_se_size=*/nullptr, isPermuteQuantize, smallKernelOptimization);

    addDPUTasks(log, nceClusterTask, rewriter, workloads, isNCEPermute);

    if (ppeAttr != nullptr) {
        addPPETask(log, rewriter, nceClusterTask, ppeAttr);
    }

    if (dpuCostAttr != nullptr) {
        nceClusterTask->setAttr(DPUCost, dpuCostAttr);
    }

    if (nceClusterTask.getOutputSparsityMap() != nullptr) {
        auto groupedOp = rewriter.create<VPUIP::GroupSparseBufferOp>(loc, nceClusterTask.getOutput(),
                                                                     nceClusterTask.getOutputSparsityMap());
        return groupedOp.getOutput();
    }

    return nceClusterTask.getOutput();
}

bool isSuperdenseOp(mlir::Operation* nceOp) {
    auto outType = nceOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    if (auto parentOp = nceOp->getParentOfType<VPU::NCEClusterTilingOp>()) {
        outType = parentOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    }
    const auto outputOrder = outType.getDimsOrder();
    const auto outputShape = outType.getShape();
    const auto outElemType = outType.getElementType();
    const auto arch = VPU::getArch(nceOp);

    // Check output shape for each cluster
    if (auto distributedTensorType = outType.dyn_cast<VPU::DistributedTensorType>()) {
        auto tiledComputeShapes = distributedTensorType.getPerClusterComputeShapes();
        for (auto& computeShape : tiledComputeShapes) {
            if (VPU::NCESparsity::isSuperdenseRequired(arch, outputOrder, computeShape, outElemType)) {
                return true;
            }
        }
        return false;
    }

    return VPU::NCESparsity::isSuperdenseRequired(arch, outputOrder, outputShape, outElemType);
}

VPU::PPETaskAttr composePPETask(const vpux::NDTypeInterface outType) {
    auto outElemType = outType.getElementType();
    auto uniformQElemType = outElemType.dyn_cast<mlir::quant::UniformQuantizedType>();
    const int64_t lReluMult = 1;
    const int64_t lReluShift = 0;
    // int64_t data types with int32_t numeric limits are used intentionally.
    // Schema requires i32 numeric limits to by-pass clamping, while PPETaskOp stores these values as i64.
    int64_t clampLow = std::numeric_limits<int32_t>::min();
    int64_t clampHigh = std::numeric_limits<int32_t>::max();
    if (uniformQElemType == nullptr) {
        return getPPETaskAttr(outType.getContext(), vpux::VPU::PPEMode::ADD, clampLow, clampHigh, lReluMult, lReluShift,
                              ArrayRef<double>{0.5});
    }
    const auto scale = uniformQElemType.getScale();
    // Since element-wise add sums the input with itself, output scale must be doubled.
    const auto newScale = static_cast<double>(scale * 2.0);
    clampLow = uniformQElemType.getStorageTypeMin();
    clampHigh = uniformQElemType.getStorageTypeMax();
    return getPPETaskAttr(outType.getContext(), vpux::VPU::PPEMode::ADD, clampLow, clampHigh, lReluMult, lReluShift,
                          ArrayRef<double>{1.0 / newScale});
}

SmallVector<int64_t> calculateWCHShape(ArrayRef<int64_t> shape) {
    const int64_t tensorSizeZ = shape[Dims4D::Act::W.ind()];
    const int64_t tensorSizeY = shape[Dims4D::Act::C.ind()];
    const int64_t tensorSizeX = shape[Dims4D::Act::H.ind()];
    const SmallVector<int64_t> targetShape = {shape[Dims4D::Act::N.ind()], tensorSizeZ, tensorSizeY, tensorSizeX};
    return targetShape;
}

}  // namespace

//
// bufferize VPU::NCEConvolutionO
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext* ctx, VPU::NCEConvolutionOp origOp,
                                      VPU::NCEConvolutionOp::Adaptor newArgs, mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-NCEConvolutionOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    //
    // Get dimensions
    //

    const auto filterShape = Shape(parseIntArrayAttr<int64_t>(origOp.getRawFilterShape()));

    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto inOrder = DimsOrder::fromValue(newArgs.getInput());
    const auto isCMajor = inOrder == DimsOrder::NCHW;

    //
    // Prepare output buffer for DPU
    //

    const auto outputBuffers =
            allocateBuffers(log, origOp.getLoc(), rewriter, {origOp.getOutput()}, /*individualBuffers=*/true);

    //
    // Create NCE per-cluster Operation
    //

    const auto kernelSizeAttr = getIntArrayAttr(ctx, ArrayRef({KY, KX}));
    const auto taskType = isCMajor ? VPUIP::NCETaskType::CMCONV : VPUIP::NCETaskType::CONV;
    auto ppeAttr = origOp.getPpe().has_value() ? origOp.getPpeAttr() : nullptr;
    auto dpuCostAttr = origOp->hasAttr(DPUCost) ? origOp->getAttr(DPUCost) : nullptr;

    log.nest().trace("Creating VPUIP::NCEClusterTaskOp");
    mlir::UnitAttr isSuperdenseAttr = nullptr;
    if (isSuperdenseOp(origOp)) {
        VPUX_THROW_WHEN(origOp->getResult(0).getType().isa<VPU::SparseTensorType>(),
                        "Output cannot be sparse and super-dense at the same time");
        isSuperdenseAttr = mlir::UnitAttr::get(ctx);
    }

    auto nceOp = createNCEClusterTask(
            rewriter, origOp->getLoc(), newArgs.getInput(), newArgs.getFilter(), newArgs.getWeightsTable(),
            newArgs.getInstructionListTable(), newArgs.getActivationWindow(), outputBuffers, taskType, kernelSizeAttr,
            origOp.getStrides(), origOp.getPadAttr(), origOp.getActivationWindowChannelLengthAttr(),
            origOp.getWorkloads(), isSuperdenseAttr, ppeAttr, dpuCostAttr,
            /*isInplace=*/nullptr, /*isPermuteQuantize=*/nullptr, /*cmSpPattern=*/nullptr,
            /*inputChannelsCompression=*/nullptr, /*isNCEPermute*/ false, /*smallKernelOptimization=*/nullptr, log);

    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, nceOp);

    return mlir::success();
}

//
// bufferize VPU::NCEMaxPoolOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext* ctx, VPU::NCEMaxPoolOp origOp,
                                      VPU::NCEMaxPoolOp::Adaptor newArgs, mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-NCEMaxPoolOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    //
    // Prepare output buffer for DPU
    //

    const auto outputBuffers =
            allocateBuffers(log, origOp.getLoc(), rewriter, {origOp.getOutput()}, /*individualBuffers=*/true);

    //
    // Create NCE per-cluster Operation
    //

    auto ppeAttr = origOp.getPpe().has_value() ? origOp.getPpeAttr() : nullptr;
    auto dpuCostAttr = origOp->hasAttr(DPUCost) ? origOp->getAttr(DPUCost) : nullptr;

    log.nest().trace("Creating VPUIP::NCEClusterTaskOp");
    mlir::UnitAttr isSuperdenseAttr = nullptr;
    if (isSuperdenseOp(origOp)) {
        VPUX_THROW_WHEN(origOp->getResult(0).getType().isa<VPU::SparseTensorType>(),
                        "Output cannot be sparse and super-dense at the same time");
        isSuperdenseAttr = mlir::UnitAttr::get(ctx);
    }

    auto nceOp = createNCEClusterTask(
            rewriter, origOp->getLoc(), newArgs.getInput(), /*weights=*/nullptr, newArgs.getWeightsTable(),
            /*instruction_list_table=*/nullptr, newArgs.getActivationWindow(), outputBuffers,
            VPUIP::NCETaskType::MAXPOOL, origOp.getKernelSize(), origOp.getStrides(), origOp.getPad(),
            origOp.getActivationWindowChannelLengthAttr(), origOp.getWorkloads(), isSuperdenseAttr, ppeAttr,
            dpuCostAttr, /*isInplace=*/nullptr,
            /*isPermuteQuantize=*/nullptr, /*cmSpPattern=*/nullptr,
            /*inputChannelsCompression=*/nullptr, /*isNCEPermute=*/false, /*smallKernelOptimization=*/nullptr, log);

    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, nceOp);

    return mlir::success();
}

//
// bufferize VPU::NCEAveragePoolOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext* ctx, VPU::NCEAveragePoolOp origOp,
                                      VPU::NCEAveragePoolOp::Adaptor newArgs, mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-NCEAveragePoolOp", 0);
    //
    // Prepare output buffer for DPU
    //

    const auto outputBuffers =
            allocateBuffers(log, origOp.getLoc(), rewriter, {origOp.getOutput()}, /*individualBuffers=*/true);

    //
    // Create NCE per-cluster Operation
    //

    auto ppeAttr = origOp.getPpe().has_value() ? origOp.getPpeAttr() : nullptr;
    auto dpuCostAttr = origOp->hasAttr(DPUCost) ? origOp->getAttr(DPUCost) : nullptr;

    log.nest().trace("Creating VPUIP::NCEClusterTaskOp");
    mlir::UnitAttr isSuperdenseAttr = nullptr;
    if (isSuperdenseOp(origOp)) {
        VPUX_THROW_WHEN(origOp->getResult(0).getType().isa<VPU::SparseTensorType>(),
                        "Output cannot be sparse and super-dense at the same time");
        isSuperdenseAttr = mlir::UnitAttr::get(ctx);
    }

    mlir::UnitAttr isSmallKernelOptimizationAttr = nullptr;
    if (VPU::NCEInvariant::isSmallKernelOptimizationSupported(VPU::getArch(origOp), origOp)) {
        isSmallKernelOptimizationAttr = mlir::UnitAttr::get(ctx);
    }
    auto nceOp = createNCEClusterTask(rewriter, origOp->getLoc(), newArgs.getInput(), /*weights=*/nullptr,
                                      /*weights_table=*/nullptr,
                                      /*instruction_list_table=*/nullptr, /*activation_window=*/nullptr, outputBuffers,
                                      VPUIP::NCETaskType::AVEPOOL, origOp.getKernelSize(), origOp.getStrides(),
                                      origOp.getPad(),
                                      /*activation_window_channel_length=*/nullptr, origOp.getWorkloads(),
                                      isSuperdenseAttr, ppeAttr, dpuCostAttr, /*isInplace=*/nullptr,
                                      /*isPermuteQuantize=*/nullptr, /*cmSpPattern=*/nullptr,
                                      /*inputChannelsCompression=*/nullptr, /*isNCEPermute=*/false,
                                      /*smallKernelOptimization=*/isSmallKernelOptimizationAttr, log);

    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, nceOp);

    return mlir::success();
}

//
// bufferize VPU::NCEDepthConvolutionOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext* ctx, VPU::NCEDepthConvolutionOp origOp,
                                      VPU::NCEDepthConvolutionOp::Adaptor newArgs, mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-NCEDepthConvolutionOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    //
    // Get dimensions
    //

    const auto filterShape = Shape(parseIntArrayAttr<int64_t>(origOp.getRawFilterShape()));
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    //
    // Prepare output buffer for DPU
    //

    const auto outputBuffers =
            allocateBuffers(log, origOp.getLoc(), rewriter, {origOp.getOutput()}, /*individualBuffers=*/true);

    //
    // Create NCE per-cluster Operation
    //

    const auto kernelSizeAttr = getIntArrayAttr(ctx, ArrayRef({KY, KX}));
    auto ppeAttr = origOp.getPpe().has_value() ? origOp.getPpeAttr() : nullptr;
    auto dpuCostAttr = origOp->hasAttr(DPUCost) ? origOp->getAttr(DPUCost) : nullptr;

    log.nest().trace("Creating VPUIP::NCEClusterTaskOp");
    mlir::UnitAttr isSuperdenseAttr = nullptr;
    if (isSuperdenseOp(origOp)) {
        VPUX_THROW_WHEN(origOp->getResult(0).getType().isa<VPU::SparseTensorType>(),
                        "Output cannot be sparse and super-dense at the same time");
        isSuperdenseAttr = mlir::UnitAttr::get(ctx);
    }

    mlir::UnitAttr isSmallKernelOptimizationAttr = nullptr;
    if (VPU::NCEInvariant::isSmallKernelOptimizationSupported(VPU::getArch(origOp), origOp)) {
        isSmallKernelOptimizationAttr = mlir::UnitAttr::get(ctx);
    }

    auto nceOp = createNCEClusterTask(
            rewriter, origOp->getLoc(), newArgs.getInput(), newArgs.getFilter(), newArgs.getWeightsTable(),
            newArgs.getInstructionListTable(), newArgs.getActivationWindow(), outputBuffers, VPUIP::NCETaskType::DWCONV,
            kernelSizeAttr, origOp.getStrides(), origOp.getPad(), origOp.getActivationWindowChannelLengthAttr(),
            origOp.getWorkloads(), isSuperdenseAttr, ppeAttr, dpuCostAttr, /*isInplace=*/nullptr,
            /*isPermuteQuantize=*/nullptr, /*cmSpPattern=*/nullptr,
            /*inputChannelsCompression=*/nullptr, /*isNCEPermute=*/false,
            /*smallKernelOptimization=*/isSmallKernelOptimizationAttr, log);

    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, nceOp);

    return mlir::success();
}

//
// bufferize VPU::NCEInterpolateOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext* ctx, VPU::NCEInterpolateOp origOp,
                                      VPU::NCEInterpolateOp::Adaptor newArgs, mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-NCEInterpolateOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    const auto filterShape = Shape(parseIntArrayAttr<int64_t>(origOp.getRawFilterShape()));

    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    auto kernelSizeAttr = getIntArrayAttr(ctx, ArrayRef({KY, KX}));

    log.nest().trace("Allocating output buffer");

    auto newLoc = appendLoc(origOp.getLoc(), "_interpolate");

    const auto outputBuffers = allocateBuffers(log, newLoc, rewriter, {origOp.getOutput()}, /*individualBuffers=*/true);

    log.nest().trace("Creating VPUIP::NCEClusterTaskOp");

    auto ppeTaskAttr = origOp.getPpe().has_value() ? origOp.getPpeAttr() : nullptr;
    auto dpuCostAttr = origOp->hasAttr(DPUCost) ? origOp->getAttr(DPUCost) : nullptr;

    mlir::UnitAttr isSuperdenseAttr = nullptr;
    if (isSuperdenseOp(origOp)) {
        VPUX_THROW_WHEN(origOp->getResult(0).getType().isa<VPU::SparseTensorType>(),
                        "Output cannot be sparse and super-dense at the same time");
        isSuperdenseAttr = mlir::UnitAttr::get(ctx);
    }

    auto nceOpInterface = mlir::dyn_cast<VPU::NCEOpInterface>(origOp.getOperation());
    auto nceOp = createNCEClusterTask(
            /*rewriter=*/rewriter, /*loc=*/newLoc, /*input=*/newArgs.getInput(),
            /*weights=*/newArgs.getWeights(), /*weightsTable=*/newArgs.getWeightsTable(),
            /*instructionListTable=*/nullptr,
            /*activationWindow=*/nullptr, /*outputBuffs=*/outputBuffers, /*taskType=*/VPUIP::NCETaskType::CONV,
            /*kernelSizeAttr=*/kernelSizeAttr,
            /*kernelStridesAttr=*/getIntArrayAttr(ctx, nceOpInterface.getStridesVal()),
            /*kernelPaddingAttr=*/nceOpInterface.getPad(),
            /*activationWindowChannelLengthAttr=*/nullptr,
            /*workloads=*/origOp.getWorkloads(), /*isSuperdenseAttr=*/isSuperdenseAttr, /*ppeAttr=*/ppeTaskAttr,
            /*dpuCostAttr=*/dpuCostAttr,
            /*isInplace=*/nullptr, /*isPermuteQuantize=*/nullptr, /*cmSpPattern=*/nullptr,
            /*inputChannelsCompression=*/nullptr, /*isNCEPermute=*/false, /*smallKernelOptimization=*/nullptr,
            /*log=*/log);

    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, nceOp);

    return mlir::success();
}

//
// bufferize VPU::NCEEltwiseOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext* ctx, VPU::NCEEltwiseOp origOp,
                                      VPU::NCEEltwiseOp::Adaptor newArgs, mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-NCEEltwiseOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    //
    // Prepare output buffer for DPU
    //

    const auto outputBuffers =
            allocateBuffers(log, origOp.getLoc(), rewriter, {origOp.getOutput()}, /*individualBuffers=*/true);

    //
    // Create NCE per-cluster Operation
    //

    const auto activation_window_channel_length = getIntAttr(ctx, static_cast<int32_t>(0));
    auto ppeAttr = origOp.getPpe().has_value() ? origOp.getPpeAttr() : nullptr;
    VPUX_THROW_UNLESS(ppeAttr != nullptr, "Eltwise operation should always have PPE info");
    auto dpuCostAttr = origOp->hasAttr(DPUCost) ? origOp->getAttr(DPUCost) : nullptr;

    log.nest().trace("Creating VPUIP::NCEClusterTaskOp");
    mlir::UnitAttr isSuperdenseAttr = nullptr;
    if (isSuperdenseOp(origOp)) {
        VPUX_THROW_WHEN(origOp->getResult(0).getType().isa<VPU::SparseTensorType>(),
                        "Output cannot be sparse and super-dense at the same time");
        isSuperdenseAttr = mlir::UnitAttr::get(ctx);
    }

    auto nceOp = createNCEClusterTask(
            rewriter, origOp->getLoc(), newArgs.getInput1(), newArgs.getInput2(),
            /*weightsTable=*/nullptr,
            /*instruction_table_list=*/nullptr,
            /*activation_window=*/nullptr, outputBuffers, VPUIP::NCETaskType::ELTWISE,
            /*kernel_size=*/nullptr,
            /*kernel_strides=*/nullptr,
            /*kernel_padding=*/nullptr, activation_window_channel_length, origOp.getWorkloads(), isSuperdenseAttr,
            ppeAttr, dpuCostAttr, origOp.getIsInplaceAttr(),
            /*isPermuteQuantize=*/nullptr, /*cmSpPattern=*/nullptr,
            /*inputChannelsCompression=*/nullptr, /*isNCEPermute=*/false, /*smallKernelOptimization=*/nullptr, log);

    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, nceOp);

    return mlir::success();
}

//
// bufferize VPU::NCECompressConvolutionOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext* ctx, VPU::NCECompressConvolutionOp origOp,
                                      VPU::NCECompressConvolutionOp::Adaptor newArgs, mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-NCECompressConvolutionOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    //
    // Get dimensions
    //

    const auto filterShape = Shape(parseIntArrayAttr<int64_t>(origOp.getRawFilterShape()));

    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto channelAlignValue = VPU::NCEInvariant::getAlignment(
            newArgs.getFilter().getType().cast<vpux::NDTypeInterface>().getElementType());

    const auto finalShape = SmallVector<int64_t>({filterShape[Dims4D::Filter::OC], channelAlignValue, KY, KX});
    auto shapeCastWeightsOp = rewriter.create<VPUIP::ShapeCastOp>(origOp->getLoc(), newArgs.getFilter(),
                                                                  getIntArrayAttr(origOp.getContext(), finalShape));
    //
    // Prepare output buffer for DPU
    //

    const auto outputBuffers =
            allocateBuffers(log, origOp.getLoc(), rewriter, {origOp.getOutput()}, /*individualBuffers=*/true);

    //
    // Create NCE per-cluster Operation
    //
    auto inputType = newArgs.getInput().getType();
    const auto inputShape = inputType.cast<vpux::NDTypeInterface>().getShape();
    const auto finalInputShape = vpux::Shape(
            {inputShape[Dims4D::Act::N], channelAlignValue, inputShape[Dims4D::Act::H], inputShape[Dims4D::Act::W]});
    auto finalInputShapeAttr = getIntArrayAttr(origOp.getContext(), finalInputShape);

    const auto kernelSizeAttr = getIntArrayAttr(ctx, ArrayRef({KY, KX}));
    auto ppeAttr = origOp.getPpe().has_value() ? origOp.getPpeAttr() : nullptr;
    auto dpuCostAttr = origOp->hasAttr(DPUCost) ? origOp->getAttr(DPUCost) : nullptr;

    log.nest().trace("Creating VPUIP::NCEClusterTaskOp");
    mlir::UnitAttr isSuperdenseAttr = nullptr;
    if (isSuperdenseOp(origOp)) {
        VPUX_THROW_WHEN(origOp->getResult(0).getType().isa<VPU::SparseTensorType>(),
                        "Output cannot be sparse and super-dense at the same time");
        isSuperdenseAttr = mlir::UnitAttr::get(ctx);
    }
    auto inputShapeCastOp =
            rewriter.create<VPUIP::ShapeCastOp>(origOp->getLoc(), newArgs.getInput(), finalInputShapeAttr);
    const auto inputChannelsCompression = mlir::UnitAttr::get(origOp->getContext());

    auto nceOp = createNCEClusterTask(
            rewriter, origOp->getLoc(), inputShapeCastOp.getResult(), shapeCastWeightsOp.getResult(),
            newArgs.getWeightsTable(),
            /*instructionListTable=*/nullptr, /*activationWindow=*/nullptr, outputBuffers, VPUIP::NCETaskType::CONV,
            kernelSizeAttr, origOp.getStrides(), origOp.getPadAttr(), /*activation_window_channel_lengthAttr=*/nullptr,
            origOp.getWorkloads(), isSuperdenseAttr, ppeAttr, dpuCostAttr,
            /*isInplace=*/nullptr, /*isPermuteQuantize=*/nullptr, origOp.getCmSpPatternAttr(), inputChannelsCompression,
            /*isNCEPermute=*/false, /*smallKernelOptimization=*/nullptr, log);

    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, nceOp);

    return mlir::success();
}

//
// bufferize VPU::NCEPermuteOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext* ctx, VPU::NCEPermuteOp origOp,
                                      VPU::NCEPermuteOp::Adaptor newArgs, mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-NCEPermuteOp", 0);

    auto clusterTilingOp = origOp->getParentOfType<VPU::NCEClusterTilingOp>();
    if (clusterTilingOp != nullptr) {
        return mlir::failure();
    }

    log.trace("Got '{0}' Single Tile '{1}'", origOp->getName(), origOp->getLoc());

    // ViewOp Input
    // Reshape to NxWxCxH
    // Layout change to NHWC
    const auto inputShape = getShape(newArgs.getInput());
    const auto targetShape = calculateWCHShape(inputShape.raw());

    auto inType = newArgs.getInput().getType().cast<NDTypeInterface>();
    const auto targetInOutOrder = DimsOrder::NHWC;
    inType = inType.changeShape(ShapeRef(targetShape));
    inType = inType.changeDimsOrder(targetInOutOrder);
    auto viewOpIn = rewriter.create<VPUIP::ViewOp>(origOp.getLoc(), inType, newArgs.getInput());

    // Manual update output type
    auto outType = origOp.getOutput().getType().cast<NDTypeInterface>();
    const auto outNCEPermuteShape = calculateWCHShape(outType.getShape().raw());
    outType = outType.changeShape(ShapeRef(outNCEPermuteShape));
    outType = outType.changeDimsOrder(DimsOrder::NWCH);

    //
    // Prepare output buffer for DPU
    //
    auto bufferType = vpux::getBufferType(outType);

    log.nest().trace("Allocating result buffer of type '{0}' for value type '{1}'", bufferType, outType);
    const auto outputBuffers =
            allocateBuffersOfType(log.nest(), origOp.getLoc(), rewriter, bufferType, /*individualBuffers=*/true);

    // Add PPE task to rescale output.
    const auto ppeTaskAttr = composePPETask(outType);

    mlir::UnitAttr isSuperdenseAttr = nullptr;
    if (isSuperdenseOp(origOp)) {
        VPUX_THROW_WHEN(origOp->getResult(0).getType().isa<VPU::SparseTensorType>(),
                        "Output cannot be sparse and super-dense at the same time");
        isSuperdenseAttr = mlir::UnitAttr::get(ctx);
    }

    const auto dpuCostAttr = origOp->hasAttr(DPUCost) ? origOp->getAttr(DPUCost) : nullptr;
    const auto activationWindowChannelLength = getIntAttr(ctx, static_cast<int32_t>(0));
    const auto isPermuteQuantizeAttr = mlir::UnitAttr::get(ctx);

    log.nest().trace("Creating VPUIP::NCEClusterTaskOp");

    auto nceOp = createNCEClusterTask(rewriter, origOp->getLoc(), viewOpIn.getResult(), viewOpIn.getResult(),
                                      /*weightsTable=*/nullptr,
                                      /*instruction_table_list=*/nullptr,
                                      /*activation_window=*/nullptr, outputBuffers, VPUIP::NCETaskType::ELTWISE,
                                      /*kernel_size=*/nullptr,
                                      /*kernel_strides=*/nullptr,
                                      /*kernel_padding=*/nullptr, activationWindowChannelLength, origOp.getWorkloads(),
                                      isSuperdenseAttr, ppeTaskAttr, dpuCostAttr,
                                      /*isInplace=*/nullptr, isPermuteQuantizeAttr, /*cmSpPattern=*/nullptr,
                                      /*inputChannelsCompression=*/nullptr, /*isNCEPermute=*/true,
                                      /*smallKernelOptimization=*/nullptr, log);

    // ViewOp Output
    // Reshape to NxCxHxW
    // Layout change to NHWC
    auto viewOpOutType = nceOp.getType().cast<NDTypeInterface>().changeDimsOrder(targetInOutOrder);
    viewOpOutType = viewOpOutType.changeShape(getShape(origOp.getOutput()));
    auto viewOpOut = rewriter.create<VPUIP::ViewOp>(origOp.getLoc(), viewOpOutType, nceOp);
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, viewOpOut.getResult());

    return mlir::success();
}

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext* ctx, VPU::NCEMatMulOp origOp,
                                      VPU::NCEMatMulOp::Adaptor newArgs, mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-NCEMatMulOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());
    //
    // Get dimensions
    //
    const auto filterShape = Shape(parseIntArrayAttr<int64_t>(origOp.getRawFilterShape()));

    const auto KY = filterShape[DimsGroups5D::Filter::KY];
    const auto KX = filterShape[DimsGroups5D::Filter::KX];

    //
    // Prepare output buffer for DPU
    //

    const auto outputBuffers =
            allocateBuffers(log, origOp.getLoc(), rewriter, {origOp.getOutput()}, /*individualBuffers=*/true);

    //
    // Create NCE per-cluster Operation
    //

    const auto kernelSizeAttr = getIntArrayAttr(ctx, ArrayRef({KY, KX}));
    const auto taskType = VPUIP::NCETaskType::CONV;
    auto ppeAttr = origOp.getPpe().value_or(nullptr);
    auto dpuCostAttr = origOp->hasAttr(DPUCost) ? origOp->getAttr(DPUCost) : nullptr;

    log.nest().trace("Creating VPUIP::NCEClusterTaskOp");
    mlir::UnitAttr isSuperdenseAttr = nullptr;
    if (isSuperdenseOp(origOp)) {
        VPUX_THROW_WHEN(origOp->getResult(0).getType().isa<VPU::SparseTensorType>(),
                        "Output cannot be sparse and super-dense at the same time");
        isSuperdenseAttr = mlir::UnitAttr::get(ctx);
    }

    auto nceOp = createNCEClusterTask(
            rewriter, origOp->getLoc(), newArgs.getInput(), newArgs.getWeights(), newArgs.getWeightsTable(), nullptr,
            nullptr, outputBuffers, taskType, kernelSizeAttr, origOp.getStrides(), origOp.getPadAttr(), nullptr,
            origOp.getWorkloads(), isSuperdenseAttr, ppeAttr, dpuCostAttr,
            /*isInplace=*/nullptr, /*isPermuteQuantize=*/nullptr, /*cmSpPattern=*/nullptr,
            /*inputChannelsCompression=*/nullptr, /*isNCEPermute*/ false, /*smallKernelOptimization=*/nullptr, log);

    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, nceOp);

    return mlir::success();
}

namespace {

//
// bufferize MultiTile VPU::NCEPermuteOp
//

class NCEPermuteMultiTileRewriter final : public mlir::OpConversionPattern<VPU::NCEClusterTilingOp> {
public:
    NCEPermuteMultiTileRewriter(mlir::TypeConverter& converter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::NCEClusterTilingOp>(converter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::NCEClusterTilingOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

VPU::DistributedTensorType createCustomDistributedTensorType(VPU::ClusteredOpInterface clusteredOp,
                                                             NDTypeInterface targetType,
                                                             VPU::DistributedTensorAttr origDistTensorAttr,
                                                             mlir::UnitAttr equalMemoryAndComputeView, ShapeRef shape) {
    auto* ctx = clusteredOp->getContext();

    const auto memSpace = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));
    const auto order = mlir::AffineMapAttr::get(targetType.getDimsOrder().toAffineMap(ctx));
    auto elemType = targetType.getElementType();

    const auto origDistTensorCtx = origDistTensorAttr.getContext();

    auto newNumTilesAttr = origDistTensorAttr.getNumTiles();
    if (newNumTilesAttr != nullptr) {
        auto numTiles = parseIntArrayAttr<int64_t>(newNumTilesAttr);
        newNumTilesAttr = getIntArrayAttr(origDistTensorCtx, calculateWCHShape(numTiles));
    }

    const auto activationTensorDistributionModeAttr =
            VPU::DistributionModeAttr::get(ctx, origDistTensorAttr.getMode().getValue());
    // Padding adaptions
    auto newPadAttr = origDistTensorAttr.getPads();
    if (newPadAttr != nullptr) {
        const auto fullInputChannels =
                clusteredOp.getOperation()->getOperand(0).getType().cast<NDTypeInterface>().getShape()[Dims4D::Act::C];
        const auto fullOutputChannels =
                clusteredOp.getOperation()->getResult(0).getType().cast<NDTypeInterface>().getShape()[Dims4D::Act::C];

        newPadAttr = VPU::getPaddingAttr(origDistTensorCtx, PadInfo(origDistTensorAttr.getPads().getTop().getInt(),
                                                                    origDistTensorAttr.getPads().getBottom().getInt(),
                                                                    0, fullOutputChannels - fullInputChannels));
    }
    auto newKernelAttr = origDistTensorAttr.getKernel();
    if (newKernelAttr != nullptr) {
        auto newKernel = parseIntArrayAttr<int64_t>(newKernelAttr);
        newKernelAttr = getIntArrayAttr(origDistTensorCtx,
                                        SmallVector<int64_t>{/*neutral val*/ 1, newKernel[Dims4D::Kernel::Y.ind()]});
    }
    auto newStridesAttr = origDistTensorAttr.getStrides();
    if (newStridesAttr != nullptr) {
        auto newStrides = parseIntArrayAttr<int64_t>(newStridesAttr);
        newStridesAttr = getIntArrayAttr(origDistTensorCtx,
                                         SmallVector<int64_t>{/*neutral val*/ 1, newStrides[Dims4D::Strides::Y.ind()]});
    }
    auto newAlignmentAttr = origDistTensorAttr.getAlignment();
    if (newAlignmentAttr != nullptr) {
        auto newAlignment = parseIntArrayAttr<int64_t>(newAlignmentAttr);
        newAlignmentAttr = getIntArrayAttr(origDistTensorCtx, calculateWCHShape(newAlignment));
    }

    auto calculateWCHShapeForArrayOfArray = [origDistTensorCtx](const mlir::ArrayAttr shape) -> mlir::ArrayAttr {
        if (shape != nullptr) {
            auto newIntShape = parseIntArrayOfArrayAttr<int64_t>(shape);
            for (size_t i = 0; i < newIntShape.size(); i++) {
                newIntShape[i] = calculateWCHShape(newIntShape[i]);
            }
            return getIntArrayOfArray(origDistTensorCtx, newIntShape);
        }
        return nullptr;
    };

    auto distributedTensorAttr = VPU::DistributedTensorAttr::get(
            ctx, activationTensorDistributionModeAttr, newNumTilesAttr, newKernelAttr, newPadAttr, newStridesAttr,
            origDistTensorAttr.getNumClusters(), newAlignmentAttr, origDistTensorAttr.getUniformDistributedSegments(),
            calculateWCHShapeForArrayOfArray(origDistTensorAttr.getComputeShapes()),
            calculateWCHShapeForArrayOfArray(origDistTensorAttr.getComputeOffsets()),
            calculateWCHShapeForArrayOfArray(origDistTensorAttr.getMemoryShapes()),
            calculateWCHShapeForArrayOfArray(origDistTensorAttr.getMemoryOffsets()), equalMemoryAndComputeView);

    return VPU::DistributedTensorType::get(ctx, ArrayRef(calculateWCHShape(shape.raw())), elemType, order, memSpace,
                                           distributedTensorAttr);
}

mlir::LogicalResult NCEPermuteMultiTileRewriter::matchAndRewrite(VPU::NCEClusterTilingOp origOp, OpAdaptor newArgs,
                                                                 mlir::ConversionPatternRewriter& rewriter) const {
    auto permuteOp = origOp.getInnerTaskOpOfType<VPU::NCEPermuteOp>();
    if (permuteOp == nullptr) {
        return mlir::failure();
    }

    _log.trace("Got '{0}' Multi Tile at '{1}'", origOp->getName(), origOp->getLoc());

    auto ctx = this->getContext();
    const auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(permuteOp.getOperation());
    VPUX_THROW_UNLESS(clusteredOp != nullptr, "Operation '{0}' cannot be converted to VPU::ClusteredOpInterface",
                      permuteOp);
    VPUX_THROW_WHEN(origOp.getNumOperands() != 1,
                    "Unexpected number of operands in multi-tile NCE permute: {0}, expected 1",
                    origOp.getNumOperands());
    const auto loc = origOp.getLoc();
    const auto copyDistTensorType = origOp.getOperand(0).getType().cast<VPU::DistributedTensorType>();
    const auto copyDistTensorAttr = copyDistTensorType.getDistribution();

    auto targetType = origOp.getOperand(0).getType().cast<NDTypeInterface>();
    targetType = targetType.changeDimsOrder(DimsOrder::NHWC);

    auto castToDistType =
            createCustomDistributedTensorType(clusteredOp, targetType, copyDistTensorAttr,
                                              copyDistTensorAttr.getEqualMemoryAndComputeView(), targetType.getShape());

    auto outBufferTypeInViewOp = typeConverter->convertType(castToDistType);
    const auto castLoc = appendLoc(loc, "cast number of input tiles");

    // ViewOp Input
    // Reshape to NxWxCxH
    // Layout change to NHWC
    auto inputViewOp = rewriter.create<VPUIP::ViewOp>(castLoc, outBufferTypeInViewOp, newArgs.getOperands());

    auto outValueCastInput =
            rewriter.create<mlir::bufferization::ToTensorOp>(loc, castToDistType, inputViewOp.getResult(),
                                                             /*restrict=*/nullptr, /*writable=*/nullptr);

    // Manual update output type
    auto outType = permuteOp.getOutput().getType().cast<NDTypeInterface>();
    auto outTypeShape = outType.getShape();
    const auto outNCEPermuteShape = calculateWCHShape(outTypeShape.raw());
    outType = outType.changeShape(ShapeRef(outNCEPermuteShape));
    outType = outType.changeDimsOrder(DimsOrder::NWCH);
    targetType = targetType.changeElemType(outType.getElementType());
    auto origOutDistribution = origOp.getResult(0).getType().cast<VPU::DistributedTensorType>().getDistribution();
    auto newOutputDistType =
            createCustomDistributedTensorType(clusteredOp, targetType, origOutDistribution,
                                              origOutDistribution.getEqualMemoryAndComputeView(), outTypeShape);
    auto newClusterTilingDistType = newOutputDistType.changeDimsOrder(DimsOrder::NWCH);

    //
    // Prepare output buffer for DPU
    //
    auto bufferType = typeConverter->convertType(outType);

    // Add PPE task to rescale output.
    const auto ppeTaskAttr = composePPETask(outType);

    mlir::UnitAttr isSuperdenseAttr = nullptr;
    if (isSuperdenseOp(permuteOp)) {
        VPUX_THROW_WHEN(permuteOp->getResult(0).getType().isa<VPU::SparseTensorType>(),
                        "Output cannot be sparse and super-dense at the same time");
        isSuperdenseAttr = mlir::UnitAttr::get(ctx);
    }

    const auto dpuCostAttr = permuteOp->hasAttr(DPUCost) ? permuteOp->getAttr(DPUCost) : nullptr;
    const auto activationWindowChannelLength = getIntAttr(ctx, static_cast<int32_t>(0));
    const auto isPermuteQuantizeAttr = mlir::UnitAttr::get(ctx);

    const auto bodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        // Note: preserve unrealized_conversion_cast inside VPU.NCEClusterTiling
        // body because this is the expectation.
        auto valueCastInput = builder.create<mlir::UnrealizedConversionCastOp>(
                loc, mlir::TypeRange{typeConverter->convertType(newOperands.front().getType())},
                mlir::ValueRange{newOperands.front()});
        _log.nest().trace("Allocating result buffer of type '{0}' for value type '{1}'", bufferType, outType);
        const auto outputBuffers =
                allocateBuffersOfType(_log.nest(), loc, builder, bufferType, /*individualBuffers=*/true);

        auto nceOp = createNCEClusterTask(builder, loc, valueCastInput.getResult(0), valueCastInput.getResult(0),
                                          /*weightsTable=*/nullptr,
                                          /*instruction_table_list=*/nullptr,
                                          /*activation_window=*/nullptr, outputBuffers, VPUIP::NCETaskType::ELTWISE,
                                          /*kernel_size=*/nullptr,
                                          /*kernel_strides=*/nullptr,
                                          /*kernel_padding=*/nullptr, activationWindowChannelLength,
                                          permuteOp.getWorkloads(), isSuperdenseAttr, ppeTaskAttr, dpuCostAttr,
                                          /*isInplace=*/nullptr, isPermuteQuantizeAttr, /*cmSpPattern=*/nullptr,
                                          /*inputChannelsCompression=*/nullptr, /*isNCEPermute=*/true,
                                          /*smallKernelOptimization=*/nullptr, _log);
        const auto nceOpOutType = nceOp.getType().cast<NDTypeInterface>();

        auto valueCastOutput = builder.create<mlir::UnrealizedConversionCastOp>(
                loc,
                mlir::TypeRange{getTensorType(nceOpOutType.getShape(), nceOpOutType.getElementType(),
                                              nceOpOutType.getDimsOrder(), nceOpOutType.getMemSpace())},
                mlir::ValueRange{nceOp});
        builder.create<VPU::YieldOp>(loc, valueCastOutput->getResult(0));
    };

    auto newClusterTilingOp = rewriter.create<VPU::NCEClusterTilingOp>(
            loc, newClusterTilingDistType, mlir::ValueRange{outValueCastInput->getResult(0)}, bodyBuilder);

    // ViewOp Output
    // Reshape to NxCxHxW
    // Layout change to NHWC
    auto inValueCastOutput = rewriter.create<mlir::bufferization::ToMemrefOp>(
            loc, typeConverter->convertType(newClusterTilingDistType), newClusterTilingOp.getResult(0),
            /*read_only=*/nullptr);

    auto outputViewOp = rewriter.create<VPUIP::ViewOp>(newClusterTilingOp.getLoc(),
                                                       typeConverter->convertType(origOp.getResult(0).getType()),
                                                       inValueCastOutput->getResult(0));

    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, outputViewOp.getResult());

    return mlir::success();
}
}  // namespace

mlir::LogicalResult vpux::lowerMultiTileVpuNcePermuteOneShot(mlir::MLIRContext* ctx, mlir::Operation* func,
                                                             vpux::Logger& log) {
    mlir::ConversionTarget target(*ctx);

    // normal NCEClusterTiling will be handled separately
    target.addDynamicallyLegalOp<VPU::NCEClusterTilingOp>([&](VPU::NCEClusterTilingOp op) {
        return op.getInnerTaskOpOfType<VPU::NCEPermuteOp>() == nullptr;
    });

    target.addLegalDialect<VPUIP::VPUIPDialect>();
    target.addLegalOp<VPU::YieldOp>();
    target.addLegalOp<mlir::memref::AllocOp>();
    vpux::populateBufferizeMaterializationLegality(target);
    // allow to_tensor/to_memref in multi-tile permute, these are going to
    // appear in case of one-shot bufferization.
    target.addLegalOp<mlir::bufferization::ToTensorOp>();
    target.addLegalOp<mlir::bufferization::ToMemrefOp>();

    vpux::BufferizeOneShotTypeConverter typeConverter;
    mlir::RewritePatternSet patterns(ctx);
    patterns.add<NCEPermuteMultiTileRewriter>(typeConverter, ctx, log);

    return mlir::applyPartialConversion(func, target, std::move(patterns));
}

//
// registerVpuNceBufferizableOpInterfaces
//

void vpux::registerVpuNceBufferizableOpInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, VPU::VPUDialect*, VPUIP::VPUIPDialect*) {
        VPU::NCEConvolutionOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::NCEConvolutionOp>>(*ctx);
        VPU::NCEMaxPoolOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::NCEMaxPoolOp>>(*ctx);
        VPU::NCEAveragePoolOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::NCEAveragePoolOp>>(*ctx);
        VPU::NCEDepthConvolutionOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::NCEDepthConvolutionOp>>(*ctx);
        VPU::NCEInterpolateOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::NCEInterpolateOp>>(*ctx);
        VPU::NCEEltwiseOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::NCEEltwiseOp>>(*ctx);
        VPU::NCECompressConvolutionOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::NCECompressConvolutionOp>>(
                *ctx);
        VPU::NCEPermuteOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::NCEPermuteOp>>(*ctx);
        VPU::NCEMatMulOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::NCEMatMulOp>>(*ctx);
    });
}
