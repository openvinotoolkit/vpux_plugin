//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/resources.hpp"

#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/dpu_tiler.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/factories/split_cost_getter.hpp"

#include "vpux/utils/core/enums.hpp"

using namespace vpux;
using namespace VPU;

namespace {

//
// Upper bound for workload numbers
//

constexpr int64_t MAX_SPLIT_NUMBER = 50;

//
// generateWorkloads
//

// for workloads in sub tensors, offsets need to be from original full output tensor
void addSubTensorOffset(TileInfo& tileInfo, ShapeRef tensorOffset) {
    VPUX_THROW_WHEN(tileInfo.offsets.size() != tensorOffset.size(),
                    "Invalid size for TileInfo.offset {0} and sub tensor offset {1}", tileInfo.offsets.size(),
                    tensorOffset.size());

    for (auto d : irange(tileInfo.offsets.size())) {
        const auto dim = Dim(d);
        tileInfo.offsets[dim] += tensorOffset[dim];
    }
}

void generateWorkloads(mlir::OpBuilder& builder, VPU::NCEOpInterface origOp,
                       const VPUIP::WorkloadCostParams& costParams, VPU::MPEMode mpeMode,
                       ArrayRef<bool> isTileOverDimsSupported, VPUNN::VPUCostModel& costModel, Logger log,
                       mlir::IntegerAttr clusterId = nullptr, ShapeRef subTensorOffset = {}) {
    VPUIP::DpuTiler dpuTiler(costParams.outputShape, mpeMode);

    VPUIP::WorkloadSplitPool splitPoolSet;

    dpuTiler.tileOverH(costParams.numDPU, splitPoolSet);

    if (costParams.outputShape.size() == 5) {
        int64_t cluster = 0;
        if (clusterId != nullptr) {
            cluster = clusterId.getValue().getSExtValue();
        }
        // This logic assumes that each chunk starts right after the previous.
        // cluster 0: outOffsets [0, 0, 0, 0, 0]  outSizes [32, 1, 16, 16, 1]
        // cluster 1: outOffsets [32, 0, 0, 0, 0] outSizes [32, 1, 16, 16, 1]
        // cluster 2: outOffsets [64, 0, 0, 0, 0] outSizes [32, 1, 16, 16, 1]
        const Shape offsets = {cluster * costParams.outputShape.front(), 0, 0, 0, 0};
        auto tilePad = VPU::getPaddingAttr(builder.getContext(), 0, 0, 0, 0);
        origOp.addWorkload(builder, origOp.getLoc(), offsets, costParams.outputShape, tilePad,
                           VPU::MPEMode::CUBOID_16x16, getIntAttr(origOp->getContext(), cluster));
        return;
    } else {
        dpuTiler.tileOverH(costParams.numDPU, splitPoolSet);
        // Invariants that produce sparse activations must have the same number of channels across the variants
        const auto requiresEqualZ = (origOp->getResult(0).getType().dyn_cast<VPU::SparseTensorType>() != nullptr);

        const auto splitNumPool =
                (costParams.arch == VPU::ArchKind::NPU37XX || costParams.arch == VPU::ArchKind::NPU40XX)
                        ? dpuTiler.generateSplitNumberPool(costParams.numDPU, 1)
                        : dpuTiler.generateSplitNumberPool(costParams.numDPU, MAX_SPLIT_NUMBER);

        for (const auto& splitNum : splitNumPool) {
            if (isTileOverDimsSupported[Dims4D::Act::W.ind()] == true &&
                isTileOverDimsSupported[Dims4D::Act::H.ind()] == true) {
                dpuTiler.tileOverHW(splitNum, VPUIP::SplitDimension::SPLIT_OVER_HW, splitPoolSet);
            } else if (isTileOverDimsSupported[Dims4D::Act::W.ind()] == true) {
                dpuTiler.tileOverHW(splitNum, VPUIP::SplitDimension::SPLIT_OVER_W, splitPoolSet);
            } else if (isTileOverDimsSupported[Dims4D::Act::H.ind()] == true) {
                dpuTiler.tileOverHW(splitNum, VPUIP::SplitDimension::SPLIT_OVER_H, splitPoolSet);
            }
            if (isTileOverDimsSupported[Dims4D::Act::C.ind()] == true) {
                dpuTiler.tileOverZ(splitNum, splitPoolSet, requiresEqualZ);
            }
        }
    }

    // select workload with minimum cost
    auto splitPool = to_std_vector(splitPoolSet);
    VPUX_THROW_WHEN(splitPool.empty(), "Workload split pool is empty");

    std::vector<int64_t> splitPoolCosts(splitPool.size(), 0);
    for (const auto ind : irange(splitPool.size())) {
        auto& curSplit = splitPool[ind];

        if (clusterId != nullptr) {
            for (auto& wl : curSplit) {
                auto& outTile = std::get<0>(wl);
                addSubTensorOffset(outTile, subTensorOffset);
            }
        }
        const auto logCb = [&](const formatv_object_base& msg) {
            log.trace("{0}", msg.str());
        };
        auto computeSplitCostByArch = VPUIP::getSplitCostCb(costParams.arch);
        splitPoolCosts[ind] = computeSplitCostByArch(curSplit, costParams, costModel, logCb);
    }

    const auto bestSplitInd = std::min_element(splitPoolCosts.begin(), splitPoolCosts.end()) - splitPoolCosts.begin();
    if (splitPoolCosts[bestSplitInd] >= VPU::INVALID_COST_BASE) {
        log.setName("GenerateWorkloads");
        log.debug("An INVALID_COST is caught for bestSplit when calling VPUNN. You can pass a logCb with LOG_ERROR "
                  "level to print debug info in `computeSplitCostByArch` function and report to E#83609 if necessary");
        log.nest().debug("bestSplit cost value: {0}", splitPoolCosts[bestSplitInd]);
    }
    const auto& bestSplit = splitPool[bestSplitInd];

    origOp->setAttr(DPUCost, getIntAttr(origOp->getContext(), splitPoolCosts[bestSplitInd]));

    const auto kernel = origOp.getKernelSizeVal();
    const auto strides = origOp.getStridesVal();

    for (const auto& wl : bestSplit) {
        const auto& outTile = std::get<0>(wl);
        const auto mpeMode = std::get<1>(wl);

        const auto padsTileConf =
                backInferPadsTile(outTile, costParams.fullInputShape, costParams.padInfo, kernel, strides);
        auto tilePad = VPU::getPaddingAttr(builder.getContext(), padsTileConf);

        origOp.addWorkload(builder, origOp.getLoc(), outTile.offsets, outTile.shape, tilePad, mpeMode, clusterId);
    }
}

//
// splitOntoWorkloads
//

void splitOntoWorkloads(mlir::OpBuilder& builder, VPU::NCEOpInterface origOp, VPUIP::WorkloadCostParams& costParams,
                        VPU::MPEMode mpeMode, ArrayRef<bool> isTileOverDimsSupported, VPUNN::VPUCostModel& costModel,
                        Logger log) {
    auto distributedIf = mlir::dyn_cast<VPU::DistributedTypeInterface>(origOp->getResult(0).getType());
    if ((distributedIf != nullptr) && (distributedIf.containsDistributedTypes())) {
        const auto outputs = origOp->getResults();
        VPUX_THROW_UNLESS(outputs.size() == 1, "Wrong outputs size: {0}", outputs.size());

        const auto output = *outputs.begin();

        auto getDistributedTensor = [](const mlir::Value value) -> VPU::DistributedTensorType {
            if (auto sparseTensor = value.getType().dyn_cast<VPU::SparseTensorType>()) {
                return sparseTensor.getData().dyn_cast<VPU::DistributedTensorType>();
            }
            return value.getType().dyn_cast<VPU::DistributedTensorType>();
        };

        auto distributedOutputType = getDistributedTensor(output);
        VPUX_THROW_WHEN(distributedOutputType == nullptr, "Wrong output type {0} for NCEClusterTilingOp",
                        output.getType());

        const auto outputSubTensorShapes = distributedOutputType.getPerClusterComputeShapes();
        auto outputSubTensorOffsets = distributedOutputType.getPerClusterComputeShapeOffsets();
        VPUX_THROW_WHEN(outputSubTensorShapes.size() != outputSubTensorOffsets.size(),
                        "sub tensor size:{0} not equal to offset size:{1}", outputSubTensorShapes.size(),
                        outputSubTensorOffsets.size());

        const auto inputs = origOp->getOperands();
        VPUX_THROW_UNLESS(inputs.size() >= 1, "Wrong inputs size: {0}", inputs.size());

        const auto input = *inputs.begin();
        auto distributedInputType = getDistributedTensor(input);
        VPUX_THROW_WHEN(distributedInputType == nullptr, "Wrong input type {0} for NCEClusterTilingOp",
                        input.getType());

        // @todo When halos supported in VPUNN, we need use computeShape instead of memory shape
        // See E#87028
        const auto inputSubTensorShapes = distributedInputType.getPerClusterMemoryShapes();
        VPUX_THROW_WHEN(outputSubTensorShapes.size() != inputSubTensorShapes.size(),
                        "output tensor size:{0} not equal to input tensor size:{1}", outputSubTensorShapes.size(),
                        inputSubTensorShapes.size());

        const auto distributionAttr = distributedOutputType.getDistribution();
        if (isSegmentedOverC(distributionAttr)) {
            // Here we keep the output offset for SOC NCEPermute to keep the logic be aligned
            // with SOH because it will be lowered to SOH NCEEltwise
            if (mlir::isa<VPU::NCEPermuteOp>(origOp.getOperation())) {
                // Correct layer strategy to the real strategy after being lowered to Eltwise
                costParams.layerStrategy = VPU::MultiClusterStrategy::SplitOverHeight;
            } else {
                // In the case of an non broadcasted SOK, outputSubTensorOffsets don't need to be applied
                for (auto& shapeOffset : outputSubTensorOffsets) {
                    std::fill(shapeOffset.begin(), shapeOffset.end(), 0);
                }
            }
        }

        for (size_t clusterId = 0; clusterId < outputSubTensorShapes.size(); clusterId++) {
            auto clusterIdAttr = getIntAttr(origOp->getContext(), clusterId);
            // Update workload params for per tile
            costParams.inputShape = inputSubTensorShapes[clusterId];
            costParams.outputShape = outputSubTensorShapes[clusterId];
            costParams.numTiles = distributionAttr.getNumClusters().getInt();

            if (costParams.arch == VPU::ArchKind::NPU37XX &&
                mlir::isa<VPU::NCEConvolutionOp, VPU::NCECompressConvolutionOp, VPU::NCEInterpolateOp>(origOp)) {
                mpeMode = origOp.getMpeMode(nullptr, nullptr, outputSubTensorShapes[clusterId]);
            }
            generateWorkloads(builder, origOp, costParams, mpeMode, isTileOverDimsSupported, costModel, log,
                              clusterIdAttr, outputSubTensorOffsets[clusterId]);
        }
    } else {
        generateWorkloads(builder, origOp, costParams, mpeMode, isTileOverDimsSupported, costModel, log);
    }
}

//
// GenericNCERewrite
//

class GenericNCERewrite final : public mlir::OpInterfaceRewritePattern<VPU::NCEOpInterface> {
public:
    GenericNCERewrite(mlir::MLIRContext* ctx, int64_t numDPU, VPU::ArchKind arch,
                      std::shared_ptr<VPUNN::VPUCostModel> costModel, Logger log)
            : mlir::OpInterfaceRewritePattern<VPU::NCEOpInterface>(ctx),
              _numDPU(numDPU),
              _arch(arch),
              _costModel(std::move(costModel)),
              _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::NCEOpInterface origOp, mlir::PatternRewriter& rewriter) const final;

private:
    int64_t _numDPU;
    VPU::ArchKind _arch;
    std::shared_ptr<VPUNN::VPUCostModel> _costModel;
    Logger _log;
};

mlir::LogicalResult GenericNCERewrite::matchAndRewrite(VPU::NCEOpInterface nceOp,
                                                       mlir::PatternRewriter& rewriter) const {
    const auto inputType = nceOp->getOperand(0).getType().cast<NDTypeInterface>();
    const auto outputType = nceOp->getResult(0).getType().cast<NDTypeInterface>();

    const auto inElemType = inputType.getElementType();
    const auto outElemType = outputType.getElementType();

    const auto outputShape = outputType.getShape();

    const auto mpeMode = nceOp.getMpeMode(inElemType, outElemType, outputShape);

    auto params = VPU::getWorkloadCostParam(nceOp, _arch, _numDPU);

    SmallVector<bool> isTileOverDimsSupported = {false, mpeMode == VPU::MPEMode::VECTOR, true, true};
    if (mlir::isa<VPU::NCEConvolutionOp>(nceOp.getOperation())) {
        const auto inOrder = inputType.getDimsOrder();
        const auto isCMajor = inOrder == DimsOrder::NCHW;
        isTileOverDimsSupported[Dims4D::Act::C.ind()] |= !isCMajor;
    } else if (mlir::isa<VPU::NCEEltwiseOp>(nceOp.getOperation())) {
        isTileOverDimsSupported[Dims4D::Act::C.ind()] = false;
    } else if (mlir::isa<VPU::NCEPermuteOp>(nceOp.getOperation())) {
        // For NCE Permute operation tileOverHK is needed : See E#91637
        isTileOverDimsSupported[Dims4D::Act::W.ind()] = false;
    }

    rewriter.modifyOpInPlace(nceOp, [&]() {
        splitOntoWorkloads(rewriter, nceOp, params, mpeMode, ArrayRef(isTileOverDimsSupported), *_costModel, _log);
    });

    return mlir::success();
}

//
// SplitNCEOpsOntoWorkloads
//

class SplitNCEOpsOntoWorkloadsPass final : public SplitNCEOpsOntoWorkloadsBase<SplitNCEOpsOntoWorkloadsPass> {
public:
    explicit SplitNCEOpsOntoWorkloadsPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void SplitNCEOpsOntoWorkloadsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    auto module = func->getParentOfType<mlir::ModuleOp>();

    const auto arch = VPU::getArch(module);

    auto nceCluster = IE::getTileExecutor(module);
    VPUX_THROW_UNLESS(nceCluster != nullptr, "Failed to get NCE_Cluster information");

    auto dpuExec = nceCluster.getSubExecutor(VPU::ExecutorKind::DPU);
    VPUX_THROW_UNLESS(dpuExec != nullptr, "Failed to get DPU information");

    const auto numDPUs = dpuExec.getCount();

    const auto costModel = VPU::createCostModel(arch);

    mlir::ConversionTarget target(ctx);
    target.markUnknownOpDynamicallyLegal([&](mlir::Operation* op) {
        if (auto swOp = mlir::dyn_cast<VPU::SWOpInterface>(op)) {
            return true;
        }
        if (auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(op)) {
            return !nceOp.getWorkloads().empty();
        }
        return true;
    });
    target.addLegalOp<VPU::DPUWorkloadOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<GenericNCERewrite>(&ctx, numDPUs, arch, costModel, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createSplitNCEOpsOntoWorkloadsPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createSplitNCEOpsOntoWorkloadsPass(Logger log) {
    return std::make_unique<SplitNCEOpsOntoWorkloadsPass>(log);
}
