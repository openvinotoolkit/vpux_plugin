//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <llvm/ADT/TypeSwitch.h>

#include "vpux/compiler/core/attributes/dim.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/core/type_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/sparsity_constraint.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/se_roll_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sparsity_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/dpu_tiler.hpp"

#include <mlir/IR/IRMapping.h>
#include <vpux/compiler/dialect/VPUIP/utils/convert_to_dma_utils.hpp>

namespace vpux {
namespace VPU {

TilingMode getTilingSupportedMode(VPU::TilingBuilderOpInterface origOp, bool enablePrefetchTiling, Logger log) {
    auto tilingMode = TilingMode::ISOLATED;

    auto op = origOp.getOperation();
    auto tilingInfo = mlir::cast<VPU::TilingInfoOpInterface>(op);

    // Prefetching for HW layers
    if (enablePrefetchTiling && mlir::isa<VPU::NCEOpInterface>(op)) {
        const auto resShape = getShape(op->getResult(0));
        const Shape neutralTile(resShape.size(), 1);
        auto fillTiles = fillDividedTiles(op, neutralTile, resShape);
        const auto isSupportIsolated =
                tilingInfo.isSupportedTiling(fillTiles.value(), TilingMode::ISOLATED, log.nest());
        const auto isPrefetchable = VPU::prefetchTilingConditionSatisfied(op, log.nest());
        tilingMode = isSupportIsolated && isPrefetchable ? TilingMode::PREFETCHING : TilingMode::PIPELINING;
    }

    return tilingMode;
}

mlir::FailureOr<OutputTiling> getLayerTilingStrategy(VPU::TilingBuilderOpInterface origOp, bool enablePrefetchTiling,
                                                     Logger log) {
    auto tilingMode = TilingMode::ISOLATED;
    return getLayerTilingStrategy(origOp, enablePrefetchTiling, tilingMode, log);
}

mlir::FailureOr<OutputTiling> getLayerTilingStrategy(VPU::TilingBuilderOpInterface origOp, bool enablePrefetchTiling,
                                                     TilingMode& mode, Logger log) {
    log.trace("getLayerTilingStrategy for op '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    log.nest().trace("Enable Prefetch Tiling: {0}", enablePrefetchTiling);

    // Default to ISOLATED mode
    mode = getTilingSupportedMode(origOp, enablePrefetchTiling, log);

    log.nest().trace("Assigning {0} tiling strategy", getTilingModeStr(mode));
    return origOp.getTilingStrategy(mode, log.nest());
}

mlir::LogicalResult checkAndAlignActInputTiling(vpux::VPU::NCEOpInterface nceOp, InputTiling& inputTiling,
                                                vpux::Logger log) {
    auto origInputType = nceOp->getOperand(0).getType().cast<vpux::NDTypeInterface>();
    auto tiledInputType = origInputType.extractDenseTile(inputTiling.tiles[0].offsets, inputTiling.tiles[0].shape);
    if (mlir::succeeded(nceOp.verifyInputType(tiledInputType))) {
        return mlir::success();
    }
    log.trace("Inferred activation input tiling {0} is invalid for {1}", inputTiling.tiles[0], nceOp->getName());
    auto stride = nceOp.getStridesVal()[Dims4D::Strides::X.ind()];  // get W side strides
    int64_t bias = 0;
    auto newInputActTiling = inputTiling.tiles[0];
    while (++bias < stride) {
        auto alignedShape =
                Shape({inputTiling.tiles[0].shape[Dims4D::Act::N], inputTiling.tiles[0].shape[Dims4D::Act::C],
                       inputTiling.tiles[0].shape[Dims4D::Act::H], inputTiling.tiles[0].shape[Dims4D::Act::W] + bias});
        newInputActTiling = TileInfo(alignedShape, inputTiling.tiles[0].offsets, inputTiling.tiles[0].axis);
        auto newInputActType = origInputType.extractDenseTile(newInputActTiling.offsets, newInputActTiling.shape);
        if (mlir::succeeded(nceOp.verifyInputType(newInputActType))) {
            inputTiling.tiles[0] = newInputActTiling;
            log.trace("Input tiling is corrected to {0}", inputTiling.tiles[0]);
            return mlir::success();
        }
    }
    VPUX_THROW("Cannot find aligned act input tiling for op {0} at {1}", nceOp->getName(), nceOp->getLoc());
}

// Temporari solution until E#59988 will be implemented.
// Function can be deleted when E#59993 related interface will be added.
SmallVector<mlir::Value> reifyTileTopK(VPU::TilingBuilderOpInterface origOp, const TileInfo& outputTile,
                                       mlir::OpBuilder& builder, Logger log) {
    log = log.nest(2);
    log.trace("{0}", outputTile);

    auto inputTiling = origOp.backInferTileInfo(outputTile, log);
    auto& inTiles = inputTiling.tiles;
    VPUX_THROW_UNLESS(!inTiles.empty(), "Got empty tile information");

    mlir::IRMapping mapper;
    for (auto p : origOp->getOperands() | indexed) {
        auto origInput = p.value();
        auto inputIdx = p.index();
        const auto valName = printToString("input {0}", inputIdx);
        const auto tiledInput = vpux::VPU::makeTile(builder, origOp->getLoc(), origInput, inTiles[inputIdx], valName);
        mapper.map(origInput, tiledInput);
    }
    const auto tileLoc = appendLoc(origOp->getLoc(), "output tile {0}", outputTile.offsets);
    auto* tiledOp = builder.clone(*origOp, mapper);
    tiledOp->setLoc(tileLoc);

    auto tiledBuilderOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(tiledOp);
    VPUX_THROW_WHEN(tiledBuilderOp == nullptr, "Operation '{0}' doesn't implement TilingBuilderOpInterface",
                    tiledBuilderOp->getName());
    tiledBuilderOp.adjustAttrs(inputTiling, outputTile);

    SmallVector<mlir::Value> ret;
    for (auto p : origOp->getResults() | indexed) {
        auto idx = p.index();
        const auto baseResType = origOp->getResult(idx).getType().cast<vpux::NDTypeInterface>();
        const auto tiledResType = baseResType.extractDenseTile(outputTile.offsets, outputTile.shape);
        auto tiledRes = tiledOp->getResult(idx);
        tiledRes.setType(tiledResType);
        ret.push_back(tiledRes);
    }
    return ret;
}

// Temporari solution until E#59988 will be implemented.
// Function can be deleted when E#59993 related interface will be added.
mlir::LogicalResult applyTileStrategyTopK(VPU::TilingBuilderOpInterface origOp, const OutputTiling& tiles,
                                          mlir::PatternRewriter& rewriter, Logger log) {
    // apply the generated tiling strategy and create tiled operations
    // insert the tiled pattern with a concat to the IR
    SmallVector<SmallVector<mlir::Value>> resultTileVals(origOp->getNumResults());
    SmallVector<SmallVector<ShapeRef>> resultTileOffsets(origOp->getNumResults());
    for (const auto& outputTile : tiles) {
        const auto tiledRes = reifyTileTopK(origOp, outputTile, rewriter, log);
        for (auto p : origOp->getResults() | indexed) {
            auto idx = p.index();
            const auto tiledShape = getShape(tiledRes[idx]);
            VPUX_THROW_UNLESS(tiledShape == outputTile.shape,
                              "Inferred tiled output shape '{0}' doesn't match with generated '{1}'", tiledShape,
                              outputTile.shape);
            resultTileOffsets[idx].push_back(outputTile.offsets);
            resultTileVals[idx].push_back(tiledRes[idx]);
        }
    }

    SmallVector<mlir::Value> opsConcat;
    for (auto p : origOp->getResults() | indexed) {
        auto idx = p.index();
        auto opConcat =
                rewriter.create<VPU::ConcatOp>(origOp->getLoc(), origOp->getResult(idx).getType(),
                                               mlir::ValueRange(resultTileVals[idx]), ArrayRef(resultTileOffsets[idx]));
        opsConcat.push_back(opConcat.getOutput());
    }
    rewriter.replaceOp(origOp, opsConcat);

    return mlir::success();
}

// Function can be deleted when E#59993 related interface will be added.
SmallVector<mlir::Value> reifyTilesMultiOutput(VPU::TilingBuilderOpInterface origOp, const TileInfo& outputTile,
                                               mlir::OpBuilder& builder, Logger log) {
    log = log.nest(2);
    log.trace("{0}", outputTile);

    auto inputTiling = origOp.backInferTileInfo(outputTile, log);
    auto& inTiles = inputTiling.tiles;

    VPUX_THROW_UNLESS(!inTiles.empty(), "Got empty tile information");

    mlir::IRMapping mapper;
    for (auto p : origOp->getOperands() | indexed) {
        auto origInput = p.value();
        auto inputIdx = p.index();

        const auto valName = printToString("input {0}", inputIdx);
        const auto tiledInput = vpux::VPU::makeTile(builder, origOp->getLoc(), origInput, inTiles[inputIdx], valName);

        mapper.map(origInput, tiledInput);
    }

    const auto tileLoc = appendLoc(origOp->getLoc(), "output tile {0}", outputTile.offsets);

    auto* tiledOp = builder.clone(*origOp, mapper);
    tiledOp->setLoc(tileLoc);

    auto tiledBuilderOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(tiledOp);
    VPUX_THROW_WHEN(tiledBuilderOp == nullptr, "Operation '{0}' doesn't implement TilingBuilderOpInterface",
                    tiledBuilderOp->getName());

    tiledBuilderOp.adjustAttrs(inputTiling, outputTile);

    vpux::inferReturnTypes(tiledOp, vpux::InferShapedTypeMode::ALL);

    return tiledOp->getResults();
}

// This is an attempt to implement multi-output tiling
// Should be removed after a common approach will be implemented E#59993
mlir::LogicalResult applyTileStrategyMultiOutput(VPU::TilingBuilderOpInterface origOp, const OutputTiling& tiles,
                                                 mlir::PatternRewriter& rewriter, Logger log) {
    auto resultTileVals = SmallVector<SmallVector<mlir::Value>>(origOp->getNumResults());
    auto resultTileOffsets = SmallVector<SmallVector<Shape>>(origOp->getNumResults());

    for (const auto& outputTile : tiles) {
        const auto values = reifyTilesMultiOutput(origOp, outputTile, rewriter, log);
        const auto outputTiles = origOp.getOutputTiling(outputTile, log);

        for (const auto& p : zip(values, outputTiles)) {
            const auto tiledShape = getShape(std::get<0>(p));
            VPUX_THROW_UNLESS(tiledShape == std::get<1>(p).shape,
                              "Inferred tiled output shape '{0}' doesn't match with generated '{1}'", tiledShape,
                              outputTile.shape);
        }

        for (const auto& p : origOp->getResults() | indexed) {
            auto idx = p.index();
            resultTileOffsets[idx].push_back(outputTiles[idx].offsets);
            resultTileVals[idx].push_back(values[idx]);
        }
    }

    SmallVector<mlir::Value> opsConcat;
    for (const auto& p : origOp->getResults() | indexed) {
        auto idx = p.index();
        auto opConcat =
                rewriter.create<VPU::ConcatOp>(origOp->getLoc(), origOp->getResult(idx).getType(),
                                               mlir::ValueRange(resultTileVals[idx]), ArrayRef(resultTileOffsets[idx]));
        opsConcat.push_back(opConcat.getOutput());
    }
    rewriter.replaceOp(origOp, opsConcat);

    return mlir::success();
}

mlir::Value reifyTile(VPU::TilingBuilderOpInterface origOp, const TileInfo& outputTile, mlir::OpBuilder& builder,
                      Logger log) {
    log = log.nest(2);
    log.trace("{0}", outputTile);

    auto inputTiling = origOp.backInferTileInfo(outputTile, log);
    auto& inTiles = inputTiling.tiles;

    VPUX_THROW_UNLESS(!inTiles.empty(), "Got empty tile information");

    mlir::IRMapping mapper;
    for (auto p : origOp->getOperands() | indexed) {
        auto origInput = p.value();
        auto inputIdx = p.index();

        const auto valName = printToString("input {0}", inputIdx);
        const auto tiledInput = vpux::VPU::makeTile(builder, origOp->getLoc(), origInput, inTiles[inputIdx], valName);

        mapper.map(origInput, tiledInput);
    }

    const auto tileLoc = appendLoc(origOp->getLoc(), "output tile {0}", outputTile.offsets);

    auto* tiledOp = builder.clone(*origOp, mapper);
    tiledOp->setLoc(tileLoc);

    auto tiledBuilderOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(tiledOp);
    VPUX_THROW_WHEN(tiledBuilderOp == nullptr, "Operation '{0}' doesn't implement TilingBuilderOpInterface",
                    tiledBuilderOp->getName());

    tiledBuilderOp.adjustAttrs(inputTiling, outputTile);

    const auto baseResType = origOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    const auto tiledResType = baseResType.extractDenseTile(outputTile.offsets, outputTile.shape);

    auto tiledRes = tiledOp->getResult(0);
    tiledRes.setType(tiledResType);

    return tiledRes;
}

mlir::LogicalResult applyTileStrategy(VPU::TilingBuilderOpInterface origOp, const OutputTiling& tiles,
                                      mlir::PatternRewriter& rewriter, Logger log) {
    // TODO: delete when function will allow rewrite for multiple output ops (E#59988)
    if (mlir::isa<VPU::TopKOp>(origOp)) {
        return applyTileStrategyTopK(origOp, tiles, rewriter, log);
    }
    if (mlir::isa<VPU::DetectionOutputSortOp>(origOp) || mlir::isa<VPU::DetectionOutputNmsCaffeOp>(origOp) ||
        mlir::isa<VPU::GRUSequenceOp>(origOp) || mlir::isa<VPU::GRUSequenceLastPartOp>(origOp)) {
        return applyTileStrategyMultiOutput(origOp, tiles, rewriter, log);
    }

    // apply the generated tiling strategy and create tiled operations
    // insert the tiled pattern with a concat to the IR
    SmallVector<mlir::Value> resultTileVals;
    SmallVector<ShapeRef> resultTileOffsets;

    resultTileVals.reserve(tiles.size());
    resultTileOffsets.reserve(tiles.size());

    for (const auto& outputTile : tiles) {
        const auto tiledRes = reifyTile(origOp, outputTile, rewriter, log);

        const auto tiledShape = getShape(tiledRes);
        VPUX_THROW_UNLESS(tiledShape == outputTile.shape,
                          "Inferred tiled output shape '{0}' doesn't match with generated '{1}'", tiledShape,
                          outputTile.shape);

        resultTileVals.push_back(tiledRes);
        resultTileOffsets.push_back(outputTile.offsets);
    }

    rewriter.replaceOpWithNewOp<VPU::ConcatOp>(origOp, origOp->getResult(0).getType(), mlir::ValueRange(resultTileVals),
                                               ArrayRef(resultTileOffsets));

    // update concat users and also place correctly in the IR
    for (auto* concatOp : resultTileVals[0].getUsers()) {
        if (!mlir::isa<VPU::ConcatOp>(concatOp)) {
            continue;
        }
        origOp->replaceAllUsesWith(concatOp);
        for (auto concatConsumer : concatOp->getResult(0).getUsers()) {
            if (concatOp->isBeforeInBlock(concatConsumer)) {
                continue;
            }
            concatOp->moveBefore(concatConsumer);
            // also move the Slice+Conv pattern, first conv, then slice
            for (auto concatOperand : concatOp->getOperands()) {
                auto concatProducer = concatOperand.getDefiningOp();
                if (concatProducer->isBeforeInBlock(concatOp)) {
                    continue;
                }
                concatProducer->moveBefore(concatOp);
                auto sliceOp = concatProducer->getOperand(0).getDefiningOp();
                for (auto sliceOperand : concatProducer->getOperands()) {
                    if (mlir::isa<VPU::SliceOp>(sliceOperand.getDefiningOp())) {
                        sliceOp = sliceOperand.getDefiningOp();
                        break;
                    }
                }
                if (!mlir::isa<VPU::SliceOp>(sliceOp)) {
                    continue;
                }
                sliceOp->moveBefore(concatProducer);
            }
        }
        break;
    }

    return mlir::success();
}

bool hasMultiBranches(mlir::Operation* op) {
    // not the only result
    if (op->getResults().size() != 1) {
        return true;
    }
    // only one user
    if (op->getResult(0).hasOneUse()) {
        return false;
    }
    // only one result but multiple users
    auto user1 = op->getResult(0).user_begin();
    for (auto remainUser : llvm::drop_begin(op->getResult(0).getUsers())) {
        if (remainUser != *user1) {
            return true;
        }
    }
    return false;
}

mlir::Operation* getParentComputeOp(mlir::Operation* op) {
    // for const prefetch ignore cases where activation is handled by
    // intermediate operations and causes a stall
    // prefetch is wanted from current op to parent op
    const auto isOpIgnorable = [](mlir::Operation* op) -> bool {
        if (auto nceEltwiseAnd = mlir::dyn_cast<VPU::NCEEltwiseOp>(op)) {
            return nceEltwiseAnd.getOpType() == VPU::EltwiseType::AND;
        }
        if (mlir::isa<VPU::MemPermuteOp, VPU::DepthToSpaceOp, VPU::SpaceToDepthOp>(op) &&
            !VPUIP::isLegalAndBeneficialConvertToDMA(op)) {
            // don't ignore layers that will be converted to SW but not SWOpInterface now
            return true;
        }
        return !mlir::isa<VPU::NCEOpInterface>(op) && !mlir::isa<VPU::SWOpInterface>(op);
    };

    mlir::Operation* parentOp = op->getOperand(0).getDefiningOp();
    while (parentOp && isOpIgnorable(parentOp)) {
        // skip the Permute, Reshape and And
        if (parentOp->getOperands().size() < 1) {
            break;
        }
        if (hasMultiBranches(parentOp)) {
            // for parallel sub-graphs, the order is undecided yet
            // abandon prefetching these cases
            return nullptr;
        }
        parentOp = parentOp->getOperand(0).getDefiningOp();
    }
    // check the last op
    return (parentOp == nullptr || hasMultiBranches(parentOp)) ? nullptr : parentOp;
}

bool prefetchTilingConditionSatisfied(mlir::Operation* op, Logger log) {
    auto parentOp = getParentComputeOp(op);
    if (parentOp == nullptr) {
        return false;
    }
    const auto arch = VPU::getArch(op);
    if (arch == VPU::ArchKind::NPU30XX) {
        if (!mlir::isa<VPU::NCEOpInterface>(parentOp)) {
            return false;
        }
    }
    auto opTilingInter = mlir::dyn_cast<VPU::TilingInfoOpInterface>(op);
    auto parentTilingInter = mlir::dyn_cast<VPU::TilingInfoOpInterface>(parentOp);
    if (!opTilingInter || !parentTilingInter) {
        return false;
    }
    auto opTilingBuilder = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(op);
    if (!opTilingBuilder) {
        return false;
    }

    // For parallel sub-graphs, the order is undecided yet
    // Abandon prefetching these cases
    if (!parentOp->getResult(0).hasOneUse()) {
        auto user1 = *parentOp->getResult(0).getUsers().begin();
        for (auto remainUser : parentOp->getResult(0).getUsers()) {
            if (remainUser != user1) {
                return false;
            }
        }
    }

    // Check if tile pattern is supported
    const auto resShape = getShape(op->getResult(0));
    const Shape neutralTile(resShape.size(), 1);
    auto fillTiles = fillDividedTiles(op, neutralTile, resShape);
    if (mlir::failed(fillTiles)) {
        return false;
    }
    if (opTilingInter.isSupportedTiling(fillTiles.value(), TilingMode::PREFETCHING, log)) {
        return false;
    }
    log.nest(1).trace("Attempting to satisfy PREFETCHING tiling.");
    auto tiles = opTilingBuilder.getTilingStrategy(TilingMode::PREFETCHING, log.nest());
    if (mlir::failed(tiles)) {
        return false;
    }

    return tiles.value().begin()->axis != neutralTile;
}

bool isLargeConstOp(mlir::Operation* op, Logger log) {
    // The operation should have constant filter
    if (!mlir::isa<VPU::NCEConvolutionOp>(op) && !mlir::isa<VPU::NCEDepthConvolutionOp>(op) &&
        !mlir::isa<VPU::NCECompressConvolutionOp>(op)) {
        return false;
    }
    auto filter = op->getOperand(1).getDefiningOp<Const::DeclareOp>();
    if (filter == nullptr) {
        return false;
    }

    Byte filterSize(0);
    auto filterType = filter.getOutput().getType().cast<vpux::NDTypeInterface>();
    if (op->hasAttr(multiClusterStrategy)) {
        auto nceOp = mlir::cast<VPU::NCEOpInterface>(op);
        auto clusterOp = mlir::cast<VPU::ClusteredOpInterface>(op);
        auto numClusters = VPU::getOptimalNumClusters(
                clusterOp, filterType.getShape()[Dims4D::Filter::OC],
                clusterOp->getAttr(VPU::multiClusterStrategy).cast<VPU::MultiClusterStrategyAttr>().getValue());
        auto filterDistributedType = VPU::getDistributedFilterTypeFromOp(nceOp, filterType, numClusters);
        for (auto filterType : filterDistributedType.getDistributedTypes()) {
            filterSize += filterType.cast<VPU::DistributedTensorType>().getTotalAllocSize();
        }
    } else {
        filterSize = filterType.getTotalAllocSize();
    }

    auto cmxThreshold = Byte(static_cast<int64_t>(
            std::ceil(static_cast<double>(VPU::getTotalCMXSize(op).count()) * LARGE_CONST_THRESHOLD_RATIO)));
    if (filterSize > cmxThreshold) {
        log.nest(1).trace("filter size {0} is larger than cmxThreshold {1}", filterSize, cmxThreshold);
        return true;
    }
    return false;
}

bool largeConstPipelineConditionSatisfied(mlir::Operation* op, Logger log) {
    // Check if the operation has large constant filter
    if (!isLargeConstOp(op, log)) {
        return false;
    }
    auto opTilingBuilder = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(op);
    if (!opTilingBuilder) {
        return false;
    }

    // Find the available tiling size over C
    // The pipelining should be doable with this tiling size
    log.nest(1).trace("Checking large const pipeline tiling.");
    auto tiles = opTilingBuilder.getTilingStrategy(TilingMode::PIPELINING, log.nest());
    if (mlir::failed(tiles)) {
        return false;
    }

    if (tiles.value().begin()->axis != Shape(getShape(op->getResult(0)).size(), 1)) {
        log.nest(1).trace("Found pipelining tiling strategy {0}", tiles.value().begin()->axis);
        return true;
    }

    return false;
}

bool archSupportsSwLayerTiling(VPU::ArchKind arch) {
    return arch == VPU::ArchKind::NPU37XX || arch == VPU::ArchKind::NPU40XX;
}

bool opNeedsTiling(mlir::Operation* op, bool enablePrefetchTiling, Logger log) {
    if (mlir::isa<VPU::SliceOp, VPU::ConcatOp, VPU::NCEClusterTilingOp>(op) ||
        op->getParentOfType<VPU::NCEClusterTilingOp>()) {
        return false;
    }
    auto func = op->getParentOfType<mlir::func::FuncOp>();
    if (func == nullptr) {
        return false;
    }
    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);
    if (!mlir::isa<VPU::NCEOpInterface>(op) && !VPU::archSupportsSwLayerTiling(arch)) {
        return false;
    }

    if (auto iface = mlir::dyn_cast<VPU::TilingInfoOpInterface>(op)) {
        log.trace("Check: '{0}' at '{1}'", op->getName(), op->getLoc());
        const auto resType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();
        Shape resShape = resType.getShape().toValues();
        // TODO(E#113258): getShape needs to return shape based on upper bounds to avoid parsing bounds
        if (auto boundedTensor = resType.dyn_cast_or_null<vpux::BoundedTypeInterface>();
            boundedTensor != nullptr && boundedTensor.getBounds() != nullptr) {
            const auto bounds = parseIntArrayAttr<int64_t>(boundedTensor.getBounds());
            resShape = Shape(bounds.begin(), bounds.end());
        }
        if (!iface.isSupportedTiling({TileInfo(resShape)}, TilingMode::ISOLATED, log.nest())) {
            log.nest().trace("ISOLATED tiling or PIPELINING tiling required");
            return true;
        }
        if (enablePrefetchTiling && mlir::isa<VPU::NCEOpInterface>(op)) {
            if (VPU::prefetchTilingConditionSatisfied(op, log.nest())) {
                log.nest().trace("PREFETCHING tiling required");
                return true;
            }
            if (VPU::largeConstPipelineConditionSatisfied(op, log.nest())) {
                log.nest().trace("PIPELINING tiling for large constant weights required");
                return true;
            }
        }
    }
    return false;
}

std::optional<size_t> getMaxWorkLoadNumberPerClusterForNCEWithSparseOutput(VPU::ArchKind arch,
                                                                           ArrayRef<Shape> perClusterShapes,
                                                                           ArrayRef<int64_t> supportedChannels) {
    const auto getWorkloadNum = [&](int64_t channelSupported) {
        size_t wlMaxNumPerCluster = 0;
        for (const auto& perClusterShape : perClusterShapes) {
            size_t wlNum;
            const auto perClusterChannel = perClusterShape[vpux::Dims4D::Act::C];
            if (perClusterChannel % channelSupported == 0) {
                wlNum = perClusterChannel / channelSupported;
            } else {
                wlNum = divUp(perClusterChannel, channelSupported);
            }

            if (wlMaxNumPerCluster < wlNum) {
                wlMaxNumPerCluster = wlNum;
            }
        }
        return wlMaxNumPerCluster;
    };

    auto sparsityConstraint = VPU::getSparsityConstraint(arch);
    for (const auto channelSupported : supportedChannels) {
        if (!sparsityConstraint.areChannelsFitForSESize(channelSupported)) {
            continue;
        }

        // Only the last cluster can have the not-even channels for workloads
        // For exapmle, we need to split OC = 736 on 6 clusters, the tiled size will be
        // { {64, 64}, {64, 64}, {64, 64}, {64, 64}, {64, 64}, {64 ,32} }.
        //
        size_t numOfClusterNotEven = 0;
        size_t indexOfClusterNotEven = 0;
        for (const auto index : irange(perClusterShapes.size())) {
            if (perClusterShapes[index][vpux::Dims4D::Act::C] % channelSupported != 0) {
                numOfClusterNotEven++;
                indexOfClusterNotEven = index;
            }
        }

        if (numOfClusterNotEven == 0) {
            return getWorkloadNum(channelSupported);
        } else if (numOfClusterNotEven == 1) {
            if (indexOfClusterNotEven != perClusterShapes.size() - 1) {
                continue;
            }

            const auto clusterChannel = perClusterShapes[indexOfClusterNotEven][vpux::Dims4D::Act::C];
            const auto lastChannelNum = clusterChannel % channelSupported;
            if (llvm::count(supportedChannels, lastChannelNum)) {
                return getWorkloadNum(channelSupported);
            }
        }
    }
    return std::nullopt;
}

// All variants of a invariant update a single barrier, therefore the barrier count would be the number of variants.
// And the available slots of a barrier is limited to a architecture specific count. So the variants count must be
// less than a specific number.
bool doesNCEOpChannelSatisfyWorkload(mlir::Operation* nceOp, const TileInfo& outputTile) {
    auto channelAlignedIface = mlir::dyn_cast<VPU::AlignedWorkloadChannelsOpInterface>(nceOp);
    if (channelAlignedIface == nullptr) {
        return true;
    }
    const auto supportedChannels = channelAlignedIface.getSupportedWorkLoadChannels();
    auto log = Logger::global().nest();
    log.trace("supportedChannels - {0}", supportedChannels);
    const auto minSupportedChannel = supportedChannels.back();
    const auto tileChannel = outputTile.shape[Dims4D::Act::C];
    if (tileChannel % minSupportedChannel != 0) {
        log.trace("tileChannel {0} can not be divisible by minSupportedChannel {1}", tileChannel, minSupportedChannel);
        return false;
    }
    const auto maxNumClusters = tileChannel / minSupportedChannel;

    auto getDataType = [](mlir::Type type) {
        if (auto sparseTensor = type.dyn_cast<VPU::SparseTensorType>()) {
            return sparseTensor.getData();
        }
        return type;
    };

    const auto getPerClusterShapes = [&]() {
        auto outputType = getDataType(nceOp->getResult(0).getType()).cast<NDTypeInterface>();
        const auto outputTileType = outputType.extractDenseTile(outputTile.offsets, outputTile.shape);

        auto clusterOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(nceOp);
        if (clusterOp == nullptr || !clusterOp.getMultiClusterStrategy().has_value() ||
            clusterOp.getMultiClusterStrategy().value() != VPU::MultiClusterStrategy::SplitOverKernel) {
            return SmallVector<Shape>{outputTile.shape};
        }
        // SOK case
        auto numClusters =
                VPU::getOptimalNumClusters(clusterOp, tileChannel, VPU::MultiClusterStrategy::SplitOverKernel);
        // check wl channel on each cluster to satisfy supportedChannels
        if (numClusters.getInt() > maxNumClusters) {
            numClusters = mlir::IntegerAttr::get(getInt64Type(nceOp->getContext()), maxNumClusters);
        }
        auto distributedType = getDistributedOutputTypeFromOp(clusterOp, outputTileType, numClusters,
                                                              VPU::MultiClusterStrategy::SplitOverKernel);
        return getDataType(distributedType).cast<VPU::DistributedTensorType>().getPerClusterComputeShapes();
    };

    const auto perClusterShapes = getPerClusterShapes();

    // for some patterns, e.g. NCE(SOK and sparse output)->Concat->NCE, the sparse output would be removed in the
    // following pass
    auto nceOpIf = mlir::dyn_cast<VPU::NCEOpInterface>(nceOp);
    const auto isSparseRemoved = nceOpIf != nullptr && VPU::shouldRemoveOutputSparsity(nceOpIf);

    size_t wlMaxNumPerCluster = 0;
    if (nceOp->getResult(0).getType().isa<VPU::SparseTensorType>() && !isSparseRemoved) {
        // NCE operations with sparse outputs must have all variants with the same number of channels
        // except of the last one which can have fewer channels than the rest
        const auto maxWorkloadNum = getMaxWorkLoadNumberPerClusterForNCEWithSparseOutput(
                getArch(nceOp), perClusterShapes, supportedChannels);
        if (!maxWorkloadNum.has_value()) {
            return false;
        }
        wlMaxNumPerCluster = maxWorkloadNum.value();
    } else {
        for (const auto& perClusterShape : perClusterShapes) {
            const auto perClusterChannel = perClusterShape[vpux::Dims4D::Act::C];
            auto wlChannels = splitWorkloadChannel(perClusterChannel, supportedChannels);
            // There may be some invalid tileChannel passed into. For example, channel is 16 but supportedChannels is
            // [32]. We can't split it over C in that case.
            if (wlChannels.size() == 0) {
                log.warning("splitWorkloadChannel failed: perClusterChannel - {0}, supportedChannels - {1}",
                            perClusterChannel, supportedChannels);
                return false;
            }
            if (wlMaxNumPerCluster < wlChannels.size()) {
                wlMaxNumPerCluster = wlChannels.size();
            }
        }
    }

    // divide max available slots equally for producers and consumers to a barrier
    // for a unified solution for all architectures
    // TODO: E#107973: more bigger / relaxing availableSlot to decrease tiling
    const auto maxAvailableSlots = VPUIP::getBarrierMaxVariantCount(nceOp);
    const auto maxSlotsSum = VPUIP::getBarrierMaxVariantSum(nceOp);
    const auto availableSlot = std::min(maxAvailableSlots, maxSlotsSum) / 2;

    return wlMaxNumPerCluster <= availableSlot;
}

std::optional<DimArr> getSEPConvTilingOrder(mlir::Operation* op) {
    auto nceConv = mlir::dyn_cast<VPU::NCEConvolutionOp>(op);
    if (nceConv == nullptr) {
        return std::nullopt;
    }
    auto sparseInput = nceConv.getInput().getType().dyn_cast<VPU::SparseTensorType>();
    if (sparseInput == nullptr) {
        return std::nullopt;
    }

    auto seAttr = sparseInput.getSeAttr().dyn_cast_or_null<VPU::SERollAttr>();
    if (seAttr != nullptr) {
        return VPU::getRollSEPConvTilingOrder(seAttr);
    }
    return std::nullopt;
}

/*
 * Get supported one-dimension isolated tiling strategies on all dimensions
 * For each dimension, increase the tiling number until each tile fits into CMX
 * or the tiling number reaches the maximum limitation
 */
SmallVector<OutputTiling> getOneDimIsolatedTilingStrategies(mlir::Operation* op,
                                                            const std::pair<Dim, int64_t>& alignInfo, Logger log) {
    SmallVector<OutputTiling> supportedTilingStrategies;
    const auto outputShape = getShape(op->getResult(0));
    const auto dimToAlign = alignInfo.first;
    const auto dimAlignment = alignInfo.second;
    auto tilingBuilder = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(op);
    if (tilingBuilder == nullptr) {
        // return empty result if the op is not a tilingBuilder
        return supportedTilingStrategies;
    }
    const auto& maxNumTiles = tilingBuilder.getMaxNumTiles();
    for (const auto tileIndex : irange(outputShape.size())) {
        Shape nTilesOnDim(outputShape.size(), 1);
        auto dimToTile = Dim(tileIndex);
        // iterate to search the supported tiling strategy
        auto findSupportedTileSize = isSupportedTileSize(op, nTilesOnDim, TilingMode::ISOLATED, log);
        while (mlir::failed(findSupportedTileSize)) {
            if (!isDimLeftToTile(nTilesOnDim, maxNumTiles, dimToTile)) {
                break;
            }
            auto nextTileSearchResult =
                    getNextTiling(dimToTile, dimToAlign, dimAlignment, nTilesOnDim, maxNumTiles, outputShape);
            if (mlir::failed(nextTileSearchResult)) {
                break;
            }
            nTilesOnDim = nextTileSearchResult.value();
            findSupportedTileSize = isSupportedTileSize(op, nTilesOnDim, TilingMode::ISOLATED, log);
        }
        if (!mlir::failed(findSupportedTileSize) && nTilesOnDim[dimToTile] > 1) {
            // find an available isolated tiling strategy
            supportedTilingStrategies.push_back(findSupportedTileSize.value());
            log.trace("Got one-dimension isolated tiling strategy {0} for op {1}", nTilesOnDim, op->getLoc());
        }
    }
    return supportedTilingStrategies;
}

/*
 * Get supported one-dimension tiling strategies on all dimensions
 * Prefetching and pipelining tiling strategies are generated from isolated tiling strategy
 * i.e., increase the tiling dimension of isolated tiling until prefetching/pipelining requirement is satisfied
 */
SmallVector<OutputTiling> getOneDimTilingStrategies(mlir::Operation* op, TilingMode tilingMode, Logger log) {
    const auto alignRequirement = getAlignDimAndSize(op);
    auto supportedTilingStrategies = getOneDimIsolatedTilingStrategies(op, alignRequirement, log.nest());
    if (supportedTilingStrategies.empty() || tilingMode == TilingMode::ISOLATED) {
        return supportedTilingStrategies;
    }
    const auto dimToAlign = alignRequirement.first;
    const auto dimAlignment = alignRequirement.second;
    auto tilingBuilder = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(op);
    VPUX_THROW_WHEN(tilingBuilder == nullptr, "Operation {0} doesn't support tiling", op->getName());
    const auto& maxNumTiles = tilingBuilder.getMaxNumTiles();
    const auto outputShape = getShape(op->getResult(0));
    // Increase the tiled dimension to get PREFETCHING/PIPELINING tiling strategies
    const auto oneDimIsolatedTilingSize = supportedTilingStrategies.size();
    for (auto isolatedTilingIndex : irange(oneDimIsolatedTilingSize)) {
        auto isolatedTiling = supportedTilingStrategies[isolatedTilingIndex];
        auto prefetchableTilesOnDim = isolatedTiling[0].axis;
        const auto nonOneDims = getNonOneDim(prefetchableTilesOnDim);
        VPUX_THROW_UNLESS(nonOneDims.size() == 1,
                          "Isolated tiling strategy is not one-dimension but {0}, not supported.", nonOneDims.size());
        auto targetDim = *nonOneDims.begin();
        auto findSupportedTileSize = isSupportedTileSize(op, prefetchableTilesOnDim, tilingMode, log);
        while (mlir::failed(findSupportedTileSize)) {
            if (!isDimLeftToTile(prefetchableTilesOnDim, maxNumTiles, targetDim)) {
                break;
            }
            auto nextTileSearchResult = getNextTiling(targetDim, dimToAlign, dimAlignment, prefetchableTilesOnDim,
                                                      maxNumTiles, outputShape);
            if (mlir::failed(nextTileSearchResult)) {
                break;
            }
            prefetchableTilesOnDim = nextTileSearchResult.value();
            findSupportedTileSize = isSupportedTileSize(op, prefetchableTilesOnDim, tilingMode, log);
        }
        if (!mlir::failed(findSupportedTileSize)) {
            // find an available isolated tiling strategy
            supportedTilingStrategies.push_back(findSupportedTileSize.value());
            log.trace("Got one-dimension prefetching tiling strategy {0} for op {1}", prefetchableTilesOnDim,
                      op->getLoc());
        }
    }
    return supportedTilingStrategies;
}

mlir::FailureOr<OutputTiling> getHWLayerTilingStrategyBasedOnCost(mlir::Operation* op, TilingMode tilingMode,
                                                                  DimArrRef tileDimOrder,
                                                                  const std::shared_ptr<LayerCostModel>& costModel,
                                                                  Logger log) {
    auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(op);
    if (nceOp == nullptr || costModel == nullptr) {
        return getHWLayerTilingStrategyWithTileDimOrder(op, tilingMode, tileDimOrder, log);
    }
    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(op);
    VPUX_THROW_WHEN(tilingInfo == nullptr, "Operation '{0}' doesn't implement TilingInfoOpInterface", op->getName());
    auto tilingBuilder = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(op);
    VPUX_THROW_WHEN(tilingBuilder == nullptr, "Operation '{0}' doesn't implement TilingBuilderOpInterface",
                    op->getName());

    const auto outputShape = getShape(op->getResult(0));

    VPUX_THROW_UNLESS(outputShape.size() == 4, "Unsupported operation '{0}' at '{1}', it has non 4D result",
                      op->getName(), op->getLoc());
    auto oneDimStratgies = getOneDimTilingStrategies(op, tilingMode, log.nest());
    if (oneDimStratgies.empty()) {
        return getHWLayerTilingStrategyWithTileDimOrder(op, tilingMode, tileDimOrder, log);
    }

    // If the op does not have MC strategy, use Clustering by default
    auto mcStrategy = VPU::MultiClusterStrategy::Clustering;
    auto clusteredNCEOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(op);
    if (clusteredNCEOp != nullptr) {
        auto strategy = clusteredNCEOp.getMultiClusterStrategy();
        if (strategy.has_value()) {
            mcStrategy = strategy.value();
        }
    }

    // compare the costs and get the best one-dimension tiling strategy
    auto bestTilingStrategy = SmallVector({TileInfo(1)});
    auto bestCost = INVALID_COST_BASE;

    for (const auto& curTiling : oneDimStratgies) {
        auto curCost = costModel->getDPUandDMATimeCostWithCustomTiling(nceOp, mcStrategy, curTiling);
        if (curCost >= INVALID_COST_BASE) {
            log.warning("Invalid cost for tiling strategy {0}", bestTilingStrategy);
            return getHWLayerTilingStrategyWithTileDimOrder(op, tilingMode, tileDimOrder, log);
        } else {
            log.nest().trace("tiling strategy {0} cost is {1}", curTiling, curCost);
            if (curCost < bestCost) {
                bestTilingStrategy = curTiling;
                bestCost = curCost;
            }
        }
    }
    log.trace("Got best one-dimension tiling strategy {0} for op {1} at {2}", bestTilingStrategy, op->getName(),
              op->getLoc());
    return bestTilingStrategy;
}

static constexpr auto MODE_ON = "true";
static constexpr auto MODE_OFF = "false";
static constexpr auto MODE_AUTO = "auto";

VPU::EnableShaveDDRAccessOptimization getShaveDDRAccessOptimizationMode(StringRef enableShaveDDRAccessOptimization) {
    std::string strMode = enableShaveDDRAccessOptimization.str();
    std::transform(strMode.begin(), strMode.end(), strMode.begin(), ::tolower);

    if (strMode == MODE_ON) {
        return VPU::EnableShaveDDRAccessOptimization::TRUE;
    } else if (strMode == MODE_OFF) {
        return VPU::EnableShaveDDRAccessOptimization::FALSE;
    } else if (strMode == MODE_AUTO) {
        VPUX_THROW("auto EnableShaveDDRAccessOptimization is not supported for now");
    }

    VPUX_THROW("Unknown value for the shave DDR access optimization mode: {0}", strMode);
}

}  // namespace VPU
}  // namespace vpux
