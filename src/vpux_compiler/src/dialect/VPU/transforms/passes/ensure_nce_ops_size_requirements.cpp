//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/utils/VPU/ppe_utils.hpp"
#include "vpux/compiler/utils/sparsity.hpp"

using namespace vpux;

namespace {

SmallVector<Dim> getDimsOverKHWLimit(ShapeRef shape) {
    SmallVector<Dim> wrongDims = {};
    for (size_t i = 0; i < shape.size(); i++) {
        const auto dim = Dim(i);
        if (shape[dim] > VPU::NCEInvariant::VPU_DIMENSION_LIMIT) {
            wrongDims.push_back(dim);
        }
    }
    return wrongDims;
}

class EnsureNCEOpSizeRequirements final : public mlir::OpInterfaceRewritePattern<VPU::TilingBuilderOpInterface> {
public:
    EnsureNCEOpSizeRequirements(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpInterfaceRewritePattern<VPU::TilingBuilderOpInterface>(ctx), _log(log) {
        this->setDebugName("EnsureNCEOpSizeRequirements");
    }
    mlir::LogicalResult matchAndRewrite(VPU::TilingBuilderOpInterface origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult EnsureNCEOpSizeRequirements::matchAndRewrite(VPU::TilingBuilderOpInterface origOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp->getLoc());

    auto op = origOp.getOperation();
    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(op);
    VPUX_THROW_WHEN(tilingInfo == nullptr, "Operation '{0}' doesn't implement TilingInfoOpInterface", op->getName());
    rewriter.setInsertionPoint(op);

    const auto outputType = op->getResult(0).getType().cast<NDTypeInterface>();
    const auto outputShape = outputType.getShape();
    Shape nTilesOnDim(outputShape.size(), 1);
    const auto log = _log.nest();
    const auto tilingMode = TilingMode::ISOLATED;
    const auto tileDimOrder = getTileDimOrder(op, tilingMode, log);
    _log.nest(4).trace("Tile Dim order is {0}", tileDimOrder);

    const auto isSupportedTileSize = [&](ShapeRef nTilesOnDim, int32_t dimToTile) -> bool {
        const auto tiles = fillDividedTiles(op, nTilesOnDim, outputShape);
        if (mlir::failed(tiles)) {
            return false;
        }
        for (auto tile : tiles.value()) {
            if (tile.shape.raw()[dimToTile] > VPU::NCEInvariant::VPU_DIMENSION_LIMIT) {
                return false;
            }
            auto inputTiling = origOp.backInferTileInfo(tile, log);
            auto& inTiles = inputTiling.tiles;
            if ((dimToTile != Dims4D::Act::C.ind()) &&
                (inTiles.begin()->shape.raw()[dimToTile] > VPU::NCEInvariant::VPU_DIMENSION_LIMIT)) {
                return false;
            }
        }
        return true;
    };

    for (auto tileDimIter = tileDimOrder.begin(); tileDimIter < tileDimOrder.end(); ++tileDimIter) {
        auto dimToTile = *tileDimIter;
        while (!isSupportedTileSize(nTilesOnDim, dimToTile.ind())) {
            ++nTilesOnDim[dimToTile];
        }
    }

    // In case of single tile scheduled there is no need for tiling
    if (llvm::none_of(nTilesOnDim, [](int64_t tiles) {
            return tiles > 1;
        })) {
        return mlir::failure();
    }

    const auto tilesNew = fillDividedTiles(op, nTilesOnDim, outputShape);
    if (mlir::failed(tilesNew)) {
        return mlir::failure();
    }

    return VPU::applyTileStrategy(origOp, tilesNew.value(), rewriter, log.nest());
}

//
//  EnsureConvICRequirements
//

class EnsureConvICRequirements final : public mlir::OpRewritePattern<VPU::NCEConvolutionOp> {
public:
    EnsureConvICRequirements(mlir::MLIRContext* ctx, VPU::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<VPU::NCEConvolutionOp>(ctx), _arch(arch), _log(log) {
        this->setDebugName("EnsureConvICRequirements");
    }
    mlir::LogicalResult matchAndRewrite(VPU::NCEConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    VPU::ArchKind _arch;
    Logger _log;
};

mlir::LogicalResult EnsureConvICRequirements::matchAndRewrite(VPU::NCEConvolutionOp origOp,
                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp->getLoc());

    // Split over IC supported only for NCEConvolutionOp
    // TODO: E#70421

    // Get the NCEConvolutionOp's input and kernel sizes
    const auto inputShape = getShape(origOp.getInput());
    auto inputW = inputShape[Dims4D::Act::W];
    auto inputH = inputShape[Dims4D::Act::H];
    auto inputC = inputShape[Dims4D::Act::C];
    auto inputN = inputShape[Dims4D::Act::N];

    if (inputC <= VPU::NCEInvariant::VPU_DIMENSION_LIMIT) {
        return mlir::failure();
    }

    const auto kernelShape = getShape(origOp.getFilter());
    auto kernelW = kernelShape[Dims4D::Filter::KX];
    auto kernelH = kernelShape[Dims4D::Filter::KY];
    auto kernelN = kernelShape[Dims4D::Filter::OC];

    SmallVector<VPU::NCEConvolutionOp> convOps;
    auto maxTiles = vpux::divUp(inputC, VPU::NCEInvariant::VPU_DIMENSION_LIMIT);

    if (maxTiles == 1) {
        return mlir::failure();
    }

    Shape nTilesOnDim(inputShape.size(), 1);
    nTilesOnDim[Dims4D::Act::C] = maxTiles;
    const auto tiles = fillDividedTiles(origOp, nTilesOnDim, inputShape);
    if (mlir::failed(tiles)) {
        return mlir::failure();
    }

    auto weightsTable = origOp.getWeightsTable();
    auto weightsTableConst = weightsTable.getDefiningOp<Const::DeclareOp>();
    if (weightsTableConst == nullptr) {
        _log.trace("Could not extract constant from weights table.");
        return mlir::failure();
    }
    auto weightsTableContent = weightsTableConst.getContent();
    auto weightsTableValues = weightsTableContent.getValues<int32_t>();
    auto weightsTableVecSize = weightsTableValues.size();
    std::vector<int32_t> weightsTableVec(weightsTableVecSize);
    std::copy(weightsTableValues.begin(), weightsTableValues.end(), weightsTableVec.begin());

    auto filterType = origOp.getFilter().getType().cast<vpux::NDTypeInterface>();
    auto filterElemType = filterType.getElementType();

    // TODO: E#70371 - Remaining opens for InputChannels 8K size
    for (auto tile = 0; tile < maxTiles; tile++) {
        auto offsetIC = tiles.value()[tile].offsets[Dims4D::Act::C];
        auto sizeIC = tiles.value()[tile].shape[Dims4D::Act::C];
        _log.nest().trace("Slicing channels {0} - {1}", offsetIC, sizeIC);

        // Slice inputs
        const Shape inSliceOffsets{0, offsetIC, 0, 0};
        const Shape inSliceShape{inputN, sizeIC, inputH, inputW};
        auto convInput = rewriter.create<VPU::SliceOp>(origOp->getLoc(), origOp.getInput(),
                                                       getIntArrayAttr(rewriter, inSliceOffsets.raw()),
                                                       getIntArrayAttr(rewriter, inSliceShape.raw()));

        // Slice kernels
        const Shape kernelSliceOffsets{0, offsetIC, 0, 0};
        const Shape kernelSliceShape{kernelN, sizeIC, kernelH, kernelW};
        const auto rawKernelSliceShape = getIntArrayAttr(rewriter, kernelSliceShape);
        auto convFilter = rewriter.create<VPU::SliceOp>(origOp.getLoc(), origOp.getFilter(),
                                                        getIntArrayAttr(rewriter, kernelSliceOffsets.raw()),
                                                        getIntArrayAttr(rewriter, kernelSliceShape.raw()));

        // Adjust the weights table pointers to correspond to the new offsets of the slices
        const auto noOfBits = vpux::getElemTypeSize(filterElemType);
        const auto weightSetSize = alignMemSize(kernelH * kernelW * sizeIC * noOfBits,
                                                Byte(VPU::NCEInvariant::VPU_WEIGHT_SET_BYTE_ALIGNMENT))
                                           .to<Byte>()
                                           .count();
        const auto sparsitySetSize =
                alignValUp(divUp(kernelH * kernelW * sizeIC, CHAR_BIT * getValuesPerSparsityBit(filterElemType)),
                           static_cast<int64_t>(VPU::NCEInvariant::VPU_WEIGHT_SET_BYTE_ALIGNMENT));

        // Apply bias for the first convolution only
        if (tile != 0) {
            // Set the bias values to 0
            for (size_t i = 3; i < weightsTableVecSize; i += VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC) {
                weightsTableVec[i] = checked_cast<int32_t>(0);
            }
        }

        // Adjust the weight pointers
        for (size_t i = 0; i < weightsTableVecSize; i += VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC) {
            weightsTableVec[i] =
                    checked_cast<int32_t>((i / VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC) * weightSetSize);
        }

        // Adjust the sparsity pointers
        for (size_t i = 1; i < weightsTableVecSize; i += VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC) {
            weightsTableVec[i] =
                    checked_cast<int32_t>((i / VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC) * sparsitySetSize);
        }

        auto weightsTable = VPU::createWeightsTableTensor(rewriter, origOp->getLoc(), weightsTableVec);
        auto convOp = rewriter.create<VPU::NCEConvolutionOp>(
                origOp.getLoc(), origOp.getType(), convInput.getResult(), convFilter.getResult(), weightsTable,
                origOp.getActivationWindow(), origOp.getInstructionListTable(), origOp.getStrides(), origOp.getPad(),
                nullptr, rawKernelSliceShape, origOp.getActivationWindowChannelLengthAttr(),
                origOp.getMultiClusterStrategyAttr());

        convOps.push_back(convOp);
    }

    // Add the outputs of the convolutions with NCEEltwise Add operations. This is needed because NCEConvolutionOp
    // accumulates all its input channels into 1 output channel. Splitting the Convolutions into smaller Convolutions,
    // the outputs have to be added together.
    auto output = origOp->getResult(0);
    auto targetEltwiseOutputType = output.getType().cast<vpux::NDTypeInterface>();
    const auto opType = VPU::EltwiseType::ADD;
    SmallVector<VPU::NCEEltwiseOp> addOps;
    VPU::NCEEltwiseOp addResult;

    for (size_t index = 0; index < convOps.size() - 1; index++) {
        auto addOperand = index == 0 ? convOps[index].getOutput() : addResult.getOutput();

        // Construct ppeTaskAttr for NCEEltwise (the last NCEEltwiseAdd will get the PPE from the original Conv)
        auto ppeTaskAttr = VPU::getNCEEltwisePPETaskAttr(addOperand.getType(), convOps[index + 1].getOutput().getType(),
                                                         addOperand.getType(), nullptr, addOperand.getLoc(), opType,
                                                         addOperand.getContext(), _arch);

        // NCEEltwise inType and outType are always same with ConvOp outType
        addResult = rewriter.create<VPU::NCEEltwiseOp>(
                origOp->getLoc(), targetEltwiseOutputType, addOperand, convOps[index + 1].getOutput(), opType,
                ((index == (convOps.size() - 2) && origOp.getPpe().has_value()) ? origOp.getPpeAttr() : ppeTaskAttr),
                nullptr, nullptr);

        // change NCEConv's output layout to supported NCEEltwise input layout
        // Eg: if NCEConv (inL=NHWC,outL=NCHW) splits into 3 small NCEConv:
        //   NCEConv (inL=NHWC,out=NHWC)    NCEConv (inL=NHWC,out=NHWC)     NCEConv (inL=NHWC,out=NHWC)
        //              \                         /                                     /
        //               NCEElt (inL=NHWC,out=NHWC)                                    /
        //                             \                                              /
        //                                         NCEElt (inL=NHWC,out=NCHW)
        if (auto iface = mlir::dyn_cast<IE::LayoutInfoOpInterface>(addResult.getOperation())) {
            auto orderInfo = iface.getLayoutInfo();
            iface.inferLayoutInfo(orderInfo, /*seOpsEnabled=*/false, /*seExperimentalOpsEnabled=*/false);
            const auto supportOrder1 = orderInfo.getInput(0);
            const auto supportOrder2 = orderInfo.getInput(1);
            const auto inputOrder1 = DimsOrder::fromValue(addResult.getInput1());
            const auto inputOrder2 = DimsOrder::fromValue(addResult.getInput2());

            if (supportOrder1 != inputOrder1 && supportOrder2 != inputOrder2) {
                const auto newInput1Type =
                        addResult.getInput1().getType().dyn_cast<vpux::NDTypeInterface>().changeDimsOrder(
                                supportOrder1);
                const auto newInput2Type =
                        addResult.getInput2().getType().dyn_cast<vpux::NDTypeInterface>().changeDimsOrder(
                                supportOrder2);

                auto input1Op = addResult.getInput1().getDefiningOp();
                auto input2Op = addResult.getInput2().getDefiningOp();
                input1Op->getResult(0).setType(newInput1Type);
                input2Op->getResult(0).setType(newInput2Type);

                addResult.getOperation()->setOperands({input1Op->getResult(0), input2Op->getResult(0)});
            }
        }

        addOps.push_back(addResult);
    }

    rewriter.replaceOp(origOp, addResult.getOutput());

    return mlir::success();
}

//
// EnsureNCEOpsSizeRequirementsPass
//

class EnsureNCEOpsSizeRequirementsPass final :
        public VPU::EnsureNCEOpsSizeRequirementsBase<EnsureNCEOpsSizeRequirementsPass> {
public:
    explicit EnsureNCEOpsSizeRequirementsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void EnsureNCEOpsSizeRequirementsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);

    mlir::ConversionTarget target(ctx);
    mlir::RewritePatternSet patterns(&ctx);
    target.addLegalOp<VPU::SliceOp, VPU::ConcatOp>();

    target.markUnknownOpDynamicallyLegal([&](mlir::Operation* op) {
        if (!mlir::isa<VPU::NCEConvolutionOp>(op)) {
            return true;
        }

        const auto inputShape = getShape(op->getOperand(0));
        return inputShape[Dims4D::Act::C] <= VPU::NCEInvariant::VPU_DIMENSION_LIMIT;
    });

    patterns.add<EnsureConvICRequirements>(&ctx, arch, _log);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }

    target.markUnknownOpDynamicallyLegal([&](mlir::Operation* op) {
        if (!mlir::isa<VPU::NCEOpInterface>(op)) {
            return true;
        }

        if (mlir::isa<VPU::TilingInfoOpInterface>(op)) {
            const auto inputShape = getShape(op->getOperand(0));
            const auto outputShape = getShape(op->getResult(0));

            auto inSizeWrongDims = getDimsOverKHWLimit(inputShape);
            if (!inSizeWrongDims.empty()) {
                _log.nest(2).info("Input size has dims greater than HW requirements: {0}", inSizeWrongDims);
            }
            const auto outSizeWrongDims = getDimsOverKHWLimit(outputShape);
            if (!outSizeWrongDims.empty()) {
                _log.nest(2).info("Output size has dims greater than HW requirements: {0}", outSizeWrongDims);
            }
            return inSizeWrongDims.empty() && outSizeWrongDims.empty();
        }

        return true;
    });

    patterns.clear();
    patterns.add<EnsureNCEOpSizeRequirements>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createEnsureNCEOpsSizeRequirementsPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createEnsureNCEOpsSizeRequirementsPass(Logger log) {
    return std::make_unique<EnsureNCEOpsSizeRequirementsPass>(log);
}
