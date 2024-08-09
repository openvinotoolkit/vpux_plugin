//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <llvm/ADT/SmallVector.h>
#include <mlir/Support/LogicalResult.h>
#include <climits>
#include <cstdint>
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/type_interfaces.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/nce_invariant.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/core/type/float16.hpp"

using namespace vpux;

namespace {

// To explicitly control the patterns exec order to assure dependency
// benefitLevels[0] is highest benefit level and represent the relative pattern is the first one to run
const uint32_t levelCount = 2;
SmallVector<mlir::PatternBenefit> benefitLevels = getBenefitLevels(levelCount);

//
// MatMulInputsTo2dPass
//

class MatMulInputsTo2dPass final : public IE::MatMulInputsTo2dBase<MatMulInputsTo2dPass> {
public:
    explicit MatMulInputsTo2dPass(const bool enableGroupedMatMul, Logger log)
            : _enableGroupedMatMul(enableGroupedMatMul) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

public:
    class ReshapeNDInputConverter;
    class MatMulOpConverter;

private:
    void safeRunOnFunc() final;
    bool _enableGroupedMatMul;
};

mlir::LogicalResult MatMulInputsTo2dPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    if (!enableGroupedMatMul.hasValue()) {
        return mlir::success();
    }

    _enableGroupedMatMul = enableGroupedMatMul;
    return mlir::success();
}

//
// MatMulOpConverter
//

class MatMulInputsTo2dPass::MatMulOpConverter final : public mlir::OpRewritePattern<IE::MatMulOp> {
public:
    MatMulOpConverter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log, bool enableGroupedMatMul)
            : mlir::OpRewritePattern<IE::MatMulOp>(ctx, benefit), _log(log), _enableGroupedMatMul(enableGroupedMatMul) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MatMulOp matmulOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    bool _enableGroupedMatMul;
};

static SmallVector<mlir::Value> sliceTensor(const mlir::Value tensorToSplit, const mlir::Location location,
                                            mlir::PatternRewriter& rewriter, const std::string& tensorName) {
    const auto tensorShape = getShape(tensorToSplit);
    int64_t batch = 1;
    int64_t width = 1;
    int64_t height = 1;
    auto channelDim = Dim(0);
    if (tensorShape.size() == 3) {
        batch = tensorShape[Dim(0)];
        height = tensorShape[Dim(1)];
        width = tensorShape[Dim(2)];
        channelDim = Dim(0);
    } else if (tensorShape.size() == 4) {
        batch = tensorShape[Dim(1)];
        height = tensorShape[Dim(2)];
        width = tensorShape[Dim(3)];
        channelDim = Dim(1);
    } else if (tensorShape.size() == 2) {
        return {tensorToSplit};
    }
    SmallVector<mlir::Value> weightSlices;
    Shape rhsShape2D{height, width};
    const auto rhsShape2DAttr = getIntArrayAttr(rewriter.getContext(), rhsShape2D);
    if (batch > 1) {
        for (int64_t sliceIdx = 0; sliceIdx < batch; sliceIdx++) {
            Shape sliceOffsets = Shape(tensorShape.size(), 0);
            sliceOffsets[channelDim] = checked_cast<int64_t>(sliceIdx);
            auto staticOffsetsAttr = getIntArrayAttr(rewriter.getContext(), sliceOffsets);

            Shape sliceSizes = tensorShape.raw();
            sliceSizes[channelDim] = 1;
            auto staticSizesAttr = getIntArrayAttr(rewriter.getContext(), sliceSizes);
            auto newSubViewOp = rewriter.create<IE::SliceOp>(appendLoc(location, "{0}_slice_{1}", tensorName, sliceIdx),
                                                             tensorToSplit, staticOffsetsAttr, staticSizesAttr);

            auto rhs2d = rewriter.create<IE::ReshapeOp>(appendLoc(location, "{0}_reshape_{1}", tensorName, sliceIdx),
                                                        newSubViewOp, nullptr, false, rhsShape2DAttr);
            weightSlices.push_back(rhs2d);
        }
    } else {
        auto rhs2d = rewriter.create<IE::ReshapeOp>(appendLoc(location, "{0}_reshape", tensorName), tensorToSplit,
                                                    nullptr, false, rhsShape2DAttr);
        weightSlices.push_back(rhs2d);
    }

    return weightSlices;
}

mlir::quant::UniformQuantizedType getQuantizedTypeFromPossibleFakeQuantize(mlir::Value value) {
    auto op = value.getDefiningOp();
    if (op == nullptr) {
        return nullptr;
    }
    while (mlir::isa_and_nonnull<IE::ViewLikeOpInterface, IE::TransposeOp>(op)) {
        op = op->getOperand(0).getDefiningOp();
    }

    return IE::getQuantizedTypeFromFakeQuantize(mlir::dyn_cast_or_null<IE::FakeQuantizeOp>(op));
}

IE::FakeQuantizeOp getMatmulOutputFakeQuantize(IE::MatMulOp matmulOp) {
    auto resultUsers = matmulOp.getOutput().getUsers();
    if (!matmulOp.getOutput().hasOneUse()) {
        return nullptr;
    }
    return mlir::dyn_cast<IE::FakeQuantizeOp>(*resultUsers.begin());
}

int64_t getExpandedCMXUsage(IE::MatMulOp matmulOp, ShapeRef input1Shape, ShapeRef input2Shape) {
    VPUX_THROW_UNLESS(input1Shape.size() == 3 && input2Shape.size() == 3,
                      "Matmul Dimensions for batched Matmul must be 3d");
    const auto transposeA = matmulOp.getTransposeA();
    const auto transposeB = matmulOp.getTransposeB();
    const auto dimOfIC = transposeA ? Dims3D::Act::H : Dims3D::Act::IC;
    const auto dimOfOC = transposeB ? Dims3D::Filter::IC : Dims3D::Filter::OC;
    const int64_t sizeofIC{input1Shape[dimOfIC]};
    const int64_t sizeofOC{input2Shape[dimOfOC]};
    // Following expansion is done at expandActivationChannelsPass, example assuming no transpose A/B
    // We calculate CMX size following expansion and assumed tiling
    // WE mean W Expanded
    // Input1 Input2 Output  -> After Expansion Input1  Input2   Output
    // BxHxIC  BxICxOC BxHxOC ------------------- BxHxICE  BxICExOCE BxHxOCE

    // Here we preemptively estimate CMX usage in VPU after tiling, if data does not fit
    // into CMX we dont run with batched matmul
    // OC dimension must be expanded to be multiple of 16

    auto alignedChannelOpInterface = mlir::cast<IE::AlignedChannelsOpInterface>(matmulOp.getOperation());
    const auto inputChannelAlignment = alignedChannelOpInterface.getInputChannelAlignment();
    const auto outputChannelAlignment = alignedChannelOpInterface.getOutputChannelAlignment();
    constexpr auto float16Size = sizeof(type::float16);
    constexpr auto int32Size = sizeof(int32_t);
    const auto sizeOfICE = alignValUp(sizeofIC, inputChannelAlignment);
    const auto sizeOfOCE = alignValUp(sizeofOC, outputChannelAlignment);
    const auto input1QuantizedType = getQuantizedTypeFromPossibleFakeQuantize(matmulOp.getInput1());
    const auto elementSizeInput1 = input1QuantizedType ? input1QuantizedType.getStorageTypeIntegralWidth() / CHAR_BIT
                                                       : checked_cast<double>(float16Size);
    const auto input1Size = details::calcTotalShapeSize(input1Shape) * elementSizeInput1 * sizeOfICE / sizeofIC;
    const auto input2QuantizedType = getQuantizedTypeFromPossibleFakeQuantize(matmulOp.getInput2());
    const auto elementSizeInput2 = input2QuantizedType ? input2QuantizedType.getStorageTypeIntegralWidth() / CHAR_BIT
                                                       : checked_cast<double>(float16Size);
    // To cover transpose case without conditional we multiply all then remove expanded dimension
    const auto input2Size = input2Shape[Dims3D::Filter::B] * sizeOfICE * sizeOfOCE * elementSizeInput2;

    auto possibleOutputFQOp = getMatmulOutputFakeQuantize(matmulOp);
    const auto outputQuantizedType = IE::getQuantizedTypeFromFakeQuantize(possibleOutputFQOp);
    const auto elementSizeOutput = outputQuantizedType != nullptr
                                           ? outputQuantizedType.getStorageTypeIntegralWidth() / CHAR_BIT
                                           : checked_cast<double>(float16Size);
    const auto sizeH = transposeA ? input1Shape[Dim(2)] : input1Shape[Dim(1)];
    const auto outputSize = input1Shape[Dims3D::Act::B] * sizeH * sizeOfOCE * elementSizeOutput;

    const auto weightTableSize =
            input1Shape[Dims3D::Act::B] * sizeOfOCE * VPUIP::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC * int32Size;

    const auto module = getModuleOp(matmulOp);
    auto tileOp = IE::getTileExecutor(module);
    const auto numOfTiles = tileOp.getCount();
    // Calculation is not totally accurate, we are not considering if input2 is being duplicated instead of being tiled
    return (input1Size / numOfTiles) + (input2Size / numOfTiles) + (outputSize / numOfTiles) + weightTableSize;
}

mlir::LogicalResult MatMulInputsTo2dPass::MatMulOpConverter::matchAndRewrite(IE::MatMulOp matmulOp,
                                                                             mlir::PatternRewriter& rewriter) const {
    // E-122051:
    // MatMulInputsTo2dPass should be moved to a new pass `ConvertMatMulToFullyConnected`.
    // This check should be moved in a `addDynamicallyLegalOp<IE::MatMulOp>`.
    // Transpose shold be done after `ReshapeNDInputConverter` (not in canonicalizer), experiments show that it
    // is faster when batch dimensions are merged.
    if (VPU::MatMulOp::isSupported(matmulOp)) {
        if (matmulOp.getTransposeB()) {
            auto input2 = matmulOp.getInput2();
            auto input2Rank = getShape(input2).size();
            VPUX_THROW_UNLESS(input2Rank > 2,
                              "VPU::MatMulOp only supports input 2 rank bigger than 2. "
                              "If that changes, this code needs update. Input 2 rank = '{0}'",
                              input2Rank);
            SmallVector<uint32_t> perm(input2Rank, 0);
            std::iota(perm.begin(), perm.end(), 0);
            std::iter_swap(perm.end() - 1, perm.end() - 2);
            const auto orderAttr =
                    mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(perm, matmulOp->getContext()));
            input2 = rewriter.create<IE::TransposeOp>(takeOpLoc(matmulOp, "input_b_transpose"), input2, nullptr,
                                                      orderAttr)
                             .getOutput();
            rewriter.replaceOpWithNewOp<IE::MatMulOp>(matmulOp, matmulOp.getInput1(), input2, false, false);
            return mlir::success();
        }
        return mlir::failure();
    }

    auto input1Shape = getShape(matmulOp.getInput1());
    auto input2Shape = getShape(matmulOp.getInput2());

    // 1. Cover 3D input or weights.
    // 2. Cover 4D input and weights without batch.
    if (!(input1Shape.size() == 3 && input2Shape.size() == 3) &&
        !(input1Shape.size() == 4 &&
          ((input2Shape.size() == 4 || input2Shape.size() == 3) && input1Shape[Dim(0)] == 1))) {
        return mlir::failure();
    }

    // Ideally this should be skipped using calculation from ReshapeNDInputConverter
    if (_enableGroupedMatMul) {
        const auto availableCMXBytes = vpux::VPU::getTotalCMXSize(matmulOp);
        Shape input1Shape3d =
                input1Shape.size() > 3 ? Shape(input1Shape.begin() + 1, input1Shape.end()) : input1Shape.toValues();
        auto input2Shape3d =
                input2Shape.size() > 3 ? Shape(input2Shape.begin() + 1, input2Shape.end()) : input2Shape.toValues();
        const auto expandedCMXUsage = getExpandedCMXUsage(matmulOp, input1Shape3d, input2Shape3d);
        if (input1Shape3d[Dim(0)] == 1 || expandedCMXUsage < availableCMXBytes.count()) {
            return mlir::failure();
        }
    }

    SmallVector<mlir::Value> activationSlices =
            sliceTensor(matmulOp.getInput1(), matmulOp->getLoc(), rewriter, "activation");
    SmallVector<mlir::Value> weightSlices = sliceTensor(matmulOp.getInput2(), matmulOp->getLoc(), rewriter, "weights");

    SmallVector<mlir::Value> matmulSlices;
    VPUX_THROW_UNLESS(activationSlices.size() == weightSlices.size() || weightSlices.size() == 1,
                      "Mismatch activationSlices number '{0}' with weightSlices number '{1}'", activationSlices.size(),
                      weightSlices.size());
    for (size_t sliceIdx = 0; sliceIdx < activationSlices.size(); sliceIdx++) {
        auto lhs2d = activationSlices[sliceIdx];
        auto rhs2d = weightSlices[weightSlices.size() == 1 ? 0 : sliceIdx];
        auto op = rewriter.create<IE::MatMulOp>(takeOpLoc(matmulOp, llvm::StringLiteral("slice_{0}"), sliceIdx), lhs2d,
                                                rhs2d, matmulOp.getTransposeA(), matmulOp.getTransposeB());
        matmulSlices.push_back(op.getOutput());
    }

    VPUX_THROW_WHEN(matmulSlices.empty(), "Cannot slice MatMul operation with input shape {0}, weights' shape {1}",
                    input1Shape, input2Shape);

    auto newOp = matmulSlices.size() != 1
                         ? rewriter.create<IE::ConcatOp>(takeOpLoc(matmulOp, "slice_gather"), matmulSlices, 0)
                         : matmulSlices.front();

    const auto outShape4D = getShape(matmulOp.getOutput());
    const auto outShape4DAttr = getIntArrayAttr(rewriter.getContext(), outShape4D);
    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(matmulOp, newOp, nullptr, false, outShape4DAttr);

    return mlir::success();
}

//
// ReshapeNDInputConverter
//

class MatMulInputsTo2dPass::ReshapeNDInputConverter final : public mlir::OpRewritePattern<IE::MatMulOp> {
public:
    ReshapeNDInputConverter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log, bool enableGroupedMatMul)
            : mlir::OpRewritePattern<IE::MatMulOp>(ctx, benefit), _log(log), _enableGroupedMatMul(enableGroupedMatMul) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MatMulOp matmulOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    bool _enableGroupedMatMul;
};

mlir::LogicalResult MatMulInputsTo2dPass::ReshapeNDInputConverter::matchAndRewrite(
        IE::MatMulOp matmulOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), matmulOp->getName(), matmulOp->getLoc());

    auto transposeA = matmulOp.getTransposeA();
    auto transposeB = matmulOp.getTransposeB();
    auto input1Shape = getShape(matmulOp.getInput1());
    auto input2Shape = getShape(matmulOp.getInput2());

    auto adjustTo3DShape = [](ShapeRef origShape, const bool isFirstInput) {
        if (origShape.size() == 1) {
            return isFirstInput ? Shape{1, 1, origShape.front()} : Shape{1, origShape.front(), 1};
        }

        auto batchSize = std::accumulate(origShape.begin(), origShape.end() - 2, 1, std::multiplies<int64_t>());
        return Shape{batchSize, origShape[Dim(origShape.size() - 2)], origShape[Dim(origShape.size() - 1)]};
    };

    // Adjust second input
    // Step 1: Adjust input shape to a tensor rank of 3D [Batch, Height, Width]
    //  If the tensor is 1D, the size is assigned to Height, and both Batch and Width are set to 1
    //    - For example: [6] -> [1, 6, 1]
    //  If the tensor is larger or equal than 2D, the last two dimensions are assigned to Height and Width
    //  and Batch is set to the product of the remaining dimensions
    //    - For example: [2, 3] -> [1, 2, 3]; [1, 1, 6] -> [1, 1, 6]; [3, 1, 6, 4, 2] -> [18, 4, 2]
    // Step 2: The batch dimension can be removed if its size equals 1
    //    - For example: [1, 1, 1, 8] -> [1, 1, 8] -> [1, 8]
    //                   [1, 6, 1, 8] -> [6, 1, 8] -> [6, 1, 8]
    auto newIn2Shape = adjustTo3DShape(input2Shape, /*isFirstInput=*/false);
    if (newIn2Shape.front() == 1) {
        newIn2Shape.erase(newIn2Shape.begin());
    }

    // Adjust first input
    // Step 1: Adjust input shape to a tensor rank of 3D [Batch, Height, Width]
    //  If the tensor is 1D, the size is assigned to Width, and both Batch and Height are set to 1
    //    - For example: [6] -> [1, 1, 6]
    //  If the tensor is larger or equal than 2D, the last two dimensions are assigned to Height and Width
    //  and Batch is set to the product of the remaining dimensions
    //    - For example: [2, 3] -> [1, 2, 3]; [1, 2, 6] -> [1, 2, 6]; [3, 1, 6, 4, 2] -> [18, 4, 2]
    // Step 2: If transposeA is set to false or batch equal 1 and the new second input shape lacks a batch dimension
    //         the Batch can be integrated into the Height dimension.
    // For example:
    //   MatMul(2x3x6x4, 4x8) {transposeA = false} collapses to MatMul(36x4, 4x8) {transposeA = false}
    //   MatMul(2x3x4x6, 4x8) {transposeA = true} collapses to MatMul(6x4x6, 4x8) {transposeA = true}
    //   MatMul(1x2x6x4, 2x4x8) {transposeA = false} collapses to MatMul(2x6x4, 2x4x8) {transposeA = false}
    auto newIn1Shape = adjustTo3DShape(input1Shape, /*isFirstInput=*/true);
    if (newIn2Shape.size() == 2 && (!transposeA || newIn1Shape.front() == 1)) {
        auto batchSize = newIn1Shape.front();
        newIn1Shape.erase(newIn1Shape.begin());
        newIn1Shape[Dim(0)] = newIn1Shape[Dim(0)] * batchSize;
    }
    if (newIn1Shape == input1Shape && newIn2Shape == input2Shape) {
        return mlir::failure();
    }
    if (_enableGroupedMatMul == true && newIn1Shape.size() > 2 && newIn2Shape.size() > 2 && newIn1Shape.front() != 1 &&
        newIn2Shape.front() != 1) {
        const auto availableCMXBytes = vpux::VPU::getTotalCMXSize(matmulOp);
        const auto expandedCMXUsage = getExpandedCMXUsage(matmulOp, newIn1Shape, newIn2Shape);
        if (expandedCMXUsage < availableCMXBytes.count()) {
            return mlir::failure();
        }
    }

    // Check if the original input shapes are either both 3D or both 4D without batch
    // If they can be converted to MatMul without batch, the shapes will be adjusted
    // If not, the following conversion logic can directly slice them without the need for reshaping
    const auto isNewShapeWithoutBatch = (newIn1Shape.size() == 2) && (newIn2Shape.size() == 2);
    const auto is3DInput = (input1Shape.size() == 3) && (input2Shape.size() == 3);
    const auto is4DInputWithoutBatch =
            input1Shape.size() == 4 && (input2Shape.size() == 4 || input2Shape.size() == 3) && input1Shape[Dim(0)] == 1;
    if (!isNewShapeWithoutBatch && (is3DInput || is4DInputWithoutBatch)) {
        return mlir::failure();
    }

    // Adjust MatMul inputs
    auto adjustInputTensor = [&](mlir::Value input, ShapeRef newShape, mlir::Location newLoc) {
        return rewriter.createOrFold<IE::ReshapeOp>(newLoc, input, nullptr, false,
                                                    getIntArrayAttr(rewriter.getContext(), newShape));
    };

    auto reshapeInput1 = adjustInputTensor(matmulOp.getInput1(), newIn1Shape, takeOpLoc(matmulOp, "in1_reshape"));
    auto reshapeInput2 = adjustInputTensor(matmulOp.getInput2(), newIn2Shape, takeOpLoc(matmulOp, "in2_reshape"));

    auto newMatMul =
            rewriter.create<IE::MatMulOp>(matmulOp->getLoc(), reshapeInput1, reshapeInput2, transposeA, transposeB);

    const auto origOutShape = getShape(matmulOp.getOutput());
    const auto origOutShapeAttr = getIntArrayAttr(rewriter.getContext(), origOutShape);
    auto newOp = rewriter.replaceOpWithNewOp<IE::ReshapeOp>(matmulOp, newMatMul.getOutput(), nullptr, false,
                                                            origOutShapeAttr);
    extendOpLoc(newOp, "out_reshape");
    return mlir::success();
}

//
// safeRunOnFunc
//

void MatMulInputsTo2dPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ReshapeNDInputConverter>(&ctx, benefitLevels[0], _log, _enableGroupedMatMul);
    patterns.add<MatMulOpConverter>(&ctx, benefitLevels[1], _log, _enableGroupedMatMul);
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createMatMulInputsTo2dPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createMatMulInputsTo2dPass(const bool enableGroupedMatMul, Logger log) {
    return std::make_unique<MatMulInputsTo2dPass>(enableGroupedMatMul, log);
}
