//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/VPU30XX/dialect/IE/impl/weights_dequantize_to_fakequantize_strategy.hpp"
#include "vpux/compiler/dialect/IE/utils/fake_quantize_utils.hpp"

using namespace vpux;

namespace {

class WeightsDequantizeToFakeQuantizeRewriter final : public mlir::OpRewritePattern<Const::DeclareOp> {
public:
    WeightsDequantizeToFakeQuantizeRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<Const::DeclareOp>(ctx), _log(log) {
        setDebugName("WeightsDequantizeToFakeQuantizeRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(Const::DeclareOp origOp, mlir::PatternRewriter&) const final;

private:
    Logger _log;
};

mlir::LogicalResult WeightsDequantizeToFakeQuantizeRewriter::matchAndRewrite(Const::DeclareOp origOp,
                                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("Got Constant {0} at `{1}`.", origOp->getName(), origOp->getLoc());

    //
    // Pattern matching conditions
    //

    // +----------------------------------------------------------------+
    // | Weights Const - i8 with transformations                        |
    // |  [#const.ConvertElemType<i4>] || [#const.ConvertElemType<u4>]  |
    // | [#const.ConvertElemType<f16>] || [#const.ConvertElemType<f32>] |
    // | Weights Const - u8 with transformations                        |
    // |  [#const.ConvertElemType<i4>] || [#const.ConvertElemType<u4>]  |
    // | [#const.ConvertElemType<f16>] || [#const.ConvertElemType<f32>] |
    // | Weights Const - f16 with transformations                       |
    // |  [#const.ConvertElemType<i4>] || [#const.ConvertElemType<u4>]  |
    // +----------------------------------------------------------------+
    //           |
    //           |      +-------------+
    //           |      | Shift Const |
    //           |      +-------------+
    //           |           |
    //        +-------------------+
    //        | Optional Subtract |
    //        +-------------------+
    //                  |
    //                  |   +-------------+
    //                  |   | Scale Const |
    //                  |   +-------------+
    //                  |          |
    //              +-------------------+
    //              | Optional Multiply |
    //              +-------------------+
    //
    // Subtract and Multiply operation are optional in the dequantization pattern, because they can be folded

    const auto weightsContentAttr = origOp.getContentAttr();
    const auto weightsBaseVals = weightsContentAttr.getBaseContent();
    auto weightsBaseElemType = weightsBaseVals.getShapedType().getElementType();

    // check for I4 and U4 (sub-byte types are represented using ConvertElemType transforms)
    const auto isInt4Type = llvm::any_of(weightsContentAttr.getTransformations(), [](const auto& transform) {
        const auto convert = transform.template dyn_cast_or_null<Const::ConvertElemTypeAttr>();
        return convert != nullptr &&
               (convert.getElemType().isSignedInteger(4) || convert.getElemType().isUnsignedInteger(4));
    });
    const auto isIntType =
            weightsBaseElemType.isSignedInteger(8) || weightsBaseElemType.isUnsignedInteger(8) || isInt4Type;

    // The only supported weights data type are I8, U8, I4 and U4
    if (!isIntType) {
        _log.trace("Const data type {0} is not supported.", weightsBaseElemType);
        return mlir::failure();
    }

    // Check if the op pattern matches
    const auto structureMatch = IE::getWeightsDequantizeStructure(origOp, _log.nest());
    if (mlir::failed(structureMatch)) {
        _log.trace("Failed to match WeightsDequantize structure");
        return mlir::failure();
    }
    mlir::Operation* lastOp = nullptr;
    Const::ContentAttr shiftContentAttr = nullptr;
    Const::ContentAttr scaleContentAttr = nullptr;
    std::tie(lastOp, shiftContentAttr, scaleContentAttr) = structureMatch.value();

    const auto weightsContent = origOp.getContent();
    const auto weightsElementType = weightsContent.getType().getElementType();
    const auto weightsStorageType = weightsContent.getStorageElemType();

    // If weights storage type isn't already F16 or F32, cast them to higher precision
    Const::DeclareOp fakeQuantizeInput = nullptr;
    if (weightsElementType == weightsStorageType && (weightsStorageType.isF16() || weightsStorageType.isF32())) {
        fakeQuantizeInput = origOp;

    } else {
        const auto castWeightsOrFail = IE::castWeightStorageToHighPrecision(weightsContent, _log.nest());
        if (mlir::failed(castWeightsOrFail)) {
            _log.error("Failed to cast weights");
            return mlir::failure();
        }

        fakeQuantizeInput = rewriter.create<Const::DeclareOp>(origOp.getLoc(), weightsContentAttr.getType(),
                                                              castWeightsOrFail.value());
    }

    //
    // Compute input low, input high constants of FakeQuantize using the value interval of the weights type
    //

    const auto castedContent = fakeQuantizeInput.getContent();
    const auto castedStorageType = castedContent.getStorageElemType();
    const auto weightsMinimum = castedStorageType.isF16() ? IE::getMinWeightsValue<vpux::type::float16>(castedContent)
                                                          : IE::getMinWeightsValue<float>(castedContent);

    auto levelsOrFail = IE::getLevels(weightsContentAttr, weightsMinimum);
    if (mlir::failed(levelsOrFail)) {
        _log.trace("Weights data type {0} is not supported", weightsBaseElemType);
        return mlir::failure();
    }

    int64_t levels;
    bool signedWeights;
    std::tie(levels, signedWeights) = levelsOrFail.value();
    const auto inLow = static_cast<float>(signedWeights ? -(levels / 2) : 0);
    const auto inHigh = static_cast<float>(levels + inLow - 1);
    const auto levelsAttr = getIntAttr(origOp.getContext(), levels);

    const auto weightsConstantRank = getShape(origOp).size();
    SmallVector<int64_t> inCstShape = SmallVector<int64_t>(weightsConstantRank, 1);
    const auto inStorageType = mlir::RankedTensorType::get(inCstShape, weightsElementType);
    const auto inLowDenseElementVal = wrapData(inStorageType, inLow);
    const auto inHighDenseElementVal = wrapData(inStorageType, inHigh);
    auto inLowContentAttr = Const::ContentAttr::get(inLowDenseElementVal);
    auto inHighContentAttr = Const::ContentAttr::get(inHighDenseElementVal);

    auto inLowConst = rewriter.create<Const::DeclareOp>(origOp->getLoc(), inStorageType, inLowContentAttr);
    auto inHighConst = rewriter.create<Const::DeclareOp>(origOp->getLoc(), inStorageType, inHighContentAttr);

    //
    // Compute output low and output high constants of FakeQuantize by applying a reverse scale-shift to the inputs
    //

    // Apply scale and shift (if given)
    auto outStorageType = inLowContentAttr.getType();
    if (mlir::failed(IE::revertScaleShift(scaleContentAttr, shiftContentAttr, inLowContentAttr, inHighContentAttr,
                                          outStorageType, _log))) {
        _log.error("Failed to revert scale-shift");
        return mlir::failure();
    }

    auto outLowConst = rewriter.create<Const::DeclareOp>(origOp->getLoc(), outStorageType, inLowContentAttr);
    auto outHighConst = rewriter.create<Const::DeclareOp>(origOp->getLoc(), outStorageType, inHighContentAttr);

    const auto broadCastAttr = IE::AutoBroadcastTypeAttr::get(origOp.getContext(), IE::AutoBroadcastType::NUMPY);

    // Create the FakeQuantize to replace the weights dequantize pattern
    // Since working with intergers, only levelsAttr is given
    rewriter.replaceOpWithNewOp<IE::FakeQuantizeOp>(lastOp, fakeQuantizeInput, inLowConst, inHighConst, outLowConst,
                                                    outHighConst, levelsAttr, /*lowFpType=*/nullptr, broadCastAttr);

    return mlir::success();
}

}  // namespace

//
// WeightsDequantizeToFakeQuantizeStrategy
//

void IE::arch30xx::WeightsDequantizeToFakeQuantizeStrategy::addPatterns(mlir::RewritePatternSet& patterns,
                                                                        Logger& log) const {
    auto ctx = patterns.getContext();
    patterns.add<WeightsDequantizeToFakeQuantizeRewriter>(ctx, log);
}
