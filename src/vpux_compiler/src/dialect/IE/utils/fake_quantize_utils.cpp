//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/fake_quantize_utils.hpp"

#include "vpux/compiler/utils/loop.hpp"
#include "vpux/utils/core/numeric.hpp"

namespace vpux {
namespace IE {

mlir::LogicalResult broadcastContentAttrs(Const::ContentAttr& inLowContentAttr, Const::ContentAttr& inHighContentAttr,
                                          Const::ContentAttr& transformContentAttr, const Logger& log) {
    auto transformShape = transformContentAttr.getType().getShape();
    auto inLowShape = inLowContentAttr.getType().getShape();

    // Align ranks
    if (transformShape.size() != inLowShape.size()) {
        if (transformShape.size() == 1 && transformShape[Dim(0)] == 1) {
            transformContentAttr.reshape(Shape(SmallVector<int64_t>(inLowShape.size(), 1)));
            transformShape = transformContentAttr.getType().getShape();
        } else if (inLowShape.size() == 1 && inLowShape[Dim(0)] == 1) {
            inLowContentAttr.reshape(Shape(SmallVector<int64_t>(inLowShape.size(), 1)));
            inLowShape = transformContentAttr.getType().getShape();
            inHighContentAttr.reshape(inLowShape);
        } else {
            log.trace("The transform rank {0} is not equal with inLow rank {1}", transformShape.size(),
                      inLowShape.size());
            return mlir::failure();
        }
    }

    log.trace("broadcastContentAttrs: Aligned the transformation and input shapes");

    // Align inLowCst/inHighCst and tranformCst shapes
    for (size_t i = 0; i < transformShape.size(); i++) {
        if (transformShape[Dim(i)] == 1 && inLowShape[Dim(i)] > 1) {
            transformContentAttr = transformContentAttr.broadcast(Dim(i), inLowShape[Dim(i)]);
        } else if (inLowShape[Dim(i)] == 1 && transformShape[Dim(i)] > 1) {
            inLowContentAttr = inLowContentAttr.broadcast(Dim(i), transformShape[Dim(i)]);
            inHighContentAttr = inHighContentAttr.broadcast(Dim(i), transformShape[Dim(i)]);
        } else if (transformShape[Dim(i)] > 1 && inLowShape[Dim(i)] > 1 &&
                   transformShape[Dim(i)] != inLowShape[Dim(i)]) {
            log.trace("Cannot broadcast sizes inLow/inHigh dim = {0} and transform dim = {1}", inLowShape[Dim(i)],
                      transformShape[Dim(i)]);
            return mlir::failure();
        }
    }
    log.trace("broadcastContentAttrs: Broadcasted the transformation and input shapes");

    return mlir::success();
}

mlir::FailureOr<std::tuple<Const::ContentAttr, Const::ContentAttr, mlir::RankedTensorType>> applyTransformation(
        Const::ContentAttr inLowContentAttr, Const::ContentAttr inHighContentAttr,
        Const::ContentAttr transformContentAttr, const std::function<float(float, float)>& transformCb,
        const Logger& log) {
    if (mlir::failed(broadcastContentAttrs(inLowContentAttr, inHighContentAttr, transformContentAttr, log))) {
        log.trace("Didn't manage to broadcast const content attributes");
        return mlir::failure();
    }

    // Get content of broadcasted input low, input high and scale/shift constants
    const auto broadcastedTransformContent = transformContentAttr.fold();
    const auto broadcastedInLowContent = inLowContentAttr.fold();
    const auto broadcastedInHighContent = inHighContentAttr.fold();
    const auto broadcastedTransformBaseElemType = broadcastedTransformContent.getType().getElementType();
    const auto broadcastedInLowValues = to_small_vector(broadcastedInLowContent.getValues<float>());
    const auto broadcastedInHighValues = to_small_vector(broadcastedInHighContent.getValues<float>());
    const auto broadcastedTransformValues = to_small_vector(broadcastedTransformContent.getValues<float>());

    auto outLowValues = SmallVector<float>(broadcastedTransformValues.size(), 0);
    auto outHighValues = SmallVector<float>(broadcastedTransformValues.size(), 0);
    loop_1d(LoopExecPolicy::Parallel, transformContentAttr.getType().getElementType().getContext(),
            broadcastedTransformValues.size(), [&](size_t i) {
                outLowValues[i] = transformCb(broadcastedInLowValues[i], broadcastedTransformValues[i]);
                outHighValues[i] = transformCb(broadcastedInHighValues[i], broadcastedTransformValues[i]);
            });

    auto outConstShape = inLowContentAttr.getType().dyn_cast<vpux::NDTypeInterface>().getShape();
    auto outStorageType = mlir::RankedTensorType::get(outConstShape.raw(), broadcastedTransformBaseElemType);
    const auto outLowDenseElementVal = wrapArrayRef(outStorageType, outLowValues);
    const auto outHighDenseElementVal = wrapArrayRef(outStorageType, outHighValues);
    auto outLowContentAttr = Const::ContentAttr::get(outLowDenseElementVal);
    auto outHighContentAttr = Const::ContentAttr::get(outHighDenseElementVal);
    return std::make_tuple(outLowContentAttr, outHighContentAttr, outStorageType);
}

mlir::LogicalResult applyScaleShift(const Const::ContentAttr& scale, const Const::ContentAttr& shift,
                                    Const::ContentAttr& low, Const::ContentAttr& high,
                                    vpux::NDTypeInterface& storageType, const Logger& log) {
    // Applies X * (1/scale) + shift to the given low and high tensors

    // Apply scale (if given)
    if (scale != nullptr) {
        auto applyScaleResult = IE::applyTransformation(low, high, scale, std::divides<float>(), log);
        if (mlir::failed(applyScaleResult)) {
            return mlir::failure();
        }
        std::tie(low, high, storageType) = applyScaleResult.value();
    }

    // Apply shift (if given)
    if (shift != nullptr) {
        auto applyShiftResult = IE::applyTransformation(low, high, shift, std::plus<float>(), log);
        if (mlir::failed(applyShiftResult)) {
            return mlir::failure();
        }
        std::tie(low, high, storageType) = applyShiftResult.value();
    }

    return mlir::success();
}

mlir::LogicalResult revertScaleShift(const Const::ContentAttr& scale, const Const::ContentAttr& shift,
                                     Const::ContentAttr& low, Const::ContentAttr& high,
                                     vpux::NDTypeInterface& storageType, const Logger& log) {
    // Applies (X - shift) * scale to the given low and high tensors

    // Apply shift (if given)
    if (shift != nullptr) {
        auto applyShiftResult = IE::applyTransformation(low, high, shift, std::minus<float>(), log);
        if (mlir::failed(applyShiftResult)) {
            return mlir::failure();
        }
        std::tie(low, high, storageType) = applyShiftResult.value();
    }

    // Apply scale (if given)
    if (scale != nullptr) {
        auto applyScaleResult = IE::applyTransformation(low, high, scale, std::multiplies<float>(), log);
        if (mlir::failed(applyScaleResult)) {
            return mlir::failure();
        }
        std::tie(low, high, storageType) = applyScaleResult.value();
    }

    return mlir::success();
}

mlir::FailureOr<std::tuple<int64_t, bool>> getLevels(Const::ContentAttr weightsContentAttr, float weightsMinimum) {
    // Storing i4 values as constants is not possible in MLIR; so even when we import the model we should import
    // the model in our frontend we should create a Constant that has higher level storage such as SI8 and a
    // transformation which converts the expressed type to I4/U4
    // Example: const.ConvertElemType<si4>, const.ConvertElemType<ui4>
    for (const auto& attr : weightsContentAttr.getTransformations()) {
        if (auto convert = attr.dyn_cast_or_null<Const::ConvertElemTypeAttr>()) {
            if (convert.getElemType().isSignedInteger(4)) {
                return std::tuple<int64_t, bool>((isFloatEqual(weightsMinimum, -8.0f)) ? 16 : 15, true);
            }
            if (convert.getElemType().isUnsignedInteger(4)) {
                return std::tuple<int64_t, bool>(16, false);
            }
        }
    }

    auto weightsBaseElemType = weightsContentAttr.getBaseContent().getShapedType().getElementType();
    if (weightsBaseElemType.isSignedInteger(8)) {
        return std::tuple<int64_t, bool>((isFloatEqual(weightsMinimum, -128.0f)) ? 256 : 255, true);
    }
    if (weightsBaseElemType.isUnsignedInteger(8)) {
        return std::tuple<int64_t, bool>(256, false);
    }

    return mlir::failure();
}

mlir::FailureOr<std::tuple<mlir::Operation*, Const::ContentAttr, Const::ContentAttr>> getWeightsDequantizeStructure(
        Const::DeclareOp origOp, const Logger& _log) {
    // finds the lowest op in the weights dequantize structure: Multiply / Substract / Declare
    // and retrieves the shift and scale arguments if present

    mlir::Operation* lastOp = origOp.getOperation();
    Const::ContentAttr shiftContentAttr = nullptr;
    Const::ContentAttr scaleContentAttr = nullptr;

    if (lastOp->getResult(0).getUsers().empty()) {
        _log.trace("Const is not in use");
        return mlir::failure();
    }

    auto maybeFakeQuantize = mlir::dyn_cast_or_null<IE::FakeQuantizeOp>(*lastOp->getResult(0).getUsers().begin());
    if (maybeFakeQuantize != nullptr) {
        _log.trace("FakeQuantize already present.");
        return mlir::failure();
    }

    auto maybeSubtract = mlir::dyn_cast_or_null<IE::SubtractOp>(*lastOp->getResult(0).getUsers().begin());
    if (maybeSubtract != nullptr) {
        lastOp = maybeSubtract.getOperation();

        auto shiftCst = maybeSubtract.getInput2().getDefiningOp<Const::DeclareOp>();
        if (shiftCst == nullptr) {
            return mlir::failure();
        }
        shiftContentAttr = shiftCst.getContentAttr();
    }

    if (lastOp->getResult(0).getUsers().empty()) {
        _log.trace("Const is not in use");
        return mlir::failure();
    }

    auto maybeMultiply = mlir::dyn_cast_or_null<IE::MultiplyOp>(*lastOp->getResult(0).getUsers().begin());
    if (maybeMultiply != nullptr) {
        lastOp = maybeMultiply.getOperation();

        auto scaleCst = maybeMultiply.getInput2().getDefiningOp<Const::DeclareOp>();
        if (scaleCst == nullptr) {
            return mlir::failure();
        }
        scaleContentAttr = scaleCst.getContentAttr();
    }

    return std::make_tuple(lastOp, shiftContentAttr, scaleContentAttr);
}

mlir::FailureOr<Const::ContentAttr> castWeightStorageToHighPrecision(const Const::Content& weightsContent,
                                                                     const Logger& _log) {
    // Converts weights values from lower precision to float
    // This should ensure good precision for possible further operation done on weights along the compilation flow

    const auto weightsContentType = weightsContent.getType();
    const auto weightsBufferSize = checked_cast<size_t>(weightsContentType.getNumElements());
    if (weightsBufferSize == 0) {
        _log.trace("Weights constant is empty");
        return mlir::failure();
    }

    const auto weightsElementType = weightsContentType.getElementType();
    const auto weightsBufferByteSize = checked_cast<size_t>(weightsContentType.getTotalAllocSize().count());
    const auto weightsRankedTensorType = weightsContentType.cast<mlir::RankedTensorType>();

    if (weightsElementType.isF16()) {
        std::vector<vpux::type::float16> fp16TempWeightsBuffer(weightsBufferSize);
        weightsContent.copyTo(
                MutableArrayRef(reinterpret_cast<char*>(fp16TempWeightsBuffer.data()), weightsBufferByteSize));
        const auto weightsDenseAttr = mlir::DenseElementsAttr::getFromRawBuffer(
                weightsRankedTensorType,
                ArrayRef(reinterpret_cast<char*>(fp16TempWeightsBuffer.data()), weightsBufferByteSize));
        return Const::ContentAttr::get(weightsDenseAttr);
    }

    if (weightsElementType.isF32()) {
        std::vector<float> fp32TempWeightsBuffer(weightsBufferSize);
        weightsContent.copyTo(
                MutableArrayRef(reinterpret_cast<char*>(fp32TempWeightsBuffer.data()), weightsBufferByteSize));
        const auto weightsDenseAttr = mlir::DenseElementsAttr::getFromRawBuffer(
                weightsRankedTensorType,
                ArrayRef(reinterpret_cast<char*>(fp32TempWeightsBuffer.data()), weightsBufferByteSize));
        return Const::ContentAttr::get(weightsDenseAttr);
    }

    _log.trace("Weights element type must be FP16 or FP32 but got {0}", weightsElementType);
    return mlir::failure();
}

// option A
// float getMinWeightsValue(const Const::Content& weightsContent) {
//     std::vector<float> values;
//     const auto weightsBufferByteSize = checked_cast<size_t>(weightsContent.getType().getTotalAllocSize().count());
//     values.resize(weightsBufferByteSize);
//     weightsContent.copyTo(
//                 MutableArrayRef(reinterpret_cast<char*>(values.data()), weightsBufferByteSize));

//     const auto min = std::min_element(values.begin(), values.end());

//     VPUX_THROW_WHEN(min == values.end(), "Got empty weights content");
//     return *min;
// }

// // option B
// float getMinWeightsValue(const Const::Content& weightsContent) {
//     const auto& values = weightsContent.getValues<float>();
//     const auto min = std::min_element(values.begin(), values.end());

//     VPUX_THROW_WHEN(min == values.end(), "Got empty weights content");
//     return *min;
// }

// option C
// float getMinWeightsValue(const Const::Content& weightsContent) {
//     const auto& values = weightsContent.vec<float>();
//     const auto min = std::min_element(values.begin(), values.end());

//     VPUX_THROW_WHEN(min == values.end(), "Got empty weights content");
//     return *min;
// }

// option D

}  // namespace IE
}  // namespace vpux
