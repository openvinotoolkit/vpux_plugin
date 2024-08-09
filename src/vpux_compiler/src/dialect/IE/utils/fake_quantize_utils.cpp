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

    // Apply transformation
    auto outLowValues = SmallVector<float>(broadcastedTransformValues.size(), 0);
    auto outHighValues = SmallVector<float>(broadcastedTransformValues.size(), 0);
    loop_1d(LoopExecPolicy::Parallel, transformContentAttr.getType().getElementType().getContext(),
            broadcastedTransformValues.size(), [&](size_t i) {
                outLowValues[i] = transformCb(broadcastedInLowValues[i], broadcastedTransformValues[i]);
                outHighValues[i] = transformCb(broadcastedInHighValues[i], broadcastedTransformValues[i]);
            });

    auto outConstShape = inLowContentAttr.getType().dyn_cast<vpux::NDTypeInterface>().getShape();
    auto outStorageType = mlir::RankedTensorType::get(outConstShape.raw(), broadcastedTransformBaseElemType);
    const auto outLowDenseElementVal = wrapData(outStorageType, outLowValues);
    const auto outHighDenseElementVal = wrapData(outStorageType, outHighValues);
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

template <typename NewT>
Const::ContentAttr castStorageType(const Const::Content& content) {
    const auto contentType = content.getType();
    const auto tensorType = contentType.cast<mlir::RankedTensorType>();
    const auto size = checked_cast<size_t>(contentType.getNumElements());
    const auto byteSize = checked_cast<size_t>(contentType.getTotalAllocSize().count());

    std::vector<NewT> tempBuffer(size);
    content.copyTo(MutableArrayRef(reinterpret_cast<char*>(tempBuffer.data()), byteSize));

    auto denseAttr = mlir::DenseElementsAttr::getFromRawBuffer(
            tensorType, ArrayRef(reinterpret_cast<char*>(tempBuffer.data()), byteSize));

    return Const::ContentAttr::get(denseAttr);
}

mlir::LogicalResult WeightsDequantizeStructureInfo::initializeStructure(IE::MultiplyOp& multiplyOp) {
    lastOp = multiplyOp.getOperation();

    // Retrieve scale
    auto scaleCst = multiplyOp.getInput2().getDefiningOp<Const::DeclareOp>();
    if (scaleCst == nullptr) {
        log.trace("Match failed: Got non-const scale");
        return mlir::failure();
    }
    scale = scaleCst.getContentAttr();

    return mlir::success();
}

mlir::LogicalResult WeightsDequantizeStructureInfo::initializeStructure(IE::SubtractOp& subtractOp) {
    lastOp = subtractOp.getOperation();

    // Retrieve shift
    auto shiftCst = subtractOp.getInput2().getDefiningOp<Const::DeclareOp>();
    if (shiftCst == nullptr) {
        log.trace("Match failed: Got non-const shift");
        return mlir::failure();
    }
    shift = shiftCst.getContentAttr();

    // Check following ops
    const auto opUser = subtractOp->user_begin();
    if (opUser == subtractOp->user_end()) {
        return mlir::failure();
    }

    if (auto multiplyOp = mlir::dyn_cast<IE::MultiplyOp>(*opUser)) {
        return this->initializeStructure(multiplyOp);
    }

    return mlir::success();
}

mlir::LogicalResult WeightsDequantizeStructureInfo::initializeStructure(IE::ConvertOp& convertOp) {
    lastOp = convertOp.getOperation();

    // Retrieve non-const input properties
    inputBlock = mlir::dyn_cast_or_null<mlir::BlockArgument>(convertOp.getInput());
    if (inputBlock != nullptr) {
        inputElemBaseType = llvm::dyn_cast<mlir::ShapedType>(inputBlock.getType()).getElementType();
        if (inputElemBaseType.isInteger(4)) {
            inputVirtualI4ElemType = inputElemBaseType;
        }
        log.trace("Got block argument input: {0}", inputElemBaseType);

    } else {
        log.trace("Match failed: Got ConvertOp without Const or BlockArgument input");
        return mlir::failure();
    }

    inputElemConvertType = convertOp.getDstElemType();
    inputValue = convertOp.getOutput();

    // Check following ops
    if (!convertOp->hasOneUse()) {
        // We decided to only treat the single-use case for now
        log.trace("Match failed: Got ConvertOp with 0 or multiple users");
        return mlir::failure();
    }
    auto opUser = convertOp->user_begin();

    // Prevent rematching already processed ConvertOps (they aren't deleted by the WDtoFQ pass)
    if (auto fakeQuantOp = mlir::dyn_cast<IE::FakeQuantizeOp>(*opUser)) {
        log.trace("Match failed: FakeQuantizeOp already present at end of structure");
        return mlir::failure();
    }

    if (auto subtractOp = mlir::dyn_cast<IE::SubtractOp>(*opUser)) {
        return this->initializeStructure(subtractOp);
    }
    if (auto multiplyOp = mlir::dyn_cast<IE::MultiplyOp>(*opUser)) {
        return this->initializeStructure(multiplyOp);
    }

    // in block arg case, ConvertOps are kept, so a ConvertOp with no following SubtractOp or MultiplyOp would
    // result in a useless FakeQuantizeOp being inserted
    log.trace("Match failed: ConvertOp with no following SubractOp or MultiplyOp, match failed");
    return mlir::failure();
}

mlir::LogicalResult WeightsDequantizeStructureInfo::initializeStructure(Const::DeclareOp& declareOp) {
    lastOp = declareOp.getOperation();

    inputAttr = declareOp.getContentAttr();
    inputContent = declareOp.getContent();

    inputElemBaseType = initialInputElemStorageType = inputElemConvertType =
            inputAttr.getBaseContent().getShapedType().getElementType();

    // since U4 and I4 aren't aren't fully supported, they are represented through ConvertElemType transforms
    for (const auto& attr : inputAttr.getTransformations()) {
        if (auto convert = attr.dyn_cast_or_null<Const::ConvertElemTypeAttr>()) {
            if (convert.getElemType().isInteger(4)) {
                inputVirtualI4ElemType = convert.getElemType();
            } else {
                inputElemConvertType = convert.getElemType();
            }
        }
    }

    // Check following ops
    const auto opUser = declareOp->user_begin();
    if (opUser == declareOp->user_end()) {
        return mlir::failure();
    }

    // Prevent matching DeclareOps which were already quantized in pre-compilation (by OV)
    if (auto fakeQuantOp = mlir::dyn_cast<IE::FakeQuantizeOp>(*opUser)) {
        log.trace("Match failed: FakeQuantizeOp already present at end of structure");
        return mlir::failure();
    }
    if (auto subtractOp = mlir::dyn_cast<IE::SubtractOp>(*opUser)) {
        return this->initializeStructure(subtractOp);
    }
    if (auto multiplyOp = mlir::dyn_cast<IE::MultiplyOp>(*opUser)) {
        return this->initializeStructure(multiplyOp);
    }

    if (inputElemBaseType == inputElemConvertType) {
        // A DeclareOp is still considered a WD if it has at least one ConvertElemType transformation
        // The WDtoFQ pass must remove ConvertElemType from processed constants, otherwise this case results in an
        // infinite loop
        log.trace("Match failed: DeclareOp without conversions, shifting or scaling");
        return mlir::failure();
    }

    return mlir::success();
}

WeightsDequantizeStructureInfo::WeightsDequantizeStructureInfo(Const::DeclareOp& origOp, const Logger& log) noexcept
        : firstOp(origOp.getOperation()), log(log) {
    isValid = mlir::succeeded(this->initializeStructure(origOp));
}

WeightsDequantizeStructureInfo::WeightsDequantizeStructureInfo(IE::ConvertOp& origOp, const Logger& log) noexcept
        : firstOp(origOp.getOperation()), log(log) {
    isValid = mlir::succeeded(this->initializeStructure(origOp));
}

mlir::MLIRContext* WeightsDequantizeStructureInfo::getContext() const {
    return firstOp->getContext();
}

const mlir::Location WeightsDequantizeStructureInfo::getLocation() const {
    return firstOp->getLoc();
}

mlir::Operation* WeightsDequantizeStructureInfo::getFirstOp() const {
    return firstOp;
}

mlir::Operation* WeightsDequantizeStructureInfo::getLastOp() const {
    return lastOp;
}

bool WeightsDequantizeStructureInfo::isSuccessfulMatch() const {
    return isValid;
}

bool WeightsDequantizeStructureInfo::hasConstInput() const {
    return inputAttr != nullptr;
}

bool WeightsDequantizeStructureInfo::has8BitIntegerInput() const {
    return inputVirtualI4ElemType == nullptr && inputElemBaseType.isInteger(8);
}

bool WeightsDequantizeStructureInfo::has4BitIntegerInput() const {
    return inputVirtualI4ElemType != nullptr;
}

bool WeightsDequantizeStructureInfo::has8BitFloatInput() const {
    return inputElemBaseType.isFloat8E4M3FN() || inputElemBaseType.isFloat8E5M2();
}

bool WeightsDequantizeStructureInfo::hasSignedInput() const {
    return inputVirtualI4ElemType != nullptr ? inputVirtualI4ElemType.isSignedInteger()
                                             : inputElemBaseType.isSignedInteger();
}

mlir::ShapedType WeightsDequantizeStructureInfo::getInputShapedType() const {
    if (this->hasConstInput()) {
        return llvm::dyn_cast<mlir::ShapedType>(inputAttr.getType());
    }

    return llvm::dyn_cast<mlir::ShapedType>(inputBlock.getType());
}

mlir::Type WeightsDequantizeStructureInfo::getInputElemBaseType() const {
    return inputElemBaseType;
}

mlir::Type WeightsDequantizeStructureInfo::getInputElemConvertType() const {
    return inputVirtualI4ElemType != nullptr ? inputVirtualI4ElemType : inputElemConvertType;
}

mlir::Type WeightsDequantizeStructureInfo::getInputFinalElemConvertType() const {
    // Return the final convert type (likely f16 or f32), even if an i4 conversion is present
    return inputElemConvertType;
}

mlir::Type WeightsDequantizeStructureInfo::getInputElemStorageType() const {
    if (this->hasConstInput()) {
        // Base type is not used because high precision casting might have occurred
        return inputContent->getStorageElemType();
    }

    // For block arg. input, ConvertOps determine the type of the tensor
    return inputElemConvertType;
}

int64_t WeightsDequantizeStructureInfo::getInputShapeRank() const {
    return this->getInputShapedType().getRank();
}

int64_t WeightsDequantizeStructureInfo::getQuantizationLevels() const {
    // Note: universally use fixed quantization levels. For activations, we
    // cannot know real values, so it's impossible to adjust this anyhow. For
    // weights, we do not need to know real values, because it does not affect
    // accuracy (or, should not, at least).
    if (this->has4BitIntegerInput()) {
        return 16;
    }
    if (this->has8BitIntegerInput()) {
        return 256;
    }

    VPUX_THROW("Got unsupported type when trying to compute levels: {0}", inputElemBaseType);
}

Const::DeclareOp WeightsDequantizeStructureInfo::getInputDeclareOp() const {
    VPUX_THROW_UNLESS(this->hasConstInput(), "WeightsDequantizeStructureInfo: Illegal method call for non-const input");
    return llvm::dyn_cast<Const::DeclareOp>(*firstOp);
}

const Const::Content& WeightsDequantizeStructureInfo::getInputContent() const {
    VPUX_THROW_UNLESS(this->hasConstInput(),
                      "WeightsDequantizeStructureInfo: Tried to retrieve content from non-const input");
    return *inputContent;
}

const Const::ContentAttr& WeightsDequantizeStructureInfo::getInputContentAttr() const {
    VPUX_THROW_UNLESS(this->hasConstInput(),
                      "WeightsDequantizeStructureInfo: Tried to retrieve content from non-const input");
    return inputAttr;
}

const mlir::Value WeightsDequantizeStructureInfo::getInputValue() const {
    VPUX_THROW_WHEN(this->hasConstInput(),
                    "WeightsDequantizeStructureInfo: Tried to retrieve block argument from const input");
    return inputValue;
}

mlir::LogicalResult WeightsDequantizeStructureInfo::ensureHighPrecisionStorage() {
    // Converts weights values from lower precision to float
    // Ensures good precision for later passes (some expecting f16/f32)
    // This results in a considerable performance loss (see E#115057) and efforts are made to remove it TODO: E#107322

    if (!hasConstInput()) {
        return mlir::failure();
    }

    const auto convertType = this->getInputFinalElemConvertType();
    const auto storageType = this->getInputElemStorageType();

    // Skip casting if weights storage type is already F16 or F32
    if (convertType == storageType && (storageType.isF16() || storageType.isF32())) {
        return mlir::success();
    }

    const auto castAttr = convertType.isF16() ? castStorageType<vpux::type::float16>(*inputContent)
                                              : convertType.isF32() ? castStorageType<float>(*inputContent) : nullptr;
    if (castAttr == nullptr) {
        log.trace("Weights element type must be f16 or f32 but got {0}", convertType);
        return mlir::failure();
    }

    inputAttr = castAttr;
    inputContent = castAttr.fold();

    log.trace("Weights were casted to high precision {0}", convertType);
    return mlir::success();
}

std::pair<Const::ContentAttr, Const::ContentAttr> WeightsDequantizeStructureInfo::getInputQuantizationInterval(
        const float low, const float high) {
    const auto inputRank = this->getInputShapeRank();
    const auto inElementType = this->getInputElemConvertType();
    const auto inStorageType =
            mlir::RankedTensorType::get(SmallVector<int64_t>(inputRank, 1),
                                        inElementType.isInteger(4) ? this->getInputElemStorageType() : inElementType);

    const auto inLowDenseElementVal = wrapData(inStorageType, low);
    const auto inHighDenseElementVal = wrapData(inStorageType, high);

    return std::make_pair(Const::ContentAttr::get(inLowDenseElementVal),
                          Const::ContentAttr::get(inHighDenseElementVal));
}

std::pair<Const::ContentAttr, Const::ContentAttr> WeightsDequantizeStructureInfo::getOutputQuantizationInterval(
        std::pair<Const::ContentAttr, Const::ContentAttr> inputInterval) {
    auto outType = inputInterval.first.getType();
    if (mlir::failed(IE::revertScaleShift(scale, shift, inputInterval.first, inputInterval.second, outType, log))) {
        VPUX_THROW("Failed to revert scale-shift");
    }

    return inputInterval;
}

// findAxes returns the positions of quantization axes
// For FQ in_low = in_high = out_low = out_high = 1x1x1x1 the set is empty
// For FQ in_low = in_high = out_low = out_high = 1x3x1x1 the set contains only one value = 1
// For FQ in_low = in_high = 1x1x1x1, out_low = out_high = 1x3x1x1 the set contains only one value = 1
// For FQ in_low = in_high = out_low = out_high = 1x3x1x16 the set contains positions 1 and 3
std::set<int64_t> findAxes(IE::FakeQuantizeOp origOp) {
    const auto operandShapes = SmallVector<ShapeRef>{
            getShape(origOp.getInputLow()),
            getShape(origOp.getInputHigh()),
            getShape(origOp.getOutputLow()),
            getShape(origOp.getOutputHigh()),
    };
    std::set<int64_t> axes;
    for (const auto& shape : operandShapes) {
        for (const auto& axis : irange(shape.size())) {
            if (shape[Dim(axis)] != 1) {
                axes.insert(axis);
            }
        }
    }
    return axes;
}

}  // namespace IE
}  // namespace vpux
