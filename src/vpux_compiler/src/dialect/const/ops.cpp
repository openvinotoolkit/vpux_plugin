//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/conversion/passes/VPU2VPUIP/bufferizable_ops_interface.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/ops.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/utils.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/const/attr_interfaces.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/utils/constant_folding_cache.hpp"

#include "vpux/compiler/core/type_interfaces.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Support/LogicalResult.h>

using namespace vpux;

//
// ConstDialect::materializeConstant
//

mlir::Operation* vpux::Const::ConstDialect::materializeConstant(mlir::OpBuilder& builder, mlir::Attribute value,
                                                                mlir::Type type, mlir::Location loc) {
    if (!value.isa<Const::ContentAttr>()) {
        (void)errorAt(loc, "Can't materialize Constant from Attribute '{0}'", value);
        return nullptr;
    }

    if (!type.isa<mlir::RankedTensorType, mlir::MemRefType>()) {
        (void)errorAt(loc, "Can't materialize Constant for Type '{0}'", type);
        return nullptr;
    }

    return builder.create<Const::DeclareOp>(loc, type, value.cast<Const::ContentAttr>());
}

//
// ConstDialect: bufferize Const::DeclareOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext*, Const::DeclareOp origOp, Const::DeclareOp::Adaptor,
                                      mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-ConstDeclareOp", 0);
    log.trace("Found Constant Operation '{0}'", origOp->getLoc());

    const auto newType = vpux::getBufferType(origOp.getType());
    auto newOp = rewriter.create<Const::DeclareOp>(origOp->getLoc(), newType, origOp.getContentAttr());
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, newOp->getResults());
    return mlir::success();
}

void vpux::registerConstDeclareBufferizableOpInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, vpux::Const::ConstDialect*) {
        Const::DeclareOp::attachInterface<VpuGenericOneShotBufferizeModel<Const::DeclareOp>>(*ctx);
    });
}

//
// DeclareOp::fold
//

mlir::OpFoldResult vpux::Const::DeclareOp::fold(FoldAdaptor adaptor) {
    VPUX_THROW_UNLESS(adaptor.getOperands().empty(), "constant has no operands");
    return getContentAttr();
}

//
// DeclareOp::serialize
//

void vpux::Const::DeclareOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    vpux::Const::Content cnt = getContent();
    // int64_t typeTotalSize = cnt.getRawStorageBuf().size();

    auto ptr = binDataSection.expandData(getBinarySize());
    MutableArrayRef<char> tempBuf(reinterpret_cast<char*>(ptr), reinterpret_cast<char*>(ptr) + getBinarySize());
    cnt.copyTo(tempBuf);
}

//
// DeclareOp::getBinarySize
//

size_t vpux::Const::DeclareOp::getBinarySize() {
    vpux::Const::Content cnt = getContent();

    return cnt.getType().getTotalAllocSize().count();
}

//
// DeclareOp::getAlignmentRequirements
//

size_t vpux::Const::DeclareOp::getAlignmentRequirements() {
    return ELFNPU37XX::VPUX_NO_ALIGNMENT;
}

//
// DeclareOp::getMemorySpace
//

vpux::VPURT::BufferSection vpux::Const::DeclareOp::getMemorySpace() {
    return vpux::VPURT::BufferSection::Constant;
}

//
// DeclareOp::getAccessingProcs
//

vpux::ELFNPU37XX::SectionFlagsAttr vpux::Const::DeclareOp::getAccessingProcs() {
    auto tempFlagsVal = vpux::ELFNPU37XX::SectionFlagsAttr::SHF_NONE;

    for (auto user : getResult().getUsers()) {
        if (auto binaryIface = mlir::dyn_cast<vpux::ELFNPU37XX::BinaryOpInterface>(user)) {
            tempFlagsVal = tempFlagsVal | binaryIface.getUserProcs();
        }
    }

    return tempFlagsVal;
}

//
// DeclareOp::getUserProcs
//

vpux::ELFNPU37XX::SectionFlagsAttr vpux::Const::DeclareOp::getUserProcs() {
    return (ELFNPU37XX::SectionFlagsAttr::SHF_NONE);
}

//
// DeclareOp::verify
//

mlir::LogicalResult vpux::Const::DeclareOp::verify() {
    const auto op = getOperation();
    const auto attrType = getContentAttr().getType();
    const auto opType = getType().cast<vpux::NDTypeInterface>();
    // For type with swizzling skip the shape check as the content
    // might have been flattened to accomodate swizzled buffer.
    if (!vpux::getSwizzlingSchemeAttr(opType)) {
        if (opType.getShape() != attrType.getShape()) {
            return errorAt(op, "'Const.Declare' has mismatch in value shape '{0}' and result shape '{1}'",
                           attrType.getShape(), opType.getShape());
        }
    }
    if (opType.getElementType() != attrType.getElementType()) {
        if (!opType.getElementType().isa<mlir::quant::QuantizedType>() &&
            !attrType.getElementType().isa<mlir::IntegerType>()) {
            return errorAt(op, "'Const.Declare' has mismatch in value element type '{0}' and result element type '{1}'",
                           attrType.getElementType(), opType.getElementType());
        }
    }

    const auto attrOrder = attrType.getDimsOrder();
    const auto opOrder = opType.getDimsOrder();

    if (opOrder != attrOrder) {
        return errorAt(op, "'Const.Declare' has mismatch in value DimsOrder '{0}' and result DimsOrder '{1}'",
                       attrOrder, opOrder);
    }

    return mlir::success();
}

mlir::LogicalResult vpux::Const::DeclareOp::verifySymbolUses(mlir::SymbolTableCollection& symbolTable) {
    // check if content is of type SymElementsAttr, otherwise this doesn't apply
    auto symElementsAttr = mlir::dyn_cast_or_null<vpux::Const::SymElementsAttr>(getContentAttr().getBaseContent());

    if (symElementsAttr == nullptr) {
        return mlir::success();
    }

    // lookup and check op type
    auto op = symbolTable.lookupNearestSymbolFrom(getOperation(), symElementsAttr.getSymName());
    auto rodataOp = mlir::dyn_cast_or_null<vpux::Const::RodataOp>(op);

    if (rodataOp == nullptr) {
        return emitOpError("symbol does not point to a valid const.Rodata op");
    }

    // verify types
    auto annotatedType = symElementsAttr.getType();
    auto underlyingType = rodataOp.getContent().getType();

    if (annotatedType != underlyingType) {
        return emitOpError("annotated type and the underlying dereferenced type of 'const.Rodata' op do not match");
    }

    return mlir::success();
}

//
// DeclareOp::canonicalizer
//

void sendEquivalenceRequest([[maybe_unused]] Const::ContentAttr originalAttr,
                            [[maybe_unused]] Const::ContentAttr newAttr) {
#ifdef BACKGROUND_FOLDING_ENABLED
    auto& cacheManager = Const::ConstantFoldingCacheManager::getInstance();
    auto ctx = originalAttr.getContext();
    if (cacheManager.contains(ctx)) {
        auto& cache = cacheManager.get(ctx);
        auto request = Const::EquivalenceRequestAttr::get(ctx, originalAttr, newAttr);
        cache.enqueueRequest(Const::FoldingRequest{request, /*newTransformation=*/nullptr});
    }
#endif
}

/**
 * Fuses consecutive transformations of the same type into a single transformation. For example:
 *   SubView + SubView ---> SubView
 */
class FuseConsecutiveTransformations final : public mlir::OpRewritePattern<Const::DeclareOp> {
public:
    FuseConsecutiveTransformations(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit)
            : mlir::OpRewritePattern<Const::DeclareOp>(ctx, benefit) {
    }

public:
    mlir::LogicalResult matchAndRewrite(Const::DeclareOp constOp, mlir::PatternRewriter& rewriter) const override {
        auto contentAttr = constOp.getContentAttr();
        auto transformations = contentAttr.getTransformations();
        if (transformations.empty()) {
            return mlir::failure();
        }

        const auto hasConsecutiveTransformations = [&]() {
            const auto transformationsCount = transformations.size();
            for (size_t idx = 0; idx < transformationsCount - 1; idx++) {
                if (transformations[idx].getTransformationName() == transformations[idx + 1].getTransformationName()) {
                    return true;
                }
            }
            return false;
        };
        if (!hasConsecutiveTransformations()) {
            return mlir::failure();
        }

        bool wereTransformationsFused = false;
        SmallVector<Const::TransformAttrInterface> newTransformations;

        const auto fuseTransformations = [&](Const::TransformAttrInterface attr) {
            newTransformations.pop_back();
            newTransformations.push_back(attr);
            wereTransformationsFused = true;
        };

        for (const auto attr : transformations) {
            if (newTransformations.empty()) {
                newTransformations.push_back(attr);
                continue;
            }

            if (mlir::isa<Const::SubViewAttr>(attr) && mlir::isa<Const::SubViewAttr>(newTransformations.back())) {
                auto firstAttr = mlir::cast<Const::SubViewAttr>(newTransformations.back());
                auto secondAttr = mlir::cast<Const::SubViewAttr>(attr);
                auto firstOffset = parseIntArrayAttr<int64_t>(firstAttr.getOffset());
                auto newOffset = parseIntArrayAttr<int64_t>(secondAttr.getOffset());
                for (auto i : irange(newOffset.size())) {
                    newOffset[i] += firstOffset[i];
                }
                auto newSubViewAttr =
                        Const::SubViewAttr::get(getIntArrayAttr(getContext(), newOffset), secondAttr.getShape());
                fuseTransformations(newSubViewAttr);
            } else if (mlir::isa<Const::AddAttr>(attr) && mlir::isa<Const::AddAttr>(newTransformations.back())) {
                auto firstAttr = mlir::cast<Const::AddAttr>(newTransformations.back());
                auto secondAttr = mlir::cast<Const::AddAttr>(attr);
                auto newBias = firstAttr.getBias().getValueAsDouble() + secondAttr.getBias().getValueAsDouble();
                auto newAddAttr = Const::AddAttr::get(getFPAttr(getContext(), newBias));
                fuseTransformations(newAddAttr);
            } else if (mlir::isa<Const::RescaleAttr>(attr) &&
                       mlir::isa<Const::RescaleAttr>(newTransformations.back())) {
                auto firstAttr = mlir::cast<Const::RescaleAttr>(newTransformations.back());
                auto secondAttr = mlir::cast<Const::RescaleAttr>(attr);
                auto newScale = firstAttr.getScale().getValueAsDouble() * secondAttr.getScale().getValueAsDouble();
                auto newRescaleAttr = Const::RescaleAttr::get(getFPAttr(getContext(), newScale));
                fuseTransformations(newRescaleAttr);
            } else if ((mlir::isa<Const::ReshapeAttr>(attr) &&
                        mlir::isa<Const::ReshapeAttr>(newTransformations.back())) ||
                       (mlir::isa<Const::ReorderAttr>(attr) &&
                        mlir::isa<Const::ReorderAttr>(newTransformations.back()))) {
                fuseTransformations(attr);
            } else {
                newTransformations.push_back(attr);
            }
        }

        if (!wereTransformationsFused) {
            return mlir::failure();
        }

        const auto newContentAttr = Const::ContentAttr::get(contentAttr.getBaseContent(), newTransformations);
        rewriter.replaceOpWithNewOp<vpux::Const::DeclareOp>(constOp, constOp.getType(), newContentAttr);

        sendEquivalenceRequest(contentAttr, newContentAttr);

        return mlir::success();
    }
};

/*
 * Finds compatible transformations that are followed by SubView and swaps them. Transformations are considered
 * compatible if they perform element-wise computation, only change the metadata of the constant or the information
 * of the transformation can be reconstructed when moving SubView before. For example:
 *     Add + SubView => SubView + Add
 *
 * The benefit of this change is that less computation and memory are necessary when folding constants.
 */
class MoveSubViewBefore final : public mlir::OpRewritePattern<Const::DeclareOp> {
public:
    MoveSubViewBefore(mlir::MLIRContext* ctx): mlir::OpRewritePattern<Const::DeclareOp>(ctx) {
    }

    mlir::LogicalResult matchAndRewrite(Const::DeclareOp constOp, mlir::PatternRewriter& rewriter) const override {
        auto contentAttr = constOp.getContentAttr();
        auto transformations = contentAttr.getTransformations();
        if (transformations.empty()) {
            return mlir::failure();
        }

        const auto hasCandidate = [&]() {
            for (size_t idx = 0; idx < transformations.size() - 1; idx++) {
                if (mlir::isa<Const::SubViewAttr>(transformations[idx + 1]) &&
                    mlir::isa<Const::AddAttr, Const::RescaleAttr, Const::ConvertElemTypeAttr, Const::DequantizeAttr,
                              Const::ReorderAttr, Const::QuantCastAttr, Const::TransposeAttr, Const::ReshapeAttr,
                              Const::ChangeShapeAndElemTypeAttr, Const::RelocateWeightsTableAttr>(
                            transformations[idx])) {
                    return true;
                }
            }
            return false;
        };
        if (!hasCandidate()) {
            return mlir::failure();
        }

        bool wereTransformationsSwapped = false;

        auto newTransformations = to_small_vector(transformations);
        auto inputType = contentAttr.getBaseContent().getType().cast<NDTypeInterface>();
        const auto transformationsCount = transformations.size();
        for (size_t idx = 0; idx < transformationsCount - 1; idx++) {
            auto nextTransformation = newTransformations[idx + 1];
            if (!mlir::isa<Const::SubViewAttr>(nextTransformation)) {
                inputType = newTransformations[idx].inferOutputType(inputType);
                continue;
            }

            const auto subViewAttr = mlir::cast<Const::SubViewAttr>(nextTransformation);
            const Shape offset(parseIntArrayAttr<int64_t>(subViewAttr.getOffset()));
            const Shape shape(parseIntArrayAttr<int64_t>(subViewAttr.getShape()));

            auto currentTransformation = newTransformations[idx];
            if (mlir::isa<Const::AddAttr, Const::RescaleAttr, Const::ConvertElemTypeAttr, Const::DequantizeAttr,
                          Const::ReorderAttr>(currentTransformation)) {
                std::swap(newTransformations[idx], newTransformations[idx + 1]);
                wereTransformationsSwapped = true;
                inputType = newTransformations[idx].inferOutputType(inputType);
            } else if (auto quantCastAttr = mlir::dyn_cast<Const::QuantCastAttr>(currentTransformation)) {
                if (const auto perAxisType = mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>(
                            quantCastAttr.getElemType())) {
                    const auto newElemType = tileScalesAndZP(perAxisType, shape, offset);
                    newTransformations[idx] = Const::QuantCastAttr::get(quantCastAttr.getContext(), newElemType);
                }

                std::swap(newTransformations[idx], newTransformations[idx + 1]);
                wereTransformationsSwapped = true;
                inputType = newTransformations[idx].inferOutputType(inputType);
            } else if (auto transposeAttr = mlir::dyn_cast<Const::TransposeAttr>(currentTransformation)) {
                const auto order = DimsOrder::fromAffineMap(transposeAttr.getOrder().getValue());

                SmallVector<int64_t> newOffset(offset.size());
                SmallVector<int64_t> newShape(shape.size());
                for (size_t idx = 0; idx < newShape.size(); idx++) {
                    newOffset[order.dimAt(idx).ind()] = offset.raw()[idx];
                    newShape[order.dimAt(idx).ind()] = shape.raw()[idx];
                }
                newTransformations[idx + 1] = Const::SubViewAttr::get(getIntArrayAttr(getContext(), newOffset),
                                                                      getIntArrayAttr(getContext(), newShape));

                std::swap(newTransformations[idx], newTransformations[idx + 1]);
                wereTransformationsSwapped = true;
                inputType = newTransformations[idx].inferOutputType(inputType);
            } else if (auto reshapeAttr = mlir::dyn_cast<Const::ReshapeAttr>(currentTransformation)) {
                const auto reshapeInput = inputType.getShape();
                const auto reshapeOutput = parseIntArrayAttr<int64_t>(reshapeAttr.getShape());
                const auto newSubViewOutput =
                        sliceInputShape(reshapeInput.raw(), reshapeOutput, offset.raw(), shape.raw());
                if (mlir::failed(newSubViewOutput)) {
                    inputType = newTransformations[idx].inferOutputType(inputType);
                    continue;
                }

                newTransformations[idx] = Const::ReshapeAttr::get(subViewAttr.getShape());
                const auto newOffset = getIntArrayAttr(getContext(), newSubViewOutput->first);
                const auto newShape = getIntArrayAttr(getContext(), newSubViewOutput->second);
                newTransformations[idx + 1] = Const::SubViewAttr::get(newOffset, newShape);

                std::swap(newTransformations[idx], newTransformations[idx + 1]);
                wereTransformationsSwapped = true;
                inputType = newTransformations[idx].inferOutputType(inputType);
            } else if (auto changeShapeAttr =
                               mlir::dyn_cast<Const::ChangeShapeAndElemTypeAttr>(currentTransformation)) {
                const auto reshapeInput = inputType.getShape();
                const auto reshapeOutput = parseIntArrayAttr<int64_t>(changeShapeAttr.getShape());
                const auto newSubViewOutput =
                        sliceInputShape(reshapeInput.raw(), reshapeOutput, offset.raw(), shape.raw());
                if (mlir::failed(newSubViewOutput)) {
                    inputType = newTransformations[idx].inferOutputType(inputType);
                    continue;
                }

                auto outputType = newTransformations[idx].inferOutputType(inputType);
                outputType = newTransformations[idx + 1].inferOutputType(outputType);

                const auto newOffset = getIntArrayAttr(getContext(), newSubViewOutput->first);
                const auto newShape = getIntArrayAttr(getContext(), newSubViewOutput->second);
                newTransformations[idx + 1] = Const::SubViewAttr::get(newOffset, newShape);

                auto newTrShape = getIntArrayAttr(getContext(), outputType.getShape().raw());
                newTransformations[idx] =
                        Const::ChangeShapeAndElemTypeAttr::get(newTrShape, outputType.getElementType());

                std::swap(newTransformations[idx], newTransformations[idx + 1]);
                wereTransformationsSwapped = true;
                inputType = newTransformations[idx].inferOutputType(inputType);
            } else if (auto relocateAttr = mlir::dyn_cast<Const::RelocateWeightsTableAttr>(currentTransformation)) {
                if (mlir::failed(prepareRelocateWeightsTableSwap(relocateAttr, idx, newTransformations, inputType))) {
                    inputType = transformations[idx].inferOutputType(inputType);
                    continue;
                }
                std::swap(newTransformations[idx], newTransformations[idx + 1]);
                wereTransformationsSwapped = true;
                inputType = newTransformations[idx].inferOutputType(inputType);
            } else {
                inputType = newTransformations[idx].inferOutputType(inputType);
            }
        }

        if (!wereTransformationsSwapped) {
            return mlir::failure();
        }

        const auto newContentAttr = Const::ContentAttr::get(contentAttr.getBaseContent(), newTransformations);
        rewriter.replaceOpWithNewOp<vpux::Const::DeclareOp>(constOp, constOp.getType(), newContentAttr);

        sendEquivalenceRequest(contentAttr, newContentAttr);

        return mlir::success();
    }

private:
    mlir::FailureOr<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>> sliceInputShape(
            ArrayRef<int64_t> reshapeInput, ArrayRef<int64_t> reshapeOutput, ArrayRef<int64_t> subViewOffset,
            ArrayRef<int64_t> subViewShape) const {
        SmallVector<int64_t> newOffset(reshapeInput.size(), 0);
        SmallVector<int64_t> newShape(reshapeInput);

        int64_t inputLowerDimsSize = 1;
        int64_t outputLowerDimsSize = 1;

        auto inputDim = static_cast<int64_t>(reshapeInput.size() - 1);
        for (auto outputDim = static_cast<int64_t>(reshapeOutput.size() - 1); outputDim >= 0; --outputDim) {
            const auto isDimSliced = subViewOffset[outputDim] > 0 || subViewShape[outputDim] < reshapeOutput[outputDim];
            if (!isDimSliced) {
                outputLowerDimsSize *= reshapeOutput[outputDim];
                continue;
            }
            for (; inputDim >= 0; --inputDim) {
                if (reshapeInput[inputDim] == reshapeOutput[outputDim] && inputLowerDimsSize == outputLowerDimsSize) {
                    break;
                }
                inputLowerDimsSize *= reshapeInput[inputDim];
            }
            if (inputDim < 0 || reshapeInput[inputDim] != reshapeOutput[outputDim] ||
                inputLowerDimsSize != outputLowerDimsSize) {
                return mlir::failure();
            }
            newOffset[inputDim] = subViewOffset[outputDim];
            newShape[inputDim] = subViewShape[outputDim];

            outputLowerDimsSize *= reshapeOutput[outputDim];
        }
        return std::make_pair(newOffset, newShape);
    }

    mlir::LogicalResult prepareRelocateWeightsTableSwap(Const::RelocateWeightsTableAttr relocateAttr,
                                                        size_t relocateIdx,
                                                        SmallVector<Const::TransformAttrInterface>& transformations,
                                                        NDTypeInterface relocateInputType) const {
        const auto subViewIdx = relocateIdx + 1;
        const auto subViewAttr = mlir::cast<Const::SubViewAttr>(transformations[subViewIdx]);
        const Shape offset(parseIntArrayAttr<int64_t>(subViewAttr.getOffset()));
        const Shape shape(parseIntArrayAttr<int64_t>(subViewAttr.getShape()));

        // More than one channel must be present for the transformation to deduce the weights pointer step
        const auto subviewSize = shape.front();
        if (subviewSize <= 1) {
            return mlir::failure();
        }

        const auto relocateOutputType = transformations[relocateIdx].inferOutputType(relocateInputType);
        const auto relocateOutputShape = relocateOutputType.getShape();

        const auto isSlicedOverFirstDim = [&]() {
            for (auto outputDim = static_cast<int64_t>(shape.size() - 1); outputDim >= 0; --outputDim) {
                const auto isDimSliced =
                        offset.raw()[outputDim] > 0 || shape.raw()[outputDim] < relocateOutputShape.raw()[outputDim];
                if (isDimSliced) {
                    if (outputDim != 0) {
                        return false;
                    }
                }
            }
            return true;
        }();
        if (!isSlicedOverFirstDim) {
            return mlir::failure();
        }

        const auto totalChannels = relocateOutputShape.front();
        const auto weightsTableByteSize = relocateAttr.getWeightsTableSize().getInt();
        const auto weightsTableNumElems = weightsTableByteSize / sizeof(int32_t);
        const auto tableEntrySize = weightsTableNumElems / totalChannels;

        const auto subviewOffset = offset.front();

        const auto clusterOffsets = parseIntArrayAttr<int64_t>(relocateAttr.getOffsets());
        const auto areClustersDifferent = std::adjacent_find(clusterOffsets.begin(), clusterOffsets.end(),
                                                             std::not_equal_to<>()) != clusterOffsets.end();
        SmallVector<int32_t> newWeightsPtrs = {};
        SmallVector<int64_t> newClusterOffsets(clusterOffsets);
        mlir::IntegerAttr newChannelOffsetAttr;
        const auto origChannelOffset =
                relocateAttr.getChannelOffset() != nullptr ? relocateAttr.getChannelOffset().getInt() : 0;
        if (areClustersDifferent) {
            size_t clusterIdx = 0;

            // Ensure only the values for one cluster are sliced
            bool onlyOneClusterSliced = [&]() {
                const auto offsetIt = llvm::find(clusterOffsets, subviewOffset);
                if (offsetIt == clusterOffsets.end()) {
                    return false;
                }
                if (offsetIt + 1 != clusterOffsets.end()) {
                    const auto nextOffset = *(offsetIt + 1);
                    if ((nextOffset - subviewOffset) != subviewSize) {
                        return false;
                    }
                }
                clusterIdx = static_cast<size_t>(std::distance(clusterOffsets.begin(), offsetIt));
                return true;
            }();
            if (!onlyOneClusterSliced) {
                return mlir::failure();
            }

            const auto weightsPtrs = parseIntArrayAttr<int32_t>(relocateAttr.getWeightsPtr());
            if (clusterIdx >= weightsPtrs.size()) {
                return mlir::failure();
            }
            newWeightsPtrs = {weightsPtrs[clusterIdx]};
            newClusterOffsets = {0};
            newChannelOffsetAttr = getIntAttr(getContext(), origChannelOffset);
        } else {
            newWeightsPtrs = parseIntArrayAttr<int32_t>(relocateAttr.getWeightsPtr());
            newChannelOffsetAttr = getIntAttr(getContext(), origChannelOffset + subviewOffset);
        }

        const auto newWeightsPtrsAttr = getIntArrayAttr(getContext(), newWeightsPtrs);
        const auto newClusterOffsetsAttr = getIntArrayAttr(getContext(), newClusterOffsets);

        const auto newTableByteSize = tableEntrySize * subviewSize * sizeof(int32_t);
        const auto newTableByteSizeAttr = getIntAttr(getContext(), newTableByteSize);

        const auto newWeightsElemBitSizeAttr = relocateAttr.getWeightsElemBitSize();

        auto newWeightsCompressionAttr = relocateAttr.getWeightsCompression();
        if (newWeightsCompressionAttr != nullptr) {
            if (newWeightsCompressionAttr.getAxis().getInt() != 0) {
                return mlir::failure();
            }
            const auto numElems = to_small_vector(newWeightsCompressionAttr.getNumElems().getValues<int64_t>());
            const auto newNumElems = SmallVector<int64_t>(numElems.begin() + subviewOffset,
                                                          numElems.begin() + subviewOffset + subviewSize);
            const auto numElemsType =
                    mlir::RankedTensorType::get({static_cast<int64_t>(newNumElems.size())}, getInt64Type(getContext()));
            const auto newNumElemsAttr = mlir::DenseElementsAttr::get(numElemsType, ArrayRef(newNumElems));
            newWeightsCompressionAttr =
                    VPUIP::SparsityCompressionAttr::get(getContext(), newWeightsCompressionAttr.getAxis(),
                                                        newNumElemsAttr, newWeightsCompressionAttr.getAlignment());
        }

        transformations[relocateIdx] = Const::RelocateWeightsTableAttr::get(
                newWeightsPtrsAttr, relocateAttr.getSparsityPtr(), newClusterOffsetsAttr, newTableByteSizeAttr,
                newWeightsElemBitSizeAttr, newWeightsCompressionAttr, newChannelOffsetAttr);

        return mlir::success();
    }
};

/*
 * Finds compatible transformations that are followed by Reshape and swaps them. Although the Reshape transformation
 * does not do any computation, moving it before other transformations allows the possibility for other optimizations to
 * be done. For example, in the following pattern, the Reshape is moved before SubView:
 *     Add + Reshape + SubView => Reshape + Add + SubView
 * This allows the possibility of also moving SubView before Add, so that Add only computes the relevant slice of data:
 *     Add + Reshape + SubView => Reshape + SubView + Add
 */
class MoveReshapeBefore final : public mlir::OpRewritePattern<Const::DeclareOp> {
public:
    MoveReshapeBefore(mlir::MLIRContext* ctx): mlir::OpRewritePattern<Const::DeclareOp>(ctx) {
    }

public:
    mlir::LogicalResult matchAndRewrite(Const::DeclareOp constOp, mlir::PatternRewriter& rewriter) const override {
        auto contentAttr = constOp.getContentAttr();
        auto transformations = contentAttr.getTransformations();

        if (transformations.empty()) {
            return mlir::failure();
        }

        const auto getNewQuantizationDim = [](int32_t dim, ArrayRef<int64_t> inputShape,
                                              ArrayRef<int64_t> outputShape) -> mlir::FailureOr<int32_t> {
            const auto inputLowerDimsSize =
                    std::accumulate(inputShape.begin() + dim, inputShape.end(), int64_t(1), std::multiplies<int64_t>());
            int64_t outputLowerDimsSize = 1;
            auto newDim = static_cast<int32_t>(outputShape.size() - 1);
            for (; newDim >= 0; --newDim) {
                outputLowerDimsSize *= outputShape[newDim];
                if (outputLowerDimsSize >= inputLowerDimsSize) {
                    break;
                }
            }
            if (outputLowerDimsSize == inputLowerDimsSize && outputShape[newDim] == inputShape[dim]) {
                return newDim;
            }
            return mlir::failure();
        };

        const auto hasCandidate = [&]() {
            for (size_t idx = 0; idx < transformations.size() - 1; idx++) {
                if (mlir::isa<Const::ReshapeAttr>(transformations[idx + 1]) &&
                    mlir::isa<Const::AddAttr, Const::RescaleAttr, Const::ConvertElemTypeAttr, Const::DequantizeAttr>(
                            transformations[idx])) {
                    return true;
                }
            }
            return false;
        };
        if (!hasCandidate()) {
            return mlir::failure();
        }

        bool wereTransformationsSwapped = false;

        auto newTransformations = to_small_vector(transformations);
        auto inputType = contentAttr.getBaseContent().getType().cast<NDTypeInterface>();
        const auto transformationsCount = newTransformations.size();
        for (size_t idx = 0; idx < transformationsCount - 1; idx++) {
            auto nextTransformation = newTransformations[idx + 1];
            auto reshapeAttr = mlir::dyn_cast<Const::ReshapeAttr>(nextTransformation);
            if (reshapeAttr == nullptr) {
                inputType = newTransformations[idx].inferOutputType(inputType);
                continue;
            }

            auto currentTransformation = newTransformations[idx];
            if (mlir::isa<Const::AddAttr, Const::RescaleAttr, Const::ConvertElemTypeAttr>(currentTransformation)) {
                std::swap(newTransformations[idx], newTransformations[idx + 1]);
                wereTransformationsSwapped = true;
                inputType = newTransformations[idx].inferOutputType(inputType);
            } else if (mlir::isa<Const::DequantizeAttr>(currentTransformation)) {
                // If the input is quantized per-axis, check whether the axis is compatible with the new shape
                if (auto perAxisType =
                            mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(inputType.getElementType())) {
                    const auto dim = perAxisType.getQuantizedDimension();
                    const auto inputShape = inputType.getShape();
                    const auto outputShape = parseIntArrayAttr<int64_t>(reshapeAttr.getShape());
                    auto newDim = getNewQuantizationDim(dim, inputShape.raw(), outputShape);
                    if (mlir::failed(newDim)) {
                        inputType = newTransformations[idx].inferOutputType(inputType);
                        continue;
                    }
                    const auto newPerAxisType = changeAxis(perAxisType, newDim.value());
                    newTransformations[idx + 1] =
                            Const::ChangeShapeAndElemTypeAttr::get(reshapeAttr.getShape(), newPerAxisType);
                }
                std::swap(newTransformations[idx], newTransformations[idx + 1]);
                wereTransformationsSwapped = true;
                inputType = newTransformations[idx].inferOutputType(inputType);
            } else {
                inputType = newTransformations[idx].inferOutputType(inputType);
            }
        }

        if (!wereTransformationsSwapped) {
            return mlir::failure();
        }

        const auto newContentAttr = Const::ContentAttr::get(contentAttr.getBaseContent(), newTransformations);
        rewriter.replaceOpWithNewOp<vpux::Const::DeclareOp>(constOp, constOp.getType(), newContentAttr);

        sendEquivalenceRequest(contentAttr, newContentAttr);

        return mlir::success();
    }
};

/**
 * Removes the tiled information ('strides' attribute) from memref types of const.Declare operations. For example:
 *   Before:
 *     const.Declare memref<1x96x256x16xf16, {order = #NHWC, strides = [786432, 1, 1536, 96]}> = ...
 *   After:
 *     const.Declare memref<1x96x256x16xf16, #NHWC> = ...
 *
 * This is necessary because the previous behvaiour of this tiled info erasure relied on a "bug" in MLIR which
 * is fixed in LLVM18. This rewrite pattern is designed to mostly work in conjunction with the canonicalization
 * of SubView operations is expected to NOT work in certain situation (see
 * ./tests/lit/NPU/dialect/const/ops/invalid.mlir). At the moment only MemRefType subtypes (and not for example
 * VPUIP_Buffer) are supported because they implement the correct layout.
 *
 * See E-120399 for more information.
 *
 */
class EraseTiledInfo final : public mlir::OpRewritePattern<Const::DeclareOp> {
public:
    EraseTiledInfo(mlir::MLIRContext* ctx): mlir::OpRewritePattern<Const::DeclareOp>(ctx) {
    }

private:
    static bool hasStridesAttr(mlir::MemRefType type) {
        auto layout = type.getLayout();
        auto descAttr = mlir::dyn_cast<vpux::MemRefAttr>(layout);
        return descAttr != nullptr && descAttr.strides() != nullptr;
    }

public:
    mlir::LogicalResult matchAndRewrite(Const::DeclareOp constOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult EraseTiledInfo::matchAndRewrite(Const::DeclareOp constOp, mlir::PatternRewriter& rewriter) const {
    auto type = mlir::cast<vpux::NDTypeInterface>(constOp.getOutput().getType());

    if (mlir::isa<mlir::BaseMemRefType>(type)) {
        VPUX_THROW_WHEN(!mlir::isa<mlir::MemRefType>(type),
                        "Only mlir::MemRefType is supported for constants bufferization");
    } else {
        return mlir::failure();
    }

    // Only memref types can have a 'strides' attribute.
    auto memRefType = mlir::dyn_cast<mlir::MemRefType>(type);

    // If type is already free of 'strides' we ignore it and return failure to
    // avoid infinite loops.
    if (!hasStridesAttr(memRefType)) {
        return mlir::failure();
    }

    // Otherwise, create a new constant op with the new erased type.
    rewriter.replaceOpWithNewOp<vpux::Const::DeclareOp>(constOp, type.eraseTiledInfo(), constOp.getContentAttr());

    return mlir::success();
}

void vpux::Const::DeclareOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* ctx) {
    patterns.add<FuseConsecutiveTransformations>(ctx, vpux::benefitHigh);
    patterns.add<MoveSubViewBefore>(ctx);
    patterns.add<MoveReshapeBefore>(ctx);
    patterns.add<EraseTiledInfo>(ctx);
}

//
// setupExtraInterfaces
//

void Const::ConstDialect::setupExtraInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, mlir::BuiltinDialect*) {
        mlir::RankedTensorType::attachInterface<vpux::TensorNDTypeInterface>(*ctx);
        mlir::RankedTensorType::attachInterface<vpux::TensorBoundedTypeInterface>(*ctx);
        mlir::UnrankedTensorType::attachInterface<vpux::TensorNDTypeInterface>(*ctx);
        mlir::MemRefType::attachInterface<vpux::MemRefNDTypeInterface>(*ctx);
        mlir::UnrankedMemRefType::attachInterface<vpux::MemRefNDTypeInterface>(*ctx);
    });
}

//
// Generated
//

#include <vpux/compiler/dialect/const/dialect.cpp.inc>

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/const/ops.cpp.inc>
