//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/utils/const_logger.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/constant_folding_cache.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/AsmState.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/DialectResourceBlobManager.h>

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Transforms/InliningUtils.h>

#include <cstring>
#include <exception>
#include <numeric>
#include <utility>

using namespace vpux;

namespace {

//
// ConstInlinerInterface
//

struct ConstInlinerInterface : public mlir::DialectInlinerInterface {
    using DialectInlinerInterface::DialectInlinerInterface;

    bool isLegalToInline(mlir::Operation*, mlir::Operation*, bool) const final {
        return true;
    }

    bool isLegalToInline(mlir::Operation*, mlir::Region*, bool, mlir::IRMapping&) const final {
        return true;
    }

    bool isLegalToInline(mlir::Region*, mlir::Region*, bool, mlir::IRMapping&) const final {
        return true;
    }
};

}  // namespace

//
// Generated
//

#define GET_ATTRDEF_CLASSES
#include <vpux/compiler/dialect/const/attributes.cpp.inc>

//
// ConstDialect::initialize
//

void vpux::Const::ConstDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/const/ops.cpp.inc>
            >();

    addAttributes<
#define GET_ATTRDEF_LIST
#include <vpux/compiler/dialect/const/attributes.cpp.inc>
            >();

    addInterfaces<ConstInlinerInterface>();
}

//
// ContentAttr::verify
//

mlir::LogicalResult vpux::Const::ContentAttr::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                                     mlir::ElementsAttr baseContent,
                                                     vpux::Const::TransformAttrInterfaceArrayAttr transformations,
                                                     vpux::NDTypeInterface finalType) {
    std::ignore = finalType;  // automatically inferred

    if (baseContent == nullptr) {
        return printTo(emitError(), "Got NULL 'baseContent' in 'ContentAttr'");
    }

    if (!baseContent.getShapedType().getElementType().isIntOrFloat()) {
        return printTo(emitError(), "Got unsupported element type for 'baseContent' in 'ContentAttr' : '{0}'",
                       baseContent.getShapedType().getElementType());
    }

    if (baseContent.isa<mlir::DenseElementsAttr>()) {
        // OK
    } else if (const auto denseResource = baseContent.dyn_cast<mlir::DenseResourceElementsAttr>()) {
        // Note: manual checks required since dense resource blob is opaque and does not perform much validation itself
        const auto bytes = denseResource.getRawHandle().getBlob()->getData();
        bool ignored = false;
        if (!mlir::DenseElementsAttr::isValidRawBuffer(baseContent.getShapedType(), bytes, ignored)) {
            return printTo(emitError(),
                           "Size of dense resource buffer '{0}' in 'baseContent' doesn't match its type '{1}'",
                           bytes.size(), denseResource.getShapedType());
        }
    } else {
        return printTo(emitError(), "Got unsupported 'baseContent' in 'ContentAttr'");
    }

    const auto isValid = [](const vpux::Const::TransformAttrInterface& value) -> bool {
        return value != nullptr;
    };
    if (transformations && !llvm::all_of(transformations.getValue(), isValid)) {
        return printTo(emitError(), "Got invalid transformations attribute in 'ContentAttr'");
    }

    return mlir::success();
}

namespace {

std::pair<mlir::ArrayRef<char>, bool> detectSplatElementWise(mlir::ArrayRef<char> data, size_t elementSize) {
    VPUX_THROW_WHEN((data.size() < elementSize), "The data must contain at least one element");
    VPUX_THROW_WHEN(((data.size() % elementSize) != 0), "The data array has unexpected length");
    if (data.size() == elementSize) {
        return {data, true};
    }

    const char* firstElement = data.data();
    for (size_t i = elementSize; i < data.size(); i += elementSize) {
        if (std::memcmp(data.data() + i, firstElement, elementSize) != 0) {
            return {data, false};
        }
    }

    return {data.take_front(elementSize), true};
}

// Returns whether the data is a splat, correcting the data array when it is.
std::pair<mlir::ArrayRef<char>, bool> detectSplatManually(mlir::ShapedType type, mlir::ArrayRef<char> data) {
    if (data.empty()) {
        return {data, false};  // empty data is not a splat
    }

    // use isValidRawBuffer() for the side effects to detect whether a buffer is a splat
    bool isSplat = false;
    std::ignore = mlir::DenseElementsAttr::isValidRawBuffer(type, data, isSplat);
    if (isSplat) {
        return {data, true};
    }

    // isValidRawBuffer() only checks single-element splats but if the the data
    // array has identical elements, a manual check is required
    const vpux::Byte elemTypeSize = vpux::getElemTypeSize(type);
    return detectSplatElementWise(data, static_cast<size_t>(elemTypeSize.count()));
}

//
// wrapBaseContent
//

Const::Content wrapBaseContent(mlir::ElementsAttr baseContent) {
    ArrayRef<char> data = {};
    bool isSplat = false;

    if (const auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(baseContent)) {
        data = dense.getRawData();
        isSplat = dense.isSplat();
    } else {
        const auto denseResource = mlir::cast<mlir::DenseResourceElementsAttr>(baseContent);
        // dense resource doesn't support splat detection
        std::tie(data, isSplat) =
                detectSplatManually(baseContent.getShapedType(), denseResource.getRawHandle().getBlob()->getData());
    }

    return Const::Content::fromRawBuffer(baseContent.getShapedType().cast<vpux::NDTypeInterface>(), data,
                                         baseContent.getShapedType().getElementType(), isSplat);
}

}  // namespace

//
// ContentAttr::fold
//

Const::Content vpux::Const::ContentAttr::fold(bool bypassCache) const {
    auto baseContent = getBaseContent();

#ifdef BACKGROUND_FOLDING_ENABLED
    if (!bypassCache) {
        auto& cacheManager = Const::ConstantFoldingCacheManager::getInstance();
        auto ctx = baseContent.getContext();
        if (cacheManager.contains(ctx)) {
            auto& cache = cacheManager.get(ctx);
            auto content = cache.getContent(*this);
            if (content.has_value()) {
                return content.value();
            }
        }
    }
#else
    VPUX_UNUSED(bypassCache);
#endif

    auto res = wrapBaseContent(baseContent);

    for (const auto& attr : getTransformations()) {
        const auto storageElemTypeSize = vpux::getElemTypeSize(res.getStorageElemType()).count();
        VPUX_THROW_WHEN(storageElemTypeSize < CHAR_BIT && !attr.supportsSubByteStorageType(),
                        "Unsupported storage type of size '{0}' bits.", storageElemTypeSize);
        Const::logger().trace("Applying transformation: {0}", attr);
        res = attr.transform(res);
    }

    return res;
}

//
// ContentAttr::getBaseContent
//

mlir::ElementsAttr vpux::Const::ContentAttr::getBaseContent() const {
    return getImpl()->baseContent;
}

//
// ContentAttr::getTransformations
//

mlir::ArrayRef<Const::TransformAttrInterface> vpux::Const::ContentAttr::getTransformations() const {
    if (const auto transformations = getImpl()->transformations) {
        return transformations.getValue();
    }

    return {};
}

//
// ContentAttr::getType
//

vpux::NDTypeInterface vpux::Const::ContentAttr::getType() const {
    return getImpl()->finalType;
}

//
// ContentAttr::print
//

void vpux::Const::ContentAttr::print(mlir::AsmPrinter& printer) const {
    printer.printAttribute(getBaseContent());
    if (const auto transformations = getTransformations(); !transformations.empty()) {
        printer << ", " << '[' << transformations << ']';
    }
}

//
// ContentAttr::parse
//

mlir::Attribute vpux::Const::ContentAttr::parse(mlir::AsmParser& parser, mlir::Type) {
    mlir::ElementsAttr baseContent;
    if (mlir::failed(parser.parseAttribute(baseContent))) {
        return nullptr;
    }

    if (mlir::succeeded(parser.parseOptionalComma())) {
        mlir::ArrayAttr arrayAttr;
        if (mlir::failed(parser.parseAttribute(arrayAttr))) {
            return nullptr;
        }

        mlir::SmallVector<vpux::Const::TransformAttrInterface> transformations;
        transformations.reserve(arrayAttr.size());
        for (const auto attr : arrayAttr.getValue()) {
            const auto trAttr = attr.dyn_cast<Const::TransformAttrInterface>();
            VPUX_THROW_WHEN(trAttr == nullptr, "Got non transformation attribute : '{0}'", attr);
            transformations.push_back(trAttr);
        }

        return parser.getChecked<Const::ContentAttr>(baseContent, ArrayRef(transformations));
    }

    return parser.getChecked<Const::ContentAttr>(baseContent);
}

// The list of transformations can have the following position requirements if all types are present:
//   [NONE]* -> [PREFERRED_LAST]* -> [LAST]
// No two transformations with the LAST requirement can exist.
// The order of elements with the same requirement is stable.
Const::TransformAttrInterface* getInsertionPosition(SmallVector<Const::TransformAttrInterface>& transformations,
                                                    Const::TransformAttrInterface newTransformation) {
    auto endPosition = transformations.end();
    if (transformations.empty()) {
        return endPosition;
    }

    const auto lastTransformation = transformations.back();

    const auto newTransformationReq = newTransformation.getPositionRequirement();
    const auto lastTransformationReq = lastTransformation.getPositionRequirement();

    const auto newTransformationReqLast = newTransformationReq == vpux::Const::details::PositionRequirement::LAST;
    const auto lastTransformationReqLast = lastTransformationReq == vpux::Const::details::PositionRequirement::LAST;
    VPUX_THROW_WHEN(newTransformationReqLast && lastTransformationReqLast,
                    "Existing transformation with LAST position requirement");

    if (newTransformationReqLast) {
        return endPosition;
    }

    auto it = transformations.end();
    for (; it > transformations.begin(); --it) {
        const auto transformationReq = (it - 1)->getPositionRequirement();
        if (transformationReq == vpux::Const::details::PositionRequirement::NONE ||
            (transformationReq == vpux::Const::details::PositionRequirement::PREFERRED_LAST &&
             newTransformationReq == vpux::Const::details::PositionRequirement::PREFERRED_LAST)) {
            return it;
        }
    }
    return it;
}

//
// ContentAttr::addTransformation
//

Const::ContentAttr Const::ContentAttr::addTransformation(Const::ContentAttr input,
                                                         Const::TransformAttrInterface newTransformation) {
    auto transformations = to_small_vector(input.getTransformations());
    auto insertionPosition = getInsertionPosition(transformations, newTransformation);
    insertionPosition = transformations.insert(insertionPosition, newTransformation);

    // Update all transformations attributes starting from inserted transformation
    auto baseContent = input.getBaseContent();
    for (auto it = insertionPosition; it != transformations.end(); it++) {
        SmallVector<mlir::Attribute> prevTransformations(transformations.begin(), it + 1);
        auto updatedTransformation = it->updateAttributes(baseContent, prevTransformations);
        if (updatedTransformation != nullptr) {
            *it = updatedTransformation;
        }
    }

    auto newContentAttr = Const::ContentAttr::get(baseContent, ArrayRef(transformations));

#ifdef BACKGROUND_FOLDING_ENABLED
    auto& cacheManager = Const::ConstantFoldingCacheManager::getInstance();
    auto ctx = newContentAttr.getContext();
    if (cacheManager.contains(ctx)) {
        auto& cache = cacheManager.get(ctx);
        cache.enqueueRequest(Const::FoldingRequest{newContentAttr, newTransformation});
    }
#endif

    return newContentAttr;
}

// Returns a ContentAttr containing a subset of the transformations from the original ContentAttr. The subset is
// determined by the transformation given as a parameter to the function, so that all the transformations from the list
// are included up to the one given as a parameter.
//
// For example:
// Original ContentAttr: [Transformation1, Transformation2, Transformation3, Transformation4]
// If Transformation3 is given as a parameter, a ContentAttr containing [Transformation1, Transformation2] is returned
// Note: in case there are multiple instances of Transformation3 in the list, which have the same underlying
// parameters, the rightmost one will be used
Const::ContentAttr Const::ContentAttr::stripTransformationsFrom(Const::TransformAttrInterface transformation) {
    const auto transformations = getTransformations();
    const auto rIt = std::find(transformations.rbegin(), transformations.rend(), transformation);
    if (rIt == transformations.rend()) {
        return nullptr;
    }
    auto it = (rIt + 1).base();
    const auto headTransformations = SmallVector<Const::TransformAttrInterface>(transformations.begin(), it);
    return Const::ContentAttr::get(getBaseContent(), ArrayRef(headTransformations));
}

// Returns the list of transformations of the ContentAttr starting from the transformation given as a parameter.
//
// For example:
// ContentAttr: [Transformation1, Transformation2, Transformation3, Transformation4]
// If Transformation3 is given as a parameter, a list containing [Transformation3, Transformation4] is returned
// Note: in case there are multiple instances of NewTransformation in the list, which have the same underlying
// parameters, the rightmost one will be used
SmallVector<Const::TransformAttrInterface> Const::ContentAttr::getLastTransformationsFrom(
        Const::TransformAttrInterface transformation) {
    const auto transformations = getTransformations();
    const auto rIt = std::find(transformations.rbegin(), transformations.rend(), transformation);
    if (rIt == transformations.rend()) {
        return {};
    }
    auto it = (rIt + 1).base();
    return SmallVector<Const::TransformAttrInterface>(it, transformations.end());
}

//
// ContentAttr::inferFinalType
//

vpux::NDTypeInterface vpux::Const::ContentAttr::inferFinalType(
        mlir::ElementsAttr content, mlir::ArrayRef<vpux::Const::TransformAttrInterface> transformations) {
    auto inferredType = content.getType().cast<vpux::NDTypeInterface>();
    for (const auto& attr : transformations) {
        inferredType = attr.inferOutputType(inferredType);
    }
    return inferredType;
}
