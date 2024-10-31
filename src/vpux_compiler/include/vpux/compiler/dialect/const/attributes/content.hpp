//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/const/attr_interfaces.hpp"
#include "vpux/compiler/dialect/const/utils/content.hpp"
#include "vpux/compiler/dialect/const/utils/resource_management.hpp"
#include "vpux/utils/core/func_ref.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>

namespace vpux::Const {

class ContentAttr;

class ContentSetup {
    mlir::ElementsAttr _baseContent;
    SmallVector<TransformAttrInterface> _transformations;

    // Require the user to explicitly use clone() when he/she desires a copy.
    ContentSetup(const ContentSetup&) = default;
    ContentSetup& operator=(const ContentSetup&) = default;

public:
    ContentSetup() = default;
    ~ContentSetup() = default;
    ContentSetup(ContentSetup&& other);
    ContentSetup& operator=(ContentSetup&& other);

    // This constructor throws an exception when _baseContent is null.
    ContentSetup(mlir::ElementsAttr baseContent, ArrayRef<TransformAttrInterface> transformations = {});

    // getters
    mlir::MLIRContext* getContext() const {
        return _baseContent.getContext();
    }

    mlir::ElementsAttr getBaseContent() const {
        return _baseContent;
    }

    ArrayRef<TransformAttrInterface> getTransformations() const {
        return _transformations;
    }

    ContentSetup clone() const {
        checkInvalidated();
        return *this;
    }

    // transformations
    [[nodiscard]] ContentSetup addTransformation(TransformAttrInterface newTransformation);

    // implemented by <concrete transformation attribute>.cpp
    [[nodiscard]] ContentSetup broadcast(Dim axis, int64_t value);
    [[nodiscard]] ContentSetup castElemType(mlir::Type newElemType);
    [[nodiscard]] ContentSetup convertElemType(mlir::Type newElemType);
    [[nodiscard]] ContentSetup quantCast(mlir::Type newElemType);
    [[nodiscard]] ContentSetup dequantize();
    [[nodiscard]] ContentSetup rescale(double scale);
    [[nodiscard]] ContentSetup relocateWeightsTablePointers(ArrayRef<uint32_t> weightsPtr, uint64_t sparsityPtr,
                                                            vpux::ShapeRef offsets, uint64_t weightsTableSize,
                                                            uint64_t weightsElemBitSize,
                                                            VPUIP::SparsityCompressionAttr weightsCompression,
                                                            uint64_t channelOffset);
    [[nodiscard]] ContentSetup swizzleConstant(uint64_t swizzleKey, uint64_t arch);
    [[nodiscard]] ContentSetup add(double bias);
    [[nodiscard]] ContentSetup reshape(vpux::ShapeRef newShape);
    [[nodiscard]] ContentSetup reverse(Dim axis);
    [[nodiscard]] ContentSetup reorder(vpux::DimsOrder newOrder);
    [[nodiscard]] ContentSetup padWithZero(vpux::ShapeRef padBefore, vpux::ShapeRef padAfter);
    [[nodiscard]] ContentSetup subview(vpux::ShapeRef offset, vpux::ShapeRef shape);
    [[nodiscard]] ContentSetup bitPack(int64_t width);
    [[nodiscard]] ContentSetup transpose(vpux::DimsOrder newOrder);
    [[nodiscard]] ContentSetup memPermute(vpux::DimsOrder dstOrder, vpux::DimsOrder memPerm);
    [[nodiscard]] ContentSetup layoutCast(vpux::DimsOrder dstOrder);
    [[nodiscard]] ContentSetup expandDilated(vpux::ShapeRef dilations);
    [[nodiscard]] ContentSetup getSparsityMap();
    [[nodiscard]] ContentSetup sparsify(bool compressOutputType, mlir::ElementsAttr numActualElements = nullptr);
    [[nodiscard]] ContentSetup changeShapeAndElemType(vpux::ShapeRef newShape, mlir::Type newElemType);
    [[nodiscard]] ContentSetup scalarMultInverse();
    [[nodiscard]] ContentSetup fuse(mlir::RankedTensorType fusedTensorType, ContentAttr weightsTable,
                                    ContentAttr weights, ContentAttr sparsity, ContentAttr activations);
    [[nodiscard]] ContentSetup quantize(mlir::quant::QuantizedType newElemType);

    // get
    [[nodiscard]] ContentAttr get();

private:
    bool isInvalidated() const {
        return _baseContent == nullptr;
    }

    // This function throws if a move from this class instance happened. This is the case when baseContent is null.
    void checkInvalidated() const;
};

// Returns the output type "as if" the transformations were applied to a tensor of type contentType.
vpux::NDTypeInterface inferFinalType(vpux::NDTypeInterface contentType,
                                     mlir::ArrayRef<TransformAttrInterface> transformations);
// Returns the output type and splatness of the content with transformations "as
// if" applied to this content.
// Use inferFinalType() instead if you are only interested in the type.
std::pair<vpux::NDTypeInterface, bool> inferFinalTypeAndSplat(mlir::ElementsAttr content,
                                                              mlir::ArrayRef<TransformAttrInterface> transformations);

}  // namespace vpux::Const

//
// Generated
//

#define GET_ATTRDEF_CLASSES
#include <vpux/compiler/dialect/const/attributes.hpp.inc>

namespace vpux::Const {
class ContentAttr {
    mlir::ElementsAttr _baseContent = nullptr;
    TransformAttrInterfaceArrayAttr _transformations = nullptr;
    NDTypeInterface _finalType = {};
    bool _isSplat = false;

public:
    friend bool operator==(const ContentAttr& x, const ContentAttr& y) {
        // Note: only base content and transformations are non-inferred fields -
        // if they differ, other parameters also differ.
        return x._baseContent == y._baseContent && x._transformations == y._transformations;
    }

    friend bool operator!=(const ContentAttr& x, const ContentAttr& y) {
        return !(x == y);
    }

    friend void swap(ContentAttr& x, ContentAttr& y) {
        using std::swap;
        swap(x._baseContent, y._baseContent);
        swap(x._transformations, y._transformations);
        swap(x._finalType, y._finalType);
        swap(x._isSplat, y._isSplat);
    }

    friend llvm::hash_code hash_value(const ContentAttr& x) {
        using ::llvm::hash_value;
        // Note: only base content and transformations are non-inferred fields -
        // if they differ, other parameters also differ.
        return llvm::hash_combine(hash_value(x._baseContent), hash_value(x._transformations));
    }

    // Compatibility APIs for current users
    static ContentAttr get(mlir::ElementsAttr base, ArrayRef<TransformAttrInterface> transformations = {});
    static ContentAttr getChecked(FuncRef<mlir::InFlightDiagnostic()> emitError, mlir::ElementsAttr base,
                                  ArrayRef<TransformAttrInterface> transformations = {});
    static mlir::LogicalResult verify(FuncRef<mlir::InFlightDiagnostic()> emitError, mlir::ElementsAttr baseContent,
                                      ArrayRef<TransformAttrInterface> transformations = {});

    mlir::MLIRContext* getContext() const {
        return _baseContent.getContext();
    }

    operator mlir::OpFoldResult() const {
        return EphemeralContentAttr::get(_baseContent.getContext(), _baseContent, _transformations);
    }

    friend bool operator==(const ContentAttr& x, std::nullptr_t) {
        return x._baseContent == nullptr;
    }
    friend bool operator!=(const ContentAttr& x, std::nullptr_t) {
        return !(x == nullptr);
    }

    // Existing APIs
    vpux::Const::Content fold(bool bypassCache = false) const;

    mlir::ElementsAttr getBaseContent() const {
        return _baseContent;
    }
    mlir::ArrayRef<vpux::Const::TransformAttrInterface> getTransformations() const {
        return _transformations == nullptr ? mlir::ArrayRef<vpux::Const::TransformAttrInterface>{}
                                           : _transformations.getValue();
    }
    vpux::NDTypeInterface getType() const {
        return _finalType;
    }
    bool isSplat() const {
        return _isSplat;
    }

    // ContentSetup interface
    vpux::Const::ContentSetup transform() const {
        return ContentSetup(getBaseContent(), getTransformations());
    }

    static vpux::Const::ContentSetup transform(
            mlir::ElementsAttr baseContent, llvm::ArrayRef<vpux::Const::TransformAttrInterface> transformations = {}) {
        return ContentSetup(baseContent, transformations);
    }

    // Parsing & printing
    void print(mlir::AsmPrinter&) const;
    static mlir::FailureOr<ContentAttr> parse(mlir::AsmParser&);
};

// Helper methods for tablegen's property boilerplate
mlir::LogicalResult convertFromAttribute(ContentAttr& x, mlir::Attribute attr,
                                         llvm::function_ref<mlir::InFlightDiagnostic()> emitError);
mlir::Attribute convertToAttribute(mlir::MLIRContext* ctx, const ContentAttr& x);
mlir::LogicalResult readFromMlirBytecode(mlir::DialectBytecodeReader&, ContentAttr&);
void writeToMlirBytecode(mlir::DialectBytecodeWriter&, const ContentAttr&);

// Default custom<ContentAttr> parsing & printing
mlir::ParseResult parseContentAttr(mlir::AsmParser& parser, ContentAttr& content);
void printContentAttr(mlir::AsmPrinter& printer, const ContentAttr& content);
}  // namespace vpux::Const
