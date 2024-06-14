//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"

//
// BoundedBuffer
//

namespace vpux {

ShapeRef VPUIP::BoundedBufferType::getShape() const {
    const auto data = getData().cast<NDTypeInterface>();
    return data.getShape();
}

MemShape VPUIP::BoundedBufferType::getMemShape() const {
    const auto data = getData().cast<NDTypeInterface>();
    return data.getMemShape();
}

bool VPUIP::BoundedBufferType::hasRank() const {
    const auto data = getData().cast<NDTypeInterface>();
    return data.hasRank();
}

int64_t VPUIP::BoundedBufferType::getRank() const {
    const auto data = getData().cast<NDTypeInterface>();
    return data.getRank();
}

int64_t VPUIP::BoundedBufferType::getNumElements() const {
    const auto data = getData().cast<NDTypeInterface>();
    return data.getNumElements();
}

mlir::Type VPUIP::BoundedBufferType::getElementType() const {
    const auto data = getData().cast<NDTypeInterface>();
    return data.getElementType();
}

DimsOrder VPUIP::BoundedBufferType::getDimsOrder() const {
    const auto data = getData().cast<NDTypeInterface>();
    return data.getDimsOrder();
}

IndexedSymbolAttr VPUIP::BoundedBufferType::getMemSpace() const {
    const auto data = getData().cast<NDTypeInterface>();
    return data.getMemSpace();
}

VPU::MemoryKind VPUIP::BoundedBufferType::getMemoryKind() const {
    const auto data = getData().cast<NDTypeInterface>();
    return data.getMemoryKind();
}

Strides VPUIP::BoundedBufferType::getStrides() const {
    const auto data = getData().cast<NDTypeInterface>();
    return data.getStrides();
}

MemStrides VPUIP::BoundedBufferType::getMemStrides() const {
    const auto data = getData().cast<NDTypeInterface>();
    return data.getMemStrides();
}

Bit VPUIP::BoundedBufferType::getElemTypeSize() const {
    const auto data = getData().cast<NDTypeInterface>();
    return data.getElemTypeSize();
}

Byte VPUIP::BoundedBufferType::getTotalAllocSize() const {
    const auto data = getData().cast<NDTypeInterface>();
    const auto shape = getDynamicShape().cast<NDTypeInterface>();
    return data.getTotalAllocSize() + shape.getTotalAllocSize();
}

Byte VPUIP::BoundedBufferType::getCompactAllocSize() const {
    const auto data = getData().cast<NDTypeInterface>();
    const auto shape = getDynamicShape().cast<NDTypeInterface>();
    return data.getCompactAllocSize() + shape.getCompactAllocSize();
}

NDTypeInterface VPUIP::BoundedBufferType::changeShape(ShapeRef shape) const {
    const auto data = getData().cast<NDTypeInterface>();
    const auto newData = data.changeShape(shape);

    const auto dynamicShape = getDynamicShape().cast<NDTypeInterface>();
    const auto newShape = Shape({checked_cast<Shape::ValueType>(shape.size())});
    const auto newDynamicShape = dynamicShape.changeShape(newShape);

    return VPUIP::BoundedBufferType::get(newData, newDynamicShape);
}

NDTypeInterface VPUIP::BoundedBufferType::changeElemType(mlir::Type elemType) const {
    const auto data = getData().cast<NDTypeInterface>();
    const auto newData = data.changeElemType(elemType);
    return VPUIP::BoundedBufferType::get(newData, getDynamicShape());
}

NDTypeInterface VPUIP::BoundedBufferType::changeShapeElemType(ShapeRef shape, mlir::Type elemType) const {
    const auto data = getData().cast<NDTypeInterface>();
    const auto newData = data.changeShapeElemType(shape, elemType);

    const auto dynamicShape = getDynamicShape().cast<NDTypeInterface>();
    const auto newShape = Shape({checked_cast<Shape::ValueType>(shape.size())});
    const auto newDynamicShape = dynamicShape.changeShape(newShape);

    return VPUIP::BoundedBufferType::get(newData, newDynamicShape);
}

NDTypeInterface VPUIP::BoundedBufferType::changeDimsOrder(DimsOrder order) const {
    const auto data = getData().cast<NDTypeInterface>();
    const auto newData = data.changeDimsOrder(order);
    return VPUIP::BoundedBufferType::get(newData, getDynamicShape());
}

NDTypeInterface VPUIP::BoundedBufferType::changeMemSpace(IndexedSymbolAttr memSpace) const {
    const auto data = getData().cast<NDTypeInterface>();
    const auto newData = data.changeMemSpace(memSpace);

    const auto dynamicShape = getDynamicShape().cast<NDTypeInterface>();
    const auto newDynamicShape = dynamicShape.changeMemSpace(memSpace);

    return VPUIP::BoundedBufferType::get(newData, newDynamicShape);
}

NDTypeInterface VPUIP::BoundedBufferType::changeStrides(StridesRef strides) const {
    const auto data = getData().cast<NDTypeInterface>();
    const auto newData = data.changeStrides(strides);
    return VPUIP::BoundedBufferType::get(newData, getDynamicShape());
}

NDTypeInterface VPUIP::BoundedBufferType::changeTypeComponents(const TypeComponents& typeComponents) const {
    const auto ndData = getData().cast<NDTypeInterface>();
    const auto data = ndData.changeTypeComponents(typeComponents);
    return VPUIP::BoundedBufferType::get(data, getDynamicShape());
}

NDTypeInterface VPUIP::BoundedBufferType::extractDenseTile(ShapeRef tileOffsets, ShapeRef tileShape) const {
    const auto data = getData().cast<NDTypeInterface>();
    const auto newData = data.extractDenseTile(tileOffsets, tileShape);
    return VPUIP::BoundedBufferType::get(newData, getDynamicShape());
}

NDTypeInterface VPUIP::BoundedBufferType::extractViewTile(ShapeRef tileOffsets, ShapeRef tileShape,
                                                          ShapeRef tileElemStrides) const {
    const auto data = getData().cast<NDTypeInterface>();
    const auto newData = data.extractViewTile(tileOffsets, tileShape, tileElemStrides);
    return VPUIP::BoundedBufferType::get(newData, getDynamicShape());
}

NDTypeInterface VPUIP::BoundedBufferType::eraseTiledInfo() const {
    const auto data = getData().cast<NDTypeInterface>();
    const auto newData = data.eraseTiledInfo();
    return VPUIP::BoundedBufferType::get(newData, getDynamicShape());
}

NDTypeInterface VPUIP::BoundedBufferType::pad(ShapeRef padBefore, ShapeRef padAfter) const {
    const auto data = getData().cast<NDTypeInterface>();
    const auto newData = data.pad(padBefore, padAfter);
    return VPUIP::BoundedBufferType::get(newData, getDynamicShape());
}

void VPUIP::BoundedBufferType::print(mlir::AsmPrinter& printer) const {
    printer << "<data=" << getData() << ", dynamic_shape=" << getDynamicShape() << ">";
}

mlir::Type VPUIP::BoundedBufferType::parse(mlir::AsmParser& parser) {
    if (parser.parseLess()) {
        return Type();
    }

    mlir::Type data;
    if (parser.parseKeyword("data")) {
        return Type();
    }
    if (parser.parseEqual()) {
        return Type();
    }
    if (parser.parseType<mlir::Type>(data)) {
        return Type();
    }
    if (mlir::succeeded(parser.parseOptionalGreater())) {
        return get(data, mlir::Type{});
    }

    if (parser.parseComma()) {
        return Type();
    }

    mlir::Type dynamicShape;
    if (parser.parseKeyword("dynamic_shape")) {
        return Type();
    }
    if (parser.parseEqual()) {
        return Type();
    }
    if (parser.parseType<mlir::Type>(dynamicShape)) {
        return Type();
    }

    if (parser.parseGreater()) {
        return Type();
    }

    return get(data, dynamicShape);
}

mlir::LogicalResult VPUIP::BoundedBufferType::verify(llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
                                                     mlir::Type data, mlir::Type dynamicShape) {
    if (!data.isa<mlir::MemRefType>()) {
        return printTo(emitError(), "Data type is not a memref. Got {0}", data);
    }

    if (!dynamicShape.isa<mlir::MemRefType>()) {
        return printTo(emitError(), "Dynamic shape type is not a memref. Got {0}", dynamicShape);
    }

    return mlir::success();
}

vpux::VPUIP::BoundedBufferType vpux::VPUIP::BoundedBufferType::cloneWith(std::optional<mlir::ArrayRef<int64_t>> shape,
                                                                         mlir::Type elementType) const {
    if (!shape.has_value()) {
        return changeElemType(elementType).cast<vpux::VPUIP::BoundedBufferType>();
    }
    return changeShapeElemType(ShapeRef(shape.value()), elementType).cast<vpux::VPUIP::BoundedBufferType>();
}

mlir::Attribute vpux::VPUIP::BoundedBufferType::getMemorySpace() const {
    return getMemSpace();
}
}  // namespace vpux
