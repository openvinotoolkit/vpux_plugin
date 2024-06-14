//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

//
// Generated
//

#define GET_TYPEDEF_CLASSES
#include <vpux/compiler/dialect/VPUIP/types.cpp.inc>
#undef GET_TYPEDEF_CLASSES

//
// VPUIPDialect::registerTypes
//

void vpux::VPUIP::VPUIPDialect::registerTypes() {
    addTypes<
#define GET_TYPEDEF_LIST
#include <vpux/compiler/dialect/VPUIP/types.cpp.inc>
            >();
}

//
// BufferType::Accessors
//

vpux::ShapeRef vpux::VPUIP::BufferType::getShape() const {
    return vpux::ShapeRef(getImpl()->shape);
}

mlir::Type vpux::VPUIP::BufferType::getElementType() const {
    return getImpl()->elementType;
}

mlir::MemRefLayoutAttrInterface vpux::VPUIP::BufferType::getLayout() const {
    return getImpl()->layout;
}

vpux::IndexedSymbolAttr vpux::VPUIP::BufferType::getMemSpace() const {
    return getImpl()->memSpace;
}

mlir::IntegerAttr vpux::VPUIP::BufferType::getSwizzlingKey() const {
    return getImpl()->swizzlingKey;
}

//
// SparseBufferType::Accessors
//

// Note: getMemorySpace and clonewith are defined for compliance with BaseMemRefTypeInterface.

vpux::VPUIP::SparseBufferType vpux::VPUIP::SparseBufferType::cloneWith(std::optional<mlir::ArrayRef<int64_t>> shape,
                                                                       mlir::Type elementType) const {
    if (!shape.has_value()) {
        return changeElemType(elementType).cast<vpux::VPUIP::SparseBufferType>();
    }
    return changeShapeElemType(ShapeRef(shape.value()), elementType).cast<vpux::VPUIP::SparseBufferType>();
}

mlir::Attribute vpux::VPUIP::SparseBufferType::getMemorySpace() const {
    return getMemSpace();
}

//
// DistributedBufferType::Accessors
//

// Note: getMemorySpace and clonewith are defined for compliance with BaseMemRefTypeInterface.

mlir::Attribute vpux::VPUIP::DistributedBufferType::getMemorySpace() const {
    return getMemSpace();
}

vpux::VPUIP::DistributedBufferType vpux::VPUIP::DistributedBufferType::cloneWith(
        std::optional<mlir::ArrayRef<int64_t>> shape, mlir::Type elementType) const {
    if (!shape.has_value()) {
        return changeElemType(elementType).cast<vpux::VPUIP::DistributedBufferType>();
    }
    return changeShapeElemType(ShapeRef(shape.value()), elementType).cast<vpux::VPUIP::DistributedBufferType>();
}

vpux::ShapeRef vpux::VPUIP::DistributedBufferType::getShape() const {
    return vpux::ShapeRef(getImpl()->shape);
}

mlir::Type vpux::VPUIP::DistributedBufferType::getElementType() const {
    return getImpl()->elementType;
}

mlir::MemRefLayoutAttrInterface vpux::VPUIP::DistributedBufferType::getLayout() const {
    return getImpl()->layout;
}

vpux::IndexedSymbolAttr vpux::VPUIP::DistributedBufferType::getMemSpace() const {
    return getImpl()->memSpace;
}

VPU::DistributedTensorAttr vpux::VPUIP::DistributedBufferType::getDistribution() const {
    return getImpl()->distribution;
}

VPUIP::SparsityCompressionAttr vpux::VPUIP::DistributedBufferType::getSparsityCompression() const {
    return getImpl()->sparsityCompression;
}

//
// ITIBufferType::Accessors
//

vpux::ShapeRef vpux::VPUIP::ITIBufferType::getShape() const {
    return vpux::ShapeRef(getImpl()->shape);
}

mlir::Type vpux::VPUIP::ITIBufferType::getElementType() const {
    return getImpl()->elementType;
}

mlir::MemRefLayoutAttrInterface vpux::VPUIP::ITIBufferType::getLayout() const {
    return getImpl()->layout;
}

vpux::IndexedSymbolAttr vpux::VPUIP::ITIBufferType::getMemSpace() const {
    return getImpl()->memSpace;
}

mlir::UnitAttr vpux::VPUIP::ITIBufferType::getIduSegmentation() const {
    return getImpl()->iduSegmentation;
}

ArrayRef<vpux::VPUIP::HaloRegionAttr> vpux::VPUIP::ITIBufferType::getInwardHaloRegions() const {
    return getImpl()->inwardHaloRegions;
}

ArrayRef<vpux::VPUIP::OutwardHaloRegionAttr> vpux::VPUIP::ITIBufferType::getOutwardHaloRegions() const {
    return getImpl()->outwardHaloRegions;
}
