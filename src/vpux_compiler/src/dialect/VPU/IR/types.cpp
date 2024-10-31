//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

//
// Generated
//

#define GET_TYPEDEF_CLASSES
#include <vpux/compiler/dialect/VPU/types.cpp.inc>
#undef GET_TYPEDEF_CLASSES

//
// VPUDialect::registerTypes
//

void VPU::VPUDialect::registerTypes() {
    addTypes<
#define GET_TYPEDEF_LIST
#include <vpux/compiler/dialect/VPU/types.cpp.inc>
            >();
}

//
// VPU::DistributedTensorType accessors
//

ShapeRef VPU::DistributedTensorType::getShape() const {
    return ShapeRef(getImpl()->shape);
}

mlir::Type VPU::DistributedTensorType::getElementType() const {
    return getImpl()->elementType;
}

mlir::AffineMapAttr VPU::DistributedTensorType::getOrder() const {
    return getImpl()->order;
}

IndexedSymbolAttr VPU::DistributedTensorType::getMemSpace() const {
    return getImpl()->memSpace;
}

VPU::DistributionInfoAttr VPU::DistributedTensorType::getDistribution() const {
    return getImpl()->distribution;
}

VPU::DistributedTensorType VPU::DistributedTensorType::cloneWith(std::optional<mlir::ArrayRef<int64_t>> shape,
                                                                 mlir::Type elementType) const {
    if (!shape.has_value()) {
        return changeElemType(elementType).cast<VPU::DistributedTensorType>();
    }
    return changeShapeElemType(ShapeRef(shape.value()), elementType).cast<VPU::DistributedTensorType>();
}

//
// VPU::SparseTensorType accessors
//

VPU::SparseTensorType VPU::SparseTensorType::cloneWith(std::optional<mlir::ArrayRef<int64_t>> shape,
                                                       mlir::Type elementType) const {
    if (!shape.has_value()) {
        return changeElemType(elementType).cast<VPU::SparseTensorType>();
    }
    return changeShapeElemType(ShapeRef(shape.value()), elementType).cast<VPU::SparseTensorType>();
}
