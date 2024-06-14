//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/types.hpp"

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

//
// get<scalar>Type
//

mlir::IntegerType vpux::getInt1Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 1);
}

mlir::IntegerType vpux::getInt4Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 4);
}

mlir::IntegerType vpux::getInt8Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 8);
}

mlir::IntegerType vpux::getInt16Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 16);
}

mlir::IntegerType vpux::getInt32Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 32);
}

mlir::IntegerType vpux::getInt64Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 64);
}

mlir::IntegerType vpux::getSInt4Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 4, mlir::IntegerType::Signed);
}

mlir::IntegerType vpux::getSInt8Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 8, mlir::IntegerType::Signed);
}

mlir::IntegerType vpux::getSInt16Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 16, mlir::IntegerType::Signed);
}

mlir::IntegerType vpux::getSInt32Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Signed);
}

mlir::IntegerType vpux::getSInt64Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Signed);
}

mlir::IntegerType vpux::getUInt4Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 4, mlir::IntegerType::Unsigned);
}

mlir::IntegerType vpux::getUInt8Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 8, mlir::IntegerType::Unsigned);
}

mlir::IntegerType vpux::getUInt16Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 16, mlir::IntegerType::Unsigned);
}

mlir::IntegerType vpux::getUInt32Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Unsigned);
}

mlir::IntegerType vpux::getUInt64Type(mlir::MLIRContext* ctx) {
    return mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);
}

mlir::IntegerType vpux::getBool8Type(mlir::MLIRContext* ctx) {
    // Signless 8-bit integer use for BOOL, to distinguish it from U8
    return mlir::IntegerType::get(ctx, 8, mlir::IntegerType::Signless);
}

//
// TypeSize
//

Bit vpux::getElemTypeSize(mlir::Type type) {
    if (const auto ndType = type.dyn_cast<vpux::NDTypeInterface>()) {
        return getElemTypeSize(ndType.getElementType());
    }

    if (type.isIntOrFloat()) {
        return Bit(type.getIntOrFloatBitWidth());
    }

    if (const auto qType = type.dyn_cast<mlir::quant::QuantizedType>()) {
        return Bit(qType.getStorageTypeIntegralWidth());
    }

    VPUX_THROW("Can't get type size for '{0}'", type);
}

Byte vpux::getTotalSize(mlir::Value val) {
    const auto type = val.getType().dyn_cast<vpux::NDTypeInterface>();
    VPUX_THROW_UNLESS(type != nullptr, "Value '{0}' has non vpux::NDTypeInterface '{1}'", val, val.getType());
    return type.getTotalAllocSize();
}

Byte vpux::getCompactSize(mlir::Value val) {
    const auto type = val.getType().dyn_cast<vpux::NDTypeInterface>();
    VPUX_THROW_UNLESS(type != nullptr, "Value '{0}' has non vpux::NDTypeInterface '{1}'", val, val.getType());
    return type.getCompactAllocSize();
}

std::optional<int32_t> vpux::getQuantizedAxis(int32_t axis, ShapeRef prevShape, ShapeRef newShape) {
    auto prevArray = to_small_vector(prevShape.toValues());
    auto newArray = to_small_vector(newShape.toValues());

    if (checked_cast<size_t>(axis) >= prevShape.size()) {
        return std::nullopt;
    }

    const auto isEqualOne = [](int64_t val) {
        return val == 1;
    };

    auto firstPrevIter = std::find_if_not(prevArray.begin(), prevArray.end(), isEqualOne);
    auto firstNewIter = std::find_if_not(newArray.begin(), newArray.end(), isEqualOne);

    if (firstPrevIter == prevArray.end() || firstNewIter == newArray.end()) {
        return axis;
    }

    const auto gap = checked_cast<int32_t>(std::distance(newArray.begin(), firstNewIter) -
                                           std::distance(prevArray.begin(), firstPrevIter));
    const auto newArraySize = newArray.size();

    prevArray.erase(std::remove_if(prevArray.begin(), prevArray.end(), isEqualOne), prevArray.end());
    newArray.erase(std::remove_if(newArray.begin(), newArray.end(), isEqualOne), newArray.end());

    if (!std::equal(prevArray.begin(), prevArray.end(), newArray.begin(), newArray.end())) {
        return std::nullopt;
    }

    if (axis < -gap || axis + gap >= checked_cast<int32_t>(newArraySize)) {
        return std::nullopt;
    }

    return axis + gap;
}

//
// MemRefType utilities
//

mlir::MemRefType vpux::getMemRefType(ShapeRef shape, mlir::Type elemType, DimsOrder order, IndexedSymbolAttr memSpace,
                                     StridesRef strides, VPUIP::SwizzlingSchemeAttr swizzlingSchemeAttr,
                                     VPUIP::SparsityCompressionAttr sparsityCompressionAttr,
                                     mlir::IntegerAttr allocSizeAttr, VPUIP::CompressionStateAttr compressionState) {
    VPUX_THROW_UNLESS(order.numDims() == shape.size(), "Shape '{0}' doesn't match order '{1}'", shape, order);
    VPUX_THROW_UNLESS(strides.empty() || shape.size() == strides.size(), "Strides '{0}' doesn't match shape '{1}'",
                      strides, shape);

    auto* ctx = elemType.getContext();
    const auto orderAttr = mlir::AffineMapAttr::get(order.toAffineMap(ctx));

    mlir::ArrayAttr stridesAttr = nullptr;
    if (strides != StridesRef()) {
        const Bit elemSize = getElemTypeSize(elemType);
        const auto memShape = order.toMemoryOrder(shape);
        const auto memStrides = order.toMemoryOrder(strides);
        const auto compactReqs = StrideReqs::compact(shape.size());
        if (!compactReqs.checkStrides(memStrides, elemSize, memShape)) {
            // Have strides only if they are not compact
            const auto elemStrides = to_small_vector(strides | transformed([&](Bit stride) {
                                                         return stride.count() / elemSize.count();
                                                     }));

            stridesAttr = getIntArrayAttr(ctx, elemStrides);
        }
    }

    mlir::MemRefType::Builder builder(shape.raw(), elemType);
    builder.setMemorySpace(memSpace);
    if (stridesAttr == nullptr && swizzlingSchemeAttr == nullptr && sparsityCompressionAttr == nullptr &&
        allocSizeAttr == nullptr && compressionState == nullptr) {
        builder.setLayout(orderAttr);
    } else {
        const auto layoutAttr =
                vpux::MemRefAttr::get(orderAttr, stridesAttr, allocSizeAttr,
                                      {swizzlingSchemeAttr, sparsityCompressionAttr, compressionState}, ctx);
        builder.setLayout(layoutAttr.cast<mlir::MemRefLayoutAttrInterface>());
    }
    return builder;
}

mlir::SmallVector<float> vpux::getFloatStrides(StridesRef strides) {
    Strides temp(strides.begin(), strides.end());

    const auto cvtBitStrideToByteFP = [](Bit val) {
        if (val.count() % CHAR_BIT == 0) {
            return checked_cast<float>(Byte(val).count());
        }

        return checked_cast<float>(val.count()) / CHAR_BIT;
    };

    return to_small_vector(temp | transformed(cvtBitStrideToByteFP));
}

//
// RankedTensorType utilities
//

mlir::RankedTensorType vpux::getTensorType(ShapeRef shape, mlir::Type elemType, DimsOrder order,
                                           IndexedSymbolAttr memSpace, mlir::ArrayAttr bounds) {
    VPUX_THROW_UNLESS(order.numDims() == shape.size(), "DimsOrder '{0}' doesn't match to shape '{1}'", order, shape);

    const auto tensorDesc = vpux::getTensorAttr(elemType.getContext(), order, memSpace, bounds);
    const auto newType = mlir::RankedTensorType::get(shape.raw(), elemType, tensorDesc);

    const auto loc = mlir::UnknownLoc::get(elemType.getContext());
    VPUX_THROW_UNLESS(validateQuantElemType(loc, newType).succeeded(), "Got invalid ShapedType '{0}'", newType);

    return newType;
}

mlir::MemRefType vpux::convertToMemRef(mlir::RankedTensorType tensorType) {
    const auto type = tensorType.cast<vpux::NDTypeInterface>();
    const auto shape = type.getShape();
    const auto elemType = type.getElementType();
    const auto order = type.getDimsOrder();
    const auto memSpace = type.getMemSpace();
    return getMemRefType(shape, elemType, order, memSpace);
}

//
// NDTypeInterface utilities
//

bool vpux::isCompatibleForInplaceOp(vpux::NDTypeInterface inInterface, vpux::NDTypeInterface outInterface,
                                    vpux::Logger log) {
    if (inInterface.getShape() != outInterface.getShape()) {
        log.trace("Different in and out shape {0} != {1}", inInterface.getShape(), outInterface.getShape());
        return false;
    }

    if (inInterface.getStrides() != outInterface.getStrides()) {
        log.trace("Different in and out strides {0} != {1}", inInterface.getStrides(), outInterface.getStrides());
        return false;
    }

    if (inInterface.getDimsOrder() != outInterface.getDimsOrder()) {
        log.trace("Different in and out order {0} != {1}", inInterface.getDimsOrder(), outInterface.getDimsOrder());
        return false;
    }

    if (inInterface.getTotalAllocSize() < outInterface.getTotalAllocSize()) {
        /* #65422 Case with different tensor sizes.
        If op is eltwise and it is a part of dequantize chain then input is U8 and output is float. */
        log.trace("Input tensor size is smaller than output tensor size, the case is not supported {0} < {1}",
                  inInterface.getTotalAllocSize(), outInterface.getTotalAllocSize());
        return false;
    }

    return true;
}

//
// Type comparison
//

bool vpux::areTypesCompatible(mlir::TypeRange lhs, mlir::TypeRange rhs, IE::TypeComparisonMode elemComparisonModes,
                              bool checkDimsOrder, bool checkMemSpace) {
    if (lhs.size() != rhs.size()) {
        return false;
    }

    for (const auto p : zip(lhs, rhs)) {
        auto lhsOrigType = std::get<0>(p);
        auto rhsOrigType = std::get<1>(p);

        if (lhsOrigType.getTypeID() != rhsOrigType.getTypeID()) {
            if (IE::bitEnumContainsAny(elemComparisonModes, IE::TypeComparisonMode::ALLOW_GROUPED_OUTPUT)) {
                const auto oneIsGrouped = (lhsOrigType.isa<vpux::GroupedTypeInterface>() &&
                                           !rhsOrigType.isa<vpux::GroupedTypeInterface>()) ||
                                          (!lhsOrigType.isa<vpux::GroupedTypeInterface>() &&
                                           rhsOrigType.isa<vpux::GroupedTypeInterface>());
                if (!oneIsGrouped) {
                    return false;
                }
            }
        }

        auto lhsType = lhsOrigType.dyn_cast<NDTypeInterface>();
        auto rhsType = rhsOrigType.dyn_cast<NDTypeInterface>();

        if (lhsType == nullptr || rhsType == nullptr) {
            return false;
        }

        if (lhsType.getShape() != rhsType.getShape()) {
            return false;
        }

        if (lhsType.getElementType() != rhsType.getElementType()) {
            if (IE::bitEnumContainsAny(elemComparisonModes, IE::TypeComparisonMode::STRICT_EQUAL)) {
                return false;
            }

            const auto lhsQuantizedType = lhsType.getElementType().dyn_cast<mlir::quant::QuantizedType>();
            const auto rhsQuantizedType = rhsType.getElementType().dyn_cast<mlir::quant::QuantizedType>();

            if (!lhsQuantizedType && !rhsQuantizedType) {
                return false;
            } else if (lhsQuantizedType && rhsQuantizedType) {
                if ((lhsQuantizedType.getExpressedType() != rhsQuantizedType.getExpressedType()) ||
                    (lhsQuantizedType.getStorageType() != rhsQuantizedType.getStorageType())) {
                    if (!IE::bitEnumContainsAny(elemComparisonModes, IE::TypeComparisonMode::ALLOW_DIFFERENT_QUANT)) {
                        return false;
                    }
                }
            } else {
                if (!IE::bitEnumContainsAny(elemComparisonModes, IE::TypeComparisonMode::ALLOW_QUANT_MIXED_PRECISION)) {
                    return false;
                }
            }
        }

        if (checkDimsOrder) {
            const auto order1 = lhsType.getDimsOrder();
            const auto order2 = rhsType.getDimsOrder();

            if (order1 != order2) {
                return false;
            }
        }

        if (checkMemSpace) {
            const auto memSpace1 = lhsType.getMemSpace();
            const auto memSpace2 = rhsType.getMemSpace();

            if (memSpace1 != memSpace2) {
                // Allow different memory spaces only if both types are in DDR, since a null value also represents DDR
                if (!(lhsType.getMemoryKind() == VPU::MemoryKind::DDR &&
                      rhsType.getMemoryKind() == VPU::MemoryKind::DDR)) {
                    return false;
                }
            }
        }
    }

    return true;
}

//
// Quantized dimension permutation
//

bool vpux::isQuantizedDimensionPermutation(mlir::quant::UniformQuantizedPerAxisType inputElemType,
                                           mlir::quant::UniformQuantizedPerAxisType newElemType) {
    if (inputElemType.getExpressedType() != newElemType.getExpressedType() ||
        inputElemType.getStorageType() != newElemType.getStorageType() ||
        inputElemType.isSigned() != newElemType.isSigned() || inputElemType.getFlags() != newElemType.getFlags() ||
        inputElemType.getStorageTypeMin() != newElemType.getStorageTypeMin() ||
        inputElemType.getStorageTypeMax() != newElemType.getStorageTypeMax() ||
        inputElemType.getQuantizedDimension() == newElemType.getQuantizedDimension()) {
        return false;
    }

    auto inputElemTypeScales = inputElemType.getScales();
    auto newElemTypeScales = newElemType.getScales();
    if (inputElemTypeScales.size() != newElemTypeScales.size() ||
        !std::equal(inputElemTypeScales.begin(), inputElemTypeScales.end(), newElemTypeScales.begin())) {
        return false;
    }

    auto inputElemTypeZPs = inputElemType.getZeroPoints();
    auto newElemTypeZPs = newElemType.getZeroPoints();
    if (inputElemTypeZPs.size() != newElemTypeZPs.size() ||
        !std::equal(inputElemTypeZPs.begin(), inputElemTypeZPs.end(), newElemTypeZPs.begin())) {
        return false;
    }

    return true;
}

bool vpux::isSubByteType(mlir::Type elemType) {
    return getElemTypeSize(elemType).count() < CHAR_BIT;
}

bool vpux::isBufferType(mlir::Type type) {
    // Note: BaseMemRefType covers MemRefType, UnrankedMemRefType,
    // VPUIP::DistributedBufferType, VPUIP::SparseBufferType and VPUIP::BoundedBufferType
    return mlir::isa<mlir::BaseMemRefType, VPUIP::BufferType>(type);
}
