//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

#include "vpux/compiler/utils/compression_utils.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

//
// print/parse
//

void VPUIP::BufferType::print(mlir::AsmPrinter& printer) const {
    printer << "<";
    for (auto& dim : getShape()) {
        printer << dim << "x";
    }
    printer << getElementType();

    const auto layout = getLayout();
    if (const auto mapAttr = layout.dyn_cast<mlir::AffineMapAttr>()) {
        printer << ", " << mapAttr;
    } else if (const auto descAttr = layout.dyn_cast<vpux::MemRefAttr>()) {
        printer << ", " << descAttr;
    } else {
        VPUX_THROW("Unsupported MemRefType layout '{0}'", layout);
    }

    printer << ", " << getMemSpace();

    if (auto swizzlingKey = getSwizzlingKey()) {
        printer << ", swizzlingKey = " << swizzlingKey.getInt();
    }

    printer << ">";
}

mlir::Type VPUIP::BufferType::parse(mlir::AsmParser& parser) {
    if (parser.parseLess()) {
        return Type();
    }

    SmallVector<int64_t> shape;
    int64_t dim = 0;
    while (parser.parseOptionalInteger(dim).has_value() && parser.parseXInDimensionList().succeeded()) {
        shape.push_back(dim);
    }

    mlir::Type elemType;
    if (parser.parseType(elemType)) {
        return Type();
    }
    if (parser.parseComma()) {
        return Type();
    }

    mlir::MemRefLayoutAttrInterface layout;

    mlir::AffineMapAttr mapAttr;
    vpux::MemRefAttr memRefAttr;
    if (parser.parseOptionalAttribute(mapAttr).has_value()) {
        layout = mapAttr;
    } else if (parser.parseOptionalAttribute(memRefAttr).has_value()) {
        layout = memRefAttr;
    } else {
        return Type();
    }

    if (parser.parseComma()) {
        return Type();
    }

    IndexedSymbolAttr memSpace;
    if (parser.parseAttribute(memSpace)) {
        return Type();
    }

    mlir::IntegerAttr swizzlingKey;
    if (mlir::succeeded(parser.parseOptionalComma())) {
        if (parser.parseKeyword("swizzlingKey")) {
            return Type();
        }

        if (parser.parseEqual()) {
            return Type();
        }

        if (parser.parseAttribute(swizzlingKey)) {
            return Type();
        }
    }

    if (parser.parseGreater()) {
        return Type();
    }

    return static_cast<mlir::Type>(get(parser.getContext(), ArrayRef(shape), elemType, layout, memSpace, swizzlingKey));
}

//
// NDTypeInterface
//

MemShape VPUIP::BufferType::getMemShape() const {
    const auto dimsOrder = getDimsOrder();
    const auto shape = getShape();
    return dimsOrder.toMemoryOrder(shape);
}

bool VPUIP::BufferType::hasRank() const {
    return true;
}

int64_t VPUIP::BufferType::getRank() const {
    return checked_cast<int64_t>(getShape().size());
}

int64_t VPUIP::BufferType::getNumElements() const {
    auto shape = getShape().raw();
    VPUX_THROW_UNLESS(!details::isDynamicDimValues(shape), "Cannot get element count of dynamic shaped type");
    return details::calcTotalShapeSize(shape);
}

DimsOrder VPUIP::BufferType::getDimsOrder() const {
    const auto layout = getLayout();
    if (const auto mapAttr = layout.dyn_cast<mlir::AffineMapAttr>()) {
        return DimsOrder::fromAffineMap(mapAttr.getValue());
    }

    if (const auto descAttr = layout.dyn_cast<vpux::MemRefAttr>()) {
        return DimsOrder::fromAffineMap(descAttr.order().getValue());
    }

    VPUX_THROW("Missing layout information");
}

VPU::MemoryKind VPUIP::BufferType::getMemoryKind() const {
    const auto memSpace = getMemSpace();
    if (memSpace == nullptr) {
        return VPU::MemoryKind::DDR;
    }

    return VPU::symbolizeEnum<VPU::MemoryKind>(memSpace.getLeafName()).value();
}

Strides VPUIP::BufferType::getStrides() const {
    const auto layout = getLayout();

    if (const auto mapAttr = layout.dyn_cast<mlir::AffineMapAttr>()) {
        VPUX_THROW_UNLESS(mapAttr.getValue().isPermutation(), "Got non permutation layout attribute '{0}'", layout);
    }

    if (const auto descAttr = layout.dyn_cast<vpux::MemRefAttr>()) {
        if (auto stridesAttr = descAttr.strides()) {
            const auto elemStrides = parseIntArrayAttr<int64_t>(stridesAttr);
            const Bit elemSize = getElemTypeSize();

            return Strides(to_small_vector(elemStrides | transformed([&](int64_t stride) {
                                               return stride * elemSize;
                                           })));
        }
    }

    // Missing strides specification means compact strides.
    const auto order = getDimsOrder();
    const auto memShape = getMemShape();
    const auto memStrides = StrideReqs::compact(order.numDims()).calcStrides(getElemTypeSize(), memShape);

    return order.toLogicalOrder(memStrides);
}

MemStrides VPUIP::BufferType::getMemStrides() const {
    const auto order = getDimsOrder();
    const auto strides = getStrides();
    return order.toMemoryOrder(strides);
}

Bit VPUIP::BufferType::getElemTypeSize() const {
    return vpux::getElemTypeSize(getElementType());
}

Byte VPUIP::BufferType::getTotalAllocSize() const {
    if (getRank() == 0) {
        return alignMemSize(getElemTypeSize(), Byte(1));
    }

    const auto memShape = getMemShape();
    const auto memStrides = getMemStrides();

    VPUX_THROW_UNLESS(memShape.size() == memStrides.size(), "Shape and strides mismatch : {0} vs {1}", memShape,
                      memStrides);

    auto allocSizeByte = alignMemSize(memStrides.front() * memShape.front(), Byte(1)).to<Byte>();
    auto swizzlingScheme = getSwizzlingSchemeAttr(*this);
    if (swizzlingScheme && swizzlingScheme.getKey().getInt() != 0) {
        // If swizzling is enabled total buffer size needs to be aligned to 512 or 1024 as required by HW
        allocSizeByte = Byte(alignSizeForSwizzling(allocSizeByte.count(), swizzlingScheme.getSizeAlignment().getInt()));
    }

    auto compressionState = getCompressionState(*this);
    if (compressionState == VPUIP::CompressionState::CompressionCandidate ||
        compressionState == VPUIP::CompressionState::RuntimeCompressed) {
        allocSizeByte = Byte(updateSizeForCompression(allocSizeByte.count()));
    }

    return allocSizeByte;
}

Byte VPUIP::BufferType::getCompactAllocSize() const {
    const Bit typeSize = getElemTypeSize();
    if (getRank() == 0) {
        return alignMemSize(typeSize, Byte(1));
    }

    const auto shape = getShape();
    return alignMemSize(typeSize * shape.totalSize(), Byte(1));
}

NDTypeInterface VPUIP::BufferType::changeShape(ShapeRef shape) const {
    return changeShapeElemType(shape, getElementType());
}

NDTypeInterface VPUIP::BufferType::changeElemType(mlir::Type elemType) const {
    return VPUIP::BufferType::get(getContext(), getShape().raw(), elemType, getLayout(), getMemSpace(),
                                  getSwizzlingKey());
}

NDTypeInterface VPUIP::BufferType::changeShapeElemType(ShapeRef shape, mlir::Type elemType) const {
    const auto ctx = getContext();

    const auto origOrder = getDimsOrder();
    const auto newOrder = origOrder.isIdentity() ? DimsOrder::fromNumDims(shape.size()) : origOrder;
    VPUX_THROW_UNLESS(newOrder.numDims() == shape.size(), "Order '{0}' is incompatible with the new shape '{1}'",
                      newOrder, shape);

    const auto layout = mlir::AffineMapAttr::get(newOrder.toAffineMap(ctx));

    return VPUIP::BufferType::get(ctx, shape.raw(), elemType, layout, getMemSpace(), getSwizzlingKey());
}

NDTypeInterface VPUIP::BufferType::changeDimsOrder(DimsOrder order) const {
    const auto ctx = getContext();

    const auto shape = getShape();
    VPUX_THROW_UNLESS(order.numDims() == shape.size(), "New order '{0}' is incompatible with shape '{1}'", order,
                      shape);

    const auto layout = mlir::AffineMapAttr::get(order.toAffineMap(ctx));

    return VPUIP::BufferType::get(ctx, shape.raw(), getElementType(), layout, getMemSpace(), getSwizzlingKey());
}

NDTypeInterface VPUIP::BufferType::changeMemSpace(IndexedSymbolAttr memSpace) const {
    return VPUIP::BufferType::get(getContext(), getShape().raw(), getElementType(), getLayout(), memSpace,
                                  getSwizzlingKey());
}

NDTypeInterface VPUIP::BufferType::changeStrides(StridesRef strides) const {
    const auto ctx = getContext();
    const auto elemSize = getElemTypeSize().count();
    const auto order = mlir::AffineMapAttr::get(getDimsOrder().toAffineMap(ctx));
    const auto newStrides = to_small_vector(strides | transformed([&](Bit stride) {
                                                return stride.count() / elemSize;
                                            }));
    const auto newStridesAttr = getIntArrayAttr(ctx, newStrides);
    const auto newDescAttr = vpux::MemRefAttr::get(order, newStridesAttr,
                                                   /*allocSize=*/nullptr, ctx);
    return VPUIP::BufferType::get(ctx, getShape().raw(), getElementType(), newDescAttr, getMemSpace(),
                                  getSwizzlingKey());
}

NDTypeInterface VPUIP::BufferType::changeTypeComponents(const vpux::TypeComponents& typeComponents) const {
    const auto ctx = getContext();

    const auto shape = typeComponents.shape.value_or(Shape(getShape().toValues()));
    const auto elementType = typeComponents.elementType.value_or(getElementType());
    const auto dimsOrder = typeComponents.dimsOrder.value_or(getDimsOrder());
    const auto strides = typeComponents.strides.value_or(getStrides());
    const auto memSpace = typeComponents.memSpace.value_or(getMemSpace());

    VPUX_THROW_UNLESS(dimsOrder.numDims() == shape.size(), "New order '{0}' is incompatible with shape '{1}'",
                      dimsOrder, shape);
    const auto layout = mlir::AffineMapAttr::get(dimsOrder.toAffineMap(ctx));

    const auto elemSize = vpux::getElemTypeSize(elementType).count();
    const auto newStrides = to_small_vector(strides | transformed([&](Bit stride) {
                                                return stride.count() / elemSize;
                                            }));
    const auto newStridesAttr = getIntArrayAttr(ctx, newStrides);
    const auto newDescAttr = vpux::MemRefAttr::get(layout, newStridesAttr,
                                                   /*allocSize=*/nullptr, ctx);

    return VPUIP::BufferType::get(ctx, shape.raw(), elementType, newDescAttr, memSpace, getSwizzlingKey());
}

NDTypeInterface VPUIP::BufferType::extractDenseTile(ShapeRef tileOffsets, ShapeRef tileShape) const {
    const auto ctx = getContext();
    const auto order = mlir::AffineMapAttr::get(getDimsOrder().toAffineMap(ctx));

    auto tileElemType = getElementType();
    if (const auto perAxisQType = tileElemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        tileElemType = vpux::tileScalesAndZP(perAxisQType, tileShape, tileOffsets);
    }

    return VPUIP::BufferType::get(ctx, tileShape.raw(), tileElemType, order, getMemSpace(), getSwizzlingKey());
}

NDTypeInterface VPUIP::BufferType::extractViewTile(vpux::ShapeRef tileOffsets, vpux::ShapeRef tileShape,
                                                   vpux::ShapeRef tileElemStrides) const {
    const auto ctx = getContext();
    const auto elemSize = getElemTypeSize().count();
    const auto order = mlir::AffineMapAttr::get(getDimsOrder().toAffineMap(ctx));
    const auto memSpace = getMemSpace();

    auto tileElemType = getElementType();
    if (const auto perAxisQType = tileElemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        tileElemType = vpux::tileScalesAndZP(perAxisQType, tileShape, tileOffsets);
    }

    auto tileStrides = getStrides();
    if (!tileElemStrides.empty()) {
        VPUX_THROW_UNLESS(tileElemStrides.size() == tileStrides.size(),
                          "Tile elem strides '{0}' is not aligned with rank '{1}'", tileElemStrides,
                          tileStrides.size());

        for (auto ind : irange(tileElemStrides.size())) {
            tileStrides[Dim(ind)] *= tileElemStrides[Dim(ind)];
        }
    }

    const auto newStrides = to_small_vector(tileStrides | transformed([&](Bit stride) {
                                                return stride.count() / elemSize;
                                            }));

    const auto newStridesAttr = getIntArrayAttr(ctx, newStrides);
    const auto newDescAttr = vpux::MemRefAttr::get(order, newStridesAttr,
                                                   /*allocSize=*/nullptr, ctx);

    return VPUIP::BufferType::get(ctx, tileShape.raw(), tileElemType, newDescAttr, memSpace, getSwizzlingKey());
}

NDTypeInterface VPUIP::BufferType::eraseTiledInfo() const {
    const auto ctx = getContext();
    const auto layout = mlir::AffineMapAttr::get(getDimsOrder().toAffineMap(ctx));

    return VPUIP::BufferType::get(ctx, getShape().raw(), getElementType(), layout, getMemSpace(), getSwizzlingKey());
}

NDTypeInterface VPUIP::BufferType::pad(ShapeRef /*padBefore*/, ShapeRef /*padAfter*/) const {
    VPUX_THROW("pad method is not yet implemented for BufferType");
}
