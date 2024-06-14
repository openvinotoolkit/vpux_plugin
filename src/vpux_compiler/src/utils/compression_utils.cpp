//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//
#include "vpux/compiler/utils/compression_utils.hpp"
#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/utils/memref_attr_utils.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"

namespace vpux {

int64_t updateSizeForCompression(int64_t size) {
    // In worst case scenario depending on the content of activation, its final size after
    // compression might be bigger than original size. Compiler before performing DDR
    // allocation needs to adjust required size by this buffer
    // Formula from HAS is following:
    //   DTS = X * Y * Z * (element size in bytes)
    //   denseSize = (DTS * (65/64)) + 1
    //   DDR Allocation (32B aligned) = denseSize + ( (denseSize % 32) ? (32 – (denseSize % 32) : 0)
    auto worstCaseSize = static_cast<int64_t>(size * 65 / 64) + 1;
    if (worstCaseSize % ACT_COMPRESSION_BUF_SIZE_ALIGNMENT) {
        worstCaseSize += ACT_COMPRESSION_BUF_SIZE_ALIGNMENT - worstCaseSize % ACT_COMPRESSION_BUF_SIZE_ALIGNMENT;
    }
    return worstCaseSize;
}

VPUIP::CompressionStateAttr getCompressionStateAttr(mlir::Type type) {
    VPUIP::CompressionStateAttr compressionAttr;

    if (type == nullptr) {
        return compressionAttr;
    }

    mlir::MemRefLayoutAttrInterface layout;

    if (auto memref = type.dyn_cast<mlir::MemRefType>()) {
        layout = memref.getLayout();
    } else if (auto distributedBuffer = type.dyn_cast<VPUIP::DistributedBufferType>()) {
        layout = distributedBuffer.getLayout();
    } else if (auto itiBuffer = type.dyn_cast<VPUIP::ITIBufferType>()) {
        layout = itiBuffer.getLayout();
    } else {
        return compressionAttr;
    }

    if (layout) {
        if (const auto memRefAttr = layout.dyn_cast<vpux::MemRefAttr>()) {
            compressionAttr = memRefAttr.hwSpecificField<vpux::VPUIP::CompressionStateAttr>();
        }
    }

    return compressionAttr;
}

VPUIP::CompressionState getCompressionState(mlir::Type type) {
    auto compressionAttr = getCompressionStateAttr(type);

    if (compressionAttr == nullptr) {
        return VPUIP::CompressionState::NoCompression;
    }

    return compressionAttr.getValue();
}

mlir::Type setCompressionStateAttribute(mlir::Type type, VPUIP::CompressionStateAttr compressionAttr) {
    VPUX_THROW_WHEN(type == nullptr, "NULL type provided");

    if (!compressionAttr) {
        return type;
    }

    const auto ndType = type.cast<vpux::NDTypeInterface>();
    auto* ctx = type.getContext();

    const auto shape = ndType.getShape();
    const auto elemType = ndType.getElementType();
    const auto order = ndType.getDimsOrder();
    const auto strides = ndType.getStrides();
    const auto memSpace = ndType.getMemSpace();

    if (type.isa<mlir::MemRefType>()) {
        return vpux::getMemRefType(shape, elemType, order, memSpace, strides, getSwizzlingSchemeAttr(type),
                                   VPUIP::getSparsityCompressionAttr(type), getAllocSizeAttr(type), compressionAttr);
    } else if (type.isa<VPUIP::DistributedBufferType>() || type.isa<VPUIP::ITIBufferType>()) {
        mlir::ArrayAttr stridesAttr;
        const auto orderAttr = mlir::AffineMapAttr::get(order.toAffineMap(ctx));
        const Bit elemSize = ndType.getElemTypeSize();
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

        const auto layoutAttr =
                vpux::MemRefAttr::get(orderAttr, stridesAttr,
                                      /*allocSize=*/nullptr, {getSwizzlingSchemeAttr(type), compressionAttr}, ctx);

        if (auto itiBufferType = type.dyn_cast<VPUIP::ITIBufferType>()) {
            return VPUIP::ITIBufferType::get(ctx, shape.raw(), elemType, layoutAttr, memSpace,
                                             itiBufferType.getIduSegmentation(), itiBufferType.getInwardHaloRegions(),
                                             itiBufferType.getOutwardHaloRegions());
        }

        auto distBufferType = type.cast<VPUIP::DistributedBufferType>();
        return VPUIP::DistributedBufferType::get(ctx, shape.raw(), elemType, layoutAttr, memSpace,
                                                 distBufferType.getDistribution(),
                                                 distBufferType.getSparsityCompression());
    }

    VPUX_THROW("Unsupported type for storing swizzling setting");
}

mlir::Type setCompressionState(mlir::Type type, VPUIP::CompressionState compression) {
    VPUX_THROW_WHEN(type == nullptr, "NULL type provided");

    auto compressionAttr = VPUIP::CompressionStateAttr::get(type.getContext(), compression);

    return setCompressionStateAttribute(type, compressionAttr);
}

}  // namespace vpux
