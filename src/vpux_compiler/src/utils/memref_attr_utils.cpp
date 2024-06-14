//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//
#include "vpux/compiler/utils/memref_attr_utils.hpp"

#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"
#include "vpux/compiler/utils/types.hpp"

using namespace vpux;

mlir::IntegerAttr vpux::getAllocSizeAttr(mlir::Type type) {
    mlir::IntegerAttr allocSizeAttr;

    if (type == nullptr) {
        return allocSizeAttr;
    }

    mlir::MemRefLayoutAttrInterface layout;

    if (auto memref = type.dyn_cast<mlir::MemRefType>()) {
        layout = memref.getLayout();
    } else if (auto distributedBuffer = type.dyn_cast<VPUIP::DistributedBufferType>()) {
        layout = distributedBuffer.getLayout();
    } else if (auto itiBuffer = type.dyn_cast<VPUIP::ITIBufferType>()) {
        layout = itiBuffer.getLayout();
    }

    if (layout) {
        if (const auto memRefAttr = layout.dyn_cast<vpux::MemRefAttr>()) {
            allocSizeAttr = memRefAttr.allocSize();
        }
    }

    return allocSizeAttr;
}

// Updates the swizzling scheme, adjusts the sizeAlignment added for distributedBuffer
vpux::NDTypeInterface vpux::setAllocSizeAttr(vpux::NDTypeInterface type, int64_t allocSize) {
    auto* ctx = type.getContext();
    const auto shape = type.getShape();
    const auto elemType = type.getElementType();
    const auto order = type.getDimsOrder();
    const auto strides = type.getStrides();
    const auto memSpace = type.getMemSpace();

    mlir::IntegerAttr allocSizeAttr = getIntAttr(ctx, allocSize);

    if (type.isa<mlir::MemRefType>()) {
        return vpux::getMemRefType(shape, elemType, order, memSpace, strides, getSwizzlingSchemeAttr(type),
                                   VPUIP::getSparsityCompressionAttr(type), allocSizeAttr);
    } else if (type.isa<VPUIP::DistributedBufferType>() || type.isa<VPUIP::ITIBufferType>()) {
        vpux::MemRefAttr memRefAttr;
        const auto orderAttr = mlir::AffineMapAttr::get(order.toAffineMap(ctx));
        mlir::ArrayAttr stridesAttr = nullptr;
        vpux::MemRefAttr::HwFields hwSpecificFields{};

        auto itiBufferType = type.dyn_cast<VPUIP::ITIBufferType>();
        auto distBufferType = type.dyn_cast<VPUIP::DistributedBufferType>();

        if (itiBufferType) {
            memRefAttr = itiBufferType.getLayout().dyn_cast<vpux::MemRefAttr>();
        } else if (distBufferType) {
            memRefAttr = distBufferType.getLayout().dyn_cast<vpux::MemRefAttr>();
        }
        if (memRefAttr) {
            stridesAttr = memRefAttr.strides();
            hwSpecificFields = memRefAttr.hwSpecificFields();
        }

        const auto layoutAttr = vpux::MemRefAttr::get(orderAttr, stridesAttr, allocSizeAttr, hwSpecificFields, ctx);

        if (itiBufferType) {
            return VPUIP::ITIBufferType::get(ctx, shape.raw(), elemType, layoutAttr, memSpace,
                                             itiBufferType.getIduSegmentation(), itiBufferType.getInwardHaloRegions(),
                                             itiBufferType.getOutwardHaloRegions());
        }

        return VPUIP::DistributedBufferType::get(ctx, shape.raw(), elemType, layoutAttr, memSpace,
                                                 distBufferType.getDistribution(),
                                                 distBufferType.getSparsityCompression());
    }

    VPUX_THROW("Unsupported type for storing allocSize setting - {0}", type);
}
