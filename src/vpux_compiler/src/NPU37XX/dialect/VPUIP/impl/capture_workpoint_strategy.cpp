//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPUIP/impl/capture_workpoint_strategy.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/utils/profiling/common.hpp"

using namespace vpux;

namespace vpux::VPUIP::arch37xx {

void insertCaptureDma(mlir::OpBuilder& builder, int64_t profOutputId, size_t dstDdrOffset) {
    auto* ctx = builder.getContext();

    const auto memKindAttr = IndexedSymbolAttr::get(ctx, stringifyEnum(VPU::MemoryKind::Register));
    const vpux::NDTypeInterface hwTimerType =
            getMemRefType(ShapeRef({1}), getUInt32Type(ctx), DimsOrder::C, memKindAttr);

    const auto loc = mlir::NameLoc::get(mlir::StringAttr::get(ctx, profiling::PROFILING_WORKPOINT_READ_ATTR));
    // Create declaration for source buffer which corresponds to HW register with free-running counter
    auto hwRegOp = builder.create<VPURT::DeclareBufferOp>(loc, hwTimerType, VPURT::BufferSection::Register,
                                                          VPUIP::HW_PLL_WORKPOINT_ABSOLUTE_ADDR);

    const auto profilingOutputType = mlir::MemRefType::get(hwTimerType.getShape().raw(), hwTimerType.getElementType());
    auto dstBufProfResultOp = builder.create<VPURT::DeclareBufferOp>(
            loc, profilingOutputType, VPURT::BufferSection::ProfilingOutput, profOutputId, dstDdrOffset);

    const auto port = 0;
    // Since the payload is copied into the final destination is DDR no barriers needed, so may be inserted anywhere in
    // the network without barriers setup
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(builder, /*waitBarriers=*/{}, /*updateBarriers=*/{}, loc, hwRegOp.getBuffer(),
                                          dstBufProfResultOp.getBuffer(), port, /*is_out_of_order=*/true,
                                          /*is_critical=*/false, /*spillId=*/nullptr);
}

void CaptureWorkpointStrategy::prepareDMACapture(mlir::OpBuilder& builder, mlir::func::FuncOp& func,
                                                 const int64_t profOutputId, mlir::func::ReturnOp returnOp) {
    builder.setInsertionPoint(&func.getBody().front().front());
    // Capture setup in the begin of inference
    insertCaptureDma(builder, profOutputId, 0);
    // And in the end
    builder.setInsertionPoint(returnOp);
    insertCaptureDma(builder, profOutputId, VPUIP::HW_PLL_WORKPOINT_SIZE);
}

}  // namespace vpux::VPUIP::arch37xx
