//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/Transforms/DialectConversion.h>

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

namespace vpux::vpuip2vpumi40xx {

template <class SpecificDMAType>
struct DMARewriterBase : mlir::OpConversionPattern<SpecificDMAType> {
    DMARewriterBase(mlir::MLIRContext* context, bool isMemorySideCacheEnabled)
            : mlir::OpConversionPattern<SpecificDMAType>(context), _isMemorySideCacheEnabled(isMemorySideCacheEnabled) {
    }

    bool _isMemorySideCacheEnabled;
};

struct NNDMARewriter : DMARewriterBase<VPUIP::NNDMAOp> {
    using DMARewriterBase::DMARewriterBase;
    mlir::LogicalResult matchAndRewrite(VPUIP::NNDMAOp nnDMAOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override;
};

struct PermuteDMARewriter : DMARewriterBase<VPUIP::PermuteDMAOp> {
    using DMARewriterBase::DMARewriterBase;
    mlir::LogicalResult matchAndRewrite(VPUIP::PermuteDMAOp permuteDMAOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override;
};

struct ExpandDMARewriter : DMARewriterBase<VPUIP::ExpandDMAOp> {
    using DMARewriterBase::DMARewriterBase;
    mlir::LogicalResult matchAndRewrite(VPUIP::ExpandDMAOp expandDMAOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override;
};

struct ConvertDMARewriter : DMARewriterBase<VPUIP::ConvertDMAOp> {
    using DMARewriterBase::DMARewriterBase;
    mlir::LogicalResult matchAndRewrite(VPUIP::ConvertDMAOp convertDMAOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override;
};

struct SpaceToDepthDMARewriter : DMARewriterBase<VPUIP::SpaceToDepthDMAOp> {
    using DMARewriterBase::DMARewriterBase;
    mlir::LogicalResult matchAndRewrite(VPUIP::SpaceToDepthDMAOp spaceToDepthDMAOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override;
};

struct DepthToSpaceDMARewriter : DMARewriterBase<VPUIP::DepthToSpaceDMAOp> {
    using DMARewriterBase::DMARewriterBase;
    mlir::LogicalResult matchAndRewrite(VPUIP::DepthToSpaceDMAOp depthToSpaceDMAOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override;
};

struct UpsamplingDMARewriter : DMARewriterBase<VPUIP::UpsamplingDMAOp> {
    using DMARewriterBase::DMARewriterBase;
    mlir::LogicalResult matchAndRewrite(VPUIP::UpsamplingDMAOp upsamplingDMAOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override;
};

struct PerAxisTileDMARewriter : DMARewriterBase<VPUIP::PerAxisTileDMAOp> {
    using DMARewriterBase::DMARewriterBase;
    mlir::LogicalResult matchAndRewrite(VPUIP::PerAxisTileDMAOp perAxisTileDMAOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override;
};

struct DecompressDMARewriter : DMARewriterBase<VPUIP::DecompressDMAOp> {
    using DMARewriterBase::DMARewriterBase;
    mlir::LogicalResult matchAndRewrite(VPUIP::DecompressDMAOp decompressDMAOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override;
};

struct CompressDMARewriter : DMARewriterBase<VPUIP::CompressDMAOp> {
    using DMARewriterBase::DMARewriterBase;
    mlir::LogicalResult matchAndRewrite(VPUIP::CompressDMAOp compressDMAOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override;
};

struct GatherDMARewriter : DMARewriterBase<VPUIP::GatherDMAOp> {
    using DMARewriterBase::DMARewriterBase;
    mlir::LogicalResult matchAndRewrite(VPUIP::GatherDMAOp gatherDMAOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override;
};

struct SyncDMARewriter : DMARewriterBase<VPUIP::SyncDMAOp> {
    using DMARewriterBase::DMARewriterBase;
    mlir::LogicalResult matchAndRewrite(VPUIP::SyncDMAOp syncDMAOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override;
};

}  // namespace vpux::vpuip2vpumi40xx
