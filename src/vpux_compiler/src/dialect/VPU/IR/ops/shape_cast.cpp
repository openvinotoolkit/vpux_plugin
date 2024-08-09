//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"

using namespace vpux;
using namespace VPU;

mlir::LogicalResult vpux::VPU::ShapeCastOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                             std::optional<mlir::Location> optLoc,
                                                             mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                             mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                             mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::ShapeCastOpAdaptor shapeCast(operands, attrs, prop);
    if (mlir::failed(shapeCast.verify(loc))) {
        return mlir::failure();
    }

    const auto outShape = parseIntArrayAttr<int64_t>(shapeCast.getShape());
    const auto inType = shapeCast.getSource().getType().cast<vpux::NDTypeInterface>();

    auto outType = inType.changeShape(Shape(outShape));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

mlir::OpFoldResult vpux::VPU::ShapeCastOp::fold(FoldAdaptor adaptor) {
    auto operands = adaptor.getOperands();
    auto inputType = getSource().getType().cast<vpux::NDTypeInterface>();
    auto outputType = getResult().getType().cast<vpux::NDTypeInterface>();
    if (getSource().getType() == getResult().getType()) {
        return getSource();
    }

    VPUX_THROW_UNLESS(!operands.empty(), "Wrong number of operands : {0}", operands.size());
    if (inputType.getElementType().dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>()) {
        return nullptr;
    }
    if (const auto attr = operands[0].dyn_cast_or_null<Const::ContentAttr>()) {
        return attr.reshape(outputType.getShape());
    }

    return nullptr;
}

//
// FuseShapeCast
//

namespace {
class FuseShapeCast final : public mlir::OpRewritePattern<VPU::ShapeCastOp> {
public:
    using mlir::OpRewritePattern<VPU::ShapeCastOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(VPU::ShapeCastOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseShapeCast::matchAndRewrite(VPU::ShapeCastOp origOp, mlir::PatternRewriter& rewriter) const {
    auto prevOp = origOp.getSource().getDefiningOp<VPU::ShapeCastOp>();
    if (prevOp == nullptr) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<VPU::ShapeCastOp>(origOp, prevOp.getSource(), origOp.getShape());
    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::VPU::ShapeCastOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* ctx) {
    patterns.add<FuseShapeCast>(ctx);
}
