//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/Support/LogicalResult.h>
#include <cstdint>
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/type_infer.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::PermuteCastOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                               std::optional<mlir::Location> optLoc,
                                                               mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                               mlir::OpaqueProperties prop,
                                                               mlir::RegionRange /*regions*/,
                                                               mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::PermuteCastOpAdaptor permuteCast(operands, attrs, prop);
    if (mlir::failed(permuteCast.verify(loc))) {
        return mlir::failure();
    }

    const auto inOrder = DimsOrder::fromValue(permuteCast.getInput());
    const auto inShape = getShape(permuteCast.getInput());
    const auto inMemShape = inOrder.toMemoryOrder(inShape);
    if (!isTrivialPermute(inMemShape, permuteCast.getMemPerm())) {
        return errorAt(loc, "Operation represents non trivial permutation");
    }

    VPU::inferPermuteReturnTypes(permuteCast.getInput(), permuteCast.getMemPerm(), permuteCast.getDstOrder(),
                                 inferredReturnTypes);

    return mlir::success();
}

//
// DistributedCastOpInterface
//

mlir::FailureOr<std::pair<mlir::Type, VPU::DistributionInfo>> vpux::VPU::PermuteCastOp::inferCastedTypeAndDistribution(
        vpux::NDTypeInterface inType, VPU::DistributionInfo& distribution) {
    if (inType == nullptr || mlir::isa<VPU::DistributedTensorType>(inType) ||
        distribution.getDistributionMode() == DistributionMode::NONE) {
        return mlir::failure();
    }
    const auto srcType = getInput().getType().cast<NDTypeInterface>();
    const auto dstType = getOutput().getType().cast<NDTypeInterface>();

    auto castedOutputDistribution =
            applyPermutationOnDistributionInfo(inType, distribution, getMemPerm(), srcType.getDimsOrder(),
                                               dstType.getDimsOrder(), srcType.getShape(), dstType.getShape());
    if (mlir::failed(castedOutputDistribution)) {
        return mlir::failure();
    };
    const auto typeComponents = TypeComponents()
                                        .setShape(dstType.getShape())
                                        .setDimsOrder(dstType.getDimsOrder())
                                        .setElementType(dstType.getElementType())
                                        .setMemSpace(inType.getMemSpace());
    return std::make_pair(mlir::cast<mlir::Type>(dstType.changeTypeComponents(typeComponents)),
                          castedOutputDistribution.value());
}

namespace {

//
// PropagatePermuteCast
//

class PropagatePermuteCast final : public mlir::OpRewritePattern<VPU::PermuteCastOp> {
public:
    using mlir::OpRewritePattern<VPU::PermuteCastOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(VPU::PermuteCastOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult PropagatePermuteCast::matchAndRewrite(VPU::PermuteCastOp origOp,
                                                          mlir::PatternRewriter& rewriter) const {
    auto checkShapeChanged = [](mlir::Operation* op) -> bool {
        return getShape(op->getOperand(0)) == getShape(op->getResult(0));
    };

    if (!checkShapeChanged(origOp)) {
        return mlir::failure();
    }

    auto middleOp = origOp.getInput().getDefiningOp();
    if (middleOp == nullptr || !middleOp->hasOneUse()) {
        return mlir::failure();
    }

    // Some other ops can be added if needed
    if (!mlir::isa<VPU::ExpandOp>(middleOp)) {
        return mlir::failure();
    }

    auto prePermuteCastOp = middleOp->getOperand(0).getDefiningOp<VPU::PermuteCastOp>();
    if (prePermuteCastOp == nullptr || !prePermuteCastOp->hasOneUse() || !checkShapeChanged(prePermuteCastOp)) {
        return mlir::failure();
    }

    auto srcInOrder = DimsOrder::fromValue(prePermuteCastOp.getInput());
    auto srcOutOrder = DimsOrder::fromValue(prePermuteCastOp.getOutput());
    auto dstInOrder = DimsOrder::fromValue(origOp.getInput());
    auto dstOutOrder = DimsOrder::fromValue(origOp.getOutput());

    if (srcInOrder != dstOutOrder || srcOutOrder != dstInOrder) {
        return mlir::failure();
    }

    mlir::IRMapping mapper;
    mapper.map(middleOp->getOperand(0), prePermuteCastOp.getInput());
    auto newOp = rewriter.clone(*middleOp, mapper);
    vpux::inferReturnTypes(newOp, vpux::InferShapedTypeMode::ALL);

    rewriter.replaceOp(origOp, newOp->getResults());

    return mlir::success();
}

//
// MergeParallelPermuteCast
//

class MergeParallelPermuteCast final : public mlir::OpRewritePattern<VPU::PermuteCastOp> {
public:
    using mlir::OpRewritePattern<VPU::PermuteCastOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(VPU::PermuteCastOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult MergeParallelPermuteCast::matchAndRewrite(VPU::PermuteCastOp origOp,
                                                              mlir::PatternRewriter& rewriter) const {
    auto input = origOp.getInput();

    if (input.hasOneUse()) {
        return mlir::failure();
    }

    SmallVector<VPU::PermuteCastOp> permuteCastOps;
    for (auto user : input.getUsers()) {
        if (user == origOp.getOperation()) {
            continue;
        }

        auto permuteCastOp = mlir::dyn_cast_or_null<VPU::PermuteCastOp>(user);
        if (permuteCastOp == nullptr) {
            continue;
        }

        if (origOp.getMemPermAttr() == permuteCastOp.getMemPermAttr() &&
            origOp.getDstOrderAttr() == permuteCastOp.getDstOrderAttr()) {
            permuteCastOps.push_back(permuteCastOp);
        }
    }

    if (permuteCastOps.empty()) {
        return mlir::failure();
    }

    for (auto permuteCastOp : permuteCastOps) {
        rewriter.replaceOp(permuteCastOp, origOp.getOutput());
    }

    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::VPU::PermuteCastOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* ctx) {
    patterns.add<PropagatePermuteCast>(ctx);
    patterns.add<MergeParallelPermuteCast>(ctx);
}
