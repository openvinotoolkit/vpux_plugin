//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

using namespace vpux;

//
// build
//

void VPUIP::GroupSparseBufferOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value data,
                                       bool isWeights, VPUIP::SparsityCompressionAttr sparsityCompression) {
    build(builder, state, data, nullptr, nullptr, isWeights, sparsityCompression);
}

void VPUIP::GroupSparseBufferOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value data,
                                       mlir::Value sparsityMap, bool isWeights,
                                       VPUIP::SparsityCompressionAttr sparsityCompression) {
    build(builder, state, data, sparsityMap, nullptr, isWeights, sparsityCompression);
}

void VPUIP::GroupSparseBufferOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value data,
                                       mlir::Value sparsityMap, mlir::Value storageElementTable, bool isWeights,
                                       VPUIP::SparsityCompressionAttr sparsityCompression) {
    const auto isWeightsAttr = isWeights ? mlir::UnitAttr::get(builder.getContext()) : nullptr;
    build(builder, state, data, sparsityMap, storageElementTable, isWeightsAttr, sparsityCompression, nullptr);
}

void VPUIP::GroupSparseBufferOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value data,
                                       mlir::Value sparsityMap, mlir::Value storageElementTable, VPU::SEAttr seAttr) {
    build(builder, state, data, sparsityMap, storageElementTable, nullptr, nullptr, seAttr);
}

//
// getViewSources
//

mlir::ValueRange VPUIP::GroupSparseBufferOp::getViewSources() {
    return getOperands();
}

//
// inferReturnTypes
//

mlir::LogicalResult VPUIP::GroupSparseBufferOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                                 std::optional<mlir::Location> optLoc,
                                                                 mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                                 mlir::OpaqueProperties, mlir::RegionRange /*ranges*/,
                                                                 SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPUIP::GroupSparseBufferOpAdaptor groupOp(operands, attrs);
    if (mlir::failed(groupOp.verify(loc))) {
        return mlir::failure();
    }

    const auto dataType = groupOp.getData().getType();
    const auto sparsityMapType = groupOp.getSparsityMap() != nullptr ? groupOp.getSparsityMap().getType() : nullptr;
    const auto storageElementTableType =
            groupOp.getStorageElementTable() != nullptr ? groupOp.getStorageElementTable().getType() : nullptr;

    inferredReturnTypes.push_back(
            VPUIP::SparseBufferType::get(dataType, sparsityMapType, storageElementTableType, groupOp.getIsWeightsAttr(),
                                         groupOp.getSparsityCompressionAttr(), groupOp.getSeAttrAttr()));

    return mlir::success();
}

namespace {

//
// RemoveGroupUngroup
//

class RemoveGroupUngroup final : public mlir::OpRewritePattern<VPUIP::GroupSparseBufferOp> {
public:
    using mlir::OpRewritePattern<VPUIP::GroupSparseBufferOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::GroupSparseBufferOp groupOp,
                                        mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult RemoveGroupUngroup::matchAndRewrite(VPUIP::GroupSparseBufferOp groupOp,
                                                        mlir::PatternRewriter& /*rewriter*/) const {
    if (llvm::any_of(groupOp.getOutput().getUsers(), [](mlir::Operation* userOp) {
            return !mlir::isa<VPUIP::UngroupSparseBufferOp>(userOp);
        })) {
        return mlir::failure();
    }

    const auto operands = groupOp.getOperands();
    for (auto userOp : groupOp.getOutput().getUsers()) {
        for (auto userResult : userOp->getResults() | indexed) {
            userResult.value().replaceAllUsesWith(operands[userResult.index()]);
        }
    }

    return mlir::success();
}

}  // namespace

void VPUIP::GroupSparseBufferOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                             mlir::MLIRContext* context) {
    patterns.add<RemoveGroupUngroup>(context);
}
