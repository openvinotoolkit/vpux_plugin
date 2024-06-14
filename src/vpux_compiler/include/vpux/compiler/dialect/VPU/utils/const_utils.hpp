//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"

namespace vpux {
namespace VPU {

mlir::Value createActivationWindowTensor(mlir::OpBuilder& builder, mlir::Location loc, ArrayRef<uint8_t> fakeSparsity);

std::vector<int32_t> createWeightsTableData(mlir::Value opInput, mlir::Value opOutput, mlir::Value weights,
                                            Const::ContentAttr bias, int64_t OC, vpux::VPU::PPETaskAttr ppeTaskAttr,
                                            VPU::ArchKind _arch, vpux::IE::PostOpAttr postOpAttr,
                                            mlir::FloatAttr constScale);
mlir::Value createWeightsTableTensor(mlir::OpBuilder& builder, mlir::Location loc, ArrayRef<int32_t> weightsTable);
std::optional<SmallVector<int32_t>> createInstructionListTableData(mlir::Value opOutput, vpux::IE::PostOpAttr postOp,
                                                                   VPU::ArchKind _arch);
mlir::Value createInstructionListTableTensor(mlir::OpBuilder& builder, mlir::Location loc,
                                             const std::optional<SmallVector<int32_t>>& instructionListTable);

mlir::Value alignDepthWiseWeightsTensor(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value origFilter);
mlir::Value alignConvWeightsTensor(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value origFilter,
                                   const bool isCMajorConv);
mlir::Value getZerosConst(mlir::PatternRewriter& rewriter, ShapeRef constShape, mlir::Value input, mlir::Location loc);
mlir::Value buildWeightsConst(vpux::ShapeRef weightsShape, DimsOrder weightsOrder, ArrayRef<float> weightsValue,
                              mlir::Value activation, mlir::PatternRewriter& rewriter);

/**
 * @brief calculate memory requirement for given buffer sizes and architecture-dependent allocation requirements
 *
 * @param arch - architecture type
 * @param bufferSizes - vector containing sizes [bytes] of buffers to be allocated
 *
 * @return required memory taking into account the allocation requirements for swizzled buffers [bytes].
 *
 * For NPU30XX this returns the size of combined vectors. Starting with NPU37XX the required memory size is
 * calculated according to requirements for CMX allocation for swizzled buffers.
 *
 * NOTE: see also vpux::calculateAlignedBuffersMemoryRequirement
 */
Byte calculateAlignedBuffersMemoryRequirement(VPU::ArchKind arch, mlir::SmallVector<Byte>& bufferSizes);

Const::DeclareOp declareFloatConst(mlir::OpBuilder& builder, mlir::Location loc, float val,
                                   mlir::RankedTensorType argType);

mlir::DenseElementsAttr wrapData(const mlir::RankedTensorType dataStorageType, ArrayRef<float> values);
mlir::FailureOr<Const::DeclareOp> updateConstStorageValues(Const::DeclareOp origConst, ArrayRef<float> constValues,
                                                           mlir::PatternRewriter& rewriter, Logger log);

bool hasNegativeValues(const Const::Content& content);
Const::DeclareOp createFloatConst(mlir::RankedTensorType constType, ArrayRef<float> constValues, mlir::Location loc,
                                  mlir::PatternRewriter& rewriter);
mlir::Value createIntConst(ShapeRef constShape, DimsOrder constOrder, ArrayRef<int32_t> constData, mlir::Location loc,
                           mlir::PatternRewriter& rewriter);
bool isLowPrecisionTypeRange(const Const::Content& lowVal, const Const::Content& highVal, int64_t levels, bool isSigned,
                             mlir::PatternRewriter& rewriter);

}  // namespace VPU
}  // namespace vpux
