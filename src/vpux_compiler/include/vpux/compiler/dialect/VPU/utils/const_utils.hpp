//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"

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
bool isNullOrConstWithSingleValue(mlir::Value value);

/**
 * @brief calculate memory requirement for given buffer sizes and architecture-dependent allocation requirements
 *
 * @param arch - architecture type
 * @param bufferSizes - vector containing sizes [bytes] of buffers to be allocated
 *
 * @return required memory taking into account the allocation requirements for swizzled buffers [bytes].
 *
 * Starting with NPU37XX the required memory size is
 * calculated according to requirements for CMX allocation for swizzled buffers.
 *
 * NOTE: see also vpux::calculateAlignedBuffersMemoryRequirement
 */
Byte calculateAlignedBuffersMemoryRequirement(VPU::ArchKind arch, mlir::SmallVector<Byte>& bufferSizes);

}  // namespace VPU
}  // namespace vpux
