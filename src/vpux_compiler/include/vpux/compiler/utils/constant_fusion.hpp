//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/logging.hpp"

#include "vpux/compiler/utils/types.hpp"

namespace vpux {
namespace ConstantFusing {

using ConstantVector = SmallVector<std::pair<VPUIP::CopyOp, Const::DeclareOp>>;

constexpr StringLiteral constantsFused = "constantsFused";
constexpr int8_t numberOfConstantsToFuse = 3;

///
/// \brief Get underlying DeclareOp and Op for passed constant
/// \param [in] constant - Constant to get DeclareOp and Op for
/// \param [out] constOp - Op if the DeclareOp is found
/// \return Const::DeclareOp when found
///

Const::DeclareOp getConstAndDma(mlir::Value constant, mlir::Operation** constOp);

///
/// \brief Get static offset for the constant
/// \param [in] constant - mlir::Value constant
/// \return int32_t offset for the constant
///

int32_t getOffsetForConstant(VPUIP::NCEClusterTaskOp& nceOp, mlir::Value constant);

/// @brief Function creates a new distributed buffer type, used for creating a alloc op for distributed buffer
/// @param origDistType [in] Used to get the original Distribute Mode
/// @param declOp [in] constant used for reference
/// @param rewriter [in] rewriter
/// @return VPUIP::DistributedBufferType [out]

VPUIP::DistributedBufferType getDistributedBufferType(VPUIP::DistributedBufferType origDistType,
                                                      Const::DeclareOp constant, mlir::PatternRewriter& rewriter);

/// @brief Gets the CopyOp and DeclareOp for constants to be fused
/// @param nceOp [in] The NCECluster task to which the constant belongs
/// @param constant [in] Constant (mlir::Value) for which the copy and const op are needed
/// @param copyOp [out] stores the copyOp found
/// @param declareOp [out] stores the declareOp found
/// @param allocDistributed [out] stores the allocDistributed if found for future use
/// @param tilingOp [out] stores the top TilingOp if found for future use

void getCopyAndDeclareOpForFusion(mlir::Value constant, VPUIP::CopyOp& copyOp, Const::DeclareOp& declareOp,
                                  VPURT::AllocDistributed& foundAllocDistributed);
}  // namespace ConstantFusing
}  // namespace vpux
