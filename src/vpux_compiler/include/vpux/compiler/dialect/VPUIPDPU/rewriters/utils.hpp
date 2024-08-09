//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/attributes.hpp"

#include <mlir/IR/Builders.h>

namespace vpux {
namespace VPUIPDPU {

enum class IOType { INT, FP, IOTypeNum };

enum class BlockArg {
    ACT_IN,
    ACT_SE_IN,
    ACT_SPARSE_MAP_IN,
    WEIGHTS_TABLE,
    WEIGHTS,
    WEIGHTS_SPARSE_MAP,
    SPR_LOOKUP_TABLE,
    ACT_OUT,
    ACT_SPARSE_MAP_OUT,
    Count
};

IOType getIOType(mlir::Type type);

template <typename AttrType, typename ValueType>
AttrType getEnumAttrOrNull(mlir::OpBuilder& builder, const std::optional<ValueType>& attr) {
    if (attr.has_value()) {
        return AttrType::get(builder.getContext(), attr.value());
    }

    return nullptr;
}
mlir::FloatAttr getF32FloatAttrOrNull(mlir::OpBuilder& builder, const std::optional<float>& attr);

mlir::MemRefType getBufferType(mlir::Operation* bufferRef);

uint64_t getSwizzlingKey(mlir::Operation* bufferRef);

mlir::BlockArgument getInvBlockArg(BlockArg invBlockArg, mlir::Block* invBlock,
                                   const std::unordered_map<BlockArg, size_t>& invBlockArgsPos);

mlir::Type getBaseType(mlir::Type type);

mlir::LogicalResult getQuantConfig(const Logger&, mlir::Type type, SmallVector<int64_t>& quantMult,
                                   SmallVector<int64_t>& quantShift, SmallVector<uint8_t>& quantZero,
                                   VPU::ArchKind arch);

mlir::IntegerAttr getI64IntegerAttrOrNull(mlir::OpBuilder& builder, const std::optional<int64_t>& attr);

}  // namespace VPUIPDPU
}  // namespace vpux
