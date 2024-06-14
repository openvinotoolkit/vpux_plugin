//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/passes/VPUMI37XX2VPUASM/symbolization_type_converter.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/ops.hpp"

namespace vpux {
namespace vpumi37xx2vpuasm {

namespace {

template <typename Type>
std::optional<mlir::LogicalResult> doNotConvert(Type, llvm::SmallVectorImpl<mlir::Type>& results) {
    results.clear();
    return mlir::success();
}

}  // namespace

SymbolizationTypeConverter::SymbolizationTypeConverter() {
    addConversion(doNotConvert<VPURegMapped::IndexType>);
    addConversion(doNotConvert<mlir::MemRefType>);
    addConversion(doNotConvert<VPUIP::DistributedBufferType>);
}

}  // namespace vpumi37xx2vpuasm
}  // namespace vpux
