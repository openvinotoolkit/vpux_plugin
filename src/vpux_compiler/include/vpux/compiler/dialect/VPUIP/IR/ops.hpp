//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Quant/QuantOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/CopyOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPUIP/ops.hpp.inc>

//
// Operation verifiers
//

namespace vpux {
namespace VPUIP {

constexpr Bit FP16_SIZE = 16_Bit;
constexpr KB SHAVE_LIB_DATA_SIZE = 112_KB;

// According to the documentation, total transfer length (LEN) field is stored in 24 bits that means max value is 16MB
constexpr Byte DMA_LIMIT = MB(16).to<Byte>() - Byte(1);
constexpr int64_t CMX_DMA_MAX_NUM_PLANES_30XX_37XX = 255;
// According to the documentation, size of the highest dimension is stored in 16 bits on NPU40XX
constexpr int64_t CMX_DMA_MAX_NUM_PLANES_40XX = INT16_MAX;
constexpr int64_t CMX_DMA_MAX_STRIDING_LEVEL_30XX_37XX = 2;
constexpr int64_t CMX_DMA_MAX_STRIDING_LEVEL_40XX = 6;

}  // namespace VPUIP
}  // namespace vpux

//
// Template methods
//

namespace vpux {
namespace VPUIP {

template <typename... Args>
VPUIP::PPETaskOp NCEClusterTaskOp::addPPETask(mlir::OpBuilder& builder, Args&&... args) {
    if (getPpe().empty()) {
        getPpe().emplaceBlock();
    }

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&getPpe().front());

    return builder.create<VPUIP::PPETaskOp>(getLoc(), std::forward<Args>(args)...);
}

template <typename T>
T vpux::VPUIP::NCEClusterTilingOp::getInnerTaskOpOfType() {
    return mlir::dyn_cast<T>(&getBody().front().front());
}
}  // namespace VPUIP
}  // namespace vpux
