//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Bufferization/Transforms/Bufferize.h>
#include <mlir/IR/Value.h>

#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"

namespace vpux {

SmallVector<mlir::Value> allocateBuffersOfType(const Logger& log, mlir::Location loc, mlir::OpBuilder& builder,
                                               mlir::Type bufferType, bool individualBuffers = false);

SmallVector<mlir::Value> allocateBuffersOfType(const Logger& log, mlir::Location loc, mlir::RewriterBase& rewriter,
                                               mlir::Value value, vpux::IndexedSymbolAttr memSpace,
                                               bool individualBuffers = false);

//
// allocateBuffers & allocateBuffersForValue using bufferizable interface
//

SmallVector<mlir::Value> allocateBuffersForValue(const Logger& log, mlir::Location loc, mlir::OpBuilder& builder,
                                                 mlir::Value value, bool individualBuffers = false);

SmallVector<mlir::Value> allocateBuffers(const Logger& log, mlir::Location loc, mlir::OpBuilder& builder,
                                         mlir::ValueRange values, bool individualBuffers = false);

mlir::Value allocateBuffer(const Logger& log, mlir::Location loc, mlir::RewriterBase& rewriter, mlir::Value value,
                           vpux::IndexedSymbolAttr memSpace);

}  // namespace vpux
