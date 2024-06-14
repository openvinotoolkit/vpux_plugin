//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/nce_invariant.hpp"
#include "vpux/compiler/utils/asm.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Quant/QuantOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OpImplementation.h>

using namespace vpux;

bool IE::isActShaveKernel(mlir::Operation* operation) {
    return VPU::NCEInvariant::isSupported(operation, Logger::global()).failed();
}

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/IE/ops.cpp.inc>
