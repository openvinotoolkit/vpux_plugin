//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUASM/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUASM/types.hpp"

#include "vpux/compiler/NPU40XX/dialect/ELF/metadata.hpp"
#include "vpux/compiler/NPU40XX/dialect/ELF/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/ELF/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPURegMapped/attributes.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"

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

#include <vpux/compiler/dialect/VPUASM/dialect.hpp.inc>

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPUASM/ops.hpp.inc>
