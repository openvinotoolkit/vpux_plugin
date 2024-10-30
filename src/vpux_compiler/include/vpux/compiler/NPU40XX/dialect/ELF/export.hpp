//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/compiler.hpp"

#include "vpux/utils/core/logger.hpp"

#include <vpux_elf/writer.hpp>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/Timing.h>

#include <transformations/utils/utils.hpp>

#include "vpux/compiler/NPU40XX/dialect/ELF/ops.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"

namespace vpux {
namespace ELF {

std::vector<uint8_t> exportToELF(mlir::ModuleOp module, Logger log = Logger::global());
BlobView exportToELF(mlir::ModuleOp module, BlobAllocator& allocator, Logger log = Logger::global());

}  // namespace ELF
}  // namespace vpux
