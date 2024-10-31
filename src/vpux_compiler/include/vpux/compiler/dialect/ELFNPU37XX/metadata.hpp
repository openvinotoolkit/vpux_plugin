//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/core/logger.hpp"

#include <vpux_elf/writer.hpp>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/Timing.h>

#include <transformations/utils/utils.hpp>

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

#include <vpux_headers/metadata.hpp>

namespace vpux {
namespace ELFNPU37XX {

std::unique_ptr<elf::NetworkMetadata> constructMetadata(mlir::ModuleOp module, Logger log);

elf::TensorRef createTensorRef(mlir::Value val, StringRef name);
elf::TensorRef createTensorRef(vpux::NDTypeInterface type, StringRef name);

elf::DType createDType(mlir::Type type);
elf::OVNodeType createOVNodeType(mlir::Type type);

}  // namespace ELFNPU37XX
}  // namespace vpux
