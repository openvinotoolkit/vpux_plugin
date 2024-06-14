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

#include "vpux/compiler/dialect/ELFNPU37XX/ops.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"

namespace vpux {
namespace ELFNPU37XX {

std::vector<uint8_t> exportToELF(
        mlir::ModuleOp module,
        const std::vector<std::shared_ptr<const ov::Node>>& parameters = std::vector<std::shared_ptr<const ov::Node>>(),
        const std::vector<std::shared_ptr<const ov::Node>>& results = std::vector<std::shared_ptr<const ov::Node>>(),
        Logger log = Logger::global());

}  // namespace ELFNPU37XX
}  // namespace vpux
