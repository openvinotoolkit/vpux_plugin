//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/IR/Dialect.h>
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"

namespace vpux {

// TODO: needs refactoring. Ticket: E#50937
// Dummy op interfaces will end up being deleted if we properly refactor this dummy op feature
enum class DummyOpMode { ENABLED = 0, DISABLED = 1 };

// instantiates mlir::DialectRegistry and registers interfaces that are common across generations
mlir::DialectRegistry createDialectRegistry(DummyOpMode = DummyOpMode::DISABLED);

}  // namespace vpux
