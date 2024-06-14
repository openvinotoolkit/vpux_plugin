//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/core/logger.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassOptions.h>

namespace vpux {

enum class LocationsVerifierMarker { BEGIN, END };
enum class LocationsVerificationMode { OFF, FAST, FULL, THOROUGH };

LocationsVerificationMode getLocationsVerificationMode(mlir::ModuleOp moduleOp);

LocationsVerificationMode getLocationsVerificationMode(
        const mlir::detail::PassOptions::Option<std::string>& locationsVerificationMode);

void setLocationsVerificationMode(mlir::ModuleOp moduleOp, LocationsVerificationMode mode);

std::string stringifyLocationsVerificationMode(LocationsVerificationMode mode);

LocationsVerificationMode symbolizeLocationsVerificationMode(StringRef strMode);

void addLocationsVerifier(mlir::PassManager& pm);

mlir::LogicalResult verifyLocationsUniquenessFull(mlir::Operation* op, StringRef passName);
mlir::LogicalResult verifyLocationsUniquenessFast(mlir::Operation* op, StringRef passName);

// verifyLocations is a wrapper for verifyLocationsUniquenessFull and verifyLocationsUniquenessFast
// depending on the current locations verification mode in Module
mlir::LogicalResult verifyLocations(mlir::Operation* op, StringRef passName);

};  // namespace vpux
