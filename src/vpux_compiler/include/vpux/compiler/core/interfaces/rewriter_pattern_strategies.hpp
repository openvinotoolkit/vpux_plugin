//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/core/logger.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux {

/*
   Interface for implementing platform specific rewriter patterns applied using the Greedy driver
*/
class IGreedilyPassStrategy {
public:
    virtual ~IGreedilyPassStrategy() = default;

    virtual void addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const = 0;
};

/*
   Interface for implementing platform specific rewriter patterns applied using the Conversion driver
*/
class IConversionPassStrategy {
public:
    virtual ~IConversionPassStrategy() = default;

    virtual void addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const = 0;
    virtual void markOpLegality(mlir::ConversionTarget& target, Logger& log) const = 0;
};

}  // namespace vpux
