//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"

namespace vpux::VPU {

/*
 Interface for set of functions which determine how to change values
 of tiles for axes
*/
class IVFAxisIncrement {
public:
    virtual ~IVFAxisIncrement() = default;

    /*
     Get median value between two values
    */
    virtual int64_t getMiddleValue(int64_t min, int64_t max) const = 0;

    /*
     Increase the value of tiles, not exceeding the limit
    */
    virtual void increasedValue(int64_t& value, int64_t limit) const = 0;

    /*
     Decrease the value of tiles, not exceeding the limit
    */
    virtual void decreasedValue(int64_t& value, int64_t limit) const = 0;

    /*
     Get the limit of tiles based on several values
    */
    virtual int64_t getLimitValue(mlir::ArrayRef<int64_t> values) const = 0;
};

}  // namespace vpux::VPU
