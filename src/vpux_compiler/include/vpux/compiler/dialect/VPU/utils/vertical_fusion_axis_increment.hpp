//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/vf_axis_increment.hpp"

namespace vpux {
namespace VPU {

/*
 Class defines functions to operate with spatial dimensions
*/
class DefaultVFAxisIncrement : public IVFAxisIncrement {
public:
    /*
     Get median value between two values
    */
    int64_t getMiddleValue(int64_t min, int64_t max) const override;

    /*
     Increase the value of tiles, not exceeding the limit
    */
    void increasedValue(int64_t& value, int64_t limit) const override;

    /*
     Decrease the value of tiles, not exceeding the limit
    */
    void decreasedValue(int64_t& value, int64_t limit) const override;

    /*
     Get the limit of tiles based on several values
    */
    int64_t getLimitValue(mlir::ArrayRef<int64_t> values) const override;
};

/*
 Class defines functions to operate with channels dimension
*/
class ChannelsVFAxisIncrement : public IVFAxisIncrement {
public:
    /*
     Get median value between two values
    */
    int64_t getMiddleValue(int64_t min, int64_t max) const override;

    /*
     Increase the value of tiles, not exceeding the limit
    */
    void increasedValue(int64_t& value, int64_t limit) const override;

    /*
     Decrease the value of tiles, not exceeding the limit
    */
    void decreasedValue(int64_t& value, int64_t limit) const override;

    /*
     Get the limit of tiles based on several values
    */
    int64_t getLimitValue(mlir::ArrayRef<int64_t> values) const override;
};

}  // namespace VPU
}  // namespace vpux
