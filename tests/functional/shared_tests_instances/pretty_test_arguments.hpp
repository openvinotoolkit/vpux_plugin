//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/partial_shape.hpp>
#include <vpux/utils/core/checked_cast.hpp>
#include "shared_test_classes/base/ov_subgraph.hpp"

#include <string>
#include <vector>

#define PRETTY_PARAM(name, type)                                                 \
    class name {                                                                 \
    public:                                                                      \
        using paramType = type;                                                  \
        name(paramType arg = paramType()): val_(arg) {                           \
        }                                                                        \
        operator paramType() const {                                             \
            return val_;                                                         \
        }                                                                        \
                                                                                 \
    private:                                                                     \
        paramType val_;                                                          \
    };                                                                           \
    static inline void PrintTo(name param, ::std::ostream* os) {                 \
        *os << #name ": " << ::testing::PrintToString((name::paramType)(param)); \
    }

PRETTY_PARAM(Device, std::string);

template <typename... Dims>
ov::Shape makeShape(Dims... dims) {
    return ov::Shape{vpux::checked_cast<size_t>(static_cast<int>(dims))...};
}

inline ov::test::InputShape staticShape(ov::Shape shape) {
    auto partialShape = ov::PartialShape(shape);
    return ov::test::InputShape(std::move(partialShape), {std::move(shape)});
}

template <typename... Dims>
ov::test::InputShape staticShape(Dims... dims) {
    return staticShape(makeShape(dims...));
}
