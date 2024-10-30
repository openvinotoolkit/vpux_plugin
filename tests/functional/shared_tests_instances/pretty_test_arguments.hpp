//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <openvino/core/dimension.hpp>
#include <openvino/core/partial_shape.hpp>
#include <openvino/core/shape.hpp>
#include <type_traits>
#include <vpux/utils/core/checked_cast.hpp>
#include "shared_test_classes/base/ov_subgraph.hpp"

#include <string>
#include <vector>

#define PRETTY_PARAM(name, type)                                                 \
    class name {                                                                 \
    public:                                                                      \
        using paramType = type;                                                  \
        name(paramType arg = paramType()): val_(std::move(arg)) {                \
        }                                                                        \
        operator paramType() const {                                             \
            return val_;                                                         \
        }                                                                        \
                                                                                 \
    private:                                                                     \
        paramType val_;                                                          \
    };                                                                           \
    static inline void PrintTo(const name& param, ::std::ostream* os) {          \
        *os << #name ": " << ::testing::PrintToString((name::paramType)(param)); \
    }

PRETTY_PARAM(Device, std::string);

template <typename... Dims>
ov::Shape makeShape(Dims... dims) {
    using signed_dims_t = std::make_signed_t<ov::Shape::value_type>;
    return ov::Shape{vpux::checked_cast<ov::Shape::value_type, signed_dims_t>(dims)...};
}

inline ov::test::InputShape staticShape(const ov::Shape& shape) {
    auto partialShape = ov::PartialShape(shape);
    return ov::test::InputShape(std::move(partialShape), {shape});
}

template <typename... Dims>
ov::test::InputShape staticShape(Dims... dims) {
    return staticShape(makeShape(dims...));
}

inline ov::test::InputShape boundedShape(const ov::Shape& bounds) {
    auto boundedDims = std::vector<ov::Dimension>(bounds.size());

    auto toBoundedDim = [](const auto dim) {
        return ov::Dimension(1, dim);
    };

    std::transform(std::begin(bounds), std::end(bounds), std::begin(boundedDims), toBoundedDim);

    return ov::test::InputShape(ov::PartialShape(boundedDims), {bounds});
}

template <typename... Dims>
ov::test::InputShape boundedShape(Dims... dims) {
    return boundedShape(makeShape(dims...));
}
