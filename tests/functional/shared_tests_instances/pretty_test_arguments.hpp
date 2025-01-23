//
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"

#include <openvino/core/dimension.hpp>
#include <openvino/core/partial_shape.hpp>
#include <openvino/core/shape.hpp>
#include <vpux/utils/core/checked_cast.hpp>

#include <string>
#include <type_traits>
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

struct BoundedDim {
    int dim;
    int bound;

    BoundedDim(int value): dim(value), bound(value) {
        VPUX_THROW_UNLESS(value > 0, "Static dimension must have positive value, got: {0}", value);
    }

    BoundedDim(int dim, int bound): dim(dim), bound(bound) {
        VPUX_THROW_UNLESS(dim == -1, "Dynamic dimension must have value -1, got: {0}", dim);
        VPUX_THROW_UNLESS(bound > 0, "Upper bound must have positive value, got: {0}", bound);
    }
};

inline BoundedDim operator"" _Dyn(unsigned long long value) {
    return BoundedDim{-1, vpux::checked_cast<int>(value)};
}

template <typename... Dims>
auto makeShape(Dims... dims) {
    if constexpr ((std::is_same_v<Dims, BoundedDim> || ...)) {
        return std::vector<BoundedDim>{BoundedDim(dims)...};
    } else {
        using signed_dims_t = std::make_signed_t<ov::Shape::value_type>;
        return ov::Shape{vpux::checked_cast<ov::Shape::value_type, signed_dims_t>(dims)...};
    }
}

ov::test::InputShape staticShape(const ov::Shape& shape);

template <typename... Dims>
ov::test::InputShape staticShape(Dims... dims) {
    return staticShape(makeShape(dims...));
}

ov::test::InputShape boundedShape(const ov::Shape& bounds);
ov::test::InputShape boundedShape(const std::vector<BoundedDim>& boundedDims);

template <typename... Dims>
ov::test::InputShape boundedShape(Dims... dims) {
    return boundedShape(makeShape(dims...));
}

std::vector<ov::Shape> generateStaticShapes(const std::vector<BoundedDim>& dims);

template <typename... Dims>
ov::test::InputShape generateShapes(Dims... dims) {
    if constexpr ((std::is_same_v<Dims, BoundedDim> || ...)) {
        auto staticShapes = generateStaticShapes(std::vector<BoundedDim>{dims...});
        auto testShape = boundedShape(dims...);
        testShape.second = staticShapes;
        return testShape;
    } else {
        return staticShape(dims...);
    }
}
