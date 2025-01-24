//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pretty_test_arguments.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include <openvino/core/dimension.hpp>
#include <openvino/core/partial_shape.hpp>
#include <openvino/core/shape.hpp>
#include <vpux/utils/core/checked_cast.hpp>

#include <algorithm>
#include <string>
#include <vector>

ov::test::InputShape staticShape(const ov::Shape& shape) {
    auto partialShape = ov::PartialShape(shape);
    return ov::test::InputShape(std::move(partialShape), {shape});
}

ov::test::InputShape boundedShape(const ov::Shape& bounds) {
    auto boundedDims = std::vector<ov::Dimension>(bounds.size());

    auto toBoundedDim = [](const auto dim) {
        return ov::Dimension(1, dim);
    };

    std::transform(std::begin(bounds), std::end(bounds), std::begin(boundedDims), toBoundedDim);

    return ov::test::InputShape(ov::PartialShape(boundedDims), {bounds});
}

ov::test::InputShape boundedShape(const std::vector<BoundedDim>& boundedDims) {
    auto dimensions = std::vector<ov::Dimension>(boundedDims.size());

    auto toPartialDim = [](const BoundedDim boundedDim) {
        if (boundedDim.dim == -1) {
            return ov::Dimension(1, boundedDim.bound);
        }
        return ov::Dimension(boundedDim.dim);
    };
    std::transform(std::begin(boundedDims), std::end(boundedDims), std::begin(dimensions), toPartialDim);

    auto bounds = ov::Shape(boundedDims.size());
    auto toUpperBound = [](const BoundedDim boundedDim) {
        return vpux::checked_cast<ov::Shape::value_type>(boundedDim.bound);
    };
    std::transform(std::begin(boundedDims), std::end(boundedDims), std::begin(bounds), toUpperBound);

    return ov::test::InputShape(ov::PartialShape(dimensions), {bounds});
}

namespace {
using DimType = decltype(BoundedDim::dim);

std::vector<DimType> generateStaticDims(const BoundedDim& dimInfo) {
    if (dimInfo.dim != -1) {
        return {dimInfo.dim};
    } else {
        std::set<DimType> values{1, dimInfo.bound / 2, dimInfo.bound};
        return std::vector<DimType>(values.begin(), values.end());
    }
}

}  // namespace

std::vector<ov::Shape> generateStaticShapes(const std::vector<BoundedDim>& dims) {
    std::vector<std::vector<DimType>> staticDims;
    staticDims.reserve(dims.size());

    for (const auto& dimInfo : dims) {
        staticDims.push_back(generateStaticDims(dimInfo));
    }

    std::vector<int> indices(staticDims.size(), 0);
    std::set<ov::Shape> allShapes;

    auto incrementIndex = [&]() {
        for (auto i : vpux::irange(indices.size()) | vpux::reversed) {
            auto maxDimIndex = static_cast<int>(staticDims[i].size());
            if (++indices[i] < maxDimIndex) {
                return true;
            } else {
                indices[i] = 0;
            }
        }

        return false;
    };

    while (true) {
        ov::Shape currentShape;
        currentShape.reserve(indices.size());

        for (size_t i = 0; i < indices.size(); ++i) {
            currentShape.push_back(staticDims[i][indices[i]]);
        }

        allShapes.insert(currentShape);

        if (!incrementIndex()) {
            break;
        }
    }

    return std::vector<ov::Shape>(allShapes.begin(), allShapes.end());
}
