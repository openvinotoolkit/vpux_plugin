//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/strides_utils.hpp"

namespace vpux {
namespace VPUIP {

MemDimArr getStridesMemDims(vpux::NDTypeInterface tensorType) {
    const auto elemSize = tensorType.getElemTypeSize();
    const auto memShapes = tensorType.getMemShape().raw();
    const auto memStrides = tensorType.getMemStrides().raw();

    MemDimArr stridesDims;
    for (auto ind : irange(memShapes.size()) | reversed) {
        auto dim = MemDim(ind);
        if (ind == memShapes.size() - 1 && memStrides[ind] != elemSize) {
            stridesDims.push_back(dim);
        } else if (ind != memShapes.size() - 1) {
            const auto prevMemDim = ind + 1;
            if (memStrides[ind] != memStrides[prevMemDim] * memShapes[prevMemDim]) {
                stridesDims.push_back(dim);
            }
        }
    }

    return stridesDims;
}

}  // namespace VPUIP
}  // namespace vpux
