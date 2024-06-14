//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "intel_npu/al/profiling.hpp"
#include "vpux/IMD/infer_request.hpp"
#include "vpux/utils/profiling/parser/api.hpp"
#include "vpux/utils/profiling/reports/api.hpp"

#include <vector>

namespace vpux::profiling {

LayerStatistics getLayerStatistics(const uint8_t* profData, size_t profSize, const std::vector<uint8_t>& blob) {
    auto layerData = getLayerProfilingInfoHook(profData, profSize, blob);
    return intel_npu::profiling::convertLayersToIeProfilingInfo(layerData);
}

}  // namespace vpux::profiling
