//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/utils/core/string_ref.hpp"

namespace vpux {

constexpr StringLiteral multiClusterStrategy = "multiClusterStrategy";  // only be used for manual strategy utils
constexpr StringLiteral tilingStrategy = "tilingStrategy";
constexpr StringLiteral defaultNoValue = "NONE";
constexpr StringLiteral verticalFusion = "verticalFusion";  // only be used for manual strategy utils
constexpr StringLiteral verticalFusionHash = "verticalFusionHash";
constexpr StringLiteral layerTypeName = "layerType";
constexpr StringLiteral updatedVFTiling = "updatedVFTiling";
constexpr StringLiteral outputPipelining = "outputPipelining";

}  // namespace vpux
