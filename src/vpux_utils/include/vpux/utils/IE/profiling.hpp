//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#pragma once

#include <ie_blob.h>
#include <ie_common.h>
#include "vpux/utils/plugin/profiling_parser.hpp"

namespace vpux {

std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> convertProfilingLayersToIEInfo(
        std::vector<vpux::ProfilingLayerInfo>& layerInfo);

}  // namespace vpux
