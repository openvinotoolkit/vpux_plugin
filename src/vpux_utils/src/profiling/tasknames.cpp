//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// Task name helpers

#include "vpux/utils/profiling/tasknames.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/profiling/location.hpp"

#include <sstream>

namespace vpux::profiling {

const std::string CLUSTER_LEVEL_PROFILING_SUFFIX = "cluster";
const std::string VARIANT_LEVEL_PROFILING_SUFFIX = "variant";

std::vector<std::string> splitBySeparator(const std::string& s, char separator) {
    std::istringstream iss(s);
    std::string part;
    std::vector<std::string> parts;
    while (std::getline(iss, part, separator)) {
        parts.push_back(part);
    }
    return parts;
}

TokenizedTaskName tokenizeTaskName(const std::string& taskName) {
    auto nameSepPos = taskName.rfind(vpux::LOCATION_ORIGIN_SEPARATOR);
    VPUX_THROW_WHEN(nameSepPos == std::string::npos, "Malformed task name: '{0}'", taskName);
    auto layerName = taskName.substr(0, nameSepPos);
    auto afterNameSep = taskName.substr(nameSepPos + 1);
    std::vector<std::string> parts = splitBySeparator(afterNameSep, vpux::LOCATION_SEPARATOR);
    return {std::move(layerName), std::move(parts)};
}

ParsedTaskName deserializeTaskName(const std::string& fullTaskName) {
    const auto LOC_METADATA_SEPARATOR = '_';  // conventional separator used for attaching metadata to MLIR Locations

    auto tokenized = tokenizeTaskName(fullTaskName);
    std::string layerType = "<unknown>";
    std::string& layerName = tokenized.layerName;

    for (const auto& token : tokenized.tokens) {
        VPUX_THROW_WHEN(token.empty(), "Empty task name token");

        auto parts = splitBySeparator(token, LOC_METADATA_SEPARATOR);
        auto partsNum = parts.size();

        if (partsNum == 2 && parts[0] == "t") {
            layerType = parts[1];
        }
    }

    return {std::move(layerName), std::move(layerType)};
}

std::string getLayerName(const std::string& taskName) {
    return taskName.substr(0, taskName.rfind(vpux::LOCATION_ORIGIN_SEPARATOR));
}

std::string getTaskNameSuffixes(const std::string& name) {
    const auto startPos = name.rfind(LOCATION_ORIGIN_SEPARATOR);
    if (startPos == std::string::npos) {
        return "";
    }
    return name.substr(startPos + 1);
}

std::string getClusterFromName(const std::string& name) {
    return getValueFromStructuredTaskName(name, CLUSTER_LEVEL_PROFILING_SUFFIX);
}

std::string getVariantFromName(const std::string& name) {
    return getValueFromStructuredTaskName(name, VARIANT_LEVEL_PROFILING_SUFFIX);
}

std::string getValueFromStructuredTaskName(const std::string& name, const std::string& key) {
    auto taskNameSuffixes = getTaskNameSuffixes(name);
    auto suffixes = splitBySeparator(taskNameSuffixes, LOCATION_SEPARATOR);

    for (auto& suffix : suffixes) {
        auto parts = splitBySeparator(suffix, '_');
        if (parts.size() == 2) {
            auto extractedKey = parts[0];
            auto extractedValue = parts[1];
            if (extractedKey == key) {
                return extractedValue;
            }
        }
    }
    return "";
}

}  // namespace vpux::profiling
