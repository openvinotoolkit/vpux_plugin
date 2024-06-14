//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/utils/profiling/reports/api.hpp"

#include "vpux/utils/core/env.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/profiling/parser/api.hpp"
#include "vpux/utils/profiling/reports/tasklist.hpp"
#include "vpux/utils/profiling/taskinfo.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <ostream>
#include <string>
#include <vector>

using namespace vpux;

namespace vpux::profiling {

namespace {

enum class ProfilingFormat { NONE, JSON, TEXT, RAW };

std::string capitalize(const std::string& str) {
    std::string capStr(str);
    std::transform(capStr.begin(), capStr.end(), capStr.begin(), ::toupper);
    return capStr;
}

std::string getProfilingFileName(ProfilingFormat format) {
    const auto filename = env::getEnvVar("NPU_PROFILING_OUTPUT_FILE");
    if (filename.has_value()) {
        return filename.value();
    }
    switch (format) {
    case ProfilingFormat::JSON:
        return "profiling.json";
    case ProfilingFormat::TEXT:
        return "profiling.txt";
    default:
        return "profiling.out";
    }
}

ProfilingFormat getProfilingFormat(const std::string& format) {
    if (format == "JSON")
        return ProfilingFormat::JSON;
    if (format == "TEXT")
        return ProfilingFormat::TEXT;
    if (format == "RAW")
        return ProfilingFormat::RAW;

    VPUX_THROW("Invalid profiling format '{0}'", format);
}

std::ofstream openProfilingStream(ProfilingFormat* format) {
    VPUX_THROW_WHEN(format == nullptr, "Invalid argument");
    *format = ProfilingFormat::NONE;
    const auto printProfiling = env::getEnvVar("NPU_PRINT_PROFILING");
    if (printProfiling.has_value()) {
        *format = getProfilingFormat(capitalize(printProfiling.value()));
    }

    std::ofstream output;
    if (*format != ProfilingFormat::NONE) {
        const auto outFileName = getProfilingFileName(*format);
        auto flags = std::ios::out | std::ios::trunc;
        if (*format == ProfilingFormat::RAW) {
            flags |= std::ios::binary;
        }
        output.open(outFileName, flags);
        if (!output) {
            VPUX_THROW("Can't write into file '{0}'", outFileName);
        }
    }
    return output;
}

void saveProfilingDataToFile(ProfilingFormat format, std::ostream& output, const std::vector<LayerInfo>& layers,
                             const std::vector<TaskInfo>& tasks) {
    static const std::map<std::string, size_t> VERBOSITY_TO_NUM_FILTERS = {
            {"LOW", 1},
            {"MEDIUM", 0},
            {"HIGH", 0},
    };
    auto verbosityValue = capitalize(env::getEnvVar("NPU_PROFILING_VERBOSITY", "HIGH"));
    if (VERBOSITY_TO_NUM_FILTERS.count(verbosityValue) == 0) {
        VPUX_THROW("Invalid NPU_PROFILING_VERBOSITY");
    }
    std::vector<decltype(&isVariantLevelProfilingTask)> verbosityFilters = {&isVariantLevelProfilingTask,
                                                                            &isClusterLevelProfilingTask};
    auto numFilters = VERBOSITY_TO_NUM_FILTERS.at(verbosityValue);
    std::vector<TaskInfo> filteredTasks;
    // Driver return tasks at maximum verbosity, so filter them to needed level
    std::copy_if(tasks.begin(), tasks.end(), std::back_inserter(filteredTasks), [&](const TaskInfo& task) {
        bool toKeep = true;
        for (size_t filterId = 0; filterId < numFilters; ++filterId) {
            toKeep &= !verbosityFilters[filterId](task);
        }
        return toKeep;
    });

    output.exceptions(std::ios::badbit | std::ios::failbit);
    switch (format) {
    case ProfilingFormat::JSON:
        printProfilingAsTraceEvent(filteredTasks, layers, output);
        break;
    case ProfilingFormat::TEXT:
        printProfilingAsText(filteredTasks, layers, output);
        break;
    case ProfilingFormat::RAW:
    case ProfilingFormat::NONE:
        VPUX_THROW("Unsupported profiling format");
    }
}

void saveRawDataToFile(const uint8_t* rawBuffer, size_t size, std::ostream& output) {
    output.write(reinterpret_cast<const char*>(rawBuffer), size);
    output.flush();
}

}  // namespace

std::vector<LayerInfo> getLayerProfilingInfoHook(const uint8_t* profData, size_t profSize, const uint8_t* blobData,
                                                 size_t blobSize) {
    ProfilingFormat format = ProfilingFormat::NONE;
    std::ofstream outFile = openProfilingStream(&format);

    std::vector<LayerInfo> layerData;
    if (outFile.is_open()) {
        if (format == ProfilingFormat::RAW) {
            saveRawDataToFile(profData, profSize, outFile);
            layerData = getLayerInfo(blobData, blobSize, profData, profSize);
        } else {
            auto taskData = getTaskInfo(blobData, blobSize, profData, profSize, VerbosityLevel::HIGH);
            layerData = getLayerInfo(taskData);
            saveProfilingDataToFile(format, outFile, layerData, taskData);
        }
    } else {
        layerData = getLayerInfo(blobData, blobSize, profData, profSize);
    }
    return layerData;
}

std::vector<LayerInfo> getLayerProfilingInfoHook(const std::vector<uint8_t>& data, const std::vector<uint8_t>& blob) {
    return getLayerProfilingInfoHook(data.data(), data.size(), blob.data(), blob.size());
}

std::vector<LayerInfo> getLayerProfilingInfoHook(const uint8_t* profData, size_t profSize,
                                                 const std::vector<uint8_t>& blob) {
    return getLayerProfilingInfoHook(profData, profSize, blob.data(), blob.size());
}

}  // namespace vpux::profiling
