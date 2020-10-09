// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kmb_test_tool.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>

#include <vpu/utils/error.hpp>

using namespace InferenceEngine;

namespace LayerTestsUtils {

KmbTestTool::KmbTestTool(const KmbTestEnvConfig& envCfg): envConfig(envCfg),
    DEVICE_NAME(envConfig.IE_KMB_TESTS_DEVICE_NAME.empty() ? "VPUX" : envConfig.IE_KMB_TESTS_DEVICE_NAME) {
}

void KmbTestTool::exportNetwork(ExecutableNetwork& exeNet, const std::string& testName) {
    if (envConfig.IE_KMB_TESTS_DUMP_PATH.empty()) {
        std::cout << "IE_KMB_TESTS_DUMP_PATH is not set. Skip blob export" << std::endl;
        return;
    }

    const auto fileName = vpu::formatString("%v/%v.net", envConfig.IE_KMB_TESTS_DUMP_PATH, testName);
    std::cout << "Exporting nn into file " << fileName << std::endl;

    if (envConfig.IE_KMB_TESTS_RAW_EXPORT) {
        exeNet.Export(fileName);
    } else {
        std::ofstream file(fileName, std::ios_base::out | std::ios_base::binary);
        if (!file.is_open())
            THROW_IE_EXCEPTION << "exportNetwork(). Can't open file " << fileName;

        exeNet.Export(file);
    }
    std::cout << "Exported nn into file " << fileName << std::endl;
}

ExecutableNetwork KmbTestTool::importNetwork(const std::shared_ptr<InferenceEngine::Core>& core, const std::string& testName) {
    IE_ASSERT(!envConfig.IE_KMB_TESTS_DUMP_PATH.empty());

    const auto fileName = vpu::formatString("%v/%v.net", envConfig.IE_KMB_TESTS_DUMP_PATH, testName);
    std::cout << "Importing nn from " << (envConfig.IE_KMB_TESTS_RAW_EXPORT ? "" : "not ") << "raw file " << fileName
        << ", device " << DEVICE_NAME << std::endl;

    if (envConfig.IE_KMB_TESTS_RAW_EXPORT) {
        return core->ImportNetwork(fileName, DEVICE_NAME);
    } else {
        std::ifstream file(fileName, std::ios_base::in | std::ios_base::binary);
        if (!file.is_open())
            THROW_IE_EXCEPTION << "importNetwork(). Can't open file " << fileName;

        return core->ImportNetwork(file, DEVICE_NAME);
    }
}

std::string cleanName(std::string name) {
    std::replace_if(
        name.begin(), name.end(),
        [](char c) {
            return !std::isalnum(c);
        },
        '_');
    return name;
}

std::string filesysTestName(const testing::TestInfo* testInfo) {
    return cleanName(vpu::formatString("%v_%v", testInfo->test_case_name(), testInfo->name()));
}

}  // namespace LayerTestsUtils
