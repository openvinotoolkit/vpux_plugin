//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/vpu_test_env_cfg.hpp"
#include "intel_npu/al/config/common.hpp"
#include "vpux/utils/IE/config.hpp"

#include <cstdlib>
#include <stdexcept>

namespace ov::test::utils {

VpuTestEnvConfig::VpuTestEnvConfig() {
    // start reading obsolete environment variables
    if (auto var = std::getenv("IE_KMB_TESTS_DEVICE_NAME")) {
        IE_NPU_TESTS_DEVICE_NAME = var;
    }

    if (auto var = std::getenv("IE_KMB_TESTS_DUMP_PATH")) {
        IE_NPU_TESTS_DUMP_PATH = var;
    }

    if (auto var = std::getenv("IE_KMB_TESTS_LOG_LEVEL")) {
        IE_NPU_TESTS_LOG_LEVEL = var;
    }

    if (auto var = std::getenv("IE_KMB_TESTS_RUN_COMPILER")) {
        IE_NPU_TESTS_RUN_COMPILER = vpux::envVarStrToBool("IE_KMB_TESTS_RUN_COMPILER", var);
    }

    if (auto var = std::getenv("IE_KMB_TESTS_RUN_EXPORT")) {
        IE_NPU_TESTS_RUN_EXPORT = vpux::envVarStrToBool("IE_KMB_TESTS_RUN_EXPORT", var);
    }

    if (auto var = std::getenv("IE_KMB_TESTS_RUN_IMPORT")) {
        IE_NPU_TESTS_RUN_IMPORT = vpux::envVarStrToBool("IE_KMB_TESTS_RUN_IMPORT", var);
    }

    if (auto var = std::getenv("IE_KMB_TESTS_RUN_INFER")) {
        IE_NPU_TESTS_RUN_INFER = vpux::envVarStrToBool("IE_KMB_TESTS_RUN_INFER", var);
    }

    if (auto var = std::getenv("IE_KMB_TESTS_EXPORT_INPUT")) {
        IE_NPU_TESTS_EXPORT_INPUT = vpux::envVarStrToBool("IE_KMB_TESTS_EXPORT_INPUT", var);
    }

    if (auto var = std::getenv("IE_KMB_TESTS_EXPORT_OUTPUT")) {
        IE_NPU_TESTS_EXPORT_OUTPUT = vpux::envVarStrToBool("IE_KMB_TESTS_EXPORT_OUTPUT", var);
    }

    if (auto var = std::getenv("IE_KMB_TESTS_EXPORT_REF")) {
        IE_NPU_TESTS_EXPORT_REF = vpux::envVarStrToBool("IE_KMB_TESTS_EXPORT_REF", var);
    }

    if (auto var = std::getenv("IE_KMB_TESTS_IMPORT_INPUT")) {
        IE_NPU_TESTS_IMPORT_INPUT = vpux::envVarStrToBool("IE_KMB_TESTS_IMPORT_INPUT", var);
    }

    if (auto var = std::getenv("IE_KMB_TESTS_IMPORT_REF")) {
        IE_NPU_TESTS_IMPORT_REF = vpux::envVarStrToBool("IE_KMB_TESTS_IMPORT_REF", var);
    }

    if (auto var = std::getenv("IE_KMB_TESTS_RAW_EXPORT")) {
        IE_NPU_TESTS_RAW_EXPORT = vpux::envVarStrToBool("IE_KMB_TESTS_RAW_EXPORT", var);
    }

    if (auto var = std::getenv("IE_KMB_TESTS_LONG_FILE_NAME")) {
        IE_NPU_TESTS_LONG_FILE_NAME = vpux::envVarStrToBool("IE_KMB_TESTS_LONG_FILE_NAME", var);
    }

    if (auto var = std::getenv("IE_KMB_TESTS_PLATFORM")) {
        IE_NPU_TESTS_PLATFORM = var;
    }
    // end reading obsolete environment variables

    if (auto var = std::getenv("IE_NPU_TESTS_DEVICE_NAME")) {
        IE_NPU_TESTS_DEVICE_NAME = var;
    }

    if (auto var = std::getenv("IE_NPU_TESTS_DUMP_PATH")) {
        IE_NPU_TESTS_DUMP_PATH = var;
    }

    if (auto var = std::getenv("IE_NPU_TESTS_LOG_LEVEL")) {
        IE_NPU_TESTS_LOG_LEVEL = var;
    }

    if (auto var = std::getenv("IE_NPU_TESTS_RUN_COMPILER")) {
        IE_NPU_TESTS_RUN_COMPILER = vpux::envVarStrToBool("IE_NPU_TESTS_RUN_COMPILER", var);
    }

    if (auto var = std::getenv("IE_NPU_TESTS_RUN_EXPORT")) {
        IE_NPU_TESTS_RUN_EXPORT = vpux::envVarStrToBool("IE_NPU_TESTS_RUN_EXPORT", var);
    }

    if (auto var = std::getenv("IE_NPU_TESTS_RUN_IMPORT")) {
        IE_NPU_TESTS_RUN_IMPORT = vpux::envVarStrToBool("IE_NPU_TESTS_RUN_IMPORT", var);
    }

    if (auto var = std::getenv("IE_NPU_TESTS_RUN_INFER")) {
        IE_NPU_TESTS_RUN_INFER = vpux::envVarStrToBool("IE_NPU_TESTS_RUN_INFER", var);
    }

    if (auto var = std::getenv("IE_NPU_TESTS_EXPORT_INPUT")) {
        IE_NPU_TESTS_EXPORT_INPUT = vpux::envVarStrToBool("IE_NPU_TESTS_EXPORT_INPUT", var);
    }

    if (auto var = std::getenv("IE_NPU_TESTS_EXPORT_OUTPUT")) {
        IE_NPU_TESTS_EXPORT_OUTPUT = vpux::envVarStrToBool("IE_NPU_TESTS_EXPORT_OUTPUT", var);
    }

    if (auto var = std::getenv("IE_NPU_TESTS_EXPORT_REF")) {
        IE_NPU_TESTS_EXPORT_REF = vpux::envVarStrToBool("IE_NPU_TESTS_EXPORT_REF", var);
    }

    if (auto var = std::getenv("IE_NPU_TESTS_IMPORT_INPUT")) {
        IE_NPU_TESTS_IMPORT_INPUT = vpux::envVarStrToBool("IE_NPU_TESTS_IMPORT_INPUT", var);
    }

    if (auto var = std::getenv("IE_NPU_TESTS_IMPORT_REF")) {
        IE_NPU_TESTS_IMPORT_REF = vpux::envVarStrToBool("IE_NPU_TESTS_IMPORT_REF", var);
    }

    if (auto var = std::getenv("IE_NPU_TESTS_RAW_EXPORT")) {
        IE_NPU_TESTS_RAW_EXPORT = vpux::envVarStrToBool("IE_NPU_TESTS_RAW_EXPORT", var);
    }

    if (auto var = std::getenv("IE_NPU_TESTS_LONG_FILE_NAME")) {
        IE_NPU_TESTS_LONG_FILE_NAME = vpux::envVarStrToBool("IE_NPU_TESTS_LONG_FILE_NAME", var);
    }

    if (auto var = std::getenv("IE_NPU_TESTS_PLATFORM")) {
        IE_NPU_TESTS_PLATFORM = var;
    }
}

const VpuTestEnvConfig& VpuTestEnvConfig::getInstance() {
    static VpuTestEnvConfig instance{};
    return instance;
}

std::string getTestsDeviceNameFromEnvironmentOr(const std::string& instead) {
    return (!VpuTestEnvConfig::getInstance().IE_NPU_TESTS_DEVICE_NAME.empty())
                   ? VpuTestEnvConfig::getInstance().IE_NPU_TESTS_DEVICE_NAME
                   : instead;
}

std::string getTestsPlatformFromEnvironmentOr(const std::string& instead) {
    return (!VpuTestEnvConfig::getInstance().IE_NPU_TESTS_PLATFORM.empty())
                   ? VpuTestEnvConfig::getInstance().IE_NPU_TESTS_PLATFORM
                   : instead;
}

std::string getTestsPlatformCompilerInPlugin() {
    return getTestsPlatformFromEnvironmentOr(
            getTestsDeviceNameFromEnvironmentOr(std::string(ov::intel_npu::Platform::AUTO_DETECT)));
}

std::string getDeviceNameTestCase(const std::string& str) {
    ov::DeviceIDParser parser = ov::DeviceIDParser(str);
    return parser.get_device_name() + parser.get_device_id();
}

std::string getDeviceName() {
    return LayerTestsUtils::getTestsDeviceNameFromEnvironmentOr("NPU.3700");
}

std::string getDeviceNameID(const std::string& str) {
    ov::DeviceIDParser parser = ov::DeviceIDParser(str);
    return parser.get_device_id();
}

}  // namespace ov::test::utils
