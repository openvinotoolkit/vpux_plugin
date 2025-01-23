//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/properties_tests.hpp"
#include <array>
#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "intel_npu/config/common.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "openvino/runtime/intel_cpu/properties.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

namespace {

template <typename T>
constexpr std::vector<T> operator+(const std::vector<T>& vector1, const std::vector<T>& vector2) {
    std::vector<T> result;
    result.insert(std::end(result), std::begin(vector1), std::end(vector1));
    result.insert(std::end(result), std::begin(vector2), std::end(vector2));
    return result;
}

ov::log::Level getTestsLogLevelFromEnvironmentOr(ov::log::Level instead) {
    if (auto var = std::getenv("OV_NPU_LOG_LEVEL")) {
        std::istringstream stringStream = std::istringstream(var);
        ov::log::Level level;

        stringStream >> level;

        return level;
    }
    return instead;
}

const std::vector<ov::AnyMap> configsDeviceProperties = {
        {ov::device::properties(ov::test::utils::DEVICE_NPU, ov::num_streams(ov::streams::AUTO))},
        {ov::device::properties(
                ov::AnyMap{{ov::test::utils::DEVICE_NPU, ov::AnyMap{ov::num_streams(ov::streams::AUTO)}}})}};

INSTANTIATE_TEST_SUITE_P(BehaviorTests_OVGetConfigTest_nightly, OVGetConfigTest,
                         ::testing::Values(ov::test::utils::DEVICE_NPU),
                         ov::test::utils::appendPlatformTypeTestName<OVGetConfigTest>);

// IE Class load and check network with ov::device::properties
// OVClassCompileModelAndCheckSecondaryPropertiesTest only works with property num_streams of type int32_t
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_BehaviorTests_OVClassLoadNetworkAndCheckWithSecondaryPropertiesTest,
                         OVClassCompileModelAndCheckSecondaryPropertiesTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU, "AUTO:NPU", "MULTI:NPU"),
                                            ::testing::ValuesIn(configsDeviceProperties)));

};  // namespace
