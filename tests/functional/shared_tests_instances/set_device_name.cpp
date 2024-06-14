// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <stdexcept>
#include <string>

#include "functional_test_utils/ov_plugin_cache.hpp"

#include "common/utils.hpp"
#include "set_device_name.hpp"
namespace ov {
namespace test {

void set_device_suffix(const std::string& suffix) {
    std::string new_vpux_name = ov::test::utils::DEVICE_NPU;
    new_vpux_name += ov::test::utils::DEVICE_SUFFIX_SEPARATOR;
    new_vpux_name += suffix;
    auto available_devices = utils::PluginCache::get().core()->get_available_devices();
    if (std::find(available_devices.begin(), available_devices.end(), new_vpux_name) == available_devices.end()) {
        std::string msg("The device " + new_vpux_name + " is not in the available devices! Please use other on!");
        throw std::runtime_error(msg);
    }
    ov::test::utils::DEVICE_NPU = new_vpux_name.c_str();
}

}  // namespace test
}  // namespace ov
