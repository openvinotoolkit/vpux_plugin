//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/ppe_version_config.hpp"

using namespace vpux::VPU;

std::unique_ptr<IPpeFactory>& PpeVersionConfig::_getFactory() {
    static std::unique_ptr<IPpeFactory> instance;
    return instance;
}

const IPpeFactory& PpeVersionConfig::getFactory() {
    VPUX_THROW_WHEN(_getFactory() == nullptr, "Tried to access an uninitialized PpeFactory");
    return *_getFactory();
}
