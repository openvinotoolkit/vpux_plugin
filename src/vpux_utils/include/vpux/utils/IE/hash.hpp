//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/utils/core/hash.hpp"

#include <openvino/core/node_output.hpp>

//
// Hash
//

namespace std {

template <>
struct hash<ov::Output<ov::Node>> final {
    size_t operator()(const ov::Output<ov::Node>& out) const {
        return vpux::getHash(out.get_node(), out.get_index());
    }
};

}  // namespace std
