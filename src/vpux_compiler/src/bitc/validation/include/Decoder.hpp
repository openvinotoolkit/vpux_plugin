//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>
#include "bitc.hpp"

namespace vpux {
namespace bitc {
class Decoder {
public:
    Decoder(const std::vector<uint8_t>& bits, BitCompactorConfig config);
    Decoder(const std::vector<uint8_t>&& bits, BitCompactorConfig config);
    void decode(std::vector<uint8_t>& out);
    void decode(std::vector<uint8_t>& out, std::vector<uint8_t>& bitmap, unsigned sparse_block_size);
    ~Decoder();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace bitc
}  // namespace vpux
