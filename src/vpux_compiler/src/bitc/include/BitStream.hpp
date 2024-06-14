//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>

namespace vpux {
class BitStream {
public:
    BitStream() = default;
    BitStream(const std::vector<uint8_t>& bits);
    bool read(const uint32_t& bit_start, uint64_t& bits) const;
    uint32_t write(const uint64_t& bits, const uint32_t& bit_count);
    uint8_t* get_byte_pointer(const uint32_t& byte_index) {
        assert(byte_index < array_bytes_);
        return reinterpret_cast<uint8_t*>(bit_array_.data()) + byte_index;
    }
    uint32_t stream_length();
    uint32_t source_stream_length() {
        return source_byte_count_;
    }
    void allocate_bits(const uint32_t& bits);
    void append(const BitStream& stream);
    uint32_t get_bit_count() const {
        return bit_position_;
    }
    void read(std::vector<uint8_t>& out, const uint32_t& byte_alignment) const;

private:
    std::vector<uint64_t> bit_array_;
    uint32_t array_words_{};
    uint32_t array_bytes_{};
    uint32_t bit_position_{};
    uint32_t bit_count_{};
    uint32_t source_byte_count_{};
};
}  // namespace vpux
