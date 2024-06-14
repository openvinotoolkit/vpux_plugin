//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "BitStream.hpp"

using namespace vpux;

BitStream::BitStream(const std::vector<uint8_t>& bits) {
    source_byte_count_ = static_cast<uint32_t>(bits.size());
    auto size{source_byte_count_ + 7u};
    size >>= 3;

    bit_array_.resize(size + 1u);
    array_words_ = static_cast<uint32_t>(bit_array_.size());
    array_bytes_ = array_words_ << 3;
    std::memcpy(reinterpret_cast<void*>(bit_array_.data()), reinterpret_cast<const void*>(bits.data()), bits.size());
}

void BitStream::allocate_bits(const uint32_t& bits) {
    bit_count_ = bits;
    auto size{bits + 63ull};
    size >>= 6;

    bit_array_.resize(size + 1ull);
    array_words_ = static_cast<uint32_t>(bit_array_.size());
    array_bytes_ = array_words_ << 3;
}

void BitStream::append(const BitStream& stream) {
    int32_t bits{static_cast<int32_t>(stream.get_bit_count())};
    uint64_t stream_bits{};
    uint32_t bit_start{};

    while (bits > 0) {
        auto add_bits{bits >= 64 ? 64 : bits};
        stream.read(bit_start, stream_bits);

        write(stream_bits, static_cast<uint32_t>(add_bits));
        bits -= add_bits;
        bit_start += static_cast<uint32_t>(add_bits);
    }
}

bool BitStream::read(const uint32_t& bit_start, uint64_t& bits) const {
    bits = 0ull;
    const auto bit_start_word{bit_start >> 6};

    if (bit_start_word >= array_words_) {
        return false;
    }

    const auto bs{bit_start - (bit_start_word << 6)};
    bits = bit_array_[bit_start_word] >> bs;

    if (bs) {
        bits |= (bit_array_[bit_start_word + 1u] << (64 - bs));
    }

    return true;
}

uint32_t BitStream::write(const uint64_t& bits, const uint32_t& bit_count) {
    const auto bit_start_word{bit_position_ >> 6};
    // 64 bits max
    const auto insert_bit_count{bit_count > 64 ? 64 : bit_count};
    uint64_t insert_mask0 = insert_bit_count == 64u ? ~0ull : ~(~0ull << insert_bit_count);
    const uint64_t bits_sel{bits & insert_mask0};
    const auto bs{bit_position_ - (bit_start_word << 6)};
    const uint64_t insert_mask1{~(~0ull << bs)};

    bit_array_[bit_start_word] |= (bits_sel << bs);

    if (bs) {
        bit_array_[bit_start_word + 1u] |= ((bits_sel >> (64 - bs)) & insert_mask1);
    }

    bit_position_ += insert_bit_count;
    return insert_bit_count;
}

uint32_t BitStream::stream_length() {
    uint32_t len{static_cast<uint32_t>(bit_array_.size()) << 3};
    return len;
}

void BitStream::read(std::vector<uint8_t>& out, const uint32_t& byte_alignment) const {
    const auto alignment_bits{byte_alignment << 3};

    auto size{bit_position_ + alignment_bits - 1u};
    size /= alignment_bits;
    size *= alignment_bits;
    size >>= 3;

    const auto copy_bytes{(bit_position_ + 7u) >> 3};

    out.resize(size);

    std::memcpy(reinterpret_cast<void*>(out.data()), reinterpret_cast<const void*>(bit_array_.data()), copy_bytes);
}
