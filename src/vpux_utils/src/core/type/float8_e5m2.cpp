// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpux/utils/core/type/float16.hpp>
#include <vpux/utils/core/type/float8_e5m2.hpp>

#include <array>
#include <cmath>
#include <limits>

static_assert(sizeof(vpux::type::float8_e5m2) == 1, "class f8e5m2 must be exactly 1 byte");
static_assert(std::is_trivially_constructible<vpux::type::float8_e5m2, vpux::type::float8_e5m2>::value,
              "should be trivially constructible");
static_assert(std::is_trivially_copyable<vpux::type::float8_e5m2>::value, "must be trivially copyable");
static_assert(std::is_trivially_destructible<vpux::type::float8_e5m2>::value, "must be trivially destructible");
static_assert(std::numeric_limits<vpux::type::float8_e5m2>::is_specialized, "numeric_limits must be specialized");
static_assert(!std::numeric_limits<vpux::type::float8_e5m2>::is_integer, "numeric_limits::is_integer must be false");

constexpr uint8_t byte_shift = 8;

constexpr uint8_t f8e5m2_e_size = 5;     // f8e5m2 exponent bit size
constexpr uint8_t f8e5m2_e_mask = 0x7c;  // f8e5m2 exponent bit mask
constexpr uint8_t f8e5m2_m_size = 2;     // f8e5m2 mantissa bits size
constexpr uint8_t f8e5m2_m_mask = 0x03;  // f8e5m2 mantissa bit mask

void emulate_f8e5m2_on_fp16(const vpux::type::float16* const arg_f, vpux::type::float16* out_f, size_t count) {
    const auto arg_u = reinterpret_cast<const uint16_t*>(arg_f);
    auto out_u = reinterpret_cast<uint16_t*>(out_f);
    uint16_t val_bit_repr;

    constexpr auto exp_bits = 5;
    constexpr auto mbits = 8;
    constexpr auto non_mant_bits = exp_bits + 1;  /// exponent + sign
    constexpr auto lshift = 10 - (mbits - non_mant_bits);
    constexpr uint16_t mask_mant = static_cast<uint16_t>(0xFFFF << lshift);  /// 1111111111111111 -> 1 11111 1100000000
    constexpr uint16_t grs_bitmask = 0x00FF;  /// 0 00000 0011111111, grs denotes guard, round, sticky bits
    constexpr uint16_t rne_tie = 0x0180;      /// 0 00000 0110000000, rne denotes round to nearest even
    constexpr uint16_t fp16_inf = 0x7C00;

    for (size_t i = 0; i < count; ++i) {
        /// converts float number to half precision in round-to-nearest-even mode and returns half with converted value.
        val_bit_repr = arg_u[i];
        /// 0x7c00 = 0111110000000000 - exponent mask
        /// s 11111 xxx xxxx xxxx - is nan (if some x is 1) or inf (if all x is 0)
        /// 0x7800 is 0111100000000000 and 0x400 is 0000010000000000
        /// number is not normal if all exponent is 1 or 0
        /// 0x7f00 is 0 11111 1100000000
        /// 0x7b00 is 0 11110 1100000000
        const bool can_round = ((val_bit_repr & 0x7F00) < 0x7B00) ? true : false;
        /// s 11111 xxx xxxx xxxx - is nan (if some x is 1) or inf (if all x is 0)
        const bool is_naninf = ((val_bit_repr & fp16_inf) == fp16_inf) ? true : false;
        /* nearest rounding masks */
        /// grs_bitmask - grs_bitmask is 0 00000 0011111111 or 0 00000 00grs11111
        uint16_t rnmask = (val_bit_repr & grs_bitmask);
        /// rne_tie - 0x180 is      0 00000 0110000000 or 384.0
        uint16_t rnmask_tie = (val_bit_repr & rne_tie);

        if (!is_naninf && can_round) {
            /* round to nearest even, if rne_mask is enabled */
            /* 0 00000 0010000000, find grs patterns */
            // 0xx - do nothing
            // 100 - this is a tie : round up if the mantissa's bit just before G is 1, else do nothing
            // 101, 110, 111 - round up > 0x0080
            val_bit_repr += (((rnmask > 0x0080) || (rnmask_tie == rne_tie)) << lshift);
        }
        val_bit_repr &= mask_mant; /* truncation */
        out_u[i] = val_bit_repr;
    }
}

uint8_t f32_to_f8e5m2_bits(const float value) {
    auto f16 = static_cast<vpux::type::float16>(value);
    emulate_f8e5m2_on_fp16(&f16, &f16, 1);
    return static_cast<uint8_t>((f16.to_bits() >> byte_shift));
}

vpux::type::float8_e5m2::float8_e5m2(uint32_t sign, uint32_t biased_exponent, uint32_t fraction)
        : m_value((sign & 0x01) << (f8e5m2_e_size + f8e5m2_m_size) |
                  (biased_exponent & (f8e5m2_e_mask >> f8e5m2_m_size)) << f8e5m2_m_size | (fraction & f8e5m2_m_mask)) {
}

vpux::type::float8_e5m2::float8_e5m2(const float value): m_value(f32_to_f8e5m2_bits(value)){};

vpux::type::float8_e5m2::operator float() const {
    return static_cast<float>(float16::from_bits((static_cast<uint16_t>(m_value) << byte_shift)));
}

uint8_t vpux::type::float8_e5m2::to_bits() const {
    return m_value;
}
