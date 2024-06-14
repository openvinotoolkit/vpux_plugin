// {% copyright %}

#pragma once

#include <map>
#include <string>

#include <llvm/ADT/bit.h>

#include <mlir/IR/Builders.h>

#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPURegMapped/attributes.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"

namespace vpux {
namespace VPURegMapped {

void updateRegMappedInitializationValues(std::map<std::string, std::map<std::string, uint64_t>>& values,
                                         const std::map<std::string, std::map<std::string, uint64_t>>& newValues);

size_t calcMinBitsRequirement(uint64_t value);

template <typename RegType>
vpux::VPURegMapped::RegMappedType getRegMappedWithValues(
        mlir::OpBuilder builder, const std::map<std::string, std::map<std::string, uint64_t>>& newValues =
                                         std::map<std::string, std::map<std::string, uint64_t>>()) {
    auto resetValues = RegType::getResetInitilizationValues();
    updateRegMappedInitializationValues(resetValues, newValues);
    return RegType::get(builder, resetValues);
}

template <typename RegType>
vpux::VPURegMapped::RegisterMappedAttr getRegMappedAttributeWithValues(
        mlir::OpBuilder builder, const std::map<std::string, std::map<std::string, uint64_t>>& newValues =
                                         std::map<std::string, std::map<std::string, uint64_t>>()) {
    return vpux::VPURegMapped::RegisterMappedAttr::get(builder.getContext(),
                                                       getRegMappedWithValues<RegType>(builder, newValues));
}

// src is unsigned int or enum or bool; reg. field is SINT or UINT

// example: BIT_WIDTH = 4, RF_TYPE=UINT, src = 8:
// src:             0000000000000000000000000000000000000000000000000000000000001000
// bitMask:         1111111111111111111111111111111111111111111111111111111111110000
// src & bitMask:   0000000000000000000000000000000000000000000000000000000000000000
// check:           passed
// result:          0000000000000000000000000000000000000000000000000000000000001000 (8:uint64_t)
//
// example: BIT_WIDTH = 4, RF_TYPE=SINT, src = 8:
// src:             0000000000000000000000000000000000000000000000000000000000001000
// bitMask:         1111111111111111111111111111111111111111111111111111111111111000
// src & bitMask:   0000000000000000000000000000000000000000000000000000000000001000
// check:           failed
// result:          throw error
//
// example: BIT_WIDTH = 4, RF_TYPE=UINT, src = 16:
// src:             0000000000000000000000000000000000000000000000000000000000010000
// bitMask:         1111111111111111111111111111111111111111111111111111111111110000
// src & bitMask:   0000000000000000000000000000000000000000000000000000000000010000
// check:           failed
// result:          throw error

template <typename REG_TYPE, typename SrcType>
uint64_t checked_cast_reg(SrcType src) {
    constexpr auto RF_TYPE = REG_TYPE::getRegFieldDataType();
    static_assert(RF_TYPE == VPURegMapped::RegFieldDataType::UINT || RF_TYPE == VPURegMapped::RegFieldDataType::SINT,
                  "checked_cast_reg: invalid RegField data type. Expected UINT or SINT");
    constexpr auto BIT_WIDTH = RF_TYPE == VPURegMapped::RegFieldDataType::SINT ? REG_TYPE::getRegFieldWidth() - 1
                                                                               : REG_TYPE::getRegFieldWidth();
    static_assert(BIT_WIDTH > 0 && BIT_WIDTH <= 64 && REG_TYPE::getRegFieldWidth() <= 64, "invalid BIT_WIDTH");
    static_assert(
            std::is_unsigned<SrcType>::value || std::is_enum<SrcType>::value || std::is_same<SrcType, bool>::value,
            "checked_cast_reg: invalid src type. Expected uint64_t or enum or bool");
    static_assert(!(std::is_enum<SrcType>::value || std::is_same<SrcType, bool>::value) ||
                          RF_TYPE == VPURegMapped::RegFieldDataType::UINT,
                  "UINT reg field type only is supported for enum and bool src types");
    if (BIT_WIDTH < 64) {
        uint64_t bitMask = ~((1ull << BIT_WIDTH) - 1);
        VPUX_THROW_WHEN(static_cast<uint64_t>(src) & bitMask,
                        "checked_cast_reg: value {0} can't be placed into {1}-bits", static_cast<uint64_t>(src),
                        BIT_WIDTH);
    }

    return static_cast<uint64_t>(src);
}

// src is int64; reg. field is SINT or UINT
// example: BIT_WIDTH = 4, RF_TYPE=SINT src = -8:
// src:             1111111111111111111111111111111111111111111111111111111111111000
//                                                                        sign bit there
//                                                                              |
// bitMask:         1111111111111111111111111111111111111111111111111111111111111000
// src & bitMask:   1111111111111111111111111111111111111111111111111111111111111000
// check:           passed
// cropBitMask:     0000000000000000000000000000000000000000000000000000000000001111
// result:          0000000000000000000000000000000000000000000000000000000000001000 (8:uint64_t)
//
// example: BIT_WIDTH = 4, src = -9:
// src:             1111111111111111111111111111111111111111111111111111111111110111
//                                                                        sign bit there
//                                                                              |
// bitMask:         1111111111111111111111111111111111111111111111111111111111111000
// src & bitMask:   1111111111111111111111111111111111111111111111111111111111110000
// check:           failed
// result:          throw error
//
// example: BIT_WIDTH = 9, src = -128:
// src:             1111111111111111111111111111111111111111111111111111111110000000
//                                                                   sign bit there
//                                                                         |
// bitMask:         1111111111111111111111111111111111111111111111111111111100000000
// src & bitMask:   1111111111111111111111111111111111111111111111111111111100000000
// check:           passed
// cropBitMask:     0000000000000000000000000000000000000000000000000000000111111111
// result:          0000000000000000000000000000000000000000000000000000000110000000 (384:uint64_t)
template <typename REG_TYPE>
uint64_t checked_cast_reg(int64_t src) {
    constexpr auto BIT_WIDTH = REG_TYPE::getRegFieldWidth();
    constexpr auto RF_TYPE = REG_TYPE::getRegFieldDataType();

    static_assert(BIT_WIDTH > 0 && BIT_WIDTH <= 64, "checked_cast_reg: invalid BIT_WIDTH");
    static_assert(RF_TYPE == VPURegMapped::RegFieldDataType::UINT || RF_TYPE == VPURegMapped::RegFieldDataType::SINT,
                  "checked_cast_reg: invalid RegField data type. Expected UINT or SINT");

    if (RF_TYPE == VPURegMapped::RegFieldDataType::UINT || src >= 0) {
        return checked_cast_reg<REG_TYPE>(checked_cast<uint64_t>(src));
    }

    uint64_t bitMask = ~((1ull << (BIT_WIDTH - 1)) - 1);
    VPUX_THROW_WHEN((src & bitMask) != bitMask, "checked_cast_reg: value {0} can't be placed into {1}-bits", src,
                    BIT_WIDTH);
    auto cropBitMask = ~(bitMask << 1);
    return static_cast<uint64_t>(src & cropBitMask);
}

// src is double; reg. field is fp32
template <typename REG_TYPE>
uint64_t checked_cast_reg(double src) {
    constexpr auto BIT_WIDTH = REG_TYPE::getRegFieldWidth();
    constexpr auto RF_TYPE = REG_TYPE::getRegFieldDataType();
    static_assert(BIT_WIDTH == 64 && RF_TYPE == VPURegMapped::RegFieldDataType::FP,
                  "checked_cast_reg: invalid RegField params");
    return llvm::bit_cast<uint64_t>(src);
}

template <typename REG_TYPE>
uint32_t checked_cast_reg(float src) {
    constexpr auto BIT_WIDTH = REG_TYPE::getRegFieldWidth();
    constexpr auto RF_TYPE = REG_TYPE::getRegFieldDataType();
    static_assert(BIT_WIDTH == 32 && RF_TYPE == VPURegMapped::RegFieldDataType::FP,
                  "checked_cast_reg: invalid RegField params");
    return llvm::bit_cast<uint32_t>(src);
}

template <typename REG_TYPE>
uint16_t checked_cast_reg(vpux::type::float16 src) {
    constexpr auto BIT_WIDTH = REG_TYPE::getRegFieldWidth();
    constexpr auto RF_TYPE = REG_TYPE::getRegFieldDataType();
    static_assert(BIT_WIDTH == 16 && RF_TYPE == VPURegMapped::RegFieldDataType::FP,
                  "checked_cast_reg: invalid RegField params");
    return llvm::bit_cast<uint16_t>(src);
}

}  // namespace VPURegMapped
}  // namespace vpux
