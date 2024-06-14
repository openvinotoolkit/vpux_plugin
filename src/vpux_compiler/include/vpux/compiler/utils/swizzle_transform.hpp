//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <mlir/IR/DialectImplementation.h>

namespace vpux {

namespace BufferTransform {
const uint32_t LOG2_RAM_CUT_DATA_WIDTH{4};                       // 4 -> 16 bytes
const uint32_t LOG2_RAM_CUT_BYTES{LOG2_RAM_CUT_DATA_WIDTH + 5};  // log2(Ram_cut_data_stride)
const uint32_t RAM_CUT_ADDRESS_MASK{(1u << LOG2_RAM_CUT_BYTES) - 1u};
const uint32_t CUT_ADDRESS_MASK_10b{(1u << 11) - 1u};
const uint32_t MAX_SWIZZLE_KEY{5};

class AddressTransform {
public:
    AddressTransform(uint32_t staggerBits, VPU::ArchKind archKind)
            : _staggerAddressBits{staggerBits}, _archKind{archKind} {
        setStaggerBits(staggerBits);
    }

    ~AddressTransform() {
    }

    void setStaggerBits(uint32_t bits);
    uint32_t getRamCut(uint32_t addr);
    uint32_t getPhysicalAddress(uint32_t dpuAddr);
    inline uint32_t getLog2RamCutDataWidth() {
        return _log2RamCutDataWidth;
    }

private:
    uint32_t _staggerAddressBits{};
    uint32_t _staggerAddressMask{};
    uint32_t _shift{};
    uint32_t _log2RamCutDataWidth{LOG2_RAM_CUT_DATA_WIDTH};
    VPU::ArchKind _archKind{};
    uint32_t _ramCutAddressMask{RAM_CUT_ADDRESS_MASK};
};

class BufferSwizzleTransform {
public:
    BufferSwizzleTransform(uint32_t swizzleKey = 5, VPU::ArchKind archKind = VPU::ArchKind::NPU37XX);
    uint32_t getSwizzlePatternStride();

    template <typename OutT>
    void swizzle(ArrayRef<char> in, MutableArrayRef<OutT>& swizzledBuffer) {
        const auto logToRamCutDataWidth{_addressTransform.getLog2RamCutDataWidth()};
        const auto copyDataWidth{1u << logToRamCutDataWidth};
        const auto dataWidthM1{copyDataWidth - 1u};

        auto inSize{in.size() * sizeof(char)};
        auto rawData = in.data();
        // Make sure in buffer size is a multiple of the NN CMX data width (VPUX37XX: 16B, VPUX40XX: 32B)
        inSize = ((inSize + dataWidthM1) >> logToRamCutDataWidth) << logToRamCutDataWidth;
        const auto iterations{inSize >> logToRamCutDataWidth};

        uint32_t dpuAddr{};
        auto pIn = reinterpret_cast<const uint8_t*>(rawData);
        auto pOut = reinterpret_cast<uint8_t*>(&swizzledBuffer[0]);

        for (size_t i{}; i < iterations; ++i) {
            auto phyAddr{_addressTransform.getPhysicalAddress(dpuAddr)};
            memcpy(reinterpret_cast<void*>(pOut + phyAddr), reinterpret_cast<const void*>(pIn + dpuAddr),
                   copyDataWidth);
            dpuAddr += copyDataWidth;
        }
    }

    // Function to be used for testing
    template <typename OutT>
    void deswizzle(MutableArrayRef<OutT>& in, SmallVector<OutT>& deSwizzledBuffer) {
        const auto logToRamCutDataWidth{_addressTransform.getLog2RamCutDataWidth()};
        const auto copyDataWidth{1u << logToRamCutDataWidth};
        const auto dataWidthM1{copyDataWidth - 1u};

        auto inSize{in.size() * sizeof(OutT)};
        // Make sure in buffer size is a multiple of the NN CMX data width (VPUX37XX: 16B, VPUX40XX: 32B)
        inSize = ((inSize + dataWidthM1) >> logToRamCutDataWidth) << logToRamCutDataWidth;
        const auto iterations{inSize >> logToRamCutDataWidth};

        uint32_t dpuAddr{};
        auto pIn = reinterpret_cast<uint8_t*>(&in[0]);
        auto pOut = reinterpret_cast<uint8_t*>(&deSwizzledBuffer[0]);

        for (size_t i{}; i < iterations; ++i) {
            auto phyAddr{_addressTransform.getPhysicalAddress(dpuAddr)};
            memcpy(reinterpret_cast<void*>(pOut + dpuAddr), reinterpret_cast<const void*>(pIn + phyAddr),
                   copyDataWidth);
            dpuAddr += copyDataWidth;
        }
    }

private:
    AddressTransform _addressTransform;
};

}  // namespace BufferTransform

}  // namespace vpux
