//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0.
//

#include "vpux/utils/plugin/device_info.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"

#include <algorithm>
#include <cctype>
#include <unordered_set>

using namespace vpux;

namespace {

constexpr StringLiteral VPUX_PREFIX = "NPU";
constexpr char ANY_SYM = 'X';
constexpr size_t VPU_GEN_NUM_COMPONENTS = 2;
constexpr size_t SOC_NUM_COMPONENTS = 2;
constexpr size_t SOC_REV_NUM_COMPONENTS = 2;
constexpr StringLiteral EMPTY_SOC_REV = "XX";

}  // namespace

const DeviceInfo vpux::DeviceInfo::VPUX30XX("30XX", 0);
const DeviceInfo vpux::DeviceInfo::VPUX31XX("31XX", 0);
const DeviceInfo vpux::DeviceInfo::VPUX37XX("37XX", 0);
const DeviceInfo vpux::DeviceInfo::VPUX40XX("40XX", 0);

vpux::DeviceInfo::DeviceInfo(StringRef codeStr, int) {
    std::fill_n(_code.data(), _code.size(), ANY_SYM);
    std::copy_n(codeStr.data(), codeStr.size(), _code.data());
}

vpux::DeviceInfo::DeviceInfo(StringRef codeStr) {
    static const std::unordered_set<StringRef> supportedVpuGen = {
            "30",  //
            "31",  //
            "37",  //
            "40",  //
    };

    std::string canonicalCodeStr = codeStr.upper();
    StringRef canonicalCode = canonicalCodeStr;
    if (canonicalCode.startswith(VPUX_PREFIX)) {
        canonicalCode = canonicalCode.drop_front(VPUX_PREFIX.size());
    }

    VPUX_THROW_WHEN(canonicalCode.size() < VPU_GEN_NUM_COMPONENTS,
                    "The device info code '{0}' is too short : not enough VPU generation components", codeStr);
    VPUX_THROW_WHEN(canonicalCode.size() > TOTAL_NUM_COMPONENTS, "The device info code '{0}' is too long", codeStr);

    const auto vpuGen = canonicalCode.take_front(VPU_GEN_NUM_COMPONENTS);
    VPUX_THROW_UNLESS(supportedVpuGen.find(vpuGen) != supportedVpuGen.end(),
                      "Unsupported device info code '{0}' : unknown VPU generation '{1}'", codeStr, vpuGen);

    if (canonicalCode.size() > VPU_GEN_NUM_COMPONENTS) {
        VPUX_THROW_UNLESS(canonicalCode.size() >= (VPU_GEN_NUM_COMPONENTS + SOC_NUM_COMPONENTS),
                          "The device info code '{0}' is too short : not enough SoC number components", codeStr);

        const auto socNum = canonicalCode.substr(VPU_GEN_NUM_COMPONENTS, SOC_NUM_COMPONENTS);

        VPUX_THROW_UNLESS(std::isdigit(socNum[0]) || socNum[0] == ANY_SYM,
                          "Unsupported device info code '{0}' : wrong SoC number part '{1}'", codeStr, socNum);
        if (socNum[0] == ANY_SYM) {
            VPUX_THROW_UNLESS(socNum[1] == ANY_SYM, "Unsupported device info code '{0}' : wrong SoC number part '{1}'",
                              codeStr, socNum);
        } else {
            VPUX_THROW_UNLESS(std::isdigit(socNum[1]) || socNum[1] == ANY_SYM,
                              "Unsupported device info code '{0}' : wrong SoC number part '{1}'", codeStr, socNum);
        }
    }

    if (canonicalCode.size() > (VPU_GEN_NUM_COMPONENTS + SOC_NUM_COMPONENTS)) {
        VPUX_THROW_UNLESS(canonicalCode.size() == TOTAL_NUM_COMPONENTS,
                          "The device info code '{0}' is too short : not enough SoC revision components", codeStr);

        const auto socNum = canonicalCode.substr(VPU_GEN_NUM_COMPONENTS, SOC_NUM_COMPONENTS);
        const auto socRev = canonicalCode.substr(VPU_GEN_NUM_COMPONENTS + SOC_NUM_COMPONENTS, SOC_REV_NUM_COMPONENTS);

        if (socNum[1] == ANY_SYM) {
            VPUX_THROW_UNLESS(socRev == EMPTY_SOC_REV,
                              "Unsupported device info code '{0}' : wrong SoC revision part '{1}'", codeStr, socRev);
        } else {
            VPUX_THROW_UNLESS(std::isdigit(socRev[0]) || socRev[0] == 'A' || socRev[0] == 'B' || socRev[0] == ANY_SYM,
                              "Unsupported device info code '{0}' : wrong SoC revision part '{1}'", codeStr, socRev);
            if (socRev[0] == ANY_SYM) {
                VPUX_THROW_UNLESS(socRev[1] == ANY_SYM,
                                  "Unsupported device info code '{0}' : wrong SoC revision part '{1}'", codeStr,
                                  socRev);
            } else {
                VPUX_THROW_UNLESS(std::isdigit(socRev[1]) || socRev[1] == ANY_SYM,
                                  "Unsupported device info code '{0}' : wrong SoC revision part '{1}'", codeStr,
                                  socRev);
            }
        }
    }

    std::fill_n(_code.data(), _code.size(), ANY_SYM);
    std::copy_n(canonicalCode.data(), canonicalCode.size(), _code.data());
}

StringRef vpux::DeviceInfo::strref() const {
    StringRef codeStr(_code.data(), _code.size());
    if (codeStr.endswith(EMPTY_SOC_REV)) {
        return codeStr.drop_back(EMPTY_SOC_REV.size());
    }
    return codeStr;
}

bool vpux::DeviceInfo::operator==(const DeviceInfo& other) const {
    for (const auto ind : irange(TOTAL_NUM_COMPONENTS)) {
        if (_code[ind] == ANY_SYM || other._code[ind] == ANY_SYM) {
            continue;
        }

        if (_code[ind] != other._code[ind]) {
            return false;
        }
    }

    return true;
}

bool vpux::DeviceInfo::operator!=(const DeviceInfo& other) const {
    return !(*this == other);
}

void vpux::DeviceInfo::printFormat(llvm::raw_ostream& os) const {
    os << "NPU" << strref();
}
