//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPURegMapped/descriptors.hpp"

namespace vpux::VPURegMapped::detail {

std::pair<mlir::ParseResult, std::optional<elf::Version>> parseVersion(mlir::AsmParser& parser) {
    // E#135397: parseOptionalKeyword seems to signal failed parsing
    // even if assembly contained some keyword not equal to given one
    // here and in other parts of descriptors parsing methods it's treated
    // as missing keyword and "successful" parsing
    // revisit this place, hopefully making it more robust and signal failure
    // in case of unexpected tokens in assembly (allow only nothing or exactly
    // given keyword)
    if (parser.parseOptionalKeyword("requires").failed()) {
        return {mlir::success(), {}};
    }

    uint32_t major = 0;
    if (parser.parseInteger(major).failed()) {
        return {mlir::failure(), {}};
    }

    if (parser.parseColon().failed()) {
        return {mlir::failure(), {}};
    }

    uint32_t minor = 0;
    if (parser.parseInteger(minor).failed()) {
        return {mlir::failure(), {}};
    }

    if (parser.parseColon().failed()) {
        return {mlir::failure(), {}};
    }

    uint32_t patch = 0;
    if (parser.parseInteger(patch).failed()) {
        return {mlir::failure(), {}};
    }

    return {mlir::success(), std::optional<elf::Version>{std::in_place_t{}, elf::Version{major, minor, patch}}};
}

}  // namespace vpux::VPURegMapped::detail
