//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0.
//

#pragma once

#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <array>

namespace vpux {

class DeviceInfo final {
public:
    static const DeviceInfo VPUX30XX;
    static const DeviceInfo VPUX31XX;
    static const DeviceInfo VPUX37XX;
    static const DeviceInfo VPUX40XX;

public:
    explicit DeviceInfo(StringRef codeStr);

public:
    StringRef strref() const;

public:
    bool operator==(const DeviceInfo& other) const;
    bool operator!=(const DeviceInfo& other) const;

public:
    void printFormat(llvm::raw_ostream& os) const;

private:
    // Internal ctor with skipped validation
    DeviceInfo(StringRef codeStr, int);

private:
    static constexpr size_t TOTAL_NUM_COMPONENTS = 6;

    std::array<char, TOTAL_NUM_COMPONENTS> _code;
};

}  // namespace vpux
