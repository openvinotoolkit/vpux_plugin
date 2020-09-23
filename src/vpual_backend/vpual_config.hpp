//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#pragma once

#include <ie_common.h>

#include <map>
#include <string>
#include <unordered_set>
#include <vpux_config.hpp>

namespace vpux {

class VpualConfig final : public vpux::VPUXConfig {
public:
    bool forceNCHWToNHWC() const { return _forceNCHWToNHWC; }

protected:
    const std::unordered_set<std::string>& getRunTimeOptions() const override;
    void parse(const std::map<std::string, std::string>& config) override;

private:
    bool _forceNCHWToNHWC = false;
};

}  // namespace vpux
