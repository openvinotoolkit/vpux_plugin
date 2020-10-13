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

#include "vpux/utils/core/error.hpp"

#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/logger.hpp"

#include <stdexcept>

using namespace vpux;

//
// Exceptions
//

[[noreturn]] void vpux::details::throwFormat(StringRef file,
                                             int line,
                                             std::string message) {
    VPUX_UNUSED(file);
    VPUX_UNUSED(line);

#ifdef NDEBUG
    VPUX_GLOG_ERROR("Got exception : {0}", message);
#else
    VPUX_GLOG_ERROR("Got exception in {0}:{1} : {2}", file, line, message);
#endif

    throw std::runtime_error(message);
}
