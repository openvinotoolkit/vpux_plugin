//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/utils/core/error.hpp"

#include "vpux/utils/core/logger.hpp"

#include <sstream>

using namespace vpux;

//
// Exceptions
//

template <typename ExceptionT>
void vpux::details::throwFormat(const char* file, int line, const std::string& message) {
    VPUX_UNUSED(file);
    VPUX_UNUSED(line);

#ifdef VPUX_DEVELOPER_BUILD
    Logger::global().error("Got exception in {0}:{1} : {2}", file, line, message);
#else
    Logger::global().error("Got exception : {0}", message);
#endif

    std::stringstream strm;
    strm
#ifndef NDEBUG
            << '\n'
            << file << ':' << line << ' '
#endif
            << message;
    // E#94973 TODO catch this exception and rethrow as ov::Exception in OV linked layer
    throw ExceptionT(strm.str());
}

template void vpux::details::throwFormat<vpux::Exception>(const char* file, int line, const std::string& message);
template void vpux::details::throwFormat<vpux::WlmRollbackException>(const char* file, int line,
                                                                     const std::string& message);
