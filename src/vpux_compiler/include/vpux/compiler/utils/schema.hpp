//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/core/helper_macros.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <flatbuffers/flatbuffers.h>

#include "schema/graphfile_generated.h"

//
// stringifyEnum
//

namespace MVCNN {

#define VPUX_STRINGIFY_SCHEMA_ENUM(_name_)             \
    inline vpux::StringRef stringifyEnum(_name_ val) { \
        return VPUX_COMBINE(EnumName, _name_)(val);    \
    }

VPUX_STRINGIFY_SCHEMA_ENUM(MemoryLocation)
VPUX_STRINGIFY_SCHEMA_ENUM(DType)
VPUX_STRINGIFY_SCHEMA_ENUM(PPELayerType)
VPUX_STRINGIFY_SCHEMA_ENUM(MPE_Mode)

#undef VPUX_STRINGIFY_SCHEMA_ENUM

}  // namespace MVCNN
