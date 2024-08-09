//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/ELF/ops.hpp"
#include "vpux/compiler/utils/stl_extras.hpp"

#include "vpux/utils/core/optional.hpp"

#include <vpux_elf/writer.hpp>

using namespace vpux;

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/NPU40XX/dialect/ELF/ops.cpp.inc>
