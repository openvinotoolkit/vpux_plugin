//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/const/utils/sub_byte.hpp"

//
// Const::isSubByte
//

bool vpux::Const::isSubByte(const size_t bitWidth) {
    return bitWidth < CHAR_BIT && bitWidth >= 1;
}
