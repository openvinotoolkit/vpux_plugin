//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

//
// A SmallString is just a SmallVector with methods and accessors
// that make it work better as a string
//

#pragma once

#include <llvm/ADT/SmallString.h>

namespace vpux {

using SmallString = llvm::SmallString<128>;

}  // namespace vpux
