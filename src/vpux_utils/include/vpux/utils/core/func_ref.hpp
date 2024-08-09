//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

//
// Light-weight non-owning wrapper for callback functions.
//
// Can be used to pass some callback to another function in case if the callback
// wo't be stored for future use.
//

#pragma once

#include <llvm/ADT/STLFunctionalExtras.h>

namespace vpux {

template <typename Fn>
using FuncRef = llvm::function_ref<Fn>;

}  // namespace vpux
