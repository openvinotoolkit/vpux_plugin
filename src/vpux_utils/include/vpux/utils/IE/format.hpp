//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/utils/core/format.hpp"

#include <openvino/core/partial_shape.hpp>
#include <openvino/core/shape.hpp>
#include <openvino/core/strides.hpp>
#include <openvino/core/type/element_type.hpp>

#include <sstream>

//
// Format
//

namespace llvm {

template <>
struct format_provider<ov::element::Type> final {
    static void format(const ov::element::Type& elemType, llvm::raw_ostream& stream, StringRef style) {
        llvm::detail::build_format_adapter(elemType.get_type_name()).format(stream, style);
    }
};

template <>
struct format_provider<ov::Shape> final : vpux::ListFormatProvider {};

template <>
struct format_provider<ov::Strides> final : vpux::ListFormatProvider {};

template <>
struct format_provider<ov::PartialShape> {
    static void format(const ov::PartialShape& pshape, llvm::raw_ostream& stream, StringRef style) {
        std::ostringstream strm;
        strm << pshape;

        llvm::detail::build_format_adapter(strm.str()).format(stream, style);
    }
};

}  // namespace llvm
