//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/ELF/ops_interfaces.hpp"

using namespace vpux;

//
// SectionSignature
//

ELF::SectionSignature::SectionSignature(std::string name, SectionFlagsAttr flags, SectionTypeAttr type)
        : _name(std::move(name)), _flags(flags), _type(type) {
}

std::string_view ELF::SectionSignature::getName() const {
    return _name;
}

StringRef ELF::SectionSignature::getNameRef() const {
    return _name;
}

ELF::SectionFlagsAttr ELF::SectionSignature::getFlags() const {
    return _flags;
}

ELF::SectionTypeAttr ELF::SectionSignature::getType() const {
    return _type;
}

bool ELF::operator==(const ELF::SectionSignature& lhs, const ELF::SectionSignature& rhs) {
    return lhs.getName() == rhs.getName() && lhs.getFlags() == rhs.getFlags() && lhs.getType() == rhs.getType();
}

//
// SymbolSignature
//

ELF::SymbolSignature::SymbolSignature(mlir::SymbolRefAttr reference, std::string name, ELF::SymbolType type,
                                      size_t size, size_t value)
        : reference(reference), name(std::move(name)), type(type), size(size), value(value) {
}

bool ELF::operator==(const ELF::SymbolSignature& lhs, const ELF::SymbolSignature& rhs) {
    return lhs.name == rhs.name && lhs.type == rhs.type && lhs.size == rhs.size && lhs.value == rhs.value;
}

//
// Generated
//

#include <vpux/compiler/NPU40XX/dialect/ELF/ops_interfaces.cpp.inc>
