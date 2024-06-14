//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/NPU40XX/dialect/ELF/attributes.hpp"
#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/stl_extras.hpp"

#include "vpux/utils/core/dense_map.hpp"
#include "vpux/utils/core/small_string.hpp"

#include <tuple>
#include <vpux_elf/writer.hpp>
namespace vpux {
namespace ELF {

typedef DenseMap<mlir::Operation*, elf::writer::Section*> SectionMapType;
// Note that this works since in our case the IR is immutable troughout the life-time of the map.
typedef DenseMap<mlir::Operation*, elf::writer::Symbol*> SymbolMapType;

//
// ElfSectionInterface
//

template <typename ConcreteOp>
mlir::Block* getSectionBlock(ConcreteOp op) {
    mlir::Operation* operation = op.getOperation();
    auto& region = operation->getRegion(0);

    if (region.empty()) {
        region.emplaceBlock();
    }

    return &region.front();
}

//
// SectionSignature
//

class SectionSignature {
public:
    SectionSignature(std::string name, SectionFlagsAttr flags, SectionTypeAttr type = SectionTypeAttr::SHT_PROGBITS)
            : _name(std::move(name)), _flags(flags), _type(type) {
    }
    SectionSignature() = delete;

    const std::string& getName() const {
        return _name;
    }

    StringRef getNameRef() const {
        return _name;
    }

    SectionFlagsAttr getFlags() const {
        return _flags;
    }

    SectionTypeAttr getType() const {
        return _type;
    }

private:
    std::string _name;
    SectionFlagsAttr _flags;
    SectionTypeAttr _type;
};

struct RelocationInfo;

class SymbolReferenceMap;

}  // namespace ELF
}  // namespace vpux

//
// Generated
//

#include <vpux/compiler/NPU40XX/dialect/ELF/ops_interfaces.hpp.inc>

struct vpux::ELF::RelocationInfo {
    RelocationInfo(mlir::SymbolRefAttr source, vpux::ELF::ElfSectionInterface targetSection, size_t offset,
                   vpux::ELF::RelocationType relocType, size_t addend, bool isOffsetRelative = true,
                   std::string description = "")
            : source{source},
              targetSection{targetSection},
              offset{offset},
              relocType{relocType},
              addend{addend},
              isOffsetRelative{isOffsetRelative},
              description(std::move(description)){};

    mlir::SymbolRefAttr source;
    vpux::ELF::ElfSectionInterface targetSection;
    size_t offset;
    vpux::ELF::RelocationType relocType;
    size_t addend;
    bool isOffsetRelative;
    std::string description;
};
