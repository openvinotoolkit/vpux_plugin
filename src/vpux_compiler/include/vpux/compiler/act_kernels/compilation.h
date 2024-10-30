//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <string>
#include <vector>

#include <llvm/ADT/StringRef.h>
#include <mlir/IR/BuiltinOps.h>

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/utils/schema.hpp"
#include "vpux/utils/core/small_vector.hpp"

namespace vpux {

struct KernelDataDesc {
    std::string name;
    SmallVector<uint8_t> data;
    // unpadded size
    size_t size;
};

struct ActKernelDesc {
    KernelDataDesc text;
    KernelDataDesc data;
};

struct SerializedKernelDataDesc {
    std::string name;
    flatbuffers::Offset<MVCNN::KernelData> data;
    // unpadded size
    size_t size;
};

struct CompilationUnitDesc {
    mlir::StringRef name;
    mlir::StringRef entry;
};

ActKernelDesc compileKernelForACTShave(const CompilationUnitDesc& unitDesc, VPU::ArchKind archKind,
                                       std::optional<VPU::RevisionID> revisionID, bool hasInputsInDDR);

const CompilationUnitDesc& managementKernelCompilationDesc();

ActKernelDesc compileManagementKernelForACTShave();

flatbuffers::Offset<MVCNN::KernelData> buildKernelData(flatbuffers::FlatBufferBuilder& fbb,
                                                       llvm::ArrayRef<uint8_t> content);

}  // namespace vpux
