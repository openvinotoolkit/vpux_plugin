//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vector>
#include "vpux/compiler/utils/bit_compactor_codec.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux {

std::unique_ptr<ICodec> getBitCompactorCodec(VPU::ArchKind arch) {
    return std::make_unique<vpux::BitCompactorCodec>(arch);
}

std::unique_ptr<ICodec> makeCodec(const ICodec::CompressionAlgorithm algo, VPU::ArchKind arch) {
    switch (algo) {
    case ICodec::CompressionAlgorithm::BITCOMPACTOR_CODEC:
        return getBitCompactorCodec(arch);
    default:
        VPUX_THROW("vpux::makeCodec: unsupported compression algorithm");
    }
    VPUX_THROW("vpux::makeCodec: unsupported compression algorithm");
}

bool ICodec::supportsFP16compression() const {
    return false;
}

std::string ICodec::compressionModeToStr(ICodec::CompressionMode mode) {
    switch (mode) {
    case CompressionMode::UINT8:
        return "U8";
    case CompressionMode::FP16:
        return "FP16";
    default:
        VPUX_THROW("Unsupported compression mode");
        break;
    }
}

}  // namespace vpux
