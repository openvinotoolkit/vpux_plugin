//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <vector>

#include "vpux/compiler/bitc/bitc.hpp"
#include "vpux/compiler/utils/codec_factory.hpp"

namespace vpux {
class BitCompactorCodec final : public ICodec {
public:
    BitCompactorCodec(VPU::ArchKind arch_kind);
    bool supportsFP16compression() const override;
    mlir::FailureOr<std::vector<uint8_t>> compress(std::vector<uint8_t>& data, const CompressionMode mode,
                                                   const Logger& _log) const override;

private:
    vpux::bitc::ArchType arch_type_;
};

}  // namespace vpux
