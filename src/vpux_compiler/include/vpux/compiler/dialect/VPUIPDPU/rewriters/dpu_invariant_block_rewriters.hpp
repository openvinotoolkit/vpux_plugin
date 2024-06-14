//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

namespace vpux {
namespace VPUIPDPU {

class DPUInvariantBlockRewriter {
public:
    enum class BlockArg {
        ACT_IN,
        ACT_SE_IN,
        ACT_SPARSE_MAP_IN,
        WEIGHTS_TABLE,
        WEIGHTS,
        WEIGHTS_SPARSE_MAP,
        SPR_LOOKUP_TABLE,
        ACT_OUT,
        ACT_SPARSE_MAP_OUT,
        Count
    };

    DPUInvariantBlockRewriter(VPUASM::DPUInvariantOp origInvOp, mlir::Block* invBlock,
                              std::map<BlockArg, size_t>& invBlockArgsPos, mlir::PatternRewriter& rewriter,
                              const Logger& log);

    static mlir::LogicalResult insertInvBlockArgs(VPUASM::DPUInvariantOp op, mlir::Block* invBlock,
                                                  std::map<BlockArg, size_t>& invBlockArgsPos, const Logger& log,
                                                  ELF::SymbolReferenceMap& symRefMap);
    static mlir::Type getBaseType(mlir::Type type);

    static mlir::LogicalResult getQuantConfig(const Logger&, mlir::Type type, SmallVector<int64_t>& quantMult,
                                              SmallVector<int64_t>& quantShift, SmallVector<uint8_t>& quantZero);

protected:
    VPUASM::DPUInvariantOp _origInvOp;
    mlir::Block* _invBlock;
    std::map<BlockArg, size_t>& _invBlockArgsPos;
    mlir::PatternRewriter& _rewriter;
    const Logger _log;

    mlir::BlockArgument getInvBlockArg(BlockArg invBlockArg) const;

    template <typename CfgOp>
    mlir::LogicalResult insertEntryBlock() {
        _rewriter.setInsertionPointToEnd(_invBlock);

        CfgOp cfgOp = _rewriter.create<CfgOp>(_origInvOp.getLoc());
        auto& region = cfgOp.getOperation()->getRegion(0);
        auto entryBlock = _rewriter.createBlock(&region);

        if (!entryBlock) {
            _log.error("Error creating entry block for {0}", typeid(CfgOp).name());
            return mlir::failure();
        }

        return mlir::success();
    }
};

class DPUInvariantIDURewriter : public DPUInvariantBlockRewriter {
public:
    DPUInvariantIDURewriter(VPUASM::DPUInvariantOp origInvOp, mlir::Block* invBlock,
                            std::map<BlockArg, size_t>& invBlockArgsPos, mlir::PatternRewriter& rewriter,
                            const Logger& log);

    mlir::LogicalResult rewrite();
};

class DPUInvariantMPERewriter : public DPUInvariantBlockRewriter {
public:
    DPUInvariantMPERewriter(VPUASM::DPUInvariantOp origInvOp, mlir::Block* invBlock,
                            std::map<BlockArg, size_t>& invBlockArgsPos, mlir::PatternRewriter& rewriter,
                            const Logger& log);

    mlir::LogicalResult rewrite();
};

class DPUInvariantPPERewriter : public DPUInvariantBlockRewriter {
public:
    DPUInvariantPPERewriter(VPUASM::DPUInvariantOp origInvOp, mlir::Block* invBlock,
                            std::map<BlockArg, size_t>& invBlockArgsPos, mlir::PatternRewriter& rewriter,
                            const Logger& log);

    mlir::LogicalResult rewrite();
};

class DPUInvariantPPEFpRewriter : public DPUInvariantBlockRewriter {
public:
    DPUInvariantPPEFpRewriter(VPUASM::DPUInvariantOp origInvOp, mlir::Block* invBlock,
                              std::map<BlockArg, size_t>& invBlockArgsPos, mlir::PatternRewriter& rewriter,
                              const Logger& log);

    mlir::LogicalResult rewrite();
};

class DPUInvariantODURewriter : public DPUInvariantBlockRewriter {
public:
    DPUInvariantODURewriter(VPUASM::DPUInvariantOp origInvOp, mlir::Block* invBlock,
                            std::map<BlockArg, size_t>& invBlockArgsPos, mlir::PatternRewriter& rewriter,
                            const Logger& log);

    mlir::LogicalResult rewrite(ELF::SymbolReferenceMap& symRefMap);
};

}  // namespace VPUIPDPU
}  // namespace vpux
