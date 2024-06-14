//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/rewriters/dpu_variant_block_rewriters.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/rewriters/utils.hpp"

namespace {

using namespace vpux;

mlir::LogicalResult buildODUOutSubtensor(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger& log,
                                         const SmallVector<int64_t>&& start, const SmallVector<int64_t>&& end) {
    if (start.size() != 3 || end.size() != 3) {
        log.error("ODU out subtensor start/end coordinates not properly specified: expected 3 start/end coordinates; "
                  "actual {0}/{1}",
                  start.size(), end.size());
        return mlir::failure();
    }

    builder.create<VPUIPDPU::ODUOutSubtensorOp>(loc, start[0], start[1], start[2], end[0], end[1], end[2]);

    return mlir::success();
}

mlir::LogicalResult buildODUHaloRegionOp(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger& log,
                                         mlir::ArrayAttr haloRegions, bool outSparsityEnabled) {
    for (const auto& attr : haloRegions) {
        auto haloRegion = attr.dyn_cast<VPUIP::DPUHaloRegionAttr>();
        if (!haloRegion) {
            log.error("Got non DPUHaloRegion attribute '{0}' in array", haloRegion);
            return mlir::failure();
        }

        if (haloRegion.getSparsityOffset() && !outSparsityEnabled) {
            log.error("HaloRegion '{0}': sparsity_offset should only exist in case an output sparsity map "
                      "buffer is defined",
                      haloRegion);
            return mlir::failure();
        }

        const auto targetClusters = parseIntArrayAttr<uint8_t>(haloRegion.getTargetClusters());
        uint64_t tileMask = 0;
        for (const auto clusterIdx : targetClusters) {
            tileMask |= 1ull << clusterIdx;
        }
        auto dpuTiles = VPUIPDPU::symbolizeDPUTiles(tileMask);
        if (!dpuTiles.has_value()) {
            log.error("Incorrect target clusters configuration in DPUHaloRegion: {0}", tileMask);
            return mlir::failure();
        }
        builder.create<VPUIPDPU::ODUHaloRegionOp>(
                loc, haloRegion.getXStart(), haloRegion.getYStart(), haloRegion.getXEnd(), haloRegion.getYEnd(),
                haloRegion.getTargetOffset(), haloRegion.getSparsityOffset(), haloRegion.getTargetWidth(),
                VPUIPDPU::DPUTilesAttr::get(builder.getContext(), dpuTiles.value()));
    }

    return mlir::success();
}

}  // namespace

namespace vpux {
namespace VPUIPDPU {

DPUVariantODURewriter::DPUVariantODURewriter(VPUASM::DPUVariantOp origVarOp, mlir::PatternRewriter& rewriter,
                                             const Logger& log)
        : DPUVariantBlockRewriter(origVarOp, rewriter, log) {
}

mlir::LogicalResult DPUVariantODURewriter::rewrite(ELF::SymbolReferenceMap& symRefMap) {
    auto origInvOp = mlir::cast<VPUASM::DPUInvariantOp>(symRefMap.lookupSymbol(_origVarOp.getInvariant()));

    if (buildODUOutSubtensor(_rewriter, _origVarOp.getLoc(), _log, parseIntArrayAttr<int64_t>(_origVarOp.getStart()),
                             parseIntArrayAttr<int64_t>(_origVarOp.getEnd()))
                .failed()) {
        return mlir::failure();
    }

    if (auto haloRegions = _origVarOp.getHaloRegionsAttr()) {
        if (buildODUHaloRegionOp(_rewriter, _origVarOp.getLoc(), _log, haloRegions,
                                 origInvOp.getOutputSparsityMap().has_value())
                    .failed()) {
            return mlir::failure();
        }
    }

    return mlir::success();
}

}  // namespace VPUIPDPU
}  // namespace vpux
