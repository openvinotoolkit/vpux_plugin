//
// Copyright Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "vpux/compiler/dialect/IERT/passes.hpp"

#include "vpux/compiler/core/aliases_info.hpp"

using namespace vpux;

namespace {

//
// SetInternalMemorySpacePass
//

class SetInternalMemorySpacePass final : public IERT::SetInternalMemorySpaceBase<SetInternalMemorySpacePass> {
public:
    SetInternalMemorySpacePass(IERT::AttrCreateFunc memSpaceCb, Logger log);

public:
    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;

private:
    IERT::AttrCreateFunc _memSpaceCb;
    mlir::Attribute _memSpace;
};

SetInternalMemorySpacePass::SetInternalMemorySpacePass(IERT::AttrCreateFunc memSpaceCb, Logger log)
        : _memSpaceCb(std::move(memSpaceCb)) {
    VPUX_THROW_UNLESS(_memSpaceCb != nullptr, "Missing memSpaceCb");
    Base::initLogger(log, Base::getArgumentName());
}

mlir::LogicalResult SetInternalMemorySpacePass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    _memSpace = _memSpaceCb(ctx, memSpaceName.getValue());

    if (_memSpace == nullptr) {
        return mlir::failure();
    }

    return mlir::success();
}

//
// safeRunOnFunc
//

void SetInternalMemorySpacePass::safeRunOnFunc() {
    auto& aliasInfo = getAnalysis<AliasesInfo>();

    const auto callback = [&](mlir::memref::AllocOp allocOp) {
        _log.trace("Got Alloc Operation '{0}'", allocOp->getLoc());

        const auto& aliases = aliasInfo.getAliases(allocOp.memref());

        for (auto var : aliases) {
            _log.trace("Process alias buffer '{0}'", var);

            const auto origType = var.getType().dyn_cast<mlir::MemRefType>();
            VPUX_THROW_UNLESS(origType != nullptr, "Got non MemRef Type '{0}'", var.getType());

            const auto newType = mlir::MemRefType::get(origType.getShape(), origType.getElementType(),
                                                       origType.getAffineMaps(), _memSpace);

            var.setType(newType);
        }
    };

    getFunction().walk(callback);
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IERT::createSetInternalMemorySpacePass(AttrCreateFunc memSpaceCb, Logger log) {
    return std::make_unique<SetInternalMemorySpacePass>(std::move(memSpaceCb), log);
}
