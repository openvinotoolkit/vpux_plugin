//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/conversion/passes/VPU2VPUIP/bufferizable_op_interface.hpp"
#include "vpux/compiler/NPU40XX/utils.hpp"
#include "vpux/compiler/conversion/passes/VPU2VPUIP/bufferize_sw_ops_interface.hpp"

using namespace vpux;

namespace {

//
// ConvertOpBufferizeModel
//

class ConvertOpBufferizeModel :
        public BufferizableOpInterfaceExternalModelBase<ConvertOpBufferizeModel, VPU::ConvertOp> {
public:
    mlir::LogicalResult bufferizeImpl(VPU::ConvertOp origOp, mlir::RewriterBase& rewriter,
                                      const mlir::bufferization::BufferizationOptions& options,
                                      VPU::ConvertOp::Adaptor adaptor) const;
};

mlir::LogicalResult ConvertOpBufferizeModel::bufferizeImpl(VPU::ConvertOp origOp, mlir::RewriterBase& rewriter,
                                                           const mlir::bufferization::BufferizationOptions& options,
                                                           VPU::ConvertOp::Adaptor adaptor) const {
    // If conversion can be done on DMA, bufferize it to DMA operation.
    if (isConvertSupportedOnDMA<VPU::ConvertOp>(origOp)) {
        return vpux::bufferizeOp(origOp->getContext(), origOp, adaptor, rewriter);
    }

    // If ConvertOp can not be converted to DMA operation, bufferize it to software layer operation instead.
    SoftwareLayerOpBufferizeModel<VPU::ConvertOp> convertOpSoftwareModel;
    return convertOpSoftwareModel.bufferizeImpl(origOp, rewriter, options, adaptor);
}

void registerConvertOpBufferizableOpInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, VPU::VPUDialect*, VPUIP::VPUIPDialect*) {
        VPU::ConvertOp::attachInterface<ConvertOpBufferizeModel>(*ctx);
    });
}

class GatherDMAOpBufferizeModel :
        public BufferizableOpInterfaceExternalModelBase<GatherDMAOpBufferizeModel, VPU::GatherDMAOp> {
public:
    mlir::LogicalResult bufferizeImpl(VPU::GatherDMAOp origOp, mlir::RewriterBase& rewriter,
                                      const mlir::bufferization::BufferizationOptions& options,
                                      VPU::GatherDMAOp::Adaptor adaptor) const;
};

mlir::LogicalResult GatherDMAOpBufferizeModel::bufferizeImpl(VPU::GatherDMAOp origOp, mlir::RewriterBase& rewriter,
                                                             const mlir::bufferization::BufferizationOptions&,
                                                             VPU::GatherDMAOp::Adaptor adaptor) const {
    return vpux::bufferizeOp(origOp->getContext(), origOp, adaptor, rewriter);
}

void registerGatherDMAOpBufferizableOpInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, VPU::VPUDialect*, VPUIP::VPUIPDialect*) {
        VPU::GatherDMAOp::attachInterface<GatherDMAOpBufferizeModel>(*ctx);
    });
}

}  // namespace

//
// registerBufferizableOpInterfaces
//

void vpux::arch40xx::registerBufferizableOpInterfaces(mlir::DialectRegistry& registry) {
    vpux::registerConstDeclareBufferizableOpInterfaces(registry);
    vpux::registerFuncAndReturnBufferizableOpInterfaces(registry);
    vpux::registerSoftwareLayerBufferizableOpInterfaces(registry);
    vpux::registerVpuNceBufferizableOpInterfaces(registry);
    vpux::registerVPUBufferizableOpInterfaces(registry);
    vpux::registerNCEClusterTilingBufferizableOpInterfaces(registry);
    registerConvertOpBufferizableOpInterfaces(registry);
    registerGatherDMAOpBufferizableOpInterfaces(registry);
}
