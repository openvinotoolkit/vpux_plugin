//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/VPU30XX/dialect/VPU/IR/ops_interfaces.hpp"

#include "vpux/compiler/dialect/VPU/utils/layer_post_ops_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/nce_invariant.hpp"
#include "vpux/compiler/utils/custom_pwl_table.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

namespace {

//
// LayerWithPostOpModel
//

auto isSupportedHWPostOp(mlir::Operation* mainOp, mlir::Operation* postOp, const LogCb& logCb) {
    return llvm::TypeSwitch<mlir::Operation*, bool>(postOp)
            .Case<IE::ReLUOp>([](IE::ReLUOp) {
                return true;
            })
            .Case<IE::SigmoidOp, IE::TanhOp>([&](mlir::Operation* postOpp) {
                const auto isQuantized = vpux::VPU::checkForQuantization(mainOp, postOpp);
                // These ops do not get fused for float cases to avoid dropping accuracy. Because PWL is not
                // accurate for FP16
                if (!isQuantized) {
                    logCb(llvm::formatv("{0} is not Quantized so it's not supported for this HW platform at `{1}`",
                                        postOp->getName(), postOp->getLoc()));
                    return false;
                }
                return true;
            })
            // TODO: remove option after E-83187
            .Case<IE::ClampOp>([&](IE::ClampOp clampOp) {
                if (clampOp != nullptr) {
                    const auto minVal = clampOp.getMinAttr().getValueAsDouble();
                    if (!isDoubleEqual(minVal, 0.0) && !vpux::VPU::checkForQuantization(mainOp, postOp)) {
                        logCb(llvm::formatv("{0} is not quantized and does not have 0 as minVal so it's not "
                                            "supported for this HW "
                                            "platform at `{1}`",
                                            postOp->getName(), postOp->getLoc()));
                        return false;
                    }
                }
                return true;
            })
            .Case<IE::LeakyReluOp>([&](IE::LeakyReluOp leakyReluOp) {
                const auto getUniformQuantizedType =
                        [](IE::FakeQuantizeOp fakeQuantizeOp) -> mlir::quant::UniformQuantizedType {
                    if (fakeQuantizeOp == nullptr) {
                        return nullptr;
                    }
                    if (!fakeQuantizeOp.getLevels().has_value()) {
                        return nullptr;
                    }
                    auto outLoConst = fakeQuantizeOp.getOutputLow().getDefiningOp<Const::DeclareOp>();
                    auto outHiConst = fakeQuantizeOp.getOutputHigh().getDefiningOp<Const::DeclareOp>();
                    const auto realType = fakeQuantizeOp.getInput().getType().cast<vpux::NDTypeInterface>();
                    const auto realElemType = realType.getElementType().cast<mlir::FloatType>();

                    const auto outElemType =
                            getQuantizedType(outLoConst.getContentAttr(), outHiConst.getContentAttr(),
                                             fakeQuantizeOp.getLevels(), fakeQuantizeOp.getLowFpType(), realElemType,
                                             false, fakeQuantizeOp.getLoc(), fakeQuantizeOp.getAutoBroadcast());
                    return outElemType.dyn_cast<mlir::quant::UniformQuantizedType>();
                };

                // If mainOp is Add, Multiply or And then the PPE is already busy and cannot do LeakyRelu
                if (mlir::isa<IE::AddOp, IE::AndOp, IE::MultiplyOp>(mainOp)) {
                    logCb(llvm::formatv("Unsupported fusing of LeakyRelu because PPE is already busy with {0} at `{1}`",
                                        mainOp->getName(), mainOp->getLoc()));
                    return false;
                }

                const auto reluSlope = leakyReluOp.getNegativeSlopeAttr().getValueAsDouble();
                if (reluSlope < 0) {
                    logCb(llvm::formatv("Unsupported negative Relu slope at `{1}`", postOp->getLoc()));
                    return false;
                }

                const auto isQuantized = vpux::VPU::checkForQuantization(mainOp, postOp);
                if (!isQuantized) {
                    return true;
                }

                // Checks for quantized LeakyRelu
                if (!leakyReluOp.getOutput().hasOneUse()) {
                    logCb(llvm::formatv("Unsupported Relu with more than one use at `{0}`", postOp->getLoc()));
                    return false;
                }

                const auto fqOp = mlir::dyn_cast<IE::FakeQuantizeOp>(*(leakyReluOp.getOutput().getUsers().begin()));

                if (fqOp == nullptr) {
                    return true;
                }

                const auto uniformElemType = getUniformQuantizedType(fqOp);
                if (uniformElemType == nullptr) {
                    logCb(llvm::formatv("Unsupported per-axis quantized output at `{0}`", postOp->getLoc()));
                    return false;
                }

                const auto zeroPoint = uniformElemType.getZeroPoint();
                if (!isSupportedPReLU(static_cast<float>(reluSlope), zeroPoint)) {
                    logCb(llvm::formatv("Unsupported slope for quantized {0} at `{1}`", postOp->getName(),
                                        postOp->getLoc()));
                    return false;
                }

                return true;
            })
            .Default([&](mlir::Operation*) {
                logCb(llvm::formatv("{0} at `{1}` is not supported on this HW platform", postOp->getName(),
                                    postOp->getLoc()));
                return false;
            });
}

template <class MainOpType>
class LayerWithPostOpModel final :
        public IE::LayerWithPostOpInterface::ExternalModel<LayerWithPostOpModel<MainOpType>, MainOpType> {
public:
    bool isSupportedPostOp(mlir::Operation* mainOp, mlir::Operation* postOp, const LogCb& logCb) const {
        if (VPU::getCompilationMode(postOp) == VPU::CompilationMode::ReferenceSW) {
            return false;
        }

        if (!isSupportedHWPostOp(mainOp, postOp, logCb)) {
            return false;
        }

        return VPU::NCEInvariant::verifyKernel(mlir::cast<MainOpType>(mainOp)).succeeded();
    }

    bool isSupportedClampOp(mlir::Operation* mainOp, mlir::Operation* clampOp, const LogCb& logCb) const {
        if (VPU::getCompilationMode(clampOp) == VPU::CompilationMode::ReferenceSW) {
            return false;
        }

        if (!VPU::isSupportedHWClampOp(mainOp, clampOp, logCb)) {
            return false;
        }

        return VPU::NCEInvariant::verifyKernel(mlir::cast<MainOpType>(mainOp)).succeeded();
    }

    void setLayerClampOp(mlir::Operation* mainOp, mlir::Operation* activationOp) const {
        VPU::setHWClampOp(mainOp, activationOp);
    }
};

}  // namespace

//
// setupExtraInterfaces
//

void vpux::VPU::arch30xx::registerLayerWithPostOpModelInterface(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, IE::IEDialect*) {
        IE::ConvolutionOp::attachInterface<LayerWithPostOpModel<IE::ConvolutionOp>>(*ctx);
        IE::TransposedConvolutionOp::attachInterface<LayerWithPostOpModel<IE::TransposedConvolutionOp>>(*ctx);
        IE::GroupConvolutionOp::attachInterface<LayerWithPostOpModel<IE::GroupConvolutionOp>>(*ctx);
        IE::MaxPoolOp::attachInterface<LayerWithPostOpModel<IE::MaxPoolOp>>(*ctx);
        IE::AddOp::attachInterface<LayerWithPostOpModel<IE::AddOp>>(*ctx);
        IE::MultiplyOp::attachInterface<LayerWithPostOpModel<IE::MultiplyOp>>(*ctx);
        IE::SubtractOp::attachInterface<LayerWithPostOpModel<IE::SubtractOp>>(*ctx);
        IE::AndOp::attachInterface<LayerWithPostOpModel<IE::AndOp>>(*ctx);
    });
}
