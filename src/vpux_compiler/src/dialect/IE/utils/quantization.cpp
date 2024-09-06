//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/quantization.hpp"

using namespace vpux;

std::optional<int64_t> getFQAxisIndex(IE::FakeQuantizeOp fq, Logger log) {
    const auto extractAxis = [log](mlir::Value input) -> std::optional<int64_t> {
        const auto greaterThanOne = [](auto dim) {
            return dim > 1;
        };

        const auto shape = getShape(input);

        const auto axisCount = llvm::count_if(shape, greaterThanOne);
        if (axisCount > 1) {
            log.trace("FakeQuantize constant input with unsupported shape.");
            return std::nullopt;
        }

        auto axis = llvm::find_if(shape, greaterThanOne);
        if (axis != shape.end()) {
            return std::distance(shape.begin(), axis);
        }

        return std::nullopt;
    };

    const auto inputLowAxis = extractAxis(fq.getInputLow());
    const auto outputLowAxis = extractAxis(fq.getOutputLow());

    if (!inputLowAxis && !outputLowAxis) {
        return std::nullopt;
    }

    if (inputLowAxis && outputLowAxis) {
        VPUX_THROW_UNLESS(*inputLowAxis == *outputLowAxis, "FakeQuantize constant inputs use different axis");
    }

    return inputLowAxis ? *inputLowAxis : *outputLowAxis;
}

std::optional<int64_t> IE::getQuantAxisIndex(mlir::Operation* op, Logger log) {
    std::optional<int64_t> axis = std::nullopt;
    const auto getPerAxisQType = [](mlir::Value tensor) {
        return tensor.getType()
                .cast<NDTypeInterface>()
                .getElementType()
                .dyn_cast<mlir::quant::UniformQuantizedPerAxisType>();
    };

    if (auto fqOp = mlir::dyn_cast_or_null<IE::FakeQuantizeOp>(op)) {
        axis = getFQAxisIndex(fqOp, log);
    } else if (mlir::isa<IE::DequantizeOp, IE::QuantizeOp>(op)) {
        if (const auto perAxisQType = getPerAxisQType(op->getOperand(0))) {
            axis = perAxisQType.getQuantizedDimension();
        }
        if (const auto perAxisQType = getPerAxisQType(op->getResult(0))) {
            axis = perAxisQType.getQuantizedDimension();
        }
    }

    return axis;
}

bool IE::hasLeakyReLUPostOp(mlir::Operation* op) {
    auto layerWithPostOp = mlir::dyn_cast<IE::LayerWithPostOpInterface>(op);
    if (layerWithPostOp == nullptr) {
        return false;
    }

    const auto postOpName = layerWithPostOp.getPostOp();
    return postOpName.has_value() && postOpName.value().getStringRef() == IE::LeakyReluOp::getOperationName();
}

bool IE::areAnyUserQuantizeOps(mlir::Operation* op) {
    return llvm::any_of(op->getUsers(), [](mlir::Operation* op) {
        return mlir::isa<IE::QuantizeOp>(op);
    });
}

bool IE::checkQuantApproximation(mlir::Operation* op) {
    SmallVector<double> scales;
    const auto outElemType = op->getResult(0).getType().cast<vpux::NDTypeInterface>().getElementType();
    if (outElemType.isa<mlir::quant::UniformQuantizedPerAxisType>()) {
        const auto perAxis = outElemType.cast<mlir::quant::UniformQuantizedPerAxisType>();
        std::copy(perAxis.getScales().begin(), perAxis.getScales().end(), std::back_inserter(scales));
    } else if (outElemType.isa<mlir::quant::UniformQuantizedType>()) {
        const auto perTensor = outElemType.cast<mlir::quant::UniformQuantizedType>();
        scales = {perTensor.getScale()};
    } else {
        return false;
    }

    // Check that all scales can be approximated without post-shift (i.e. exponent must fit 15 bits).
    // Negative power is used here because rescaling is computed as scale_in * scale_w / scale_out
    // In case of float input and float weights, scale_in = 1, scale_w = 1, thus we get 1 / scale_out.
    const double scaleLimit = std::pow(2, -15);
    for (const auto& scale : scales) {
        if (std::fabs(scale) < scaleLimit) {
            return false;
        }
    }

    return true;
}

mlir::Value IE::findQuantizedInput(mlir::Value opInput, bool allowPerAxisQuantize) {
    if (opInput == nullptr) {
        return nullptr;
    }

    // When the input is not a DequantizeOp, the pass is not applicable
    auto maybeDequant = opInput.getDefiningOp<IE::DequantizeOp>();
    if (maybeDequant == nullptr) {
        return nullptr;
    }

    const auto dequantType = maybeDequant.getInput().getType().cast<vpux::NDTypeInterface>();
    if (!allowPerAxisQuantize && !dequantType.getElementType().isa<mlir::quant::UniformQuantizedType>()) {
        return nullptr;
    }

    return maybeDequant.getInput();
}

bool IE::isSymmetricQuantType(mlir::quant::QuantizedType type) {
    // Check that zero points are all 0s
    if (const auto uniformQuantType = type.dyn_cast<mlir::quant::UniformQuantizedType>()) {
        return uniformQuantType.getZeroPoint() == 0;
    } else if (const auto uniformPerAxisQuantType = type.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        const auto zeroPoints = uniformPerAxisQuantType.getZeroPoints();
        return std::all_of(zeroPoints.begin(), zeroPoints.end(), [](const int64_t zp) {
            return zp == 0;
        });
    }

    return false;
}

mlir::quant::UniformQuantizedType IE::getQuantizedTypeFromFakeQuantize(IE::FakeQuantizeOp fqOp) {
    if (fqOp == nullptr) {
        return nullptr;
    }
    const auto iLoShape = getShape(fqOp.getInputLow());
    const auto iHiShape = getShape(fqOp.getInputHigh());
    const auto oLoShape = getShape(fqOp.getOutputLow());
    const auto oHiShape = getShape(fqOp.getOutputHigh());
    const auto expectedShape = Shape{1, 1, 1, 1};
    if (iLoShape != expectedShape || iHiShape != expectedShape || oLoShape != expectedShape ||
        oHiShape != expectedShape) {
        return nullptr;
    }
    auto inLowConst = fqOp.getInputLow().getDefiningOp<Const::DeclareOp>();
    auto inHighConst = fqOp.getInputHigh().getDefiningOp<Const::DeclareOp>();
    auto outLowConst = fqOp.getOutputLow().getDefiningOp<Const::DeclareOp>();
    auto outHighConst = fqOp.getOutputHigh().getDefiningOp<Const::DeclareOp>();
    if (inLowConst == nullptr || inHighConst == nullptr || outLowConst == nullptr || outHighConst == nullptr) {
        return nullptr;
    }
    const auto realType = fqOp.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto realElemType = realType.getElementType().dyn_cast<mlir::FloatType>();
    const auto outQuantizeElemType =
            getQuantizedType(outLowConst.getContentAttr(), outHighConst.getContentAttr(), fqOp.getLevels(),
                             fqOp.getLowFpType(), realElemType, false, fqOp.getLoc(), fqOp.getAutoBroadcast());
    if (outQuantizeElemType == nullptr) {
        return nullptr;
    }

    return outQuantizeElemType.dyn_cast<mlir::quant::UniformQuantizedType>();
}

//
// IE::arch37xx
//

bool IE::arch37xx::isMixPrecisionSupported(mlir::Operation* origOp, const bool isPReLUSupported, Logger log) {
    if (!mlir::isa<IE::ConvolutionOp, IE::GroupConvolutionOp, IE::AddOp, IE::AvgPoolOp, IE::TransposedConvolutionOp>(
                origOp)) {
        return false;
    }

    // Check that the kernel size are not exceding the NCE HW limits
    if (VPU::NCEInvariant::verifyKernel(origOp, log).failed()) {
        return false;
    }

    // If the Add operands have different shapes the operation will be mapped on SHAVE, which does not support mixed
    // precision operations
    if (mlir::isa<IE::AddOp>(origOp)) {
        auto addOp = mlir::dyn_cast<IE::AddOp>(origOp);
        const auto shape1 = getShape(addOp.getInput1());
        const auto shape2 = getShape(addOp.getInput2());
        if (shape1 != shape2)
            return false;
    }

    // Float input with quantized output supports leaky ReLU when quantize out is per-tensor.
    // Further checks are not necessary, bail out.
    if (isPReLUSupported) {
        return true;
    }

    // HW limitations below do not apply to NPU37XX
    // However, leaky ReLU does not work accurately in quant in / float out mode.
    // In quant in / float out flow, PReLU alpha coefficient can only be represented as prelu_mult.
    // prelu_shift is not available in such configuration.
    // Therefore, it becomes problematic to express rational negative slopes.
    // See E#58368 for details.
    const auto hasLeakyReLUConsumer = llvm::any_of(origOp->getUsers(), [](mlir::Operation* op) {
        return mlir::isa<IE::LeakyReluOp>(op);
    });

    // Thus, mixed precision is supported only when consumers and post-ops are not leaky ReLU
    return !hasLeakyReLUConsumer && !hasLeakyReLUPostOp(origOp);
}

bool vpux::IE::isPerTensorFQ(ArrayRef<IE::FakeQuantizeOp> fqOps) {
    const auto checkFQAxis = [](IE::FakeQuantizeOp fq) -> bool {
        const auto greaterThanOne = [](auto dim) {
            return dim > 1;
        };
        const auto inputLowShape = getShape(fq.getInputLow());
        const auto outputLowShape = getShape(fq.getOutputLow());
        const auto inputAxisCount = llvm::count_if(inputLowShape, greaterThanOne);
        const auto outputAxisCount = llvm::count_if(outputLowShape, greaterThanOne);
        // In case of per axis FQ, make sure that the quantization axis is the same between input and output
        if (inputAxisCount > 0 && outputAxisCount > 0) {
            VPUX_THROW_WHEN(inputLowShape.size() != outputLowShape.size(),
                            "Unaligned tensor rank for FakeQuantize constant inputs.");
            for (size_t i = 0; i < inputLowShape.size(); ++i) {
                VPUX_THROW_WHEN((inputLowShape[Dim(i)] > 1) ^ (outputLowShape[Dim(i)] > 1),
                                "FakeQuantize constant inputs use different axis");
            }
        }
        return (inputAxisCount > 0 || outputAxisCount > 0);
    };

    for (const auto& fqOp : fqOps) {
        if (checkFQAxis(fqOp)) {
            return false;
        }
    }
    return true;
}

Const::DeclareOp vpux::IE::createFQConst(mlir::MLIRContext* ctx, mlir::Location loc, float val,
                                         mlir::RankedTensorType argType, mlir::PatternRewriter& rewriter) {
    const auto denseElementVal = wrapData(mlir::RankedTensorType::get({1, 1, 1, 1}, mlir::Float32Type::get(ctx)), val);
    VPUX_THROW_UNLESS(denseElementVal != nullptr, "Failed to generate the denseElementVal.");
    const auto cstAttr = Const::ContentAttr::get(denseElementVal)
                                 .convertElemType(argType.cast<vpux::NDTypeInterface>().getElementType());
    return rewriter.create<Const::DeclareOp>(loc, argType, cstAttr);
}

Const::details::ContentRange<float> vpux::IE::getConst(Const::DeclareOp declOp) {
    const auto content = declOp.getContentAttr().fold();
    return content.getValues<float>();
}

bool vpux::IE::checkRescaledQuantApproximationForConvBasedOp(mlir::Operation* op) {
    if (!mlir::isa<IE::ConvolutionOp, IE::GroupConvolutionOp, IE::TransposedConvolutionOp>(op)) {
        return true;
    }

    auto inElemType = op->getOperand(0).getType().cast<NDTypeInterface>().getElementType();
    auto outElemType = op->getResult(0).getType().cast<NDTypeInterface>().getElementType();
    auto weightsType = op->getOperand(1).getType().cast<NDTypeInterface>().getElementType();

    auto inQuantScale = inElemType.isa<mlir::quant::QuantizedType>() ? extractScalesAndZeroPoints(inElemType).first
                                                                     : SmallVector<double>{1.0};
    auto outQuantScale = outElemType.isa<mlir::quant::QuantizedType>() ? extractScalesAndZeroPoints(outElemType).first
                                                                       : SmallVector<double>{1.0};
    auto weightsQuantScales = exractWeightsScales(weightsType);

    const auto OC = getShape(op->getOperand(1))[Dims4D::Filter::OC];
    broadcast(inQuantScale, OC);
    broadcast(outQuantScale, OC);
    broadcast(weightsQuantScales, OC);

    for (int64_t i = 0; i < OC; i++) {
        uint16_t mult = 0;
        uint8_t shift = 0;
        int8_t postShift = 0;
        double rescale = (weightsQuantScales[i] * inQuantScale[i]) / outQuantScale[i];
        std::tie(mult, shift, postShift) = approximate<decltype(mult)>(15, rescale);
        if (postShift != 0) {
            return false;
        }
    }

    return true;
}
