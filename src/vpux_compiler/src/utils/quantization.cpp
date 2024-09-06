//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/quantization.hpp"

#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/subspaces.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/compiler/utils/loop.hpp"

#include <mlir/IR/BuiltinTypes.h>

#include <cmath>

using namespace vpux;

//
// Utilities for quantized types
//

mlir::LogicalResult vpux::validateQuantElemType(mlir::Location loc, vpux::NDTypeInterface mainType) {
    if (auto perAxisQType = mainType.getElementType().dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        const auto qDim = perAxisQType.getQuantizedDimension();

        if (qDim < 0 || static_cast<int64_t>(qDim) >= mainType.getRank()) {
            return errorAt(loc, "Quantized axis '{0}' is out of main type rank '{1}'", qDim, mainType.getRank());
        }

        const auto qDimSize = mainType.getShape()[Dim(static_cast<uint32_t>(qDim))];
        const auto numScales = perAxisQType.getScales().size();

        if (qDimSize != mlir::ShapedType::kDynamic) {
            if (checked_cast<size_t>(qDimSize) != numScales) {
                return errorAt(loc,
                               "Number of scales '{0}' in per-axis quantized type do not match the quantized dimension "
                               "size '{1}'",
                               numScales, qDimSize);
            }
        }
    }

    return mlir::success();
}

mlir::Type vpux::normalizeQuantStorageType(mlir::quant::QuantizedType qType) {
    auto elemType = qType.getStorageType();
    if (const auto intType = elemType.dyn_cast_or_null<mlir::IntegerType>()) {
        return mlir::IntegerType::get(intType.getContext(), intType.getWidth(),
                                      qType.isSigned() ? mlir::IntegerType::Signed : mlir::IntegerType::Unsigned);
    }
    if (const auto lowFpType = elemType.dyn_cast_or_null<mlir::Float8E4M3FNType>()) {
        return mlir::FloatType::getFloat8E4M3FN(lowFpType.getContext());
    }
    if (const auto lowFpType = elemType.dyn_cast_or_null<mlir::Float8E5M2Type>()) {
        return mlir::FloatType::getFloat8E5M2(lowFpType.getContext());
    }

    VPUX_THROW("Unsupported storage element type {0}", elemType);
}

mlir::quant::UniformQuantizedPerAxisType vpux::expandScalesAndZP(mlir::quant::UniformQuantizedPerAxisType perAxisQType,
                                                                 ShapeRef padBefore, ShapeRef padAfter) {
    VPUX_THROW_UNLESS(padBefore.size() >= static_cast<size_t>(perAxisQType.getQuantizedDimension()),
                      "Unsupported shape size {0}. Quantized dimension index {1}", padBefore.size(),
                      perAxisQType.getQuantizedDimension());
    VPUX_THROW_UNLESS(padAfter.size() >= static_cast<size_t>(perAxisQType.getQuantizedDimension()),
                      "Unsupported shape size {0}. Quantized dimension index {1}", padAfter.size(),
                      perAxisQType.getQuantizedDimension());

    const auto quantizedDim = Dim(perAxisQType.getQuantizedDimension());

    const auto padBeforeOC = padBefore[quantizedDim];
    const auto padAfterOC = padAfter[quantizedDim];

    if (padBeforeOC == 0 && padAfterOC == 0) {
        return perAxisQType;
    }

    const auto scales = perAxisQType.getScales();
    VPUX_THROW_UNLESS(!scales.empty(), "Can't get value for expand scales.");

    const auto zeroPoints = perAxisQType.getZeroPoints();
    VPUX_THROW_UNLESS(!zeroPoints.empty(), "Can't get value for expand zero points.");
    VPUX_THROW_UNLESS(std::equal(zeroPoints.begin() + 1, zeroPoints.end(), zeroPoints.begin()),
                      "All zero points should be equal");

    // Here we need to expand scales & zero points with some values which will allow correct execution of expanded
    // convolution. Some default values (e.g. 1) does not fit here since it may lead to unsupported quantization
    // parameters (e.g. big scale value which approximation does not fit into mult & shift registers of target HW)
    // Heuristic that scales are not that different between each other is used here
    // Technically we need some way to detect if output channels we are processing are expanded ones (fake)
    // And do validation of them accordingly
    std::vector<double> newScales(padBeforeOC, scales.front());
    newScales.insert(newScales.end(), scales.begin(), scales.end());
    newScales.insert(newScales.end(), padAfterOC, scales.back());

    std::vector<int64_t> newZeroPoints(padBeforeOC, zeroPoints.front());
    newZeroPoints.insert(newZeroPoints.end(), zeroPoints.begin(), zeroPoints.end());
    newZeroPoints.insert(newZeroPoints.end(), padAfterOC, zeroPoints.back());

    VPUX_THROW_UNLESS(newScales.size() == newZeroPoints.size(),
                      "Scales & Zero Points must be of the same size, got {0} vs {1} correspondingly", newScales.size(),
                      newZeroPoints.size());

    return mlir::quant::UniformQuantizedPerAxisType::get(
            perAxisQType.getFlags(), perAxisQType.getStorageType(), perAxisQType.getExpressedType(), newScales,
            newZeroPoints, perAxisQType.getQuantizedDimension(), perAxisQType.getStorageTypeMin(),
            perAxisQType.getStorageTypeMax());
}

mlir::quant::UniformQuantizedPerAxisType vpux::tileScalesAndZP(mlir::quant::UniformQuantizedPerAxisType perAxisQType,
                                                               ShapeRef shape, ShapeRef offsets) {
    VPUX_THROW_UNLESS(offsets.size() == shape.size(), "Offsets '{0}' doesn't match shape '{1}'", offsets, shape);
    VPUX_THROW_UNLESS(shape.size() >= static_cast<size_t>(perAxisQType.getQuantizedDimension()),
                      "Unsupported shape size {0}. Quantized dimension index {1}", shape.size(),
                      perAxisQType.getQuantizedDimension());

    const auto qDim = Dim(perAxisQType.getQuantizedDimension());
    const auto qSliceSize = checked_cast<size_t>(shape[qDim]);
    const auto qSliceOffset = checked_cast<size_t>(offsets[qDim]);

    const auto scales = perAxisQType.getScales();
    const auto zeroPoints = perAxisQType.getZeroPoints();

    if (qSliceOffset == 0 && qSliceSize == scales.size()) {
        return perAxisQType;
    }

    const auto newScales = scales.slice(qSliceOffset, qSliceSize);
    const auto newZeroPoints = zeroPoints.slice(qSliceOffset, qSliceSize);

    return mlir::quant::UniformQuantizedPerAxisType::get(
            perAxisQType.getFlags(), perAxisQType.getStorageType(), perAxisQType.getExpressedType(), newScales,
            newZeroPoints, perAxisQType.getQuantizedDimension(), perAxisQType.getStorageTypeMin(),
            perAxisQType.getStorageTypeMax());
}

mlir::quant::UniformQuantizedPerAxisType vpux::changeAxis(mlir::quant::UniformQuantizedPerAxisType perAxisQType,
                                                          int32_t newAxis) {
    VPUX_THROW_UNLESS(newAxis >= 0, "Invalid axis {0} was passed", newAxis);

    if (newAxis == perAxisQType.getQuantizedDimension()) {
        return perAxisQType;
    }

    return mlir::quant::UniformQuantizedPerAxisType::get(
            perAxisQType.getFlags(), perAxisQType.getStorageType(), perAxisQType.getExpressedType(),
            perAxisQType.getScales(), perAxisQType.getZeroPoints(), newAxis, perAxisQType.getStorageTypeMin(),
            perAxisQType.getStorageTypeMax());
}

mlir::quant::QuantizedType vpux::changeStorageType(mlir::quant::QuantizedType qType, mlir::Type storageType) {
    VPUX_THROW_UNLESS(storageType.isa<mlir::IntegerType>(), "Cannot change storage type to non-integer type");

    if (qType.getStorageType() == storageType) {
        return qType;
    }

    if (auto perTensor = qType.dyn_cast<mlir::quant::UniformQuantizedType>()) {
        return mlir::quant::UniformQuantizedType::get(perTensor.getFlags(), storageType, perTensor.getExpressedType(),
                                                      perTensor.getScale(), perTensor.getZeroPoint(),
                                                      perTensor.getStorageTypeMin(), perTensor.getStorageTypeMax());
    } else if (auto perAxis = qType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        return mlir::quant::UniformQuantizedPerAxisType::get(perAxis.getFlags(), storageType,
                                                             perAxis.getExpressedType(), perAxis.getScales(),
                                                             perAxis.getZeroPoints(), perAxis.getQuantizedDimension(),
                                                             perAxis.getStorageTypeMin(), perAxis.getStorageTypeMax());
    }
    VPUX_THROW("Unsupported original type: {0}", qType);
}

bool vpux::canBeMerged(mlir::quant::UniformQuantizedPerAxisType type1, mlir::quant::UniformQuantizedPerAxisType type2) {
    const auto flags1 = type1.getFlags();
    const auto storageType1 = type1.getStorageType();
    const auto realType1 = type1.getExpressedType();
    const auto qDim1 = type1.getQuantizedDimension();
    const auto qMin1 = type1.getStorageTypeMin();
    const auto qMax1 = type1.getStorageTypeMax();

    const auto flags2 = type2.getFlags();
    const auto storageType2 = type2.getStorageType();
    const auto realType2 = type2.getExpressedType();
    const auto qDim2 = type2.getQuantizedDimension();
    const auto qMin2 = type2.getStorageTypeMin();
    const auto qMax2 = type2.getStorageTypeMax();

    return flags1 == flags2 && storageType1 == storageType2 && realType1 == realType2 && qDim1 == qDim2 &&
           qMin1 == qMin2 && qMax1 == qMax2;
}

mlir::quant::UniformQuantizedPerAxisType vpux::concatScalesAndZP(
        ArrayRef<mlir::quant::UniformQuantizedPerAxisType> types) {
    VPUX_THROW_WHEN(types.empty(), "Got empty types list in concatScalesAndZP");

    const auto flags = types.front().getFlags();
    const auto storageType = types.front().getStorageType();
    const auto realType = types.front().getExpressedType();
    const auto qDim = types.front().getQuantizedDimension();
    const auto qMin = types.front().getStorageTypeMin();
    const auto qMax = types.front().getStorageTypeMax();

    size_t newAxisSize = 0;
    for (const auto type : types) {
        VPUX_THROW_UNLESS(canBeMerged(type, types.front()), "Types '{0}' and '{1}' can't be merged", type,
                          types.front());

        newAxisSize += type.getScales().size();
    }

    SmallVector<double> newScales;
    SmallVector<int64_t> newZeroPoints;

    newScales.reserve(newAxisSize);
    newZeroPoints.reserve(newAxisSize);

    for (const auto type : types) {
        const auto scales = type.getScales();
        const auto zeroPoints = type.getZeroPoints();

        newScales.append(scales.begin(), scales.end());
        newZeroPoints.append(zeroPoints.begin(), zeroPoints.end());
    }

    return mlir::quant::UniformQuantizedPerAxisType::get(flags, storageType, realType, newScales, newZeroPoints, qDim,
                                                         qMin, qMax);
}

std::pair<Scales, ZeroPoints> vpux::extractScalesAndZeroPoints(mlir::Type tensorElemType) {
    const auto qType = tensorElemType.dyn_cast<mlir::quant::QuantizedType>();
    if (const auto uniformParams = qType.dyn_cast_or_null<mlir::quant::UniformQuantizedType>()) {
        SmallVector<double> scales{uniformParams.getScale()};
        SmallVector<int64_t> zeroPoints{uniformParams.getZeroPoint()};

        return {scales, zeroPoints};
    } else if (const auto perAxisParams = qType.dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>()) {
        SmallVector<double> scales{perAxisParams.getScales().begin(), perAxisParams.getScales().end()};
        SmallVector<int64_t> zeroPoints{perAxisParams.getZeroPoints().begin(), perAxisParams.getZeroPoints().end()};

        return {scales, zeroPoints};
    }

    VPUX_THROW("Unsupported Quantized Type {0}", qType);
}

Scales vpux::exractWeightsScales(mlir::Type weightsElemType) {
    if (weightsElemType == nullptr || !weightsElemType.isa<mlir::quant::QuantizedType>()) {
        return SmallVector<double>{1.0};
    }

    return extractScalesAndZeroPoints(weightsElemType).first;
}

vpux::QuantizationApproximation::QuantizationApproximation(VPU::ArchKind architecture, double target)
        : _mult(0), _shift(0), _postShift(0) {
    std::tie(_mult, _shift, _postShift) = approximate<decltype(_mult)>(15, target);

    VPUX_THROW_WHEN(_postShift != 0,
                    "Encountered an attempt to approximate {0} as mult = {1}, shift = {2}, postShift = {3} on {4}, "
                    "but postShift is not supported",
                    target, mult(), shift(), postShift(), architecture);
}

int64_t vpux::QuantizationApproximation::mult() const {
    return _mult;
}

int64_t vpux::QuantizationApproximation::shift() const {
    return _shift;
}

int64_t vpux::QuantizationApproximation::postShift() const {
    return _postShift;
}

void vpux::QuantizationApproximation::setMult(uint16_t mult) {
    _mult = mult;
}

void vpux::QuantizationApproximation::setShift(uint8_t shift) {
    _shift = shift;
}

vpux::EltwiseQuantizationApproximation::EltwiseQuantizationApproximation(VPU::ArchKind architecture,
                                                                         double input1Target, double input2Target,
                                                                         double outputTarget)
        : _input1(architecture, input1Target),
          _input2(architecture, input2Target),
          _output(architecture, 1 / outputTarget) {
    // We align shifts to the smaller one by dividing input MULT with 2^diff, inputs shift will be set to 0 in
    // nce_cluster_task.cpp and added to the output shift .
    //
    // what we actually do is input1 * MULT1 i32 --> + --> * MULT_OUT >> (SHIFT_OUT + SHIFT_IN) --> u8
    //                       input2 * MULT2 i32 ----^

    const auto minShift = std::min(_input1.shift(), _input2.shift());
    const auto maxShift = std::max(_input1.shift(), _input2.shift());
    // shift register is using 6 bits, so the maximum shift value is 2^6 - 1
    const int64_t maxRegisterShift = pow(2, 6) - 1;
    // mult register is using 16 bits, so the maximum mult value is 2^16 - 1
    const int64_t maxRegisterMult = pow(2, 16) - 1;

    const auto supportsShiftToMaximum = [&]() -> bool {
        if (maxShift + _output.shift() > maxRegisterShift) {
            return false;
        }
        if (_input1.mult() > maxRegisterMult >> (maxShift - _input1.shift())) {
            return false;
        }
        if (_input2.mult() > maxRegisterMult >> (maxShift - _input2.shift())) {
            return false;
        }

        return true;
    };

    if (supportsShiftToMaximum()) {
        _input1.setMult(static_cast<uint16_t>(_input1.mult() << (maxShift - _input1.shift())));
        _input2.setMult(static_cast<uint16_t>(_input2.mult() << (maxShift - _input2.shift())));
        _output.setShift(_output.shift() + maxShift);
    } else if (minShift + _output.shift() < maxRegisterShift) {
        _input1.setMult(static_cast<uint16_t>(_input1.mult() >> (_input1.shift() - minShift)));
        _input2.setMult(static_cast<uint16_t>(_input2.mult() >> (_input2.shift() - minShift)));
        _output.setShift(_output.shift() + minShift);
    } else {
        VPUX_THROW("Elwise add input1_MULT/input2_MULT/output_SHIFT out of register range");
    }
}

QuantizationApproximation vpux::EltwiseQuantizationApproximation::input1() const {
    return _input1;
}

QuantizationApproximation vpux::EltwiseQuantizationApproximation::input2() const {
    return _input2;
}

QuantizationApproximation vpux::EltwiseQuantizationApproximation::output() const {
    return _output;
}

vpux::PReLUApproximation::PReLUApproximation(VPU::ArchKind architecture, double target): _mult(0), _shift(0) {
    // TODO return logic for 11 bits for quantized case NPU37XX back as soon as it works.
    const auto bits = 11;
    int8_t postShift = 0;
    std::tie(_mult, _shift, postShift) = approximate<decltype(_mult)>(bits, target);

    VPUX_THROW_UNLESS(postShift == 0,
                      "Encountered an attempt to approximate {0} as mult = {1}, shift = {2}, postShift = {3} on {4}, "
                      "but postShift is not supported",
                      target, mult(), shift(), int64_t(postShift), architecture);
}

int64_t vpux::PReLUApproximation::mult() const {
    return _mult;
}

int64_t vpux::PReLUApproximation::shift() const {
    return _shift;
}

std::pair<int64_t, int64_t> vpux::getClampValuesForQuantizedOps(std::pair<double, double> realMinMax,
                                                                mlir::quant::QuantizedType outElemQType,
                                                                mlir::Type outElemType) {
    const auto scalesAndZp = extractScalesAndZeroPoints(outElemType);
    const auto scale = scalesAndZp.first.front();
    const auto zp = scalesAndZp.second.front();

    auto clampLowStorageMin = outElemQType.getStorageTypeMin() - zp;
    auto clampHighStorageMax = outElemQType.getStorageTypeMax() - zp;

    auto qMin = checked_cast<int64_t>(std::round(realMinMax.first / scale));
    auto qMax = checked_cast<int64_t>(std::round(realMinMax.second / scale));

    auto clampLow = std::max(clampLowStorageMin, qMin);
    auto clampHigh = std::min(clampHighStorageMax, qMax);
    return {clampLow, clampHigh};
}

//
// FakeQuantize support
//

std::tuple<double, int64_t> vpux::calcScaleAndZeroPoint(int64_t qMin, int64_t qMax, double rMin, double rMax) {
    VPUX_THROW_UNLESS(qMax >= qMin, "Wrong quantized storage values range ['{0}', '{1}']", qMin, qMax);
    VPUX_THROW_UNLESS(rMax >= rMin, "Wrong real values range ['{0}', '{1}']", rMin, rMax);

    // Is the given range actually a range or a single scalar like [-0.00, 0.00] or [3, 3]?
    if (std::fabs(rMax - rMin) < std::numeric_limits<double>::epsilon()) {
        const double scale = rMin;
        // (-inf, -eps] => scale = rMin, zp = 2
        // (-eps, eps) => scale = 1.0, zp = 0
        // [eps, inf) => scale = rMin, zp = 0

        if (std::fabs(scale) < std::numeric_limits<double>::epsilon()) {
            // "-epsilon < scale < epsilon" means that scale should be zero  ===>  scale = 1.0
            // to avoid division by zero in formula Q = R/scale + zp
            return std::make_tuple(1.0, 0);
        }
        if (scale >= std::numeric_limits<double>::epsilon()) {
            return std::make_tuple(scale, 0);
        }
        if (scale <= -std::numeric_limits<double>::epsilon()) {
            // Due to LLVM limitation scale must be >=0
            // thirdparty/llvm-project/mlir/lib/Dialect/Quant/IR/QuantTypes.cpp lines 278-280
            // quantized_value = real_value / scale + zero_point
            // real_value = (quantized_value - zero_point) * scale
            // As a workaround for a negative real value scalar -R
            // 1. apply positive scale as usual: -R/scale = -1
            // 2. set zero point to 2, which gives us Q = (-R/scale) + 2 = -1 + 2 = 1
            return std::make_tuple(scale * (-1), 2);
        }
        VPUX_THROW("Unhandled scale value.");
    }

    // Ranges that do not contain zero will generate negative zero-point which is not supported in DPU PPE pipeline
    VPUX_THROW_UNLESS(rMin <= 0 && rMax >= 0, "Real values range does not contain value zero ['{0}', '{1}']", rMin,
                      rMax);

    //
    // Determine the scale.
    //

    const auto qMinFP = static_cast<double>(qMin);
    const auto qMaxFP = static_cast<double>(qMax);
    const double scale = (rMax - rMin) / (qMaxFP - qMinFP);

    VPUX_THROW_UNLESS(std::fabs(scale) > std::numeric_limits<double>::epsilon(),
                      "Quantization scale is too small : '{0}'", scale);

    //
    // Zero point computation.
    //

    double x = qMinFP - rMin / scale;
    int64_t zp = static_cast<int64_t>(std::round(x));

    return std::make_tuple(scale, zp);
}

std::tuple<int64_t, int64_t, mlir::Type> vpux::getStorageParams(mlir::MLIRContext* ctx, int64_t levels, bool isSigned) {
    switch (levels) {
    case 256:
        if (isSigned) {
            return {-128, 127, getSInt8Type(ctx)};
        }

        return {0, levels - 1, getUInt8Type(ctx)};
    case 255:
        if (isSigned) {
            return {-127, 127, getSInt8Type(ctx)};
        }

        return {0, levels - 1, getUInt8Type(ctx)};

    case 16:
        if (isSigned) {
            return {-8, 7, getSInt4Type(ctx)};
        }

        return {0, levels - 1, getUInt4Type(ctx)};

    case 15:
        if (isSigned) {
            return {-7, 7, getSInt4Type(ctx)};
        }

        return {0, levels - 1, getUInt4Type(ctx)};

    // Because in the absence of I1 support, we must use U8 datatype.
    // [Track number: E#24341].
    case 2:
        if (isSigned) {
            return {0, 1, getSInt8Type(ctx)};
        }

        return {0, levels - 1, getUInt8Type(ctx)};

    default:
        VPUX_THROW("Got unsupported levels '{0}'", levels);
    }
}

std::tuple<int64_t, int64_t, mlir::Type> vpux::getStorageParams(mlir::MLIRContext* ctx, mlir::Type lowFpType) {
    if (lowFpType.isa<mlir::Float8E4M3FNType>()) {
        return {mlir::quant::QuantizedType::getDefaultMinimumForF8E4M3FN(),
                mlir::quant::QuantizedType::getDefaultMaximumForF8E4M3FN(), mlir::FloatType::getFloat8E4M3FN(ctx)};
    } else if (lowFpType.isa<mlir::Float8E5M2Type>()) {
        return {mlir::quant::QuantizedType::getDefaultMinimumForF8E5M2(),
                mlir::quant::QuantizedType::getDefaultMaximumForF8E5M2(), mlir::FloatType::getFloat8E5M2(ctx)};
    } else {
        VPUX_THROW("Got unsupported FP8 type '{0}'", lowFpType);
    }
}

mlir::FailureOr<std::tuple<float, float>> vpux::getFp8Range(mlir::Type lowFpType) {
    if (lowFpType.isa<mlir::Float8E4M3FNType>()) {
        return std::make_tuple(static_cast<float>(mlir::quant::QuantizedType::getDefaultMinimumForF8E4M3FN()),
                               static_cast<float>(mlir::quant::QuantizedType::getDefaultMaximumForF8E4M3FN()));
    }
    if (lowFpType.isa<mlir::Float8E5M2Type>()) {
        return std::make_tuple(static_cast<float>(mlir::quant::QuantizedType::getDefaultMinimumForF8E5M2()),
                               static_cast<float>(mlir::quant::QuantizedType::getDefaultMaximumForF8E5M2()));
    }
    return mlir::failure();
}

mlir::quant::QuantizedType vpux::getQuantizedType(mlir::Attribute lowConstAttr, mlir::Attribute highConstAttr,
                                                  std::optional<int64_t> levels, std::optional<mlir::Type> lowFpType,
                                                  mlir::FloatType realType, bool isSigned, mlir::Location loc,
                                                  IE::AutoBroadcastType broadcast, bool ignoreZPCheck,
                                                  const Logger& log) {
    const auto innerLog = log.nest("getQuantizedType");

    const auto lowConst = lowConstAttr.dyn_cast_or_null<Const::ContentAttr>();
    const auto highConst = highConstAttr.dyn_cast_or_null<Const::ContentAttr>();
    if (lowConst == nullptr || highConst == nullptr) {
        (void)errorAt(loc, "Got non constant quantization parameters (low and high values)");
        return nullptr;
    }
    // If levels is greater then MAX_LEVELS then the quantization cannot be done on HW so FakeQuantize should not be
    // split in Quantize->Dequantize
    if (levels.has_value() && *levels > MAX_LEVELS) {
        return nullptr;
    }

    mlir::Type storageType;
    int64_t qMin = 0;
    int64_t qMax = 0;
    if (levels.has_value()) {
        std::tie(qMin, qMax, storageType) = getStorageParams(lowConst.getContext(), *levels, isSigned);
    } else if (lowFpType.has_value()) {
        std::tie(qMin, qMax, storageType) = getStorageParams(lowConst.getContext(), *lowFpType);
        isSigned = true;
    } else {
        (void)errorAt(loc, "Got neither levels (for integer types) nor lowFpType (for float8 versions)");
        return nullptr;
    }

    const auto lowAttr = lowConst.fold();
    const auto highAttr = highConst.fold();
    if (lowAttr.isSplat() && highAttr.isSplat()) {
        const auto low = lowAttr.getSplatValue<double>();
        const auto high = highAttr.getSplatValue<double>();

        double scale = 0.0;
        int64_t zeroPoint = 0;
        std::tie(scale, zeroPoint) = calcScaleAndZeroPoint(qMin, qMax, low, high);

        if (lowFpType.has_value() && zeroPoint != 0) {
            innerLog.warning("HW unsupported zero point '{0}' for storage type '{1}', failed to create quantized type",
                             zeroPoint, storageType);
            return nullptr;
        }

        return mlir::quant::UniformQuantizedType::getChecked(loc, isSigned ? mlir::quant::QuantizationFlags::Signed : 0,
                                                             storageType, realType, scale, zeroPoint, qMin, qMax);
    }

    const auto lowShape = lowAttr.getType().getShape();
    const auto highShape = highAttr.getType().getShape();

    const auto broadcastShapeRes = IE::broadcastEltwiseShape(lowShape.raw(), highShape.raw(), broadcast, loc);
    if (mlir::failed(broadcastShapeRes)) {
        (void)errorAt(loc, "Low values shape '{0}' doesn't match with high values shape '{1}' and cannot be broadcast",
                      lowShape, highShape);
        return nullptr;
    }
    const auto broadcastShape = broadcastShapeRes.value();

    std::optional<int32_t> axisInd;
    for (size_t i = 0; i < broadcastShape.size(); ++i) {
        if (broadcastShape[Dim(i).ind()] == 1) {
            continue;
        }

        if (axisInd.has_value()) {
            (void)errorAt(loc, "Can't get axis index from shape '{0}'", broadcastShape);
            return nullptr;
        }

        axisInd = checked_cast<int32_t>(i);
    }
    if (!axisInd.has_value()) {
        (void)errorAt(loc, "Can't get axis index from shape '{0}'", broadcastShape);
        return nullptr;
    }

    const auto lowVals = lowAttr.getValues<double>();
    const auto highVals = highAttr.getValues<double>();

    SmallVector<double> lows(lowVals);
    SmallVector<double> highs(highVals);
    broadcastRange(lows, highs, broadcast);

    SmallVector<double> scales(lows.size());
    SmallVector<int64_t> zeroPoints(lows.size());

    loop_1d(LoopExecPolicy::Parallel, lowConst.getContext(), lows.size(), [&](size_t i) {
        std::tie(scales[i], zeroPoints[i]) = calcScaleAndZeroPoint(qMin, qMax, lows[i], highs[i]);
    });

    if (lowFpType.has_value()) {
        const auto hasUnsupportedZP = llvm::any_of(zeroPoints, [](int64_t zp) {
            return zp != 0;
        });

        if (hasUnsupportedZP) {
            innerLog.warning("HW unsupported zero point (!= 0) for storage type '{1}', failed to create quantized type",
                             storageType);
            return nullptr;
        }
    }

    if (!ignoreZPCheck && !std::equal(zeroPoints.begin() + 1, zeroPoints.end(), zeroPoints.begin())) {
        return nullptr;
    }

    return mlir::quant::UniformQuantizedPerAxisType::getChecked(
            loc, isSigned ? mlir::quant::QuantizationFlags::Signed : 0, storageType, realType, std::move(scales),
            std::move(zeroPoints), axisInd.value(), qMin, qMax);
}

void vpux::getFakeQuantParams(mlir::quant::UniformQuantizedType qElemType, int64_t& levels, float& rMin, float& rMax) {
    const auto qMin = qElemType.getStorageTypeMin();
    const auto qMax = qElemType.getStorageTypeMax();

    levels = qMax - qMin + 1;

    const auto scale = qElemType.getScale();
    const auto zeroPoint = qElemType.getZeroPoint();

    rMin = dequantize(qMin, scale, zeroPoint);
    rMax = dequantize(qMax, scale, zeroPoint);
}

void vpux::getFakeQuantParams(mlir::quant::UniformQuantizedPerAxisType qElemType, int64_t& levels,
                              SmallVectorImpl<float>& rMinVals, SmallVectorImpl<float>& rMaxVals) {
    const auto qMin = qElemType.getStorageTypeMin();
    const auto qMax = qElemType.getStorageTypeMax();

    levels = qMax - qMin + 1;

    const auto scales = qElemType.getScales();
    const auto zeroPoints = qElemType.getZeroPoints();

    rMinVals.resize(scales.size());
    rMaxVals.resize(scales.size());

    for (size_t i = 0; i < scales.size(); ++i) {
        rMinVals[i] = dequantize(qMin, scales[i], zeroPoints[i]);
        rMaxVals[i] = dequantize(qMax, scales[i], zeroPoints[i]);
    }
}

void vpux::getFakeQuantParams(vpux::NDTypeInterface qType, int64_t& levels, mlir::RankedTensorType& attrType,
                              mlir::DenseElementsAttr& rMinAttr, mlir::DenseElementsAttr& rMaxAttr) {
    const auto qElemType = qType.getElementType().dyn_cast<mlir::quant::QuantizedType>();
    VPUX_THROW_WHEN(qElemType == nullptr, "Unsupported Quantized Type '{0}'", qType.getElementType());

    if (const auto uniformType = qElemType.dyn_cast<mlir::quant::UniformQuantizedType>()) {
        float rMin, rMax;
        getFakeQuantParams(uniformType, levels, rMin, rMax);

        Shape attrShape(qType.getRank(), 1);
        attrType = mlir::RankedTensorType::get(attrShape.raw(), mlir::Float32Type::get(qType.getContext()));
        rMinAttr = mlir::DenseElementsAttr::get(attrType, rMin);
        rMaxAttr = mlir::DenseElementsAttr::get(attrType, rMax);
    } else if (const auto perAxisQType = qElemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        SmallVector<float> rMinVals, rMaxVals;
        getFakeQuantParams(perAxisQType, levels, rMinVals, rMaxVals);

        const auto axis = Dim(perAxisQType.getQuantizedDimension());

        Shape attrShape(qType.getRank(), 1);
        attrShape[axis] = rMinVals.size();

        attrType = mlir::RankedTensorType::get(attrShape.raw(), mlir::Float32Type::get(qType.getContext()));
        rMinAttr = mlir::DenseElementsAttr::get(attrType, ArrayRef(rMinVals));
        rMaxAttr = mlir::DenseElementsAttr::get(attrType, ArrayRef(rMaxVals));
    } else {
        VPUX_THROW("Unsupported Quantized Type '{0}'", qElemType);
    }
}

float vpux::fakeQuantize(float inVal, float inLow, float inHigh, float qLow, float qHigh, float fLevels) {
    if (inVal <= inLow) {
        return qLow;
    } else if (inVal > inHigh) {
        return qHigh;
    } else {
        return std::round((inVal - inLow) / (inHigh - inLow) * (fLevels - 1)) / (fLevels - 1) * (qHigh - qLow) + qLow;
    }
}

mlir::Type vpux::rescaleUniformQuantizedType(const mlir::Type tensorType, const double factor) {
    auto ndType = tensorType.cast<vpux::NDTypeInterface>();
    VPUX_THROW_UNLESS(ndType != nullptr, "Type {0} does not implement NDTypeInterface", tensorType);
    auto elemType = ndType.getElementType();
    auto uniformQElemType = elemType.dyn_cast<mlir::quant::UniformQuantizedType>();
    VPUX_THROW_UNLESS(uniformQElemType != nullptr, "Type {0} is not a UniformQuantizedType", elemType);
    const auto scale = uniformQElemType.getScale();
    const auto newScale = static_cast<double>(scale * factor);
    const auto zeroPoint = uniformQElemType.getZeroPoint();

    auto qType = elemType.dyn_cast<mlir::quant::QuantizedType>();
    auto quantizeElemType = mlir::quant::UniformQuantizedType::get(
            qType.getFlags(), qType.getStorageType(), qType.getExpressedType(), newScale, zeroPoint,
            qType.getStorageTypeMin(), qType.getStorageTypeMax());
    auto resultType = ndType.changeElemType(quantizeElemType);

    return resultType;
}
