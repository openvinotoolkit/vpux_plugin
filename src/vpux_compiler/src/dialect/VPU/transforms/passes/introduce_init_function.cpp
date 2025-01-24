//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/module_utils.hpp"
#include "vpux/compiler/core/type_interfaces.hpp"
#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/reshape_utils.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/attr_interfaces.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/transformations.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/IE/locations.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/type/float16.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/IR/Type.h>
#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/LogicalResult.h>

#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/DialectResourceBlobManager.h>

using namespace vpux;

namespace {

//
// utilities
//

std::string declareOpToString(Const::DeclareOp declareOp) {
    std::string out;
    llvm::raw_string_ostream ss(out);
    auto flags = mlir::OpPrintingFlags()
                         .elideLargeElementsAttrs()
                         .elideLargeResourceString()
                         .setAllowPrintingElementsAttrAsHex(false);
    declareOp->print(ss, flags);
    return out;
};

bool isOpenVINOConstant(Const::DeclareOp declareOp) {
    auto baseContent = declareOp.getContentAttr().getBaseContent();

    if (auto attr = mlir::dyn_cast<mlir::DenseResourceElementsAttr>(baseContent); attr != nullptr) {
        return attr.getRawHandle().getKey().starts_with(Const::OPENVINO_CONST_PREFIX);
    }

    return false;
}

SmallVector<uint32_t> computeOrder(const DimsOrder inOrder, const DimsOrder outOrder) {
    auto inPerm = inOrder.toPermutation();
    auto outPerm = outOrder.toPermutation();
    SmallVector<uint32_t> memPerm(inPerm.size());
    for (auto p : outPerm | indexed) {
        memPerm[p.index()] = static_cast<uint32_t>(inOrder.dimPos(p.value()));
    }
    return memPerm;
}

bool shouldMoveConstantToInit(Const::DeclareOp constOp) {
    // preserve splats in @main - they (should be) cheap to work with.
    const auto contentAttr = constOp.getContentAttr();

    if (contentAttr.getTransformations().empty()) {
        return false;
    }

    // ignore all non-OV constants
    if (!isOpenVINOConstant(constOp)) {
        return false;
    }

    // splat values should be quick enough to process in main()
    if (contentAttr.isSplat()) {
        return false;
    }

    auto previousType = mlir::cast<vpux::NDTypeInterface>(contentAttr.getBaseContent().getType());
    bool allViewLike = llvm::all_of(contentAttr.getTransformations(), [&constOp, &previousType](auto trans) -> bool {
        bool result = false;

        if (mlir::isa<Const::ReshapeAttr, Const::SubViewAttr, Const::CastElemTypeAttr, Const::LayoutCastAttr>(trans)) {
            result = true;
        } else if (auto memPermute = mlir::dyn_cast<Const::MemPermuteAttr>(trans)) {
            auto memPerm = memPermute.getMemPerm().getValue();
            result = memPerm.isIdentity();
        } else if (auto transpose = mlir::dyn_cast<Const::TransposeAttr>(trans)) {
            const auto inputOrder = previousType.getDimsOrder();
            const auto inPerm = inputOrder.toAffineMap(constOp.getContext());
            const auto memPerm = inPerm.compose(transpose.getOrder().getValue());
            result = memPerm.isIdentity();
        } else if (auto reorder = mlir::dyn_cast<Const::ReorderAttr>(trans)) {
            const auto inOrder = previousType.getDimsOrder();
            auto outType = trans.inferOutputType(previousType);
            const auto outOrder = outType.getDimsOrder();
            const auto memPerm =
                    mlir::AffineMap::getPermutationMap(ArrayRef(computeOrder(inOrder, outOrder)), constOp.getContext());
            result = memPerm.isIdentity();
        }

        previousType = trans.inferOutputType(previousType);
        return result;
    });

    return !allViewLike;
}

SmallVector<Const::DeclareOp> collectMoveWorthyConstantOps(mlir::func::FuncOp mainFunc) {
    SmallVector<Const::DeclareOp> toBeMovedConstants;
    mainFunc.walk([&](Const::DeclareOp constOp) {
        if (!shouldMoveConstantToInit(constOp)) {
            return;
        }
        toBeMovedConstants.push_back(constOp);
    });
    return toBeMovedConstants;
}

// The list of transformations of a particular DeclareOp can be split into two parts: One Part inside the init function
// and the other part inside the main function.
//
// For example, it is undesirable to perform a SubView operation as the last transformation inside of init. Taking two
// SubViews of the same input value produces 2 output values in init. This unnecessarily increases the required IO
// bandwidth between main and init for a pure view-like operation. It would be better to delay the SubView by performing
// it as part of main. Then we only have to transfer a single output value from init to main.
struct TransformationsSplit {
    // Quantized types are not supported for network IO => Insert two IE::QuantizeCastOps at the boundary if required.
    class QuantizedIOAdapter {
    public:
        QuantizedIOAdapter(mlir::Type quantizedType, mlir::Type storageType)
                : _quantizedType(quantizedType), _storageType(storageType) {
        }

        mlir::Value adapt(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value input) const {
            return builder.create<IE::QuantizeCastOp>(appendLoc(loc, "_quant_cast"), input, _storageType);
        }

        mlir::Value deadapt(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value input) const {
            return builder.create<VPU::QuantizeCastOp>(appendLoc(loc, "_quant_cast"), input, _quantizedType);
        }

    private:
        mlir::Type _quantizedType;
        mlir::Type _storageType;
    };

    TransformationsSplit(Const::DeclareOp declareOp): declareOp(declareOp) {
        auto transformations = declareOp.getContentAttr().getTransformations();

        if (mlir::isa<Const::SubViewAttr>(transformations.back())) {
            inInitTransformations = transformations.drop_back();
            postInitTransformations = transformations.take_back();
        } else {
            inInitTransformations = transformations;
            postInitTransformations = {};
        }

        auto baseType = declareOp.getContentAttr().getBaseContent().getType();
        auto finalType = Const::inferFinalType(baseType, inInitTransformations);

        // quantized types are not supported for network IO
        if (auto qType = mlir::dyn_cast<mlir::quant::QuantizedType>(finalType.getElementType()); qType != nullptr) {
            auto normalizedType = normalizeQuantStorageType(qType);
            ioAdaptor = QuantizedIOAdapter{qType, normalizedType};
        }
    }

    Const::DeclareOp declareOp;
    // These ArrayRefs are valid as long as the underlying attribute exists, which is until the context is destroyed.
    ArrayRef<Const::TransformAttrInterface> inInitTransformations;
    ArrayRef<Const::TransformAttrInterface> postInitTransformations;
    std::optional<QuantizedIOAdapter> ioAdaptor;
};

SmallVector<TransformationsSplit> splitTransformations(ArrayRef<Const::DeclareOp> declareOps) {
    return to_small_vector(declareOps | transformed([](auto op) {
                               return TransformationsSplit(op);
                           }));
}

// Two DeclareOps will be mapped to the same input index if they share the same underlying mlir::ElementsAttr object.
class DeclareOpInputMap {
public:
    DeclareOpInputMap(ArrayRef<Const::DeclareOp> declareOps) {
        size_t index = 0;
        for (auto declareOp : declareOps) {
            auto baseContent = declareOp.getContentAttr().getBaseContent();

            if (!_elementsToIndex.contains(baseContent)) {
                // in our case name will be something like "ov_*"
                auto name = mlir::cast<mlir::DenseResourceElementsAttr>(baseContent).getRawHandle().getKey();
                _elementsToIndex[baseContent] = index++;
                _argumentTypes.push_back(baseContent.getType());
                _argumentNames.push_back(name.str());
            }
        }
    }

    std::tuple<size_t, mlir::Type> getArgumentInfo(Const::DeclareOp declareOp) const {
        auto baseContent = declareOp.getContentAttr().getBaseContent();
        auto it = _elementsToIndex.find(baseContent);
        VPUX_THROW_WHEN(it == _elementsToIndex.end(), "Failed to find constant operation: {0}", declareOp);
        return {it->second, baseContent.getType()};
    }

    ArrayRef<mlir::Type> getArgumentTypes() const {
        return _argumentTypes;
    }

    ArrayRef<std::string> getArgumentNames() const {
        return _argumentNames;
    }

private:
    mlir::DenseMap<mlir::ElementsAttr, size_t> _elementsToIndex;
    mlir::SmallVector<mlir::Type> _argumentTypes;
    mlir::SmallVector<std::string> _argumentNames;
};

// Each DeclareOp corresponds to a single result value of init. A result value can belong to multiple DeclareOps if for
// example their underlying elements attributes and in-init-transformations are the same.
class DeclareOpOutputMap {
public:
    // We take this awkward looking ArrayRef of tuples instead of a DenseMap<mlir::Value, SmallVector<Const::DeclareOp>>
    // directly, so we can be sure about a deterministic order of result values that only depends on the IR itself and
    // not the context.
    DeclareOpOutputMap(ArrayRef<std::tuple<mlir::Value, Const::DeclareOp>> valueDeclareOpPairs) {
        // Because of CSE-caching, multiple DeclareOps can be associated with the same result value. This is why
        // we want set semantics here.
        for (auto [resultValue, declareOp] : valueDeclareOpPairs) {
            _resultValues.insert(resultValue);
        }

        // temporary map for easy lookup
        DenseMap<mlir::Value, SmallVector<Const::DeclareOp>> valueToDeclareOps;
        for (auto [resultValue, declareOp] : valueDeclareOpPairs) {
            valueToDeclareOps[resultValue].push_back(declareOp);
        }

        _resultTypes.resize(_resultValues.size());
        _resultNames.resize(_resultValues.size());

        for (auto [resultIndex, resultValue] : _resultValues.getArrayRef() | indexed) {
            auto declareOps = valueToDeclareOps.at(resultValue);

            for (auto declareOp : declareOps) {
                _declareOpToIndex[declareOp] = resultIndex;
            }

            auto ngraphName =
                    mlir::cast<mlir::DenseResourceElementsAttr>(declareOps.front().getContentAttr().getBaseContent())
                            .getRawHandle()
                            .getKey();
            _resultTypes[resultIndex] = resultValue.getType();
            _resultNames[resultIndex] = ngraphName.str();
        }
    }

    size_t getResultIndex(Const::DeclareOp declareOp) const {
        return _declareOpToIndex.at(declareOp);
    }

    // type, result index, ngraph name
    std::tuple<mlir::Type, size_t, StringRef> getResultInfo(Const::DeclareOp declareOp) const {
        size_t index = _declareOpToIndex.at(declareOp);
        return {_resultTypes[index], index, _resultNames[index]};
    }

    ArrayRef<mlir::Type> getResultTypes() const {
        return _resultTypes;
    }

    ArrayRef<mlir::Value> getResultValues() const {
        return _resultValues.getArrayRef();
    }

    std::string getUniqueResultName(size_t index) const {
        // TODO: #E-142072 Develop unique hash that is independent of MLIR context and only depends on the operations
        // themselves.
        return formatv("out_{0}_hash_{1}", _resultNames[index], index);
    }

private:
    mlir::DenseMap<Const::DeclareOp, size_t> _declareOpToIndex;
    mlir::SetVector<mlir::Value> _resultValues;
    mlir::SmallVector<mlir::Type> _resultTypes;
    mlir::SmallVector<std::string> _resultNames;
};

// We want to cache the results of mapping a list of transformations to operations to avoid the call of a
// UniquifyOps pass. Experiments showed that the load became significant in some cases.
class OperationCache {
public:
    using OperationPatternT = std::tuple<mlir::Value, ArrayRef<Const::TransformAttrInterface>>;
    using KeyT = std::tuple<mlir::Value, Const::TransformAttrInterfaceArrayAttr>;

    OperationCache(const Logger& log): _log(log) {
    }

    mlir::Value findCachedResult(const OperationPatternT& pattern) {
        auto it = _cachedResults.find(getKey(pattern));
        if (it == _cachedResults.end()) {
            return {};
        }
        return it->second;
    }

    void cacheResult(const OperationPatternT& pattern, mlir::Value value) {
        _log.trace("Mapping input {0} with transformations {1} to {2}", std::get<0>(pattern), std::get<1>(pattern),
                   value);
        _cachedResults[getKey(pattern)] = value;
    }

private:
    static KeyT getKey(const OperationPatternT& pattern) {
        auto value = std::get<0>(pattern);
        return {value, Const::TransformAttrInterfaceArrayAttr::get(value.getContext(), std::get<1>(pattern))};
    }

    llvm::DenseMap<KeyT, mlir::Value> _cachedResults;
    Logger _log;
};

//
// IntroduceInitFunctionPass
//

class IntroduceInitFunctionPass final : public VPU::IntroduceInitFunctionBase<IntroduceInitFunctionPass> {
public:
    enum class Mode { Unspecified, GenerateMain, GenerateInit, GenerateAll };

    explicit IntroduceInitFunctionPass(const Logger& log): _operationCache(log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    static mlir::Value createAvgPoolForInterQuantizedConvert(mlir::OpBuilder& builder, mlir::Value input,
                                                             mlir::Location loc, mlir::quant::QuantizedType inType,
                                                             mlir::quant::QuantizedType outType);
    std::tuple<ArrayRef<Const::TransformAttrInterface>, mlir::Value> createMatchingOperations(
            mlir::OpBuilder& builder, mlir::Value input, mlir::Location loc,
            ArrayRef<Const::TransformAttrInterface> transformations);
    std::tuple<ArrayRef<Const::TransformAttrInterface>, mlir::Value> createMatchingPostInitOperations(
            mlir::OpBuilder& builder, mlir::Value input, mlir::Location loc,
            ArrayRef<Const::TransformAttrInterface> transformations);
    std::tuple<mlir::func::FuncOp, DeclareOpOutputMap> buildInitFunction(mlir::func::FuncOp mainFuncOp,
                                                                         ArrayRef<TransformationsSplit> declareOps,
                                                                         const DeclareOpInputMap& inputMap);
    // applies necessary changes in order for main to co-exist with init.
    void updateMainToAccommodateInit(mlir::func::FuncOp mainFuncOp, mlir::func::FuncOp initFuncOp,
                                     const DeclareOpOutputMap& outputMap,
                                     ArrayRef<TransformationsSplit> preprocessedDeclareOps);

    // configures NetworkInfo to assume init-schedule is the entry-point.
    void setNetworkEntryPointToInit(IE::CNNNetworkOp mainInfo, mlir::func::FuncOp initFuncOp,
                                    const DeclareOpInputMap& inputMap, const DeclareOpOutputMap& outputMap);
    // configures NetworkInfo to assume *updated* main-schedule is the
    // entry-point. the behaviour is to be considered equivalent to setting the
    // entry-point to init.
    void setNetworkEntryPointToMain(IE::CNNNetworkOp mainInfo, mlir::func::FuncOp initFuncOp,
                                    const DeclareOpOutputMap& outputMap);
    // creates new main that calls init and main in sequence. this function
    // becomes the new entry-point.
    void buildWrapperOpForInitAndMain(IE::CNNNetworkOp mainInfo, mlir::func::FuncOp mainFuncOp,
                                      mlir::func::FuncOp initFuncOp, const DeclareOpInputMap& inputMap,
                                      ArrayRef<TransformationsSplit> preprocessedDeclareOps);
    // erases original const.Declare operations. this has to be done after
    // everything else.
    void eraseOriginalOps(ArrayRef<TransformationsSplit> preprocessedDeclareOps);

    mlir::LogicalResult initialize(mlir::MLIRContext* context) final;
    void safeRunOnModule() final;

    Mode _mode = Mode::Unspecified;
    OperationCache _operationCache;
};

// This method attempts to create a matching IE::AvgPoolOp for per-axis
// quantized ConvertElemType. This is expected to yield more efficient IR.
mlir::Value IntroduceInitFunctionPass::createAvgPoolForInterQuantizedConvert(mlir::OpBuilder& builder,
                                                                             mlir::Value input, mlir::Location loc,
                                                                             mlir::quant::QuantizedType inQType,
                                                                             mlir::quant::QuantizedType outQType) {
    auto inPerAxisQType = mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>(inQType);
    auto outPerAxisQType = mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>(outQType);

    if (inPerAxisQType == nullptr || outPerAxisQType == nullptr) {
        return nullptr;
    }

    const auto inScales = extractScalesAndZeroPoints(inPerAxisQType).first;
    const auto outScales = extractScalesAndZeroPoints(outPerAxisQType).first;
    if (inScales.size() != outScales.size()) {
        return nullptr;
    }

    auto allScalesAreEqual = llvm::all_of(llvm::zip(inScales, outScales), [](auto scales) {
        return std::get<0>(scales) == std::get<1>(scales);
    });
    if (!allScalesAreEqual) {
        return nullptr;
    }

    // Note: do what ConvertElemType does to restore the offset.
    const auto bias = Const::details::getValueRangeOffset(inQType, outQType);

    auto perTensorInQType = mlir::quant::UniformQuantizedType::get(
            inPerAxisQType.getFlags(), inPerAxisQType.getStorageType(), inPerAxisQType.getExpressedType(),
            /*scale=*/1.0, /*zeroPoint=*/0, inPerAxisQType.getStorageTypeMin(), inPerAxisQType.getStorageTypeMax());
    auto perTensorOutQType = mlir::quant::UniformQuantizedType::get(
            outPerAxisQType.getFlags(), outPerAxisQType.getStorageType(), outPerAxisQType.getExpressedType(),
            /*scale=*/1.0, /*zeroPoint=*/bias, outPerAxisQType.getStorageTypeMin(),
            outPerAxisQType.getStorageTypeMax());

    // convert per-axis case to per-tensor:
    auto normInQuantCast = builder.create<IE::QuantizeCastOp>(loc, input, normalizeQuantStorageType(inQType));
    auto avgInput = builder.create<IE::QuantizeCastOp>(loc, normInQuantCast, perTensorInQType).getResult();

    const SmallVector<int64_t> poolStrides = {1, 1};
    const SmallVector<int64_t> poolKernels = {1, 1};
    const SmallVector<int64_t> pads = {0, 0};
    auto ctx = builder.getContext();

    // implement inter-quantized-type convert via average pooling (on per-tensor
    // quantization types): such convert is essentially `IE.Add(%x, zero-point)`
    // and Add could be done via AvgPool.
    auto avgType = mlir::cast<NDTypeInterface>(avgInput.getType()).changeElemType(perTensorOutQType);
    auto avgPool = builder.create<IE::AvgPoolOp>(
            loc, avgType, avgInput, getIntArrayAttr(ctx, poolKernels), getIntArrayAttr(ctx, poolStrides),
            getIntArrayAttr(ctx, pads), getIntArrayAttr(ctx, pads),
            vpux::IE::RoundingTypeAttr::get(ctx, vpux::IE::RoundingType::FLOOR),
            mlir::UnitAttr::get(builder.getContext()), nullptr, nullptr, nullptr, nullptr, nullptr);

    // convert per-tensor case to per-axis (restore the original type):
    auto normOutQuantCast =
            builder.create<IE::QuantizeCastOp>(loc, avgPool, normalizeQuantStorageType(perTensorOutQType));
    auto resultValue = builder.create<IE::QuantizeCastOp>(loc, normOutQuantCast, outQType);

    if (mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(avgPool.getInput().getType().getElementType()) ||
        mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(avgPool.getOutput().getType().getElementType())) {
        return nullptr;
    }

    return resultValue;
}

std::tuple<ArrayRef<Const::TransformAttrInterface>, mlir::Value> IntroduceInitFunctionPass::createMatchingOperations(
        mlir::OpBuilder& builder, mlir::Value input, mlir::Location loc,
        ArrayRef<Const::TransformAttrInterface> transformations) {
    if (transformations.empty()) {
        return {ArrayRef<Const::TransformAttrInterface>{}, input};
    }

    // 1-to-n mappings
    if (auto cachedValue = _operationCache.findCachedResult({input, transformations.take_front()});
        cachedValue != nullptr) {
        return {transformations.drop_front(), cachedValue};
    }

    mlir::Value outputValue =
            llvm::TypeSwitch<Const::TransformAttrInterface, mlir::Value>(transformations.front())
                    .Case<Const::AddAttr>([&](Const::AddAttr add) {
                        const auto biasValue = checked_cast<float>(add.getBias().getValueAsDouble());

                        const auto biasLoc = appendLoc(loc, "_bias");
                        SmallVector<int64_t> shapeRank = {1};
                        auto biasType =
                                mlir::RankedTensorType::get(shapeRank, mlir::Float32Type::get(builder.getContext()));
                        auto transform = [&](Const::ContentSetup& setup) -> Const::ContentSetup {
                            return setup.castElemType(mlir::cast<NDTypeInterface>(input.getType()).getElementType());
                        };
                        auto bias = Const::createConst<float>(builder, biasLoc, biasType, {biasValue}, transform);

                        return builder.create<IE::AddOp>(loc, input, bias, IE::AutoBroadcastType::NUMPY,
                                                         /*postOp=*/nullptr, /*clamp=*/nullptr,
                                                         /*outputChannels=*/nullptr,
                                                         /*inputChannels=*/nullptr);
                    })
                    .Case<Const::BroadcastAttr>([&](Const::BroadcastAttr broadcast) {
                        const auto axis = broadcast.getAxis().getInt();
                        const auto dimValue = broadcast.getValue().getInt();
                        auto shape = SmallVector<int64_t>(input.getType().cast<NDTypeInterface>().getShape().raw());
                        shape[axis] = dimValue;

                        const auto targetShapeLoc = appendLoc(loc, "_shape");
                        SmallVector<int64_t> shapeRank = {static_cast<int64_t>(shape.size())};
                        auto targetShapeType =
                                mlir::RankedTensorType::get(shapeRank, getInt64Type(builder.getContext()));
                        auto targetShape = Const::createConst<int64_t>(builder, targetShapeLoc, targetShapeType, shape);

                        return builder.create<IE::BroadcastOp>(loc, input, targetShape, /*axesMapping=*/nullptr,
                                                               /*mode=*/nullptr);
                    })
                    .Case<Const::ChangeShapeAndElemTypeAttr>(
                            [&](Const::ChangeShapeAndElemTypeAttr changeShapeAndElemType) {
                                const auto inElemType =
                                        mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>(
                                                mlir::cast<NDTypeInterface>(input.getType()).getElementType());
                                const auto outElemType =
                                        mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>(
                                                changeShapeAndElemType.getElemType());

                                // see IE::AffineReshape::fold()
                                const bool specialCaseOfAffineReshapeFolding =
                                        inElemType != nullptr && outElemType != nullptr &&
                                        isQuantizedDimensionPermutation(inElemType, outElemType);
                                VPUX_THROW_UNLESS(specialCaseOfAffineReshapeFolding,
                                                  "Unsupported affine-reshape operation");

                                const auto outputShape = parseIntArrayAttr<int64_t>(changeShapeAndElemType.getShape());
                                const auto reassociationMap = IE::getReassociationMap(
                                        input.getType().cast<NDTypeInterface>().getShape().raw(), outputShape);
                                const auto dimMapping = getIntArrayOfArray(changeShapeAndElemType.getContext(),
                                                                           reassociationMap.value());
                                return builder.create<IE::AffineReshapeOp>(loc, input, dimMapping,
                                                                           changeShapeAndElemType.getShape());
                            })
                    .Case<Const::CastElemTypeAttr>([&](Const::CastElemTypeAttr cast) -> mlir::Value {
                        const auto inElemType = mlir::cast<NDTypeInterface>(input.getType()).getElementType();
                        const auto outElemType = cast.getElemType();

                        const bool inputQuantized = mlir::isa<mlir::quant::QuantizedType>(inElemType);
                        const bool outputQuantized = mlir::isa<mlir::quant::QuantizedType>(outElemType);
                        VPUX_THROW_WHEN(inputQuantized && outputQuantized,
                                        "Casting between quantized types {0} -> {1} is not supported", inElemType,
                                        outElemType);

                        if (inputQuantized || outputQuantized) {
                            return builder.create<IE::QuantizeCastOp>(loc, input, outElemType);
                        }
                        return builder.create<IE::ConvertOp>(loc, input, outElemType);
                    })
                    .Case<Const::ConvertElemTypeAttr>([&](Const::ConvertElemTypeAttr convert) -> mlir::Value {
                        const auto inElemType = mlir::dyn_cast_or_null<mlir::quant::QuantizedType>(
                                mlir::cast<NDTypeInterface>(input.getType()).getElementType());
                        const auto outElemType =
                                mlir::dyn_cast_or_null<mlir::quant::QuantizedType>(convert.getElemType());

                        // quantized-to-quantized conversion is special
                        if (inElemType != nullptr && outElemType != nullptr) {
                            return createAvgPoolForInterQuantizedConvert(builder, input, loc, inElemType, outElemType);
                        }
                        return builder.create<IE::ConvertOp>(loc, input, convert.getElemType());
                    })
                    .Case<Const::DequantizeAttr>([&](Const::DequantizeAttr /*dequantize*/) {
                        const auto qElemType = input.getType()
                                                       .cast<NDTypeInterface>()
                                                       .getElementType()
                                                       .cast<mlir::quant::QuantizedType>();
                        return builder.create<IE::DequantizeOp>(loc, input, qElemType.getExpressedType());
                    })
                    .Case<Const::QuantizeAttr>([&](Const::QuantizeAttr quantizeAttr) {
                        return builder.create<IE::QuantizeOp>(loc, input, quantizeAttr.getTargetType());
                    })
                    .Case<Const::LayoutCastAttr>([&](Const::LayoutCastAttr layoutCast) {
                        return builder.create<IE::LayoutCastOp>(loc, input, layoutCast.getDstOrder());
                    })
                    .Case<Const::MemPermuteAttr>([&](Const::MemPermuteAttr memPermute) {
                        return builder.create<IE::MemPermuteOp>(loc, input, memPermute.getDstOrder(),
                                                                memPermute.getMemPerm());
                    })
                    .Case<Const::PadWithZeroAttr>([&](Const::PadWithZeroAttr padWithZero) {
                        auto padOp = builder.create<IE::PadOp>(
                                loc, input, /*padsBegin=*/nullptr, /*padsEnd=*/nullptr,
                                /*padValue=*/nullptr, padWithZero.getPadBefore(), padWithZero.getPadAfter(),
                                getFPAttr(builder.getContext(), 0.0), IE::PadMode::CONSTANT, nullptr);
                        return padOp;
                    })
                    .Case<Const::ReorderAttr>([&](Const::ReorderAttr reorder) {
                        return builder.create<IE::ReorderOp>(loc, input, reorder.getOrder());
                    })
                    .Case<Const::RescaleAttr>([&](Const::RescaleAttr rescale) {
                        const auto scaleValue = checked_cast<float>(rescale.getScale().getValueAsDouble());

                        const auto scaleLoc = appendLoc(loc, "_scale");
                        SmallVector<int64_t> shapeRank = {1};
                        auto scaleType =
                                mlir::RankedTensorType::get({shapeRank}, mlir::Float32Type::get(builder.getContext()));
                        auto transform = [&](Const::ContentSetup& setup) -> Const::ContentSetup {
                            return setup.castElemType(mlir::cast<NDTypeInterface>(input.getType()).getElementType());
                        };
                        auto scale = Const::createConst<float>(builder, scaleLoc, scaleType, {scaleValue}, transform);

                        return builder.create<IE::MultiplyOp>(loc, input, scale, IE::AutoBroadcastType::NUMPY,
                                                              /*postOp=*/nullptr, /*clamp=*/nullptr,
                                                              /*outputChannels=*/nullptr, /*inputChannels=*/nullptr);
                    })
                    .Case<Const::ReshapeAttr>([&](Const::ReshapeAttr reshape) -> mlir::Value {
                        if (mlir::cast<NDTypeInterface>(input.getType()).getDimsOrder().isIdentity()) {
                            return builder.create<IE::ReshapeOp>(loc, input, nullptr, false, reshape.getShape());
                        }

                        return builder.create<IE::ShapeCastOp>(loc, input, reshape.getShape());
                    })
                    .Case<Const::ScalarMultInverseAttr>(
                            [&](Const::ScalarMultInverseAttr /*scalarMultInverse*/) -> mlir::Value {
                                const auto inverseLoc = appendLoc(loc, "_inverse");
                                SmallVector<int64_t> shapeRank = {1};
                                const auto inputElemType = input.getType().cast<NDTypeInterface>().getElementType();
                                auto inverseType = mlir::RankedTensorType::get({shapeRank}, inputElemType);

                                const auto data = [&]() -> mlir::DenseElementsAttr {
                                    if (mlir::isa<mlir::Float16Type>(inputElemType)) {
                                        return mlir::DenseElementsAttr::get(inverseType,
                                                                            ArrayRef({type::float16(1.0)}));
                                    } else if (mlir::isa<mlir::Float32Type>(inputElemType)) {
                                        return mlir::DenseElementsAttr::get(inverseType, ArrayRef({1.0f}));
                                    } else if (mlir::isa<mlir::Float64Type>(inputElemType)) {
                                        return mlir::DenseElementsAttr::get(inverseType, ArrayRef({1.0}));
                                    }
                                    return nullptr;
                                }();
                                if (data == nullptr) {
                                    return nullptr;
                                }

                                auto contentAttr = Const::ContentAttr::get(data);
                                auto inverse = builder.create<Const::DeclareOp>(inverseLoc, inverseType,
                                                                                std::move(contentAttr));
                                return builder.create<IE::DivideOp>(loc, inverse, input, IE::AutoBroadcastType::NUMPY);
                            })
                    .Case<Const::SubViewAttr>([&](Const::SubViewAttr subview) {
                        return builder.create<IE::SliceOp>(loc, input, subview.getOffset(), subview.getShape());
                    })
                    .Case<Const::TransposeAttr>([&](Const::TransposeAttr transpose) {
                        return builder.create<IE::TransposeOp>(loc, input, /*order=*/nullptr, transpose.getOrder());
                    })
                    .Case<Const::SparsifyAttr>([&](Const::SparsifyAttr sparsify) -> mlir::Value {
                        // it's fine if sparsity is disabled
                        if (!sparsify.getCompressOutputType().getValue()) {
                            return input;
                        }

                        return nullptr;
                    })
                    .Case<Const::ReverseAttr>([&](Const::ReverseAttr) -> mlir::Value {
                        return nullptr;
                    })
                    .Default([](Const::TransformAttrInterface) {
                        return nullptr;
                    });

    if (outputValue == nullptr) {
        return {transformations, mlir::Value{}};
    }

    _operationCache.cacheResult({input, transformations.take_front()}, outputValue);

    return {transformations.drop_front(), outputValue};
}

std::tuple<ArrayRef<Const::TransformAttrInterface>, mlir::Value>
IntroduceInitFunctionPass::createMatchingPostInitOperations(mlir::OpBuilder& builder, mlir::Value input,
                                                            mlir::Location loc,
                                                            ArrayRef<Const::TransformAttrInterface> transformations) {
    if (transformations.empty()) {
        return {ArrayRef<Const::TransformAttrInterface>{}, input};
    }

    auto outputValue =
            llvm::TypeSwitch<Const::TransformAttrInterface, mlir::Value>(transformations.front())
                    .Case<Const::SubViewAttr>([&](Const::SubViewAttr subView) -> mlir::Value {
                        return builder.create<VPU::SliceOp>(loc, input, subView.getOffset(), subView.getShape());
                    })
                    .Default([](Const::TransformAttrInterface) -> mlir::Value {
                        return nullptr;
                    });

    if (outputValue == nullptr) {
        return {transformations, mlir::Value{}};
    }

    _operationCache.cacheResult({input, transformations.take_front()}, outputValue);

    return {transformations.drop_front(), outputValue};
}

std::tuple<mlir::func::FuncOp, DeclareOpOutputMap> IntroduceInitFunctionPass::buildInitFunction(
        mlir::func::FuncOp mainFuncOp, ArrayRef<TransformationsSplit> declareOps, const DeclareOpInputMap& inputMap) {
    OpBuilderLogger builderLog(_log.nest());

    // create empty @init() : () -> ()
    auto initFuncOp = [&]() {
        mlir::OpBuilder moduleBuilder(&getContext(), &builderLog);
        moduleBuilder.setInsertionPoint(mainFuncOp);
        auto initLoc = appendLoc(mainFuncOp.getLoc(), "_init");
        auto initFuncType = mlir::FunctionType::get(&getContext(), {}, {});
        return moduleBuilder.create<mlir::func::FuncOp>(initLoc, "init", initFuncType);
    }();

    auto bodyBlock = initFuncOp.addEntryBlock();
    auto initBuilder = mlir::OpBuilder::atBlockEnd(bodyBlock, &builderLog);

    for (auto type : inputMap.getArgumentTypes()) {
        bodyBlock->addArgument(type, initFuncOp.getLoc());
    }

    SmallVector<std::tuple<mlir::Value, Const::DeclareOp>> valueDeclareOpPairs;

    for (auto [declareOp, inInitTransformations, postInitTransformations, ioAdaptor] : declareOps) {
        auto [argIndex, argType] = inputMap.getArgumentInfo(declareOp);
        auto argValue = bodyBlock->getArgument(argIndex);
        auto loc = appendLoc(initFuncOp.getLoc(), "_op{0}", argIndex);

        mlir::Value value = argValue;

        if (_log.isActive(LogLevel::Trace)) {
            _log.trace("Creating matching operations for '{0}'", declareOpToString(declareOp));
            _log.trace("  These transformations are put *inside* the body '{0}'",
                       Const::TransformAttrInterfaceArrayAttr::get(&getContext(), inInitTransformations));
            _log.trace("  These transformations are put *after* the body '{0}'",
                       Const::TransformAttrInterfaceArrayAttr::get(&getContext(), postInitTransformations));
        }

        while (!inInitTransformations.empty()) {
            initBuilder.setInsertionPointToEnd(bodyBlock);
            auto [remainingTransformations, resultValue] =
                    createMatchingOperations(initBuilder, value, loc, inInitTransformations);
            VPUX_THROW_WHEN(resultValue == nullptr,
                            "The following transformations cannot be mapped to equivalent IE operations: {0}",
                            inInitTransformations);

            value = resultValue;
            inInitTransformations = remainingTransformations;
        }

        if (ioAdaptor.has_value()) {
            if (auto cachedValue = _operationCache.findCachedResult({value, {}}); cachedValue != nullptr) {
                value = cachedValue;
            } else {
                auto newValue = ioAdaptor->adapt(initBuilder, loc, value);
                _operationCache.cacheResult({value, {}}, newValue);
                value = newValue;
            }
        }

        valueDeclareOpPairs.push_back({value, declareOp});
    }

    auto outputMap = DeclareOpOutputMap(valueDeclareOpPairs);

    initBuilder.setInsertionPointToEnd(bodyBlock);
    initBuilder.create<mlir::func::ReturnOp>(appendLoc(initFuncOp.getLoc(), "_return"), outputMap.getResultValues());
    auto initFuncType = mlir::FunctionType::get(&getContext(), inputMap.getArgumentTypes(), outputMap.getResultTypes());
    initFuncOp.setFunctionType(initFuncType);

    return {initFuncOp, outputMap};
}

void IntroduceInitFunctionPass::setNetworkEntryPointToInit(IE::CNNNetworkOp mainInfo, mlir::func::FuncOp initFuncOp,
                                                           const DeclareOpInputMap& inputMap,
                                                           const DeclareOpOutputMap& outputMap) {
    mainInfo.setEntryPoint(initFuncOp.getSymName());

    mlir::OpBuilder::Listener listener;
    mlir::OpBuilder builder(&getContext(), &listener);

    // update input types
    auto& inputsRegion = mainInfo.getInputsInfo();
    inputsRegion.getBlocks().clear();
    inputsRegion.getBlocks().push_back(new mlir::Block());
    builder.setInsertionPointToStart(&inputsRegion.front());

    const auto initFuncType = initFuncOp.getFunctionType();
    for (auto [type, name] : llvm::zip(initFuncType.getInputs(), inputMap.getArgumentNames())) {
        auto inputName = mlir::StringAttr::get(&getContext(), formatv("in_{0}", name));
        builder.create<IE::DataInfoOp>(appendLoc(mainInfo.getLoc(), inputName), inputName, type,
                                       /*OptionalAttr originalShape*/ nullptr,
                                       /*OptionalAttr friendlyName*/ nullptr,
                                       /*OptionalAttr inputName*/ nullptr,
                                       /*OptionalAttr tensorNames*/ nullptr,
                                       /*profilingSectionsCount=*/0);
    }

    // update output types
    auto& outputsRegion = mainInfo.getOutputsInfo();
    outputsRegion.getBlocks().clear();
    outputsRegion.getBlocks().push_back(new mlir::Block());
    builder.setInsertionPointToStart(&outputsRegion.front());

    for (auto [index, type] : outputMap.getResultTypes() | indexed) {
        auto outputName = outputMap.getUniqueResultName(index);
        builder.create<IE::DataInfoOp>(appendLoc(mainInfo.getLoc(), outputName), outputName, type,
                                       /*OptionalAttr originalShape*/ nullptr,
                                       /*OptionalAttr friendlyName*/ nullptr,
                                       /*OptionalAttr inputName*/ nullptr,
                                       /*OptionalAttr tensorNames*/ nullptr,
                                       /*profilingSectionsCount=*/0);
    }
}

void IntroduceInitFunctionPass::updateMainToAccommodateInit(mlir::func::FuncOp mainFuncOp,
                                                            mlir::func::FuncOp initFuncOp,
                                                            const DeclareOpOutputMap& outputMap,
                                                            ArrayRef<TransformationsSplit> preprocessedDeclareOps) {
    // pretend that the output values of @init() is the input to @main(). this
    // means that we change the signature of @main() and replace all the uses of
    // the constants with the argument inputs of @main() modulo the subviews.
    size_t inputIndexOffset = mainFuncOp.getFunctionType().getNumInputs();
    auto initFuncType = initFuncOp.getFunctionType();

    // append init func result types to main func input types
    auto mainFuncType = mainFuncOp.getFunctionType();
    auto inputTypes = SmallVector<mlir::Type>(mainFuncType.getInputs());
    inputTypes.append(initFuncType.getResults().begin(), initFuncType.getResults().end());
    mainFuncOp.setFunctionType(mlir::FunctionType::get(&getContext(), inputTypes, mainFuncType.getResults()));

    // Update block arguments manually, because setFunctionType() does not take care of this.
    auto& bodyBlock = mainFuncOp.getFunctionBody().front();
    for (auto result : initFuncType.getResults()) {
        bodyBlock.addArgument(result, mainFuncOp.getLoc());
    }

    // build remaining transformations and delete DeclareOps
    mlir::OpBuilder::Listener listener;
    mlir::OpBuilder builder(mainFuncOp.getFunctionBody(), &listener);

    for (auto [declareOp, _, postInitTransformations, ioAdaptor] : preprocessedDeclareOps) {
        size_t mainInputIndex = inputIndexOffset + outputMap.getResultIndex(declareOp);
        mlir::Value argValue = bodyBlock.getArgument(mainInputIndex);

        if (ioAdaptor.has_value()) {
            argValue = ioAdaptor->deadapt(builder, mainFuncOp.getLoc(), argValue);
        }

        auto [remainingTransformations, resultValue] =
                createMatchingPostInitOperations(builder, argValue, mainFuncOp.getLoc(), postInitTransformations);
        VPUX_THROW_UNLESS(remainingTransformations.empty(), "Unexpected unconsumed transformations!");

        declareOp.replaceAllUsesWith(resultValue);
    }
}

void IntroduceInitFunctionPass::eraseOriginalOps(ArrayRef<TransformationsSplit> preprocessedDeclareOps) {
    // erase in a separate loop because it conflicts with building new operations
    for (auto& split : preprocessedDeclareOps) {
        auto declareOp = split.declareOp;
        declareOp.erase();
    }
}

void IntroduceInitFunctionPass::setNetworkEntryPointToMain(IE::CNNNetworkOp mainInfo, mlir::func::FuncOp initFuncOp,
                                                           const DeclareOpOutputMap& outputMap) {
    // update network IO info
    auto& inputsRegion = mainInfo.getInputsInfo();
    mlir::OpBuilder::Listener listener;
    mlir::OpBuilder builder(&getContext(), &listener);
    builder.setInsertionPointToEnd(&inputsRegion.front());

    const auto initFuncType = initFuncOp.getFunctionType();
    for (auto [index, type] : initFuncType.getResults() | indexed) {
        auto name = mlir::StringAttr::get(&getContext(), outputMap.getUniqueResultName(index));
        builder.create<IE::DataInfoOp>(appendLoc(mainInfo.getLoc(), name), name, type,
                                       /*OptionalAttr originalShape*/ nullptr,
                                       /*OptionalAttr friendlyName*/ nullptr,
                                       /*OptionalAttr inputName*/ nullptr,
                                       /*OptionalAttr tensorNames*/ nullptr,
                                       /*profilingSectionsCount=*/0);
    }
}

void IntroduceInitFunctionPass::buildWrapperOpForInitAndMain(IE::CNNNetworkOp mainInfo, mlir::func::FuncOp mainFuncOp,
                                                             mlir::func::FuncOp initFuncOp,
                                                             const DeclareOpInputMap& inputMap,
                                                             ArrayRef<TransformationsSplit> preprocessedDeclareOps) {
    const auto mainFuncType = mainFuncOp.getFunctionType();
    const auto initFuncType = initFuncOp.getFunctionType();
    // Note: expect the below to never fail
    VPUX_THROW_WHEN(mainFuncType.getNumInputs() < initFuncType.getNumResults(),
                    "Main must be already updated to accept all init's outputs as additional inputs");
    // inputs of main are original inputs + init outputs:
    const auto inputs = mainFuncType.getInputs().drop_back(initFuncOp.getFunctionType().getNumResults());
    // results of main are untouched
    const auto results = mainFuncType.getResults();

    OpBuilderLogger builderLog(_log.nest());
    auto wrapperFuncOp = [&]() {
        mlir::OpBuilder moduleBuilder(&getContext(), &builderLog);
        moduleBuilder.setInsertionPointAfter(mainFuncOp);
        auto loc = appendLoc(mainFuncOp.getLoc(), "_wrapper");
        auto wrapperFuncType = mlir::FunctionType::get(&getContext(), inputs, results);
        return moduleBuilder.create<mlir::func::FuncOp>(loc, ("wrapper_" + mainFuncOp.getSymName()).str(),
                                                        wrapperFuncType);
    }();

    const auto locBase = wrapperFuncOp.getLoc();
    auto bodyBlock = wrapperFuncOp.addEntryBlock();
    auto builder = mlir::OpBuilder::atBlockEnd(bodyBlock, &builderLog);

    // create the declare ops without their transformations
    SmallVector<mlir::Value> inputValues(initFuncType.getNumInputs());
    for (auto split : preprocessedDeclareOps) {
        auto declareOp = split.declareOp;
        size_t argIndex = std::get<0>(inputMap.getArgumentInfo(declareOp));

        if (inputValues[argIndex] != nullptr) {
            continue;
        }

        auto baseContent = declareOp.getContentAttr().getBaseContent();
        auto contentAttr = Const::ContentAttr::get(baseContent);
        inputValues[argIndex] = builder.create<Const::DeclareOp>(appendLoc(locBase, "_cst_{0}", argIndex),
                                                                 baseContent.getType(), std::move(contentAttr));
    }

    auto initCallOp = builder.create<mlir::func::CallOp>(appendLoc(locBase, "_call_init"), initFuncOp.getSymNameAttr(),
                                                         initFuncType.getResults(), inputValues);

    auto mainInputValues = [&]() {
        const auto blockArgs = bodyBlock->getArguments();
        const auto initResults = initCallOp.getResults();
        SmallVector<mlir::Value> values;
        values.reserve(bodyBlock->getNumArguments() + initCallOp.getNumResults());
        values.append(blockArgs.begin(), blockArgs.end());
        values.append(initResults.begin(), initResults.end());
        return values;
    }();
    builder.setInsertionPointAfter(initCallOp);
    auto mainCallOp = builder.create<mlir::func::CallOp>(appendLoc(locBase, "_call_main"), mainFuncOp.getSymNameAttr(),
                                                         mainFuncType.getResults(), mainInputValues);

    builder.setInsertionPointToEnd(bodyBlock);
    builder.create<mlir::func::ReturnOp>(appendLoc(locBase, "_return"), mainCallOp.getResults());

    mainInfo.setEntryPoint(wrapperFuncOp.getSymName());
}

mlir::LogicalResult IntroduceInitFunctionPass::initialize(mlir::MLIRContext*) {
    if (extractionMode.hasValue()) {
        auto modeString = extractionMode.getValue();

        if (modeString == "gen-main") {
            _mode = Mode::GenerateMain;
        } else if (modeString == "gen-init") {
            _mode = Mode::GenerateInit;
        } else if (modeString == "gen-all") {
            _mode = Mode::GenerateAll;
        } else {
            return mlir::failure();
        }
    }

    return mlir::success();
}

void IntroduceInitFunctionPass::safeRunOnModule() {
    auto moduleOp = getOperation();

    IE::CNNNetworkOp mainInfo;
    mlir::func::FuncOp mainFuncOp;
    IE::CNNNetworkOp::getFromModule(moduleOp, mainInfo, mainFuncOp);

    auto declareOps = collectMoveWorthyConstantOps(mainFuncOp);

    if (_log.isActive(LogLevel::Debug)) {
        _log.debug("The following constants will be transformed by the 'init()' function:");

        for (auto [index, declareOp] : declareOps | indexed) {
            _log.debug("  {0}: {1}", index, declareOpToString(declareOp));
        }
    }

    if (declareOps.empty()) {
        _log.debug("  No constant candidates found!");
        return;
    }

    // Map DeclareOps to input arguments: Two DeclareOps are mapped to the same argument (and type)
    // when they contain the same underlying mlir::ElementsAttr.
    DeclareOpInputMap inputMap(declareOps);

    auto preprocessedDeclareOps = splitTransformations(declareOps);
    auto [initFuncOp, outputMap] = buildInitFunction(mainFuncOp, preprocessedDeclareOps, inputMap);
    auto initFuncType = initFuncOp.getFunctionType();

    if (_log.isActive(LogLevel::Debug)) {
        for (const auto& split : preprocessedDeclareOps) {
            auto index = std::get<1>(outputMap.getResultInfo(split.declareOp));
            _log.trace("Operation '{0}' is mapped to result value '{1}'", declareOpToString(split.declareOp), index);
        }

        int64_t inByteCount = 0;
        int64_t outByteCount = 0;

        for (auto type : initFuncType.getInputs()) {
            inByteCount += mlir::cast<NDTypeInterface>(type).getTotalAllocSize().count();
        }

        for (auto type : initFuncType.getResults()) {
            outByteCount += mlir::cast<NDTypeInterface>(type).getTotalAllocSize().count();
        }

        auto statsLogger = _log.nest("init() stats", 1);
        statsLogger.debug("Argument count: {0}", initFuncType.getNumInputs());
        statsLogger.debug("Result  count: {0}", initFuncType.getNumResults());
        statsLogger.debug("In-byte count: {0}", inByteCount);
        statsLogger.debug("Out-byte count {0}", outByteCount);
        statsLogger.debug("Signature: {0}", initFuncType);
    }

    // once init is built, there is enough information to update main. similarly
    // to how we build init schedule, make changes to the main schedule for all
    // modes to streamline the logic of the pass.
    updateMainToAccommodateInit(mainFuncOp, initFuncOp, outputMap, preprocessedDeclareOps);

    switch (_mode) {
    case Mode::GenerateInit:
        setNetworkEntryPointToInit(mainInfo, initFuncOp, inputMap, outputMap);
        mainFuncOp.erase();
        break;
    case Mode::GenerateMain:
        setNetworkEntryPointToMain(mainInfo, initFuncOp, outputMap);
        initFuncOp.erase();
        // Note: must happen after new-op-generation (because it depends on
        // original ops) and only for the case when main stays alive.
        eraseOriginalOps(preprocessedDeclareOps);
        break;
    case Mode::GenerateAll:
        initFuncOp.setPrivate();
        mainFuncOp.setPrivate();
        buildWrapperOpForInitAndMain(mainInfo, mainFuncOp, initFuncOp, inputMap, preprocessedDeclareOps);
        // Note: must happen after new-op-generation (because it depends on
        // original ops) and only for the case when main stays alive.
        eraseOriginalOps(preprocessedDeclareOps);
        break;
    default:
        // silence the unhandled case error by the compiler
        moduleOp->emitError("Encountered invalid mode: This should not happen!");
        signalPassFailure();
        break;
    }
}

}  // namespace

//
// createIntroduceInitFunctionPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createIntroduceInitFunctionPass(const Logger& log) {
    return std::make_unique<IntroduceInitFunctionPass>(log);
}
