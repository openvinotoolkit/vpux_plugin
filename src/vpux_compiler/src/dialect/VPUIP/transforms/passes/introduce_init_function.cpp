//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/module_utils.hpp"
#include "vpux/compiler/core/type_interfaces.hpp"
#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/reshape_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/attr_interfaces.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/IE/locations.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/type/float16.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/IR/Type.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/LogicalResult.h>

using namespace vpux;

namespace {

//
// IntroduceInitFunctionPass
//

class IntroduceInitFunctionPass final : public VPUIP::IntroduceInitFunctionBase<IntroduceInitFunctionPass> {
public:
    explicit IntroduceInitFunctionPass(const Logger& log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    mlir::FailureOr<Const::DataOp> insertInitResSection(mlir::ModuleOp moduleOp);
    mlir::func::FuncOp insertInitFunction(mlir::ModuleOp moduleOp, mlir::func::FuncOp mainFunc);
    mlir::LogicalResult populateInitFunction(mlir::func::FuncOp mainFunc, mlir::func::FuncOp initFunc,
                                             Const::DataOp initRes);
    mlir::Operation* createMatchingOperation(mlir::OpBuilder& builder, mlir::Value input, mlir::Location constLoc,
                                             Const::TransformAttrInterface transformation, size_t transformationIndex);

    void safeRunOnModule() final;
};

mlir::FailureOr<Const::DataOp> IntroduceInitFunctionPass::insertInitResSection(mlir::ModuleOp moduleOp) {
    const auto ovBinSym = mlir::SymbolRefAttr::get(moduleOp.getContext(), vpux::ovBinSection);
    auto ovBinOp = mlir::SymbolTable::lookupSymbolIn(moduleOp, ovBinSym);
    auto dataOp = mlir::dyn_cast_or_null<Const::DataOp>(ovBinOp);
    if (dataOp == nullptr) {
        _log.debug("Could not find @{0} section", vpux::ovBinSection);
        return mlir::failure();
    }

    auto builder = mlir::OpBuilder(moduleOp.getBodyRegion());
    builder.setInsertionPointAfter(dataOp);

    const auto initResLoc = mlir::NameLoc::get(mlir::StringAttr::get(moduleOp->getContext(), vpux::initResSection));
    return builder.create<Const::DataOp>(initResLoc, vpux::initResSection);
}

mlir::func::FuncOp IntroduceInitFunctionPass::insertInitFunction(mlir::ModuleOp moduleOp, mlir::func::FuncOp mainFunc) {
    auto builder = mlir::OpBuilder(moduleOp.getBodyRegion());
    builder.setInsertionPoint(mainFunc);

    const auto initFuncLoc = IE::createLayerLocation(moduleOp.getContext(), "init", "Func");
    const auto initFuncName = printToString("init", mainFunc.getName());
    const auto initFuncType = mlir::FunctionType::get(moduleOp.getContext(), /*inputs=*/{}, /*results=*/{});
    auto initFunc = builder.create<mlir::func::FuncOp>(initFuncLoc, initFuncName, initFuncType);
    return initFunc;
}

mlir::LogicalResult IntroduceInitFunctionPass::populateInitFunction(mlir::func::FuncOp mainFunc,
                                                                    mlir::func::FuncOp initFunc,
                                                                    Const::DataOp initRes) {
    OpBuilderLogger builderLog(_log.nest());
    auto initBuilder = mlir::OpBuilder::atBlockEnd(initFunc.addEntryBlock(), &builderLog);

    OpBuilderLogger initResBuilderLog(_log.nest());
    auto& initResBlock = initRes.getBody().emplaceBlock();
    auto initResBuilder = mlir::OpBuilder::atBlockEnd(&initResBlock, &builderLog);
    size_t constantIdx = 0;

    mlir::OpBuilder mainBuilder(mainFunc);

    auto mainWalk = mainFunc.walk([&](Const::DeclareOp constOp) {
        const auto baseContent = constOp.getContentAttr().getBaseContent();
        const auto symRefAttr = mlir::dyn_cast<Const::SymElementsAttr>(baseContent);
        if (symRefAttr == nullptr) {
            return mlir::WalkResult::skip();
        }

        const auto symRef = symRefAttr.getSymName();
        const auto section = symRef.getRootReference();
        if (section != vpux::ovBinSection) {
            return mlir::WalkResult::skip();
        }

        _log.trace("Got '{0}' at '{1}'", constOp->getName(), constOp->getLoc());

        const auto constLoc = constOp.getLoc();
        const auto loadLoc = appendLoc(constOp.getLoc(), "_load");
        auto loadOp = initBuilder.create<Const::LoadOp>(loadLoc, baseContent.getType(), symRef);

        auto lastOp = loadOp.getOperation();
        const auto transformations = constOp.getContentAttr().getTransformations();
        for (const auto& [trIndex, tr] : transformations | indexed) {
            lastOp = createMatchingOperation(initBuilder, lastOp->getResult(0), constLoc, tr, trIndex);
            if (lastOp == nullptr) {
                _log.debug("Unable to create matching operation for transformation '{0}'", tr);
                return mlir::WalkResult::interrupt();
            }
        }

        const auto foldedLoc = appendLoc(constOp.getLoc(), "_folded");
        const auto foldedSymName = formatv("cst_folded_{0}", constantIdx++).str();
        initResBuilder.create<Const::RefOp>(foldedLoc, foldedSymName, lastOp->getResult(0).getType());

        const auto storeLoc = appendLoc(constOp.getLoc(), "_store");
        const auto foldedSym =
                mlir::SymbolRefAttr::get(constOp.getContext(), initRes.getSymName(),
                                         {mlir::FlatSymbolRefAttr::get(constOp.getContext(), foldedSymName)});
        initBuilder.create<Const::StoreOp>(storeLoc, lastOp->getResult(0), foldedSym);

        mainBuilder.setInsertionPoint(constOp);
        auto mainLoadOp = mainBuilder.create<Const::LoadOp>(constOp.getLoc(), constOp.getType(), foldedSym);
        constOp->replaceAllUsesWith(mainLoadOp->getResults());
        constOp->erase();

        return mlir::WalkResult::advance();
    });

    if (mainWalk.wasInterrupted()) {
        return mlir::failure();
    }

    const auto returnLoc = appendLoc(initFunc.getLoc(), "_return");
    initBuilder.create<mlir::func::ReturnOp>(returnLoc);

    return mlir::success();
}

mlir::Operation* IntroduceInitFunctionPass::createMatchingOperation(mlir::OpBuilder& builder, mlir::Value input,
                                                                    mlir::Location constLoc,
                                                                    Const::TransformAttrInterface transformation,
                                                                    size_t transformationIndex) {
    const auto loc = appendLoc(constLoc, "_tr{0}", transformationIndex);
    return llvm::TypeSwitch<Const::TransformAttrInterface, mlir::Operation*>(transformation)
            .Case<Const::AddAttr>([&](Const::AddAttr add) {
                const auto biasValue = checked_cast<float>(add.getBias().getValueAsDouble());

                const auto biasLoc = appendLoc(constLoc, "_tr{0}_bias", transformationIndex);
                SmallVector<int64_t> shapeRank = {1};
                auto biasType = mlir::RankedTensorType::get(shapeRank, mlir::Float32Type::get(builder.getContext()));
                auto bias = Const::createConst<float>(builder, biasLoc, biasType, {biasValue});

                return builder.create<IE::AddOp>(loc, input, bias, IE::AutoBroadcastType::NUMPY,
                                                 /*postOp=*/nullptr, /*clamp=*/nullptr);
            })
            .Case<Const::BroadcastAttr>([&](Const::BroadcastAttr broadcast) {
                const auto axis = broadcast.getAxis().getInt();
                const auto dimValue = broadcast.getValue().getInt();
                auto shape = SmallVector<int64_t>(input.getType().cast<NDTypeInterface>().getShape().raw());
                shape[axis] = dimValue;

                const auto targetShapeLoc = appendLoc(constLoc, "_tr{0}_shape", transformationIndex);
                SmallVector<int64_t> shapeRank = {static_cast<int64_t>(shape.size())};
                auto targetShapeType = mlir::RankedTensorType::get(shapeRank, getInt64Type(builder.getContext()));
                auto targetShape = Const::createConst<int64_t>(builder, targetShapeLoc, targetShapeType, shape);

                return builder.create<IE::BroadcastOp>(loc, input, targetShape, /*axesMapping=*/nullptr,
                                                       /*mode=*/nullptr);
            })
            .Case<Const::ChangeShapeAndElemTypeAttr>([&](Const::ChangeShapeAndElemTypeAttr changeShapeAndElemType) {
                const auto outputShape = parseIntArrayAttr<int64_t>(changeShapeAndElemType.getShape());
                const auto reassociationMap =
                        IE::getReassociationMap(input.getType().cast<NDTypeInterface>().getShape().raw(), outputShape);
                const auto dimMapping =
                        getIntArrayOfArray(changeShapeAndElemType.getContext(), reassociationMap.value());
                return builder.create<IE::AffineReshapeOp>(loc, input, dimMapping, changeShapeAndElemType.getShape());
            })
            .Case<Const::ConvertElemTypeAttr>([&](Const::ConvertElemTypeAttr convert) {
                return builder.create<IE::ConvertOp>(loc, input, convert.getElemType());
            })
            .Case<Const::DequantizeAttr>([&](Const::DequantizeAttr /*dequantize*/) {
                const auto qElemType =
                        input.getType().cast<NDTypeInterface>().getElementType().cast<mlir::quant::QuantizedType>();
                return builder.create<IE::DequantizeOp>(loc, input, qElemType.getExpressedType());
            })
            .Case<Const::ExpandDilatedAttr>([&](Const::ExpandDilatedAttr expandDilated) {
                return builder.create<IE::ExpandDilatedOp>(loc, input, expandDilated.getDilations());
            })
            .Case<Const::LayoutCastAttr>([&](Const::LayoutCastAttr layoutCast) {
                return builder.create<IE::LayoutCastOp>(loc, input, layoutCast.getDstOrder());
            })
            .Case<Const::MemPermuteAttr>([&](Const::MemPermuteAttr memPermute) {
                return builder.create<IE::MemPermuteOp>(loc, input, memPermute.getDstOrder(), memPermute.getMemPerm());
            })
            .Case<Const::PadWithZeroAttr>([&](Const::PadWithZeroAttr padWithZero) {
                return builder.create<IE::PadOp>(loc, input, /*padsBegin=*/nullptr, /*padsEnd=*/nullptr,
                                                 /*padValue=*/nullptr, padWithZero.getPadBefore(),
                                                 padWithZero.getPadAfter(), getFPAttr(builder.getContext(), 0.0),
                                                 IE::PadMode::CONSTANT);
            })
            .Case<Const::QuantCastAttr>([&](Const::QuantCastAttr quantCast) {
                return builder.create<IE::QuantizeCastOp>(loc, input, quantCast.getElemType());
            })
            .Case<Const::ReorderAttr>([&](Const::ReorderAttr reorder) {
                return builder.create<IE::ReorderOp>(loc, input, reorder.getOrder());
            })
            .Case<Const::RescaleAttr>([&](Const::RescaleAttr rescale) {
                const auto scaleValue = checked_cast<float>(rescale.getScale().getValueAsDouble());

                const auto scaleLoc = appendLoc(constLoc, "_tr{0}_scale", transformationIndex);
                SmallVector<int64_t> shapeRank = {1};
                auto scaleType = mlir::RankedTensorType::get({shapeRank}, mlir::Float32Type::get(builder.getContext()));
                auto scale = Const::createConst<float>(builder, scaleLoc, scaleType, {scaleValue});

                return builder.create<IE::MultiplyOp>(loc, input, scale, IE::AutoBroadcastType::NUMPY,
                                                      /*postOp=*/nullptr, /*clamp=*/nullptr);
            })
            .Case<Const::ReshapeAttr>([&](Const::ReshapeAttr reshape) {
                return builder.create<IE::ReshapeOp>(loc, input, /*shape=*/nullptr, /*specialZero=*/nullptr,
                                                     reshape.getShape());
            })
            .Case<Const::ScalarMultInverseAttr>(
                    [&](Const::ScalarMultInverseAttr /*scalarMultInverse*/) -> mlir::Operation* {
                        const auto inverseLoc = appendLoc(constLoc, "_tr{0}_inverse", transformationIndex);
                        SmallVector<int64_t> shapeRank = {1};
                        const auto inputElemType = input.getType().cast<NDTypeInterface>().getElementType();
                        auto inverseType = mlir::RankedTensorType::get({shapeRank}, inputElemType);

                        const auto data = [&]() -> mlir::DenseElementsAttr {
                            if (mlir::isa<mlir::Float16Type>(inputElemType)) {
                                return mlir::DenseElementsAttr::get(inverseType, ArrayRef({type::float16(1.0)}));
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
                        auto inverse = builder.create<Const::DeclareOp>(inverseLoc, inverseType, contentAttr);

                        return builder.create<IE::DivideOp>(loc, inverse, input, IE::AutoBroadcastType::NUMPY);
                    })
            .Case<Const::SubViewAttr>([&](Const::SubViewAttr subview) {
                return builder.create<IE::SliceOp>(loc, input, subview.getOffset(), subview.getShape());
            })
            .Case<Const::TransposeAttr>([&](Const::TransposeAttr transpose) {
                return builder.create<IE::TransposeOp>(loc, input, /*order=*/nullptr, transpose.getOrder());
            })
            .Default([](Const::TransformAttrInterface) {
                return nullptr;
            });
}

void IntroduceInitFunctionPass::safeRunOnModule() {
    auto moduleOp = getOperation();

    auto initResOp = insertInitResSection(moduleOp);
    // If the @ov_bin section was not found, the @init_res section will not be created
    // The pass will do nothing in this case, as there are no constants to process during the inference
    if (mlir::failed(initResOp)) {
        return;
    }

    IE::CNNNetworkOp mainInfo;
    mlir::func::FuncOp mainFunc;
    IE::CNNNetworkOp::getFromModule(moduleOp, mainInfo, mainFunc);

    auto initFunc = insertInitFunction(moduleOp, mainFunc);
    if (mlir::failed(populateInitFunction(mainFunc, initFunc, initResOp.value()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createIntroduceInitFunctionPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createIntroduceInitFunctionPass(const Logger& log) {
    return std::make_unique<IntroduceInitFunctionPass>(log);
}
