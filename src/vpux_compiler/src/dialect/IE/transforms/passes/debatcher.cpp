//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <stack>

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/utils/core/format.hpp"

using namespace vpux;

namespace {

namespace detail {
struct DowncastedTypeDescription {
    mlir::Type downcastedType;
    bool isDowncasted;
    Shape originalShape;
};

DowncastedTypeDescription getDowncastedTypeIfApplicable(mlir::Value operand, const Shape& desiredShape) {
    auto type = operand.getType().template cast<vpux::NDTypeInterface>();
    auto originShape = type.getShape();
    bool debatched = false;
    if (desiredShape[Dims4D::Act::N] == 0 || originShape[Dims4D::Act::N] == 1) {
        return DowncastedTypeDescription{type, debatched, originShape.raw()};
    }
    type = type.changeShape(desiredShape);
    debatched = true;

    return DowncastedTypeDescription{type, debatched, originShape.raw()};
}

std::list<mlir::Value> getOperandsToDebatch(mlir::Operation* op,
                                            std::unordered_set<mlir::Operation*>& activationOperationCache) {
    if (op == nullptr || mlir::isa<vpux::Const::DeclareOp>(op)) {
        return {};
    }

    std::list<mlir::Value> ret;
    auto operands = op->getOperands();
    for (auto o : operands) {
        auto parentOp = o.getDefiningOp();
        if (parentOp == nullptr) {
            // doesn't ascend to function arguments or it's a first operation in list
            ret.push_back(o);
            continue;
        }

        if (activationOperationCache.find(parentOp) == activationOperationCache.end() &&
            getOperandsToDebatch(parentOp, activationOperationCache).empty()) {
            // the producer is not an activation kind or its grandparents are not either
            continue;
        }

        ret.push_back(o);
    }

    // put the operation in cache if it belongs to an activation kind
    if (!ret.empty()) {
        activationOperationCache.insert(op);
    }
    return ret;
}

template <class Integer>
SmallVector<Integer> consumeArrayAttrAsIntegerArray(mlir::Operation* op, std::string_view attName) {
    VPUX_THROW_UNLESS(op != nullptr, "consumeArrayAttrAsIntegerArray failed on nullptr");
    auto attr = op->getAttr(attName).dyn_cast_or_null<mlir::ArrayAttr>();
    VPUX_THROW_UNLESS(attr != nullptr, "Unexpected type for \"{0}\", only \"mlir::ArrayAttr\" supported", attName);

    return parseIntArrayAttr<Integer>(attr);
}

struct ConversionDescription {
    Shape from;
    Shape to;
};

struct OpConverter {
    virtual ~OpConverter() = default;
    virtual bool isApplicable(mlir::Operation* op) const = 0;
    virtual void apply(mlir::Operation* op, const std::vector<detail::ConversionDescription>& opArguments) const = 0;
    virtual void refineResults(mlir::Operation* op, const std::vector<ConversionDescription>& inOperands,
                               DenseMap<mlir::OpResult, Shape>& inOutResults) const = 0;
};

struct DefaultOpConverter : public OpConverter {
    DefaultOpConverter(Logger& log): _log(log) {
    }

    bool isApplicable(mlir::Operation*) const override {
        return true;
    }
    void apply(mlir::Operation*, const std::vector<detail::ConversionDescription>&) const override {
    }
    void refineResults(mlir::Operation* op, const std::vector<ConversionDescription>& inOperands,
                       DenseMap<mlir::OpResult, Shape>& inOutResults) const override {
        VPUX_THROW_UNLESS(inOutResults.size() == 0, "DefaultOpConverter expected empty results to override, got: {0}",
                          inOutResults.size());
        auto opResults = op->getResults();
        for (auto result : opResults) {
            int64_t batchDenominator = inOperands[0].from[Dims4D::Act::N] / inOperands[0].to[Dims4D::Act::N];
            auto resultShape = Shape{vpux::getShape(result).raw()};
            resultShape[Dims4D::Act::N] /= batchDenominator;
            inOutResults[result] = std::move(resultShape);
        }
    }

private:
    Logger _log;
};

struct ShapeValueAttrOpConverter : public OpConverter {
    ShapeValueAttrOpConverter(Logger& log): _log(log) {
    }

    static const StringRef getShapeValueAttrName() {
        return "shape_value";
    };

    bool isApplicable(mlir::Operation* op) const override {
        return op->hasAttr(getShapeValueAttrName());
    }

    void apply(mlir::Operation* op, const std::vector<detail::ConversionDescription>& operands) const override {
        _log.trace("Additional processing requires for an operation: {0} as it has a special attribute: \"{1}\"",
                   op->getName().getStringRef(), getShapeValueAttrName());
        auto expectedResultshapeValue = Shape(consumeArrayAttrAsIntegerArray<int64_t>(op, getShapeValueAttrName()));
        _log.trace("Attribute \"{0}\" value: {1}, original operand value: {2}, casted operand value: {3}",
                   getShapeValueAttrName(), expectedResultshapeValue, operands[0].from, operands[0].to);

        // downcast attribute by a batch size in N dimension
        expectedResultshapeValue[Dims4D::Act::N] /= (operands[0].from[Dims4D::Act::N] / operands[0].to[Dims4D::Act::N]);

        auto downcastedAttr = getIntArrayAttr(op->getContext(), expectedResultshapeValue);
        VPUX_THROW_UNLESS(downcastedAttr != nullptr, "Cannot create downcasted attribute \"{0}\"",
                          getShapeValueAttrName());
        op->setAttr(getShapeValueAttrName(), downcastedAttr);
    }

    void refineResults(mlir::Operation*, const std::vector<ConversionDescription>&,
                       DenseMap<mlir::OpResult, Shape>& results) const override {
        VPUX_THROW_UNLESS(results.size() == 1,
                          "Operation attributed by \"{0}\" is supposed to produce only one result, got: {1}",
                          getShapeValueAttrName(), results.size());
    }

private:
    Logger _log;
};

struct SizesAttrOpConverter : public OpConverter {
    SizesAttrOpConverter(Logger& log): _log(log) {
    }

    static const StringRef getShapeValueAttrName() {
        return "sizes_attr";
    };

    bool isApplicable(mlir::Operation* op) const override {
        return op->hasAttr(getShapeValueAttrName());
    }

    void apply(mlir::Operation* op, const std::vector<detail::ConversionDescription>& operands) const override {
        _log.trace("Additional processing requires for an operation: {0} as it has a special attribute: \"{1}\"",
                   op->getName().getStringRef(), getShapeValueAttrName());
        static constexpr std::string_view axesAttrName("axes_attr");
        VPUX_THROW_UNLESS(op->hasAttr(axesAttrName), "SizesAttrOpConverter requires additional \"{0}\" for processing",
                          axesAttrName);

        std::optional<Dim> batchAxisIndex;
        auto axisValues = consumeArrayAttrAsIntegerArray<int64_t>(op, axesAttrName);
        for (const auto& [dimIndex, axisValue] : axisValues | indexed) {
            if (Dim(axisValue) == Dims4D::Act::N) {
                batchAxisIndex = Dim(dimIndex);
            }
        }
        if (!batchAxisIndex.has_value()) {
            _log.trace("Attribute \"{0}\" has no N dimenstion, skip conversion step", axesAttrName);
            return;
        }

        // downcast attribute by a batch size in N dimensition
        auto expectedResultshapeValue = Shape(consumeArrayAttrAsIntegerArray<int64_t>(op, getShapeValueAttrName()));
        _log.trace("Attribute \"{0}\" value: {1}, original operand value: {2}, casted operand value: {3}",
                   getShapeValueAttrName(), expectedResultshapeValue, operands[0].from, operands[0].to);
        expectedResultshapeValue[batchAxisIndex.value()] /=
                (operands[0].from[Dims4D::Act::N] / operands[0].to[Dims4D::Act::N]);

        auto downcastedAttr = getIntArrayAttr(op->getContext(), expectedResultshapeValue);
        VPUX_THROW_UNLESS(downcastedAttr != nullptr, "Cannot create downcasted attribute \"{0}\"",
                          getShapeValueAttrName());
        op->setAttr(getShapeValueAttrName(), downcastedAttr);
    }

    void refineResults(mlir::Operation*, const std::vector<ConversionDescription>&,
                       DenseMap<mlir::OpResult, Shape>& results) const override {
        VPUX_THROW_UNLESS(results.size() == 1,
                          "Operation attributed by \"{0}\" is supposed to produce only one result, got: {1}",
                          getShapeValueAttrName(), results.size());
    }

private:
    Logger _log;
};

struct SDBroadcastConstantForceRewriteDebatchingConverter : public OpConverter {
    SDBroadcastConstantForceRewriteDebatchingConverter(Logger& log): _log(log) {
    }

    bool isApplicable(mlir::Operation* op) const override {
        if (!mlir::isa<IE::BroadcastOp>(op)) {
            return false;
        }
        // StableDiffusion comprises "an adverse `IE::Broadcast`" operations, which
        // artificially promotes N=1 input nonbatched tensors into N!=1.
        // The amount of such enlargement depends on a constant backed in IR.
        // This adverse behavior was discovered during "manual" network inspection.
        // As we didn't expect facing this situation that such pervaded changes in IR will be required,
        // and do not have strong idea at the moment whether we should empower
        // this DebatcherPass to allow making such intrusive changes in a constant definitions
        // on a regular part of this algorithm, we decided to employ "a hotfix" which is supposed
        // to turn this amendmend on once "extraArgs" has been passed through compilation options.
        // So, the further steps here is substitute a constant value from 2 to 1, which automatially
        // revokes `IE::Broadcast` result promotion from N=2 to N=1
        auto bcastOperands = op->getOperands();
        auto bcastResults = op->getResults();
        const auto inOperandShape = vpux::getShape(bcastOperands[0]);
        const auto outResultShape = vpux::getShape(bcastResults[0]);
        if (bcastOperands.size() == 2 && inOperandShape[Dims4D::Act::N] == 1 && outResultShape[Dims4D::Act::N] != 1) {
            auto constDeclareOp = bcastOperands[1].getDefiningOp<Const::DeclareOp>();
            if (constDeclareOp) {
                return true;
            }
        }
        return false;
    }

    void apply(mlir::Operation* op, const std::vector<detail::ConversionDescription>&) const override {
        auto bcastOperands = op->getOperands();
        auto constDeclareOp = bcastOperands[1].getDefiningOp<Const::DeclareOp>();
        mlir::OpBuilder builder(constDeclareOp);
        auto constType = vpux::getSInt32Type(builder.getContext());
        const auto scaleShape = mlir::RankedTensorType::get(getShape(bcastOperands[1]), constType);
        SmallVector<int32_t> scaleValue(1);
        scaleValue[0] = 1;
        auto newScaleConstantOperand =
                Const::createConst(builder, constDeclareOp.getLoc(), scaleShape, ArrayRef(scaleValue));
        bcastOperands[1].replaceAllUsesWith(newScaleConstantOperand);
    }

    void refineResults(mlir::Operation* op, const std::vector<ConversionDescription>&,
                       DenseMap<mlir::OpResult, Shape>& inOutResults) const override {
        auto opResults = op->getResults();
        for (auto result : opResults) {
            auto resultShape = Shape{vpux::getShape(result).raw()};
            resultShape[Dims4D::Act::N] = 1;
            inOutResults[result] = std::move(resultShape);
        }
    }

private:
    Logger _log;
};

struct OpCastVisitor {
    OpCastVisitor(Logger& log): _log(log.nest()) {
        converters.push_back(std::make_unique<DefaultOpConverter>(log));
        converters.push_back(std::make_unique<ShapeValueAttrOpConverter>(log));
        converters.push_back(std::make_unique<SizesAttrOpConverter>(log));
    }

    template <class Converter, class... Args>
    void addConverter(Args&&... args) {
        converters.push_back(std::make_unique<Converter>(_log, std::forward<Args>(args)...));
    }

    DenseMap<mlir::OpResult, Shape> visit(
            mlir::Operation* op, const std::list<mlir::Value>& operands,
            const DenseMap<mlir::Value, detail::ConversionDescription>& operandsConversionDescription) {
        VPUX_THROW_UNLESS(op != nullptr, "Empty operation");
        std::vector<detail::ConversionDescription> operationArguments;
        operationArguments.reserve(operands.size());
        for (mlir::Value operand : operands) {
            VPUX_THROW_UNLESS(
                    operandsConversionDescription.contains(operand),
                    "operandsConversionDescription with size: {0} doesn't contain operand {1} conversion info",
                    operandsConversionDescription.size(), operand);
            operationArguments.push_back(operandsConversionDescription.at(operand));
        }

        // apply conversion for the operation if applicable
        DenseMap<mlir::OpResult, Shape> deductedResultShapes;
        for (const auto& c : converters) {
            if (c->isApplicable(op)) {
                c->apply(op, operationArguments);
                c->refineResults(op, operationArguments, deductedResultShapes);
            }
        }

        // check deducted resultOp against predicted if exist
        auto predictedResultTypes = getPredictedResult(op);
        if (!predictedResultTypes.empty()) {
            for (auto resultPair : zip(deductedResultShapes, predictedResultTypes)) {
                auto [opResult, calculatedShape] = std::get<0>(resultPair);
                auto predictedResultType = std::get<1>(resultPair).dyn_cast<vpux::NDTypeInterface>();
                (void)opResult;
                VPUX_THROW_UNLESS(predictedResultType != nullptr,
                                  "predictedResultType has non vpux::NDTypeInterface type '{0}'",
                                  std::get<1>(resultPair));
                VPUX_THROW_UNLESS(predictedResultType.getShape() == calculatedShape,
                                  "Unexpected opResult shape for op: {0}, calculated: {1}, predicted: {2}",
                                  op->getName(), calculatedShape, predictedResultType.getShape());
            }
        }
        return deductedResultShapes;
    }

private:
    SmallVector<mlir::Type> getPredictedResult(mlir::Operation* op) {
        SmallVector<mlir::Type> predictedResultTypes;
        auto iface = mlir::dyn_cast<mlir::InferTypeOpInterface>(op);
        if (iface) {
            VPUX_THROW_WHEN(
                    iface.inferReturnTypes(op->getContext(), op->getLoc(), op->getOperands(), op->getAttrDictionary(),
                                           op->getPropertiesStorage(), op->getRegions(), predictedResultTypes)
                            .failed(),
                    "Failed to infer return types for operation '{0}'", op->getName());
        }
        return predictedResultTypes;
    }

    std::vector<std::unique_ptr<OpConverter>> converters;
    Logger _log;
};
}  // namespace detail

//
// DebatcherPass
//

class DebatcherPass final : public IE::DebatcherBase<DebatcherPass> {
public:
    explicit DebatcherPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult delegateInitializeOptions(StringRef extraArgs);

private:
    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;
    void safeRunOnFunc() final;
};

mlir::LogicalResult DebatcherPass::initialize(mlir::MLIRContext* ctx) {
    _log.trace("start initializing of {0}", getName());
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    _log.trace("{0}: {1}", extraArgs.getArgStr(), extraArgs.getValue());
    _log.trace("initializing of {0} succeeded", getName());
    return mlir::success();
}

//
// safeRunOnModule
//

void DebatcherPass::safeRunOnFunc() {
    _log.trace("{0}::safeRunOnModule", getName());

    auto main = getOperation();
    mlir::ModuleOp module = main->getParentOfType<mlir::ModuleOp>();

    // Initialize an auxiliary activation operations cache.
    // We will gather such operations gradually during our main-function traversing.
    // This approach gives us an opportunity to make decision whether or not a particular operation
    // should be debatched by the following rule: we must not debatch operands of an operation
    // which producers are constant operations or simple declarations, consequently
    // we must debatch only operands which ascend to argumens of the main-function,
    // hence if an operand of the current operation is produced by
    // a parent operation which is an activation operation then the operand must be debatched.
    // Having this cache determined, we leverage an optimization here: every time once an operation
    // is discerned as an activation-operation we will store it in the cache so that
    // we could consider this stored operation as the activation parent operation for
    // next operation from traversing graph list.
    std::unordered_set<mlir::Operation*> activationOperations;
    activationOperations.insert(main);
    // remember first generation relatives as activation operations
    llvm::for_each(main.getArguments(), [&activationOperations](auto blockArg) {
        llvm::for_each(blockArg.getUsers(), [&activationOperations](auto userOp) {
            activationOperations.insert(userOp);
        });
    });

    DenseMap<mlir::Value, detail::ConversionDescription> debatchedOperandStorage;
    DenseMap<mlir::Value, Shape> opResultsToDebatch;
    for (const auto& arg : main.getArguments()) {
        auto originalShape = Shape{vpux::getShape(arg)};
        auto desiredShape = originalShape;
        desiredShape[Dims4D::Act::N] = 1;
        // Put args of main in debatchedOperandStorage as they have already been debatched.
        // It's not true yet, though they will be "unrealized_cast'ed" later.
        // Given that assumption stated, we will leverage a generic algorithm for traversing through
        // operations and debatching operands from opResultsToDebatch collection only.
        // The reason why we had added args on main debatchedOperandStorage is tricky:
        // we shall not debatch operands of a first operation in the body of `main`
        // which operands ascent to main args. Otherwise, we will change types on main
        // inadvertently which we must not.
        debatchedOperandStorage[arg] = detail::ConversionDescription{originalShape, desiredShape};
        opResultsToDebatch[arg] = desiredShape;
        _log.trace("arg: {0}, desired shape: {1}", arg, opResultsToDebatch[arg]);
    }

    _log.trace("create builder for inserting an `unrealize_converion_cast` at region boundaries");
    mlir::OpBuilder builder(main);
    builder.setInsertionPointAfter(main);
    auto ctx = module.getContext();
    auto mainArgs = main.getArguments();
    _log.trace("Enforce cast for `main` arguments count: {0} ", mainArgs.size());
    llvm::for_each(mainArgs, [&builder, &debatchedOperandStorage, &opResultsToDebatch, this](auto& arg) {
        auto descr = detail::getDowncastedTypeIfApplicable(arg, opResultsToDebatch[arg]);
        if (!descr.isDowncasted) {
            // UnrealizedConversionCastOp doesn't distinguish whether a type is same.
            // skip it explicitly if type hasn't been changed
            return;
        }
        builder.setInsertionPointAfterValue(arg);
        auto unrealized_cast =
                builder.create<mlir::UnrealizedConversionCastOp>(arg.getLoc(), descr.downcastedType, arg);
        _log.trace("apply unrealized_cast");
        arg.replaceUsesWithIf(unrealized_cast.getResult(0), [&](mlir::OpOperand& opOperand) {
            return opOperand.getOwner() != unrealized_cast;
        });

        // Since the new operand as an opResult has been injected,
        // we must remember it in debatchedOperandStorage with the only exception
        debatchedOperandStorage[unrealized_cast.getResult(0)] = {debatchedOperandStorage[arg].from,
                                                                 opResultsToDebatch[arg]};
    });

    _log.trace("Walk through `main` region and debatch all operations");
    detail::OpCastVisitor transformation(_log);
    if (extraArgs.getValue() == "unet_hotfix") {
        transformation.addConverter<detail::SDBroadcastConstantForceRewriteDebatchingConverter>();
    }
    main.walk([this, &activationOperations, &debatchedOperandStorage, &opResultsToDebatch,
               &transformation](mlir::Operation* op) {
        // Do not debatch non-activation operations
        if (mlir::isa<vpux::Const::DeclareOp, mlir::func::ReturnOp, mlir::UnrealizedConversionCastOp>(op)) {
            mlir::OperationName name = op->getName();
            _log.trace("skip op by name: {0}, Identifier: {1} ", name.getStringRef(), name.getIdentifier());
            return;
        }

        // Check if operation is suitable to debatch.
        // The condition is met when at least one operand of the operations
        // requires for debatching
        auto operandsToDebatch = detail::getOperandsToDebatch(op, activationOperations);
        _log.trace("Operation: {0} - gathered operands as debatching candidates: ({1}/{2})",
                   op->getName().getStringRef(), operandsToDebatch.size(), op->getOperands().size());
        if (!operandsToDebatch.empty()) {
            for (mlir::Value operand : operandsToDebatch) {
                _log.nest().trace("Operation: {0}, operand: {1} ", op->getName().getStringRef(), operand);
                if (debatchedOperandStorage.contains(operand)) {
                    _log.nest().trace(
                            "Skip operand conversion from: {0}, to: {1} - which was debatched already as an opResult",
                            debatchedOperandStorage[operand].from, debatchedOperandStorage[operand].to);
                    continue;
                }
                _log.nest().trace("Operand: {0} for debatching found, desired shape: {1}", operand,
                                  opResultsToDebatch[operand]);
                auto descr = detail::getDowncastedTypeIfApplicable(operand, opResultsToDebatch[operand]);
                operand.setType(descr.downcastedType);
                debatchedOperandStorage[operand] = {descr.originalShape, opResultsToDebatch[operand]};
                _log.nest().trace("operand debatched from: {0}, to: {1}", debatchedOperandStorage[operand].from,
                                  debatchedOperandStorage[operand].to);
            }

            // apply args cast on operation and remember deducted result shapes in todo-queue
            DenseMap<mlir::OpResult, Shape> possibleResultShapes =
                    transformation.visit(op, operandsToDebatch, debatchedOperandStorage);
            for (auto val : possibleResultShapes) {
                opResultsToDebatch[val.first] = val.second;
            }

            auto opResults = op->getResults();
            _log.trace("Operation: {0} - has OpResults count: {1}", op->getName().getStringRef(), opResults.size());
            llvm::for_each(opResults, [&debatchedOperandStorage, &opResultsToDebatch, this, op](mlir::OpResult r) {
                auto descr = detail::getDowncastedTypeIfApplicable(r, opResultsToDebatch[r]);
                r.setType(descr.downcastedType);
                if (!debatchedOperandStorage.contains(r)) {
                    debatchedOperandStorage[r] = {descr.originalShape, opResultsToDebatch[r]};
                    _log.nest().trace("remember debatched result from: {1}, to: {2}", op->getName().getStringRef(),
                                      debatchedOperandStorage[r].from, debatchedOperandStorage[r].to);
                }
            });
        }
    });

    _log.trace("restoration of original ReturnOps args of 'main'");
    auto resultOriginalTypes = main.getResultTypes();
    main.walk([&builder, &ctx, &resultOriginalTypes](mlir::func::ReturnOp op) {
        auto operands = op->getOperands();
        builder.setInsertionPoint(op);
        for (auto resultOpDescriptor : zip(operands, resultOriginalTypes)) {
            auto& [operand, originalOpType] = resultOpDescriptor;
            auto casted_type = operand.getType().template cast<vpux::NDTypeInterface>();
            if (casted_type == originalOpType) {
                continue;
            }
            casted_type.changeShape(originalOpType.template cast<vpux::NDTypeInterface>().getShape().toValues());
            auto unrealized_cast = builder.create<mlir::UnrealizedConversionCastOp>(mlir::UnknownLoc::get(ctx),
                                                                                    originalOpType, operand);
            operand.replaceUsesWithIf(unrealized_cast.getResult(0), [&](mlir::OpOperand& opOperand) {
                return opOperand.getOwner() == op;
            });
        }
    });
}

mlir::LogicalResult DebatcherPass::delegateInitializeOptions(StringRef extraArgs) {
    return Base::initializeOptions(printToString("{0}={1}", this->extraArgs.getArgStr(), extraArgs));
}
}  // namespace

//
// createDebatcherPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createDebatcherPass(Logger log) {
    return std::make_unique<DebatcherPass>(log);
}

std::unique_ptr<mlir::Pass> vpux::IE::createAndInitDebatcherPass(StringRef extraArgs, Logger log) {
    auto pass = vpux::IE::createDebatcherPass(log);
    if (mlir::failed(static_cast<DebatcherPass*>(pass.get())->delegateInitializeOptions(extraArgs))) {
        VPUX_THROW("Incorrect option used for \"{0}\" pass initialization: {1}", pass->getName(), extraArgs);
    }

    return pass;
}
