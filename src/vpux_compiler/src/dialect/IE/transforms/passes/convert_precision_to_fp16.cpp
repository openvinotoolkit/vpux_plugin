//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/convert_op_types.hpp"

using namespace vpux;
using namespace IE;

namespace {

//
// ConvertPrecisionToFP16Pass
//

class ConvertPrecisionToFP16Pass final : public IE::ConvertPrecisionToFP16Base<ConvertPrecisionToFP16Pass> {
public:
    explicit ConvertPrecisionToFP16Pass(Logger log, StringRef computeLayersWithHigherPrecision)
            : _computeLayersWithHigherPrecision(computeLayersWithHigherPrecision.str()) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnModule() final;

    std::string _computeLayersWithHigherPrecision;
};

mlir::LogicalResult ConvertPrecisionToFP16Pass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    if (computeLayersWithHigherPrecision.hasValue()) {
        _computeLayersWithHigherPrecision = computeLayersWithHigherPrecision.getValue();
    }

    return mlir::success();
}

void ConvertPrecisionToFP16Pass::safeRunOnModule() {
    auto& ctx = getContext();

    mlir::TypeConverter typeConverter;
    setupConvertPrecision(typeConverter, [](mlir::Type elemType) -> mlir::Type {
        if (elemType.isF32() || elemType.isSignlessInteger(CHAR_BIT)) {
            return mlir::Float16Type::get(elemType.getContext());
        } else {
            return elemType;
        }
    });

    const auto isLegalOp = [&](mlir::Operation* op) {
        return typeConverter.isLegal(op);
    };

    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<Const::ConstDialect>();
    target.addDynamicallyLegalDialect<IE::IEDialect>(isLegalOp);
    target.addDynamicallyLegalOp<mlir::func::ReturnOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::OneHotOp>(isLegalOp);
    target.addDynamicallyLegalOp<mlir::func::CallOp>(isLegalOp);
    target.addLegalOp<mlir::ModuleOp>();
    target.addLegalOp<IE::DynamicQuantizeOp>();
    target.addLegalOp<IE::IfOp>();
    target.addLegalOp<IE::YieldOp>();
    target.addLegalOp<IE::EqualOp>();
    // AssignOp & ReadValueOp represent inputs/outputs. Cannot convert their type internally.
    target.addLegalOp<IE::AssignOp>();
    target.addLegalOp<IE::ReadValueOp>();
    target.addLegalOp<IE::BitwiseAndOp>();
    target.addLegalOp<IE::BitwiseOrOp>();
    target.addLegalOp<IE::BitwiseXorOp>();
    target.addLegalOp<IE::BitwiseNotOp>();
    target.addDynamicallyLegalOp<mlir::func::FuncOp>([&](mlir::func::FuncOp funcOp) {
        return typeConverter.isSignatureLegal(funcOp.getFunctionType());
    });

    if (!_computeLayersWithHigherPrecision.empty()) {
        std::istringstream optionsStream(_computeLayersWithHigherPrecision);
        std::string dialectNamespace = IE::IEDialect::getDialectNamespace().str() + ".";
        std::string option;
        while (std::getline(optionsStream, option, ',')) {
            std::string fullOption = dialectNamespace + option;
            StringRef opnameRef(fullOption);
            auto opname = mlir::OperationName(opnameRef, &ctx);
            VPUX_THROW_UNLESS(opname.isRegistered(), "Invalid input layer '{0}'", opname);
            // Keep the original precision for all instances of specified layer name(s) during the conversion to FP16
            target.addLegalOp(opname);
        }
    }

    auto module = getOperation();

    // For output element type is inferred based on an attribute
    auto isTargetOp = [](mlir::Operation* op) {
        return mlir::isa<IE::OneHotOp, IE::RandomUniformOp, IE::EyeOp>(op);
    };

    module.walk([&](mlir::Operation* op) {
        if (isTargetOp(op)) {
            const auto outputTypeAttrStr = "outputType";
            const auto outputTypeAttr = op->getAttr(outputTypeAttrStr);
            VPUX_THROW_UNLESS(outputTypeAttr != nullptr, "Failed to get attribute '{0}'", outputTypeAttrStr);

            if (outputTypeAttr.dyn_cast<mlir::TypeAttr>() == mlir::TypeAttr::get(mlir::Float32Type::get(&ctx))) {
                op->setAttr(outputTypeAttrStr, mlir::TypeAttr::get(mlir::Float16Type::get(&ctx)));
            }
        }
    });

    if (mlir::failed(runConvertPrecision(module, typeConverter, target, _log))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertPrecisionToFP16Pass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertPrecisionToFP16Pass(Logger log,
                                                                       StringRef computeLayersWithHigherPrecision) {
    return std::make_unique<ConvertPrecisionToFP16Pass>(log, computeLayersWithHigherPrecision);
}
