//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/IERT/passes.hpp"

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/compiler/utils/logging.hpp"

#include "vpux/utils/IE/loop.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/numeric.hpp"

#include "vpux/utils/core/range.hpp"

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// ReadValueTransPass
//

class ReadValueTransPass final : public IE::ReadValueTransBase<ReadValueTransPass> {
public:
    explicit ReadValueTransPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void ReadValueTransPass::safeRunOnModule() {
    std::cout << "ReadValueTransPass" << std::endl;

#if 1
    auto module = getOperation();
    auto* ctx = module->getContext();

    IE::CNNNetworkOp netOp;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netOp, netFunc);

    OpBuilderLogger builderLog(_log.nest());
    mlir::OpBuilder builder(&netFunc.getBody().front().front(), &builderLog);

    std::cout << "func.sym_name()" << netFunc.sym_name().str() << std::endl;

#if 0
    // correct Assign
    IE::AssignOp assign_op;
    netFunc.walk([&](IE::AssignOp op) {
        std::cout << "netFunc.walk->AssignOp" << std::endl;
        assign_op = op;
        std::cout << "netFunc.walk->AssignOp END" << std::endl;
    });

    // std::cout << "auto assignOutput = assign_op.output();" << std::endl;
    // auto assignOutput = assign_op.output();
    // std::cout << "auto assignOutput = assign_op.output() 2;" << std::endl;

    // const auto assignOutputShape = assignOutput.getType().cast<mlir::ShapedType>();

    // netFunc.getBody().front().addArgument(assignOutputShape);

#endif


    auto readValueInput = mlir::RankedTensorType::get({1, 143}, mlir::Float16Type::get(ctx));
    // auto assignOutput   = mlir::RankedTensorType::get({1, 143}, mlir::Float16Type::get(ctx));

    // change main function interface
    auto funcType = netFunc.getType();

    auto netFuncResult = netFunc.getBody().front().addArgument(readValueInput);
    // netFunc.getBody().front().addArgument(readValueInput);


#if 1
    // correct ReadValue
    netFunc.walk([&](IE::ReadValueOp op) {
        std::cout << "netFunc.walk->ReadValueOp" << std::endl;
        op.second_inputMutable().assign(netFuncResult);
    });
    // correct Assign
    // netFunc.walk([&](IE::AssignOp op) {
    //     std::cout << "netFunc.walk->ReadValueOp" << std::endl;
    //     op.output().set(assignOutput);
    // });
#endif
    std::cout << "ReadValueTransPass End" << std::endl;

    // auto copyLoc = mlir::NameLoc::get(mlir::Identifier::get("AssignOutputCopy", ctx));

    // /*auto copyOp =*/ builder.create<IE::FloorOp>(assign_op.getLoc(), assign_op.output());
    // auto assignOutputShapeType = copyOp.output().getType().cast<mlir::ShapedType>();

    // auto copyOp = builder.create<IE::FloorOp>(assign_op->getLoc(), assign_op->output());
    // assign_op->output().replaceAllUsesExcept(copyOp.output(), llvm::SmallPtrSet<mlir::Operation*, 1>{copyOp});

    std::cout << "ReadValueTransPass End" << std::endl;

    // auto assignResult = netFunc.getBody().front().addArgument(assignOutputShapeType);


    // change main function interface
    // auto assignOutputShapeType = assign_op.output().getType().cast<mlir::ShapedType>();

    // auto assignOutputRes = netFunc.getBody().front().addArgument(assignOutputShapeType);


    auto newInputsTypes =
            to_small_vector(llvm::concat<const mlir::Type>(funcType.getInputs(), makeArrayRef(readValueInput)));
    // auto newOutputsTypes =
                        //    to_small_vector(llvm::concat<const mlir::Type>(funcType.getResults(), makeArrayRef(assignOutputShapeType)));
    auto newFunctionType = mlir::FunctionType::get(ctx, newInputsTypes,  funcType.getResults());
    netFunc.setType(newFunctionType);

    // User result
    auto inputUserResult =
            getTensorType(readValueInput.getShape(), readValueInput.getElementType(), DimsOrder::fromType(readValueInput));
    auto userInfoBuilder = mlir::OpBuilder::atBlockEnd(&netOp.inputsInfo().front(), &builderLog);
    userInfoBuilder.create<IE::DataInfoOp>(mlir::UnknownLoc::get(ctx), mlir::StringAttr::get(ctx, "readValueInput"),
                                           mlir::TypeAttr::get(inputUserResult));

    // Adding output to the user info
    // auto outputUserResult =
    //                         getTensorType(assignOutputShapeType.getShape(), assignOutputShapeType.getElementType(), DimsOrder::fromType(assignOutputShapeType));

    // auto userInfoBuilderOutput = mlir::OpBuilder::atBlockEnd(&netOp.outputsInfo().front(), &builderLog);
    // userInfoBuilderOutput.create<IE::DataInfoOp>(mlir::UnknownLoc::get(ctx), mlir::StringAttr::get(ctx, "assignOutput"),
    //                                              mlir::TypeAttr::get(outputUserResult));

    std::cout << "ReadValueTransPass End" << std::endl;

#if 0
    // And to the returnOp
    netFunc.walk([&](mlir::ReturnOp op) {
        // op.operandsMutable().append(assignResult);
        std::cout << "!!!!!!!!!ReturnOp op" << std::endl;
        op.operandsMutable().append(/*assign_op.getResult()*/ assignOutputRes);
    });
#endif

#endif

}

}  // namespace

//
// ReadValueTransPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createReadValueTransPass(Logger log) {
    return std::make_unique<ReadValueTransPass>(log);
}
