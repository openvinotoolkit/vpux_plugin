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

#include "vpux/compiler/dialect/IERT/passes.hpp"

#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/attributes/enums.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/strings.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/Attributes.h>

using namespace vpux;

namespace {

//
// TimestampProfilingPass
//

class TimestampProfilingPass final : public IERT::TimestampProfilingBase<TimestampProfilingPass> {
public:
    explicit TimestampProfilingPass(IERT::AttrCreateFunc memSpaceCb, Logger log): _memSpaceCb(std::move(memSpaceCb)) {
        VPUX_THROW_UNLESS(_memSpaceCb != nullptr, "Missing memSpaceCb");
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;

private:
    IERT::AttrCreateFunc _memSpaceCb;
    mlir::Attribute _memSpace;
};

//
// safeRunOnModule
//

void TimestampProfilingPass::safeRunOnModule() {
    auto module = getOperation();
    auto* ctx = module->getContext();

    _memSpace = _memSpaceCb(ctx, "");
    if (_memSpace == nullptr) {
        _log.trace("Memory Space is not defined");
        return;
    }

    IE::CNNNetworkOp netOp;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netOp, netFunc);
    OpBuilderLogger builderLog(_log.nest());
    mlir::OpBuilder builder(&netFunc.getBody().front().front(), &builderLog);

    SmallVector<std::pair<IERT::AsyncLayerOpInterface, VPUIP::PhysicalProcessorAttr>> layerTasks;
    auto timestampType = mlir::MemRefType::get({1}, getUInt32Type(ctx), {}, _memSpace);

    netFunc.walk([&](IERT::AsyncLayerOpInterface curTask) {
        uint32_t curNumUnits = 0;
        const auto curExecutor = curTask.getExecutor(curNumUnits);
        auto physType = curExecutor.dyn_cast<VPUIP::PhysicalProcessorAttr>();
        if (physType == nullptr) {
            _log.trace("It is not a PhysicalProcessor Task");
            return;
        }
        layerTasks.push_back({curTask, physType});
    });

    VPUX_THROW_WHEN(layerTasks.empty(), "No TimestampOp was added");

    auto output_size = static_cast<int64_t>(layerTasks.size());
    auto cmxMemType = mlir::MemRefType::get({output_size}, getUInt32Type(ctx), {}, _memSpace);
    auto outputResult = mlir::MemRefType::get({output_size}, getUInt32Type(ctx));

    builder.setInsertionPointAfter(&netFunc.getBody().front().front());
    auto memOp = builder.create<mlir::memref::AllocOp>(mlir::UnknownLoc::get(ctx), cmxMemType);

    SmallVector<mlir::Value> timestamps;
    for (auto layer : layerTasks) {
        auto curTask = layer.first;
        auto physType = layer.second;
        _log.trace("Process Operation '{0}'", curTask->getLoc());

        builder.setInsertionPointAfter(curTask);
        int layerNumber = 0;
        std::string curTaskName = "[";
        curTaskName += curTask->getName().getStringRef().data();
        if ((physType.getValue() == VPUIP::PhysicalProcessor::NCE_Cluster) ||
            (physType.getValue() == VPUIP::PhysicalProcessor::NCE_PerClusterDPU)) {
            curTaskName += "_DPU]";
        } else {
            curTaskName += "_NA]";
        }

        curTaskName += stringifyLocation(curTask->getLoc());

        auto name = mlir::NameLoc::get(mlir::Identifier::get(
                curTaskName +
                        ((timestamps.size() == 0) ? "_PROFBEGIN_0"
                                                  : ("_PROFMIDDLE_" + std::to_string(timestamps.size() - 1))) +
                        "_" + std::to_string(layerNumber),
                ctx));
        auto sub = builder.create<IERT::SubViewOp>(mlir::NameLoc::get(mlir::Identifier::get("subview", ctx)), memOp,
                                                   SmallVector<int64_t>({static_cast<int>(timestamps.size())}),
                                                   timestampType.getShape());

        timestamps.push_back(builder.create<IERT::TimestampOp>(name, timestampType, sub).output());
    }

    auto concatview = builder.create<IERT::ConcatViewOp>(mlir::NameLoc::get(mlir::Identifier::get("concatview", ctx)),
                                                         timestamps, memOp.memref());

    //
    // Declare and create additional output from network
    //
    auto funcType = netFunc.getType();
    auto newResultTypes =
            to_small_vector(llvm::concat<const mlir::Type>(funcType.getResults(), makeArrayRef(outputResult)));
    auto newInputsTypes =
            to_small_vector(llvm::concat<const mlir::Type>(funcType.getInputs(), makeArrayRef(outputResult)));

    auto newFunctionType = mlir::FunctionType::get(ctx, newInputsTypes, newResultTypes);
    netFunc.setType(newFunctionType);
    auto profilngResult = netFunc.getBody().front().addArgument(outputResult);

    auto copyLoc2 = mlir::NameLoc::get(mlir::Identifier::get("profilingCMX2DDR", ctx));
    auto outputOp = builder.create<IERT::CopyOp>(copyLoc2, concatview.output(), profilngResult);

    // Adding output to the user info
    auto outputUserResult = getTensorType(outputResult.getShape(), outputResult.getElementType(),
                                          DimsOrder::fromType(outputResult), nullptr);
    auto userInfoBuilder = mlir::OpBuilder::atBlockEnd(&netOp.outputsInfo().front(), &builderLog);
    userInfoBuilder.create<IE::DataInfoOp>(mlir::UnknownLoc::get(ctx), mlir::StringAttr::get(ctx, "profilingOutput"),
                                           mlir::TypeAttr::get(outputUserResult));

    // And to the returnOp
    netFunc.walk([&](mlir::ReturnOp op) {
        op.operandsMutable().append(outputOp.output());
    });
}

}  // namespace

//
// createTimestampProfilingPass
//

std::unique_ptr<mlir::Pass> vpux::IERT::createTimestampProfilingPass(AttrCreateFunc memSpaceCb, Logger log) {
    return std::make_unique<TimestampProfilingPass>(std::move(memSpaceCb), log);
}
