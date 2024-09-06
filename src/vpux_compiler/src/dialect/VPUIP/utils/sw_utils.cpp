//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include <llvm/ADT/StringRef.h>
#include <optional>

#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/tiling_info.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/utils/core/range.hpp"

namespace vpux {
namespace VPUIP {

constexpr int64_t NPU40XX_SW_KERNEL_ADDRESS_ALIGNMENT = 32;

SmallVector<mlir::Attribute> kernelArgsRange(VPUIP::SwKernelOp swKernelOp) {
    SmallVector<mlir::Attribute> attrStorage;

    for (auto&& kernelRun : swKernelOp.getBody().getOps<VPUIP::SwKernelRun>()) {
        if (kernelRun.getAttrs().has_value()) {
            const mlir::ArrayAttr arrayAttrs = kernelRun.getAttrs().value();
            const auto& attrs = arrayAttrs.getValue();
            for (const auto& attr : attrs) {
                attrStorage.push_back(attr);
            }
        }
    }
    return attrStorage;
}

mlir::ModuleOp getVPUSWModule(mlir::ModuleOp module, const Logger& log) {
    auto* ctx = module.getContext();
    OpBuilderLogger builderLog(log);
    static constexpr StringLiteral vpuSwModuleName{"VPU.SW"};

    auto innerModule = module.lookupSymbol<mlir::ModuleOp>(vpuSwModuleName);
    // creating VPU.SW module if it is not yet created
    if (!innerModule) {
        auto mainModuleBuilder = mlir::OpBuilder::atBlockBegin(module.getBody(), &builderLog);
        innerModule = mainModuleBuilder.create<mlir::ModuleOp>(mlir::UnknownLoc::get(ctx), vpuSwModuleName);
    }
    return innerModule;
}

mlir::SymbolRefAttr createBuiltInFunction(mlir::ModuleOp module, StringRef builtInFunctionName,
                                          const ArrayRef<mlir::Type> inputTypes, StringRef kernelEntryName,
                                          StringRef kernelSourceFileName, const Logger& log) {
    auto* ctx = module.getContext();
    OpBuilderLogger builderLog(log);

    auto vpuswModule = getVPUSWModule(module, log);

    auto builtInFlatFunction = mlir::SymbolRefAttr::get(ctx, builtInFunctionName);
    auto builtInFunction = mlir::SymbolRefAttr::get(ctx, vpuswModule.getName().value(), {builtInFlatFunction});

    // check if this builtInFunction already created - consider names are unique - e.g. no overloads
    if (auto prebuiltFunction = vpuswModule.lookupSymbol<mlir::func::FuncOp>(builtInFunctionName)) {
        log.trace("Found builtin function: {0}", builtInFunctionName);
        return builtInFunction;
    }

    const auto funcType = mlir::FunctionType::get(ctx, inputTypes, {});

    auto innerModuleBuilder = mlir::OpBuilder::atBlockBegin(vpuswModule.getBody(), &builderLog);
    auto builtInOp =
            innerModuleBuilder.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(ctx), builtInFunctionName, funcType);

    // modifying attributes
    builtInOp.setSymVisibilityAttr(mlir::StringAttr::get(ctx, "private"));

    builtInOp->setAttr("VPU.kernel_entry", mlir::StringAttr::get(ctx, kernelEntryName));
    builtInOp->setAttr("VPU.kernel_code", mlir::StringAttr::get(ctx, kernelSourceFileName));
    builtInOp->setAttr("VPU.task_type",
                       mlir::SymbolRefAttr::get(ctx, VPU::stringifyActShaveTaskType(VPU::ActShaveTaskType::COMPUTE)));

    log.trace("Added new builtin function: {0}", builtInFunctionName);
    return builtInFunction;
}

void createRuntimeKernelDefinition(mlir::ModuleOp module, const Logger& log, vpux::VPU::ArchKind arch) {
    auto vpuswModule = getVPUSWModule(module, log);

    static const SmallString runtimeKernelName{"runtime"};
    static const SmallString runtimeKernelEntryName = static_cast<const SmallString>("nnActEntry");

    // check if runtimeKernel already created
    auto runtimeKernelFunction = vpuswModule.lookupSymbol<mlir::func::FuncOp>(runtimeKernelName);
    if (runtimeKernelFunction) {
        log.trace("Found builtin function: {0}", runtimeKernelName);
        return;
    }

    auto* ctx = module.getContext();
    OpBuilderLogger builderLog(log);

    // creating runtime kernel function
    const auto funcType = mlir::FunctionType::get(ctx, {}, {});
    auto innerModuleBuilder = mlir::OpBuilder::atBlockBegin(vpuswModule.getBody(), &builderLog);
    auto runtimeFunctionOp =
            innerModuleBuilder.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(ctx), runtimeKernelName, funcType);

    // modifying attributes
    runtimeFunctionOp.setSymVisibilityAttr(mlir::StringAttr::get(ctx, "private"));

    runtimeFunctionOp->setAttr("VPU.kernel_code", mlir::StringAttr::get(ctx, runtimeKernelEntryName));

    log.trace("Added runtime kernel function: {0}", runtimeKernelEntryName);

    // creating name symbol
    auto runtimeFlatSym = mlir::SymbolRefAttr::get(ctx, runtimeKernelName);
    auto runtimeSym = mlir::SymbolRefAttr::get(ctx, vpuswModule.getName().value(), {runtimeFlatSym});

    static constexpr int64_t defaultStackSize = 4096;

    // TODO: always extract num shaves info from VPURT::SW.Runtime, which can be extracted from module
    auto maxShaves = 4;
    if (arch == vpux::VPU::ArchKind::NPU40XX) {
        maxShaves = 12;
    }
    SmallVector<int64_t> stacksArray(maxShaves, defaultStackSize);

    //  adding runtime kernel configuration - stacks, etc
    auto moduleBuilder = mlir::OpBuilder::atBlockBegin(module.getBody(), &builderLog);
    moduleBuilder.create<VPURT::SWRunTimeOp>(mlir::UnknownLoc::get(ctx), runtimeSym, getIntArrayAttr(ctx, stacksArray));
}

void initSwKernel(VPUIP::SwKernelOp swKernelOp, mlir::ValueRange inputs, mlir::ValueRange outputBuffs,
                  ArrayRef<mlir::Attribute> args, const Logger& log) {
    OpBuilderLogger builderLog(log);
    auto* ctx = swKernelOp.getContext();
    auto& bodyRegion = swKernelOp.getBody();
    auto& swKernelBlock = bodyRegion.emplaceBlock();

    // embedding block args
    auto addBlockArgs = [&swKernelBlock](auto&& cnt) {
        for (auto&& arg : cnt) {
            swKernelBlock.addArgument(arg.getType(), arg.getLoc());
        }
    };

    addBlockArgs(inputs);
    addBlockArgs(outputBuffs);

    auto swKernelBlockBuilder = mlir::OpBuilder::atBlockBegin(&swKernelBlock, &builderLog);

    // pack input/outputs and constants into single call to sw_kernel_run
    SmallVector<mlir::Value> operands;
    auto fetchOperands = [&operands](auto&& cnt) {
        for (auto&& arg : cnt) {
            operands.push_back(arg);
        }
    };

    auto blockArgs = swKernelBlock.getArguments();
    fetchOperands(blockArgs);

    auto argsAttr = args.empty() ? nullptr : mlir::ArrayAttr::get(ctx, args);
    swKernelBlockBuilder.create<VPUIP::SwKernelRun>(mlir::UnknownLoc::get(ctx), mlir::ValueRange(operands), argsAttr);
}

void initSwKernel(VPUIP::SwKernelOp swKernelOp, VPUIP::SwKernelRun swKernelRunOp, const vpux::Logger& log) {
    auto& bodyRegion = swKernelOp.getBody();
    auto& swKernelBlock = bodyRegion.emplaceBlock();

    OpBuilderLogger builderLog(log);
    auto swKernelBlockBuilder = mlir::OpBuilder::atBlockBegin(&swKernelBlock, &builderLog);

    // embedding block args
    auto addBlockArgs = [&swKernelBlock](auto&& cnt) {
        for (auto&& arg : cnt) {
            swKernelBlock.addArgument(arg.getType(), arg.getLoc());
        }
    };

    addBlockArgs(swKernelOp.getInputs());
    addBlockArgs(swKernelOp.getOutputBuffs());

    auto numBlockArgs = swKernelBlock.getNumArguments();
    auto numSwKernelRunArgs = swKernelRunOp->getNumOperands();
    VPUX_THROW_UNLESS(numSwKernelRunArgs != 0, "SW Kernel Run has 0 Operands at '{0}'", swKernelOp->getLoc());
    VPUX_THROW_UNLESS(numBlockArgs % numSwKernelRunArgs == 0, "Invalid block arg num at '{0}'", swKernelOp->getLoc());
    auto tileNum = numBlockArgs / numSwKernelRunArgs;

    VPUX_THROW_UNLESS(swKernelOp.getInputs().size() % tileNum == 0 && swKernelOp.getResults().size() % tileNum == 0,
                      "Invalid block arg num at '{0}'", swKernelOp->getLoc());
    auto numSwKernelRunInputs = swKernelOp.getInputs().size() / tileNum;
    auto numSwKernelRunOutputs = swKernelOp.getResults().size() / tileNum;

    // pack input/outputs and constants into several sw_kernel_run calls
    // For example: For Operation that has 2 inputs, 1 output and tile number is 2. After tile it should be like:
    // inputs: [INPUT0_TILE0] as %arg0: First intput with 1th tile
    //         [INPUT1_TILE0] as %arg1: Second intput with 1th tile
    //         [INPUT0_TILE1] as %arg2: First intput with 2th tile
    //         [INPUT1_TILE1] as %arg3: Second intput with 2th tile
    // outputs:[OUTPUT_TILE0] as %arg4: Output of 1th tile
    //         [OUTPUT_TILE1] as %arg5: Output of 2th tile
    // Tile 0: VPUIP.SW.Kernel.run {attrs} (%arg0, %arg1, %arg4)
    // Tile 1: VPUIP.SW.Kernel.run {attrs} (%arg2, %arg3, %arg5)
    // For example: For Operation that has 1 input, 2 output and tile number is 2. After tile it should be like:
    // inputs: [INPUT0_TILE0] as %arg0: First intput with 1th tile
    //         [INPUT0_TILE1] as %arg1: First intput with 2th tile
    // outputs:[OUTPUT_TILE0] as %arg2: First Output of 1th tile
    //         [OUTPUT_TILE1] as %arg3: Second Output of 1th tile
    //         [OUTPUT_TILE0] as %arg4: First Output of 2th tile
    //         [OUTPUT_TILE1] as %arg5: Second Output of 2th tile
    // Tile 0: VPUIP.SW.Kernel.run {attrs} (%arg0, %arg2, %arg3)
    // Tile 1: VPUIP.SW.Kernel.run {attrs} (%arg1, %arg4, %arg5)
    for (auto tileIdx : irange(tileNum)) {
        auto newRunOp = swKernelBlockBuilder.clone(*swKernelRunOp.getOperation());
        for (auto argInputIdx : irange(numSwKernelRunInputs)) {
            newRunOp->setOperand(checked_cast<unsigned int>(argInputIdx),
                                 swKernelBlock.getArgument(
                                         checked_cast<unsigned int>(tileIdx * numSwKernelRunInputs + argInputIdx)));
        }

        for (auto argOutputIdx : irange(numSwKernelRunOutputs)) {
            newRunOp->setOperand(
                    checked_cast<unsigned int>(numSwKernelRunInputs + argOutputIdx),
                    swKernelBlock.getArgument(checked_cast<unsigned int>(
                            tileNum * numSwKernelRunInputs + tileIdx * numSwKernelRunOutputs + argOutputIdx)));
        }

        log.trace("create {0}th tile of SwKernelRun {1}", tileIdx, swKernelRunOp);
    }
}

SmallString getSwKernelEntryName(VPUIP::SwKernelOp swKernelOp) {
    auto module = swKernelOp->getParentOfType<mlir::ModuleOp>();
    auto kernelFunc = module.lookupSymbol<mlir::func::FuncOp>(swKernelOp.getKernelFunctionAttr());
    VPUX_THROW_WHEN(kernelFunc == nullptr, "Cannot find kernel function symbol at '{0}'", swKernelOp->getLoc());
    const auto kernelEntryPoint = kernelFunc->getAttrOfType<mlir::StringAttr>("VPU.kernel_entry");
    VPUX_THROW_WHEN(kernelEntryPoint == nullptr, "Cannot find kernel entry point at '{0}'", swKernelOp->getLoc());
    return kernelEntryPoint.getValue();
}

// Check whether SwKernelOp is activation.
bool isActivationSwKernelOp(VPUIP::SwKernelOp swKernelOp) {
    auto kernelEntryName = getSwKernelEntryName(swKernelOp);
    if (llvm::find(SW_ACTIVATION_KERNELS, kernelEntryName) != SW_ACTIVATION_KERNELS.end()) {
        return true;
    }
    return false;
}

// Check whether SwKernelOp supports tiling.
bool isSwKernelTilingSupported(VPUIP::SwKernelOp swKernelOp) {
    auto kernelEntryName = getSwKernelEntryName(swKernelOp);
    if (llvm::find(SW_KERNELS_SUPPORTING_TILING, kernelEntryName) != SW_KERNELS_SUPPORTING_TILING.end()) {
        return true;
    }
    return false;
}

// Check whether SwKernelOp support discontinuous input/output.
bool isStridedDataAccessSupported(VPUIP::SwKernelOp swKernelOp) {
    auto kernelEntryName = getSwKernelEntryName(swKernelOp);
    // SubView can be used for Softmax because it is always tilied on the highest dimension.
    if (kernelEntryName == "softmax" ||
        llvm::find(SW_KERNELS_SUPPORTING_STRIDE, kernelEntryName) != SW_KERNELS_SUPPORTING_STRIDE.end()) {
        return true;
    }
    return false;
}

namespace {
// reverse int attribute from the physical order
int64_t reverseMemDim(DimsOrder inOrder, int64_t dimIdx) {
    const auto origPerm = inOrder.toPermutation();
    return origPerm[origPerm.size() - 1 - dimIdx].ind();
}

// reverse int array attribute from the physical order
SmallVector<int64_t> reverseIntArrayAttr(DimsOrder inOrder, mlir::ArrayAttr arrayAttr) {
    const auto origPerm = inOrder.toPermutation();
    const auto origArray = parseIntArrayAttr<int64_t>(arrayAttr);
    SmallVector<int64_t> permArray(arrayAttr.size());
    for (const auto srcInd : irange(origPerm.size())) {
        const auto dstInd = origPerm[srcInd].ind();
        const auto revSrcInd = origPerm.size() - 1 - srcInd;
        const auto revDstInd = dstInd;
        permArray[revDstInd] = origArray[revSrcInd];
    }
    return permArray;
}

// permute int array attribute in the physical order
SmallVector<int64_t> permuteIntArrayAttr(DimsOrder inOrder, ArrayRef<int64_t> origArray) {
    const auto origPerm = inOrder.toPermutation();
    SmallVector<int64_t> permArray(origArray.size());
    for (const auto srcInd : irange(origPerm.size())) {
        const auto dstInd = origPerm[srcInd].ind();
        const auto revSrcInd = origPerm.size() - 1 - srcInd;
        const auto revDstInd = dstInd;
        permArray[revSrcInd] = origArray[revDstInd];
    }
    return permArray;
}

InputTiling backInferInterpolateSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile,
                                                  Logger log) {
    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);

    auto swKernelRun = *swKernelRuns.begin();
    VPUX_THROW_UNLESS(swKernelRun.getAttrs().has_value(), "SwKernelOp has no attr '{0}'", swKernelOp);
    const auto attrs = swKernelRun.getAttrs().value();
    const auto inputs = swKernelOp.getInputs();
    auto inOrder = inputs[0].getType().dyn_cast<vpux::NDTypeInterface>().getDimsOrder();

    std::optional<SmallVector<int64_t>> coordinatesShape = std::nullopt;
    std::optional<SmallVector<int64_t>> lambdasShape = std::nullopt;
    if (inputs.size() >= 2) {
        const auto coordinates = inputs[1];
        coordinatesShape = to_small_vector(coordinates.getType().dyn_cast<vpux::NDTypeInterface>().getShape());
    }
    if (inputs.size() >= 3) {
        const auto lambdas = inputs[2];
        lambdasShape = to_small_vector(lambdas.getType().dyn_cast<vpux::NDTypeInterface>().getShape());
    }

    const auto interpolateMode = static_cast<IE::InterpolateMode>(attrs[1].dyn_cast<mlir::IntegerAttr>().getInt());
    const auto coordMode = static_cast<IE::InterpolateCoordMode>(attrs[2].dyn_cast<mlir::IntegerAttr>().getInt());
    const auto nearestMode = static_cast<IE::InterpolateNearestMode>(attrs[3].dyn_cast<mlir::IntegerAttr>().getInt());
    const auto initialInputDims = reverseIntArrayAttr(inOrder, attrs[6].dyn_cast<mlir::ArrayAttr>());
    const auto initialOutputDims = reverseIntArrayAttr(inOrder, attrs[7].dyn_cast<mlir::ArrayAttr>());
    const auto initialInputOffset = reverseIntArrayAttr(inOrder, attrs[10].dyn_cast<mlir::ArrayAttr>());
    const auto initialOutputOffset = reverseIntArrayAttr(inOrder, attrs[11].dyn_cast<mlir::ArrayAttr>());

    const auto currentInputDims = to_small_vector(inputs[0].getType().dyn_cast<vpux::NDTypeInterface>().getShape());

    return vpux::backInferInterpolateTile(outputTile, initialInputDims, initialOutputDims, initialInputOffset,
                                          initialOutputOffset, currentInputDims, coordinatesShape, lambdasShape,
                                          interpolateMode, coordMode, nearestMode, log);
}

int64_t convertKernelAxisToOrigAxis(mlir::Value tensorArg, int64_t kernelAxis) {
    const auto shape = getShape(tensorArg);
    // Dims/Order sequence is not same on kernel-FW & compiler side. Convert the axis from kernel to compiler
    // representation.
    auto nDims = checked_cast<uint32_t>(shape.size());

    return nDims - 1 - kernelAxis;
}

InputTiling backInferGatherSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile,
                                             Logger log) {
    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);
    auto swKernelRun = *swKernelRuns.begin();
    VPUX_THROW_UNLESS(swKernelRun.getAttrs().has_value(), "SwKernelOp has no attr '{0}'", swKernelOp);
    const auto attrs = swKernelRun.getAttrs().value();
    const auto inputs = swKernelOp.getInputs();

    const auto kernelAxis = attrs[0].dyn_cast<mlir::IntegerAttr>().getValue().getSExtValue();
    const auto axisValue = convertKernelAxisToOrigAxis(inputs[0], kernelAxis);
    const auto batchDims = attrs[1].dyn_cast<mlir::IntegerAttr>().getValue().getSExtValue();

    const auto origInputShape = inputs[0].getType().dyn_cast<vpux::NDTypeInterface>().getShape();
    const auto origIndicesShape = inputs[1].getType().dyn_cast<vpux::NDTypeInterface>().getShape();

    return vpux::backInferGatherTile(outputTile, origInputShape, origIndicesShape, axisValue, batchDims, false, log);
}

InputTiling backInferDepthToSpaceSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile,
                                                   Logger log) {
    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);

    auto swKernelRun = *swKernelRuns.begin();
    VPUX_THROW_UNLESS(swKernelRun.getAttrs().has_value(), "SwKernelOp has no attr '{0}'", swKernelOp);
    const auto attrs = swKernelRun.getAttrs().value();

    auto inShape = swKernelOp.getInputs()[0].getType().cast<vpux::NDTypeInterface>().getShape();
    const auto blockSize = attrs[0].cast<mlir::IntegerAttr>().getInt();

    return vpux::backInferDepthToSpaceTile(outputTile, inShape, blockSize, log);
}

InputTiling backInferPadSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile, Logger log) {
    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);

    auto swKernelRun = *swKernelRuns.begin();
    VPUX_THROW_UNLESS(swKernelRun.getAttrs().has_value(), "SwKernelOp has no attr '{0}'", swKernelOp);
    const auto attrs = swKernelRun.getAttrs().value();

    const auto origInputType = swKernelOp.getInputs()[0].getType().cast<vpux::NDTypeInterface>();
    const auto origInputShape = origInputType.getShape();
    const auto origOutputShape = swKernelOp.getResults()[0].getType().cast<vpux::NDTypeInterface>().getShape();
    const auto order = origInputType.getDimsOrder();

    // Padding attr at VPUIP dialect are stored in memory order so convert to default order
    // to be aligned with shape representation
    const auto origPadsBegin = reverseIntArrayAttr(order, attrs[0].dyn_cast<mlir::ArrayAttr>());
    const auto origPadsEnd = reverseIntArrayAttr(order, attrs[1].dyn_cast<mlir::ArrayAttr>());

    return backInferPadTile(outputTile, origInputShape, origOutputShape, Shape(origPadsBegin), Shape(origPadsEnd), log);
}

InputTiling backInferReduceSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile,
                                             StringRef kernelEntryName, Logger log) {
    log.trace("Try to back infer input tiling for {0}, output tile: {1}", kernelEntryName, outputTile);

    const auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);

    auto swKernelRun = *swKernelRuns.begin();
    const auto numInputs = swKernelOp.getInputs().size();
    VPUX_THROW_UNLESS(swKernelRun.getAttrs().has_value(), "SwKernelOp has no attr '{0}'", swKernelOp);
    VPUX_THROW_UNLESS(numInputs, "SwKernelOp {0} should have 1 input, got '{1}'", swKernelOp, numInputs);

    const auto input = swKernelOp.getOperand(0);
    const auto inputOrder = input.getType().cast<vpux::NDTypeInterface>().getDimsOrder();
    const auto inputShape = getShape(input);
    const auto attrs = swKernelRun.getAttrs().value();

    VPUX_THROW_UNLESS(attrs.size() == 3, "SwKernelOp {0} should have 3 attributes, got '{1}'", swKernelOp,
                      attrs.size());
    VPUX_THROW_UNLESS(inputShape.size() == outputTile.shape.size(),
                      "Can't tile SwKernel operation '{0}' at '{1}', which has operands with different rank",
                      swKernelOp->getName(), swKernelOp->getLoc());

    auto inputTile = outputTile;
    const auto reversedAxes = parseIntArrayAttr<int64_t>(attrs[2].cast<mlir::ArrayAttr>());
    for (const auto reversedAxis : reversedAxes) {
        const auto axis = reverseMemDim(inputOrder, reversedAxis);
        const auto d = Dim(axis);
        inputTile.shape[d] = inputShape[d];
    }

    return TilingInfo{std::move(inputTile)};
}

InputTiling backInferMatMulSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile,
                                             Logger log) {
    log.trace("Try to back infer input tiling for matmul, output tile: {0}", outputTile);

    const auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);

    auto swKernelRun = *swKernelRuns.begin();
    const auto numInputs = swKernelOp.getInputs().size();
    VPUX_THROW_UNLESS(swKernelRun.getAttrs().has_value(), "SwKernelOp has no attr '{0}'", swKernelOp);
    VPUX_THROW_UNLESS(numInputs == 2, "SwKernelOp {0} should have 2 inputs, got '{1}'", swKernelOp, numInputs);

    const auto input1 = swKernelOp.getOperand(0);
    const auto input2 = swKernelOp.getOperand(1);
    const auto input1Shape = getShape(input1);
    const auto input2Shape = getShape(input2);
    const auto attrs = swKernelRun.getAttrs().value();

    VPUX_THROW_UNLESS(attrs.size() == 2, "SwKernelOp {0} should have 2 attributes, got '{1}'", swKernelOp,
                      attrs.size());
    VPUX_THROW_UNLESS(input1Shape.size() == outputTile.shape.size(),
                      "Can't tile SwKernel operation '{0}' at '{1}', which has operands with different rank",
                      swKernelOp->getName(), swKernelOp->getLoc());
    VPUX_THROW_UNLESS(input2Shape.size() == outputTile.shape.size(),
                      "Can't tile SwKernel operation '{0}' at '{1}', which has operands with different rank",
                      swKernelOp->getName(), swKernelOp->getLoc());

    auto input1Tile = outputTile;
    input1Tile.shape[Dim(input1Tile.shape.size() - 2)] = input1Shape[Dim(input1Shape.size() - 2)];
    input1Tile.shape[Dim(input1Tile.shape.size() - 1)] = input1Shape[Dim(input1Shape.size() - 1)];

    auto input2Tile = outputTile;
    input2Tile.shape[Dim(input2Tile.shape.size() - 2)] = input2Shape[Dim(input2Shape.size() - 2)];
    input2Tile.shape[Dim(input2Tile.shape.size() - 1)] = input2Shape[Dim(input2Shape.size() - 1)];

    return InputTiling{{std::move(input1Tile), std::move(input2Tile)}};
}

SmallVector<mlir::Attribute> getInterpolateSwkernelNewAttrsAfterTiling(VPUIP::SwKernelOp swKernelOp,
                                                                       ArrayRef<mlir::Attribute> origAttr,
                                                                       const TilingInfo& inputTiling,
                                                                       const TileInfo& outTile, Logger log) {
    log.trace("update attrs for SwKernel Op at '{0}' for out tile {1}", swKernelOp, outTile);
    // Get output tile against the original output
    auto kernelRun = *swKernelOp.getBody().getOps<VPUIP::SwKernelRun>().begin();
    auto attrs = kernelRun.getAttrs().value();
    VPUX_THROW_UNLESS(origAttr.size() == attrs.size(), "Unmatched attr size found at '{0}'", swKernelOp);

    SmallVector<mlir::Attribute> newAttrs(attrs.begin(), attrs.end());
    auto dim = swKernelOp.getInputs()[0].getType().dyn_cast<vpux::NDTypeInterface>().getDimsOrder();
    TileInfo inputTile = inputTiling.tiles[0];
    const auto initialInputDims = reverseIntArrayAttr(dim, attrs[6].dyn_cast<mlir::ArrayAttr>());
    const auto initialOutputDims = reverseIntArrayAttr(dim, attrs[7].dyn_cast<mlir::ArrayAttr>());
    const auto initialInputOffset = reverseIntArrayAttr(dim, attrs[10].dyn_cast<mlir::ArrayAttr>());
    const auto initialOutputOffset = reverseIntArrayAttr(dim, attrs[11].dyn_cast<mlir::ArrayAttr>());
    const auto localInputOffset = to_small_vector(inputTile.offsets);
    const auto localOutputOffset = to_small_vector(outTile.offsets);
    SmallVector<int64_t> inputTileOffset;
    SmallVector<int64_t> outputTileOffset;
    std::transform(localInputOffset.begin(), localInputOffset.end(), initialInputOffset.begin(),
                   std::back_inserter(inputTileOffset), std::plus<int64_t>());
    std::transform(localOutputOffset.begin(), localOutputOffset.end(), initialOutputOffset.begin(),
                   std::back_inserter(outputTileOffset), std::plus<int64_t>());
    auto newInputTiling = inputTiling;
    newInputTiling.tiles[0].offsets = Shape(inputTileOffset);
    auto newOutputTile = outTile;
    newOutputTile.offsets = Shape(outputTileOffset);
    newAttrs[10] = getIntArrayAttr(swKernelOp->getContext(), permuteIntArrayAttr(dim, inputTileOffset));
    newAttrs[11] = getIntArrayAttr(swKernelOp->getContext(), permuteIntArrayAttr(dim, outputTileOffset));
    return newAttrs;
}

SmallVector<mlir::Attribute> getPadSwkernelNewAttrsAfterTiling(VPUIP::SwKernelOp swKernelOp,
                                                               ArrayRef<mlir::Attribute> origAttr,
                                                               const TileInfo& outTile, Logger log) {
    log.trace("update attrs for Pad SwKernel Op at '{0}' for out tile {1}", swKernelOp, outTile);
    auto kernelRun = *swKernelOp.getBody().getOps<VPUIP::SwKernelRun>().begin();
    auto attrs = kernelRun.getAttrs().value();
    VPUX_THROW_UNLESS(origAttr.size() == attrs.size(), "Unmatched attr size found at '{0}'", swKernelOp);

    SmallVector<mlir::Attribute> newAttrs(attrs.begin(), attrs.end());
    const auto outType = swKernelOp.getResults()[0].getType().cast<vpux::NDTypeInterface>();
    const auto outShape = outType.getShape();
    auto order = outType.getDimsOrder();

    // Padding attrs at VPUIP dialect are stored in memory-order so convert to default-order
    // to be aligned with shape representation
    auto padsBegin = reverseIntArrayAttr(order, attrs[0].dyn_cast<mlir::ArrayAttr>());
    auto padsEnd = reverseIntArrayAttr(order, attrs[1].dyn_cast<mlir::ArrayAttr>());

    vpux::updatePadOpAttrsAfterTiling(outShape, outTile, padsBegin, padsEnd);

    // Convert new pads back to memory-order
    newAttrs[0] = getIntArrayAttr(swKernelOp->getContext(), permuteIntArrayAttr(order, padsBegin));
    newAttrs[1] = getIntArrayAttr(swKernelOp->getContext(), permuteIntArrayAttr(order, padsEnd));
    return newAttrs;
}

InputTiling backInferTopKSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile, Logger) {
    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);

    auto swKernelRun = *swKernelRuns.begin();
    VPUX_THROW_UNLESS(swKernelRun.getAttrs().has_value(), "SwKernelOp has no attr '{0}'", swKernelOp);
    const auto inOrder = swKernelOp.getInputs()[0].getType().cast<vpux::NDTypeInterface>().getDimsOrder();
    const auto attrs = swKernelRun.getAttrs().value();
    const auto axis = reverseMemDim(inOrder, attrs[0].cast<mlir::IntegerAttr>().getInt());

    const auto inShape = getShape(swKernelOp.getInputs()[0]);
    SmallVector<TileInfo> inputTiles;
    for (auto origInput : swKernelOp.getInputs()) {
        const auto curShape = getShape(origInput);
        VPUX_THROW_UNLESS(curShape.size() == outputTile.shape.size(),
                          "Can't tile SwKernel operation '{0}' at '{1}', which has operands with different rank",
                          swKernelOp->getName(), swKernelOp->getLoc());

        auto curTile = outputTile;
        for (auto ind : irange(curShape.size())) {
            const auto d = Dim(ind);
            if (axis == d.ind()) {
                curTile.shape[d] = inShape[d];
            }
        }

        inputTiles.push_back(curTile);
    }
    return TilingInfo{inputTiles};
}

bool isReduceKernelEntry(StringRef kernelEntryName) {
    static const std::unordered_set<std::string> reduceEntryNames = {
            "reduce_l1",   "reduce_l2",  "reduce_logical_and", "reduce_logical_or", "reduce_max",
            "reduce_mean", "reduce_min", "reduce_prod",        "reduce_sum"};

    return reduceEntryNames.find(kernelEntryName.str()) != reduceEntryNames.end();
}

InputTiling backInferGRUSequenceSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTileY,
                                                  Logger) {
    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);
    const auto inputs = swKernelOp.getInputs();

    const auto origInputShape = inputs[0].getType().dyn_cast<vpux::NDTypeInterface>().getShape();
    const auto origInitialHiddenStateShape = inputs[1].getType().dyn_cast<vpux::NDTypeInterface>().getShape();
    const auto origWShape = inputs[2].getType().dyn_cast<vpux::NDTypeInterface>().getShape();
    const auto origRShape = inputs[3].getType().dyn_cast<vpux::NDTypeInterface>().getShape();
    const auto origBShape = inputs[4].getType().dyn_cast<vpux::NDTypeInterface>().getShape();

    TileInfo inputTile(origInputShape);
    TileInfo initialHiddenStateTile(origInitialHiddenStateShape);
    TileInfo wTile(origWShape);
    TileInfo rTile(origRShape);
    TileInfo bTile(origBShape);

    inputTile.shape[Dim(0)] = outputTileY.shape[Dim(0)];
    inputTile.offsets[Dim(0)] = outputTileY.offsets[Dim(0)];

    initialHiddenStateTile.shape[Dim(0)] = outputTileY.shape[Dim(0)];
    initialHiddenStateTile.offsets[Dim(0)] = outputTileY.offsets[Dim(0)];

    return InputTiling{{std::move(inputTile), std::move(initialHiddenStateTile), std::move(wTile), std::move(rTile),
                        std::move(bTile)}};
}

InputTiling backInferGRUSequenceLastPartSwKernelInputTile(VPUIP::SwKernelOp swKernelOp,
                                                          const vpux::TileInfo& outputTileY, Logger) {
    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);
    const auto inputs = swKernelOp.getInputs();

    const auto origInputShape = inputs[0].getType().dyn_cast<vpux::NDTypeInterface>().getShape();
    const auto origInitialHiddenStateShape = inputs[1].getType().dyn_cast<vpux::NDTypeInterface>().getShape();
    const auto origRShape = inputs[2].getType().dyn_cast<vpux::NDTypeInterface>().getShape();
    const auto origBShape = inputs[3].getType().dyn_cast<vpux::NDTypeInterface>().getShape();

    TileInfo inputTile(origInputShape);
    TileInfo initialHiddenStateTile(origInitialHiddenStateShape);
    TileInfo rTile(origRShape);
    TileInfo bTile(origBShape);

    inputTile.shape[Dim(0)] = outputTileY.shape[Dim(0)];
    inputTile.offsets[Dim(0)] = outputTileY.offsets[Dim(0)];

    initialHiddenStateTile.shape[Dim(0)] = outputTileY.shape[Dim(0)];
    initialHiddenStateTile.offsets[Dim(0)] = outputTileY.offsets[Dim(0)];

    return InputTiling{{std::move(inputTile), std::move(initialHiddenStateTile), std::move(rTile), std::move(bTile)}};
}

InputTiling backInferLSTMGatesSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile,
                                                Logger) {
    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);

    SmallVector<TileInfo> inputTiles;
    for (const auto& origInput : swKernelOp.getInputs()) {
        const auto curShape = getShape(origInput);
        VPUX_THROW_UNLESS(curShape.size() == outputTile.shape.size(),
                          "Can't tile SwKernel operation '{0}' at '{1}', which has operands with different rank",
                          swKernelOp->getName(), swKernelOp->getLoc());

        auto curTile = outputTile;
        curTile.shape[Dim(curShape.size() - 1)] = curShape[Dim(curShape.size() - 1)];

        inputTiles.push_back(curTile);
    }

    return TilingInfo{inputTiles};
}

InputTiling backInferLSTMCellSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile,
                                               Logger log) {
    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);

    SmallVector<TileInfo> inputTiles;

    // inputs
    const auto inputData = swKernelOp.getInputs()[0];
    const auto initialHiddenState = swKernelOp.getInputs()[1];
    // const auto initialCellState = swKernelOp.getInputs()[2];
    //  weight
    const auto weights = swKernelOp.getInputs()[3];
    const auto weightsHidden = swKernelOp.getInputs()[4];
    const auto biases = swKernelOp.getInputs()[5];

    const auto inputDataShape = getShape(inputData);
    const auto initialHiddenStateShape = getShape(initialHiddenState);
    // const auto initialCellStateShape = getShape(initialCellState);

    const auto weightsShape = getShape(weights);
    const auto weightsHiddenShape = getShape(weightsHidden);
    const auto biasesShape = getShape(biases);

    TileInfo inputDataTile(inputDataShape);
    TileInfo initialHiddenStateTile(initialHiddenStateShape);
    TileInfo initialCellStateTile = outputTile;

    TileInfo weightsTile(weightsShape);
    weightsTile.shape[Dim(weightsShape.size() - 2)] = outputTile.shape.back();
    weightsTile.offsets[Dim(weightsShape.size() - 2)] = outputTile.offsets.back();
    weightsTile.axis[Dim(weightsShape.size() - 2)] = outputTile.axis.back();

    TileInfo weightsHiddenTile(weightsHiddenShape);
    weightsHiddenTile.shape[Dim(weightsHiddenShape.size() - 2)] = outputTile.shape.back();
    weightsHiddenTile.offsets[Dim(weightsHiddenShape.size() - 2)] = outputTile.offsets.back();
    weightsHiddenTile.axis[Dim(weightsHiddenShape.size() - 2)] = outputTile.axis.back();

    TileInfo biasesTile(biasesShape);
    // biasesTile.shape[Dim(biasesShape.size() - 1)] = outputTile.shape[Dim(outputTile.shape.size() - 1)];
    biasesTile.shape[Dim(biasesShape.size() - 1)] = outputTile.shape.back();
    biasesTile.offsets[Dim(biasesShape.size() - 1)] = outputTile.offsets.back();
    biasesTile.axis[Dim(biasesShape.size() - 1)] = outputTile.axis.back();

    inputTiles.push_back(inputDataTile);
    inputTiles.push_back(initialHiddenStateTile);
    inputTiles.push_back(initialCellStateTile);
    inputTiles.push_back(weightsTile);
    inputTiles.push_back(weightsHiddenTile);
    inputTiles.push_back(biasesTile);

    log.trace("backInferLSTMCellSwKernelInputTile  outputTile '{0}'", outputTile);
    log.trace("backInferLSTMCellSwKernelInputTile  inputTiles '{0}'", inputTiles);

    return TilingInfo{inputTiles};
}

InputTiling backInferMvn1SumSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile, Logger) {
    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);
    auto swKernelRun = *swKernelRuns.begin();
    VPUX_THROW_UNLESS(swKernelRun.getAttrs().has_value(), "SwKernelOp has no attr '{0}'", swKernelOp);
    const auto attrs = swKernelRun.getAttrs().value();

    const auto inputType = swKernelOp.getInputs()[0].getType().cast<vpux::NDTypeInterface>();
    const auto inputShape = inputType.getShape();

    TileInfo inDataTile(inputShape);
    const auto acrossChannels = attrs[0].cast<mlir::BoolAttr>().getValue();
    // When acrossChannels==true, keep initial C value
    if (!acrossChannels) {
        // Per-channel sums, thus input-tile must match output-tile on Dim[C]
        inDataTile.shape[Dims4D::Act::C] = outputTile.shape[Dims4D::Act::C];
        inDataTile.offsets[Dims4D::Act::C] = outputTile.offsets[Dims4D::Act::C];
    }
    inDataTile.shape[Dims4D::Act::N] = outputTile.shape[Dims4D::Act::N];
    inDataTile.offsets[Dims4D::Act::N] = outputTile.offsets[Dims4D::Act::N];

    return TilingInfo{std::move(inDataTile)};
}

InputTiling backInferMvn1NormSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile, Logger) {
    auto swKernelRuns = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    VPUX_THROW_UNLESS(std::distance(swKernelRuns.begin(), swKernelRuns.end()) == 1,
                      "SwKernelOp has already been tiled at '{0}'", swKernelOp);
    auto swKernelRun = *swKernelRuns.begin();
    VPUX_THROW_UNLESS(swKernelRun.getAttrs().has_value(), "SwKernelOp has no attr '{0}'", swKernelOp);
    const auto attrs = swKernelRun.getAttrs().value();

    const auto origMeanVarType = swKernelOp.getInputs()[1].getType().cast<vpux::NDTypeInterface>();
    const auto origMeanVarShape = origMeanVarType.getShape();

    TileInfo inDataTile(outputTile);
    TileInfo inMeanVarTile(origMeanVarShape);

    const auto acrossChannels = attrs[0].cast<mlir::BoolAttr>().getValue();
    if (!acrossChannels) {
        inMeanVarTile.shape[Dims4D::Act::C] = inDataTile.shape[Dims4D::Act::C];
        inMeanVarTile.offsets[Dims4D::Act::C] = inDataTile.offsets[Dims4D::Act::C];
    }
    inMeanVarTile.shape[Dims4D::Act::N] = inDataTile.shape[Dims4D::Act::N];
    inMeanVarTile.offsets[Dims4D::Act::N] = inDataTile.offsets[Dims4D::Act::N];

    return TilingInfo{{std::move(inDataTile), std::move(inMeanVarTile)}};
}

}  // namespace

InputTiling backInferSwKernelInputTile(VPUIP::SwKernelOp swKernelOp, const SmallVector<vpux::TileInfo>& outputTiles,
                                       int tileId, Logger log) {
    auto kernelEntryName = getSwKernelEntryName(swKernelOp);
    const auto& outputTile = outputTiles[tileId];
    if (kernelEntryName == "interpolate") {
        return backInferInterpolateSwKernelInputTile(swKernelOp, outputTile, log);
    } else if (kernelEntryName == "topk") {
        return backInferTopKSwKernelInputTile(swKernelOp, outputTile, log);
    } else if (kernelEntryName == "gather") {
        return backInferGatherSwKernelInputTile(swKernelOp, outputTile, log);
    } else if (kernelEntryName == "pad") {
        return backInferPadSwKernelInputTile(swKernelOp, outputTile, log);
    } else if (kernelEntryName == "mvn1_sum") {
        return backInferMvn1SumSwKernelInputTile(swKernelOp, outputTile, log);
    } else if (kernelEntryName == "mvn1_norm") {
        return backInferMvn1NormSwKernelInputTile(swKernelOp, outputTile, log);
    } else if (kernelEntryName == "depth_to_space") {
        return backInferDepthToSpaceSwKernelInputTile(swKernelOp, outputTile, log);
    } else if (kernelEntryName == "gru_sequence") {
        return backInferGRUSequenceSwKernelInputTile(swKernelOp, outputTile, log);
    } else if (kernelEntryName == "gru_sequence_last_part") {
        return backInferGRUSequenceLastPartSwKernelInputTile(swKernelOp, outputTile, log);
    } else if (isReduceKernelEntry(kernelEntryName)) {
        return backInferReduceSwKernelInputTile(swKernelOp, outputTile, kernelEntryName, log);
    } else if (kernelEntryName == "matmul") {
        return backInferMatMulSwKernelInputTile(swKernelOp, outputTile, log);
    } else if (kernelEntryName == "lstm_gates") {
        return backInferLSTMGatesSwKernelInputTile(swKernelOp, outputTile, log);
    } else if (kernelEntryName == "lstm_cell") {
        return backInferLSTMCellSwKernelInputTile(swKernelOp, outputTile, log);
    } else if (kernelEntryName == "detection_output_sort") {
        return vpux::VPU::DetectionOutputSortOpInputTilingOnShave(swKernelOp, outputTile, tileId, outputTiles.size(),
                                                                  log);
    }

    SmallVector<TileInfo> inputTiles;
    for (const auto& origInput : swKernelOp.getInputs()) {
        const auto curShape = getShape(origInput);
        VPUX_THROW_UNLESS(curShape.size() == outputTile.shape.size(),
                          "Can't tile SwKernel operation '{0}' at '{1}', which has operands with different rank",
                          swKernelOp->getName(), swKernelOp->getLoc());

        // Handle broadcasted inputs
        auto curTile = outputTile;
        for (auto ind : irange(curShape.size())) {
            const auto d = Dim(ind);
            if (curShape[d] == 1) {
                curTile.shape[d] = 1;
                curTile.offsets[d] = 0;
            }
        }

        inputTiles.push_back(curTile);
    }
    return TilingInfo{inputTiles};
}

SmallVector<mlir::Attribute> getSwkernelNewAttrsAfterTiling(VPUIP::SwKernelOp swKernelOp,
                                                            ArrayRef<mlir::Attribute> origAttr,
                                                            const TilingInfo& inputTiling, const TileInfo& outTile,
                                                            Logger log) {
    log.trace("Update SwKernel attrs after tiling at '{0}'", swKernelOp->getLoc());
    auto kernelEntryName = getSwKernelEntryName(swKernelOp);
    if (kernelEntryName == "interpolate") {
        return getInterpolateSwkernelNewAttrsAfterTiling(swKernelOp, origAttr, inputTiling, outTile, log);
    } else if (kernelEntryName == "pad") {
        return getPadSwkernelNewAttrsAfterTiling(swKernelOp, origAttr, outTile, log);
    } else {
        return SmallVector<mlir::Attribute>(origAttr.begin(), origAttr.end());
    }
}

// Return all tensor types of SwKernelOp that will be tiled
SmallVector<vpux::NDTypeInterface> getSwKernelTiledTypes(VPUIP::SwKernelOp swKernelOp) {
    auto kernelEntryName = getSwKernelEntryName(swKernelOp);
    if (kernelEntryName == "topk") {
        // For SW TopK, input, output and target shape will be tiled
        const auto inputType = swKernelOp->getOperand(0).getType();
        const auto outputType = swKernelOp->getResult(0).getType();
        const auto targetShapeType = swKernelOp->getResult(1).getType();
        return {inputType, outputType, targetShapeType};
    } else if (kernelEntryName == "gather") {
        // For SW Gather, indices and output will be tiled
        const auto indicesType = swKernelOp->getOperand(1).getType();
        const auto outputType = swKernelOp->getResult(0).getType();
        return {indicesType, outputType};
    } else if (kernelEntryName == "gru_sequence") {
        // For SW GRUSequence, inputData, initialHiddenState and outputs will be tiled
        const auto inputDataType = swKernelOp->getOperand(0).getType();
        const auto initialHiddenStateType = swKernelOp->getOperand(1).getType();
        const auto outputYType = swKernelOp->getResult(0).getType();
        const auto outputHoType = swKernelOp->getResult(1).getType();
        return {inputDataType, initialHiddenStateType, outputYType, outputHoType};
    } else if (kernelEntryName == "gru_sequence_last_part") {
        // For SW GRUSequenceLastPart, inputData, initialHiddenState and outputs will be tiled
        const auto inputDataType = swKernelOp->getOperand(0).getType();
        const auto initialHiddenStateType = swKernelOp->getOperand(1).getType();
        const auto outputYType = swKernelOp->getResult(0).getType();
        const auto outputHoType = swKernelOp->getResult(1).getType();
        return {inputDataType, initialHiddenStateType, outputYType, outputHoType};
    } else if (kernelEntryName == "eltwise_mul" || kernelEntryName == "eltwise_power" ||
               kernelEntryName == "eltwise_div" || kernelEntryName == "prelu_fp16" ||
               kernelEntryName == "eltwise_greater" || kernelEntryName == "eltwise_less" ||
               kernelEntryName == "eltwise_sub" || kernelEntryName == "eltwise_add") {
        // For SW Eltwise Op with multi inputs
        // Only the input which does not need broadcast and output will be tiled
        const auto lhsType = swKernelOp->getOperand(0).getType().cast<vpux::NDTypeInterface>();
        const auto rhsType = swKernelOp->getOperand(1).getType().cast<vpux::NDTypeInterface>();
        const auto outputType = swKernelOp->getResult(0).getType().cast<vpux::NDTypeInterface>();

        const auto lhsShape = lhsType.getShape();
        const auto rhsShape = rhsType.getShape();
        const auto outputShape = outputType.getShape();

        SmallVector<vpux::NDTypeInterface> tiledTypes;
        if (lhsShape == outputShape) {
            tiledTypes.push_back(lhsType);
        } else if (rhsShape == outputShape) {
            tiledTypes.push_back(rhsType);
        }
        tiledTypes.push_back(outputType);
        return tiledTypes;
    } else {
        // By default, all inputs and outputs will be tiled
        SmallVector<vpux::NDTypeInterface> tiledTypes;
        for (const auto& input : swKernelOp->getOperands()) {
            const auto inputType = input.getType();
            tiledTypes.push_back(inputType);
        }
        for (const auto& output : swKernelOp->getResults()) {
            const auto outputType = output.getType();
            tiledTypes.push_back(outputType);
        }
        return tiledTypes;
    }
}

bool isCacheOpTaskType(mlir::SymbolRefAttr kernelTaskType, bool includePrefetch) {
    if (!kernelTaskType) {
        return false;
    }
    auto taskTypeVal = VPU::symbolizeActShaveTaskType(kernelTaskType.getLeafReference().strref());
    VPUX_THROW_UNLESS(taskTypeVal.has_value(), "VPU::ActShaveTaskType has no value.");
    std::unordered_set<VPU::ActShaveTaskType> actShaveCacheOpTypes = {VPU::ActShaveTaskType::CACHE_FLUSH_INVALIDATE,
                                                                      VPU::ActShaveTaskType::CACHE_FLUSH,
                                                                      VPU::ActShaveTaskType::CACHE_INVALIDATE};
    if (includePrefetch) {
        actShaveCacheOpTypes.insert(VPU::ActShaveTaskType::CACHE_PREFETCH);
    }

    return actShaveCacheOpTypes.count(taskTypeVal.value()) > 0;
}

bool isCacheOpTaskType(std::optional<::mlir::SymbolRefAttr> kernelTaskType, bool includePrefetch) {
    return kernelTaskType.has_value() ? isCacheOpTaskType(kernelTaskType.value(), includePrefetch) : false;
}

bool isCacheHandlingOp(VPUIP::SwKernelOp swKernelOp) {
    auto moduleOp = swKernelOp->getParentOfType<mlir::ModuleOp>();
    auto kernelFunc = moduleOp.lookupSymbol<mlir::func::FuncOp>(swKernelOp.getKernelFunctionAttr());
    VPUX_THROW_UNLESS(kernelFunc != nullptr, "SwKernel has no kernel function.");

    auto kernelTaskType = kernelFunc->getAttrOfType<mlir::SymbolRefAttr>("VPU.task_type");
    if (kernelTaskType == nullptr) {
        return false;
    }

    return isCacheOpTaskType(kernelTaskType);
}

mlir::SmallVector<mlir::Value> getDDRBuffers(mlir::ValueRange buffers) {
    mlir::SmallVector<mlir::Value> ddrBuffers;
    llvm::copy(buffers | vpux::filtered([](mlir::Value buffer) {
                   auto bufferType = mlir::cast<vpux::NDTypeInterface>(buffer.getType());
                   return bufferType.getMemoryKind() == VPU::MemoryKind::DDR;
               }),
               std::back_inserter(ddrBuffers));

    return ddrBuffers;
}

bool hasInputsInDDR(VPUIP::SwKernelOp swKernelTask) {
    return llvm::any_of(swKernelTask.getInputs(), [](mlir::Value buffer) {
        auto bufferType = mlir::cast<vpux::NDTypeInterface>(buffer.getType());
        if (bufferType.getMemoryKind() == VPU::MemoryKind::DDR) {
            return true;
        }
        return false;
    });
}

int64_t getSwKernelTilingAddressAlignment(VPUIP::SwKernelOp swkernelOp, VPU::ArchKind arch) {
    if (arch == VPU::ArchKind::NPU37XX) {
        return 1;
    }

    auto name = getSwKernelEntryName(swkernelOp);
    if (llvm::find(SW_KERNELS_NEED_TILING_ALIGNMENT, name) == SW_KERNELS_NEED_TILING_ALIGNMENT.end()) {
        return 1;
    }
    return NPU40XX_SW_KERNEL_ADDRESS_ALIGNMENT;
}
}  // namespace VPUIP
}  // namespace vpux
