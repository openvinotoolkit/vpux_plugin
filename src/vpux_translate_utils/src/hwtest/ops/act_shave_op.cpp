//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
#include <vpux/hwtest/ops/act_shave_op.hpp>

#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"

namespace vpux {
namespace hwtest {

namespace {

uint64_t packAsI32intoU64(int64_t val1, int64_t val2) {
    static constexpr uint64_t bitWidth = sizeof(uint32_t) * CHAR_BIT;
    auto v1 = checked_cast<uint32_t>(val1);
    auto v2 = checked_cast<uint32_t>(val2);
    uint64_t patch = (static_cast<uint64_t>(v2) << bitWidth) | (static_cast<uint64_t>(v1));
    return patch;
}

VPUIP::KernelInfo getKernelInfo(nb::ActivationLayer activation, mlir::MLIRContext* ctx) {
    switch (activation.activationType) {
    case nb::ActivationType::HSwish:
        return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"activation_hswish"}};
    case nb::ActivationType::Sigmoid:
        return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"activation_sigmoid"}};
    case nb::ActivationType::Softmax: {
        SmallVector<uint64_t> storage;
        storage.push_back(packAsI32intoU64(activation.axis, /*padSize*/ 0));
        storage.push_back(packAsI32intoU64(/*mode*/ 0, /*nDims*/ 0));
        const auto paramsAttr = getIntArrayAttr(ctx, storage);
        const auto newAttrs = SmallVector<mlir::Attribute>{paramsAttr};
        return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{newAttrs}, {"softmax"}};
    }
    case nb::ActivationType::round_trip_b8h8_to_fp16:
        return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"round_trip_b8h8_to_fp16"}};
    case nb::ActivationType::sau_sumx_fp16_to_fp32:
        return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{}, {"sau_sumx_fp16_to_fp32"}};
    case nb::ActivationType::cmu_perm_x8:
    case nb::ActivationType::cmu_perm: {
        const auto permBlendParamAttr = getIntAttr(ctx, activation.permBlend);
        return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{permBlendParamAttr}, {"cmu_perm"}};
    }
    case nb::ActivationType::PopulateWeightTable: {
        const auto baseAttr = getIntAttr(ctx, activation.weightsOffset.value_or(0));
        const auto stepAttr = getIntAttr(ctx, activation.weightsPtrStep.value_or(0));
        return VPUIP::KernelInfo{SmallVector<mlir::Attribute>{baseAttr, stepAttr}, {"populate_weight_table"}};
    }
    default:
        VPUX_THROW("Activation is not supported for ActShave tests");
    }
}

}  // namespace

void buildActShaveTask(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                       Logger& log, ArrayRef<mlir::Type> inputTypes,
                       SmallVector<vpux::VPURT::DeclareBufferOp>& inputCMX, vpux::VPURT::DeclareBufferOp outputCMX,
                       vpux::VPURT::DeclareBufferOp profilingDataCMX, mlir::ValueRange waitBarrier,
                       mlir::ValueRange updateBarrier, size_t cluster, size_t /*unit*/) {
    auto* ctx = builder.getContext();
    auto activation = testDesc.getActivationLayer();
    auto profilingParams = testDesc.getProfilingParams();

    // consider the idx & the order when mapping axis:
    // NB idx starts from the left and order is NCHW
    // shave idx starts from the right and order in builder is NHWC
    //
    // NB: N C H W   ->   shave: N H W C
    //     0 1 2 3               3 2 1 0
    if (activation.axis != 0) {
        const auto inputTypesIf = inputTypes[0].cast<vpux::NDTypeInterface>();
        const auto order = inputTypesIf.getDimsOrder();
        const auto maxIter = inputTypesIf.getShape().size() - 1;
        activation.axis = maxIter - order.dimPos((DimsOrder::NCHW).dimAt(activation.axis));
    }
    auto kernelInfo = getKernelInfo(activation, ctx);

    const auto convertToUnrankedType = [ctx](mlir::Type srcType) -> mlir::Type {
        auto type = srcType.dyn_cast_or_null<mlir::MemRefType>();
        VPUX_THROW_UNLESS(type != nullptr, "Only MemRef type is supported");

        return mlir::UnrankedMemRefType::get(type.getElementType(),
                                             mlir::SymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN)));
    };
    SmallVector<mlir::Type> inputTypesUnranked;
    std::transform(inputTypes.begin(), inputTypes.end(), std::back_inserter(inputTypesUnranked), convertToUnrankedType);
    std::transform(kernelInfo.args.begin(), kernelInfo.args.end(), std::back_inserter(inputTypesUnranked),
                   [ctx](mlir::Attribute arg) {
                       const auto typedAttr = arg.dyn_cast<mlir::TypedAttr>();
                       return typedAttr != nullptr ? typedAttr.getType() : mlir::NoneType::get(ctx);
                   });

    // first creating management kernel definition
    VPUIP::createRuntimeKernelDefinition(module, log, testDesc.getArchitecture());

    // Create built-in function ------------------------------------------------

    SmallString builtInFunctionName{"builtin_actshave_"};

    auto builtInFunction = VPUIP::createBuiltInFunction(module, builtInFunctionName, inputTypesUnranked,
                                                        kernelInfo.entryName, kernelInfo.sourceFileName, log);

    // Spawn Task: Kernel ------------------------------------------------------

    auto kernelBuilder = [&](auto /*fn object*/ kernelTaskBody) {
        auto taskOp = builder.create<vpux::VPURT::TaskOp>(builder.getUnknownLoc(), waitBarrier, updateBarrier);

        mlir::OpBuilder::InsertPoint lastInsertionPoint = builder.saveInsertionPoint();
        auto& block = taskOp.getBody().emplaceBlock();
        builder.setInsertionPointToStart(&block);

        kernelTaskBody();

        builder.restoreInsertionPoint(lastInsertionPoint);
    };

    kernelBuilder([&]() {
        const int64_t tile = checked_cast<int64_t>(cluster);

        SmallVector<mlir::Value> inputCMXValues;
        for (auto& input : inputCMX) {
            inputCMXValues.push_back(input.getBuffer());
        }

        auto loc = mlir::NameLoc::get(mlir::StringAttr::get(ctx, "convert?t_Convert"));
        auto swKernelOp = builder.create<VPUIP::SwKernelOp>(
                loc, mlir::ValueRange{inputCMXValues}, outputCMX.getBuffer(),
                profilingParams.swProfilingEnabled ? profilingDataCMX.getBuffer() : nullptr, builtInFunction,
                getIntAttr(ctx, tile));

        if (profilingParams.swProfilingEnabled) {
            auto profAttr = VPUIP::SwProfilingMetadataAttr::get(
                    ctx, /* bufferId */ getIntAttr(ctx, 0), /* bufferOffset */ getIntAttr(ctx, 0),
                    /* clusterSize */ getIntAttr(ctx, 1), /* dataIndex */ getIntAttr(ctx, 0),
                    /* tileId */ getIntAttr(ctx, tile), /* clusterId */ getIntAttr(ctx, tile));
            swKernelOp.setProfilingMetadataAttr(profAttr);
        }

        VPUIP::initSwKernel(swKernelOp, mlir::ValueRange{inputCMXValues}, outputCMX.getBuffer(), kernelInfo.args, log);
    });
}

}  // namespace hwtest
}  // namespace vpux
