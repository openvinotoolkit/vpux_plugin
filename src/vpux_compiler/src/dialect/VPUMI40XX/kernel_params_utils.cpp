//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUMI40XX/kernel_params_utils.hpp"

namespace vpux {
namespace VPUMI40XX {

sw_params::DataType KernelParamsSerializer::getDataTypeFromMlirType(mlir::Type type) {
    if (auto floatType = type.dyn_cast<mlir::FloatType>()) {
        auto typeWidth = floatType.getWidth();
        switch (typeWidth) {
        case 64:
            return sw_params::DataType::NN_FP64;
        case 32:
            return sw_params::DataType::NN_FP32;
        case 16:
            return sw_params::DataType::NN_FP16;
        case 8:
            return sw_params::DataType::NN_FP8;
        }
    } else if (auto integerType = type.dyn_cast<mlir::IntegerType>()) {
        if (integerType.isSigned()) {
            auto typeWidth = integerType.getWidth();
            switch (typeWidth) {
            case 64:
                return sw_params::DataType::NN_I64;
            case 32:
                return sw_params::DataType::NN_I32;
            case 16:
                return sw_params::DataType::NN_I16;
            case 8:
                return sw_params::DataType::NN_I8;
            case 4:
                return sw_params::DataType::NN_I4;
            case 2:
                return sw_params::DataType::NN_I2;
            case 1:
                return sw_params::DataType::NN_BIN;
            }
        } else if (integerType.isUnsigned()) {
            auto typeWidth = integerType.getWidth();
            switch (typeWidth) {
            case 64:
                return sw_params::DataType::NN_U64;
            case 32:
                return sw_params::DataType::NN_U32;
            case 16:
                return sw_params::DataType::NN_U16;
            case 8:
                return sw_params::DataType::NN_U8;
            case 1:
                return sw_params::DataType::NN_BIN;
            }
        } else if (integerType.isSignless()) {
            auto typeWidth = integerType.getWidth();
            switch (typeWidth) {
            case 64:
                return sw_params::DataType::NN_I64;
            case 32:
                return sw_params::DataType::NN_I32;
            case 16:
                return sw_params::DataType::NN_I16;
            case 8:
                return sw_params::DataType::NN_I8;
            case 4:
                return sw_params::DataType::NN_I4;
            case 2:
                return sw_params::DataType::NN_I2;
            case 1:
                return sw_params::DataType::NN_BIN;
            }
        }
    } else if (type.isBF16()) {
        return sw_params::DataType::NN_BF16;
    }
    return sw_params::DataType::NN_UNDEFINED;
}

sw_params::Location KernelParamsSerializer::getSwParamsLocationFromMemKind(VPU::MemoryKind memKind) {
    static const EnumMap<VPU::MemoryKind, sw_params::Location> memKindMapping = {
            {VPU::MemoryKind::DDR, sw_params::Location::DDR},
            {VPU::MemoryKind::CMX_NN, sw_params::Location::NN_CMX},
            {VPU::MemoryKind::CMX_UPA, sw_params::Location::UPA_CMX},
            {VPU::MemoryKind::CSRAM, sw_params::Location::NONE},
            {VPU::MemoryKind::Register, sw_params::Location::NONE},
    };
    return memKindMapping.at(memKind);
}

void KernelParamsSerializer::addBasicAttrToVector(SmallVector<uint8_t>& vec, mlir::Attribute attr) {
    if (auto val = attr.dyn_cast_or_null<mlir::IntegerAttr>()) {
        appendValueToVector(vec, val.getValue().getSExtValue());
    } else if (auto val = attr.dyn_cast_or_null<mlir::FloatAttr>()) {
        appendValueToVector(vec, static_cast<float>(val.getValue().convertToDouble()));
    } else {
        const auto typedAttr = attr.dyn_cast_or_null<mlir::TypedAttr>();
        const auto type = typedAttr != nullptr ? typedAttr.getType() : nullptr;
        VPUX_THROW("Act Shave Invocation: cannot store arg of type {0}", type);
    }
}

void KernelParamsSerializer::addAttrsToVector(SmallVector<uint8_t>& vec, mlir::Attribute attr) {
    if (auto arr = attr.dyn_cast_or_null<mlir::ArrayAttr>()) {
        auto vals = arr.getValue();
        for (auto val : vals) {
            addBasicAttrToVector(vec, val);
        }
    } else {
        addBasicAttrToVector(vec, attr);
    }
}

void KernelParamsSerializer::addTensorArgToVector(SmallVector<uint8_t>& vec, mlir::Value value) {
    sw_params::MemRefData memrefData{};

    const auto shape = getShape(value);
    memrefData.numDims = checked_cast<uint32_t>(shape.size());

    // order
    const auto inOrder = DimsOrder::fromValue(value);
    const auto memShape = inOrder.toMemoryOrder(shape);
    memrefData.dimsOrder = inOrder.invertedCode();

    auto type = value.getType();
    auto ndType = type.cast<vpux::NDTypeInterface>();

    auto ndTypeMemSpace = ndType.getMemSpace();
    auto tileIndex = ndTypeMemSpace == nullptr ? 0 : ndType.getMemSpace().getIndex().value_or(0);
    uint32_t tileMask = 1 << (tileIndex + 21);
    memrefData.dataAddr = tileMask;

    memrefData.dataType = getDataTypeFromMlirType(ndType.getElementType());
    memrefData.location = getSwParamsLocationFromMemKind(ndType.getMemoryKind());

    appendValueToVector(vec, memrefData);
}

SmallVector<uint8_t> KernelParamsSerializer::createKernelParams(VPUIP::SwKernelOp swKernelOp) {
    SmallVector<uint8_t> paramsVector;

    const auto insSize = swKernelOp.getInputs().size();
    const auto outsSize = swKernelOp.getOutputBuffs().size();

    const auto kernelOpArgsCount = insSize + outsSize;

    for (auto&& kernelRun : swKernelOp.getBody().getOps<VPUIP::SwKernelRun>()) {
        for (auto&& operand : kernelRun.getArgs()) {
            auto blockArg = operand.dyn_cast_or_null<mlir::BlockArgument>();
            if (blockArg) {
                auto id = blockArg.getArgNumber();
                VPUX_THROW_UNLESS(id < kernelOpArgsCount,
                                  "Index '{0}' of argument of Kernel.Run operation is out of range {1}'", id,
                                  kernelOpArgsCount);

                auto blockArgType = blockArg.getType();
                auto blockArgNdTypeIf = blockArgType.cast<vpux::NDTypeInterface>();
                auto ioType = id < insSize ? swKernelOp.getInputs()[id].getType()
                                           : swKernelOp.getOutputBuffs()[id - insSize].getType();
                auto ioNdTypeIf = ioType.cast<vpux::NDTypeInterface>();
                VPUX_THROW_UNLESS(blockArgNdTypeIf != nullptr || ioNdTypeIf != nullptr,
                                  "createKernelParams: sw kernel I/O does not implement NDTypeInterface");
                VPUX_THROW_UNLESS(vpux::areTypesCompatible(blockArgType, ioType,
                                                           vpux::IE::TypeComparisonMode::STRICT_EQUAL, true, true),
                                  "createKernelParams: types of sw kernel I/O do not match");
                VPUX_THROW_UNLESS(blockArgNdTypeIf.getShape() == ioNdTypeIf.getShape(),
                                  "createKernelParams: shapes of I/O do not match");

                const auto operandVal = swKernelOp->getOpOperand(id).get();
                addTensorArgToVector(paramsVector, operandVal);
            } else {
                VPUX_THROW("Only block arguments are supported");
            }
        }
        if (kernelRun.getAttrs().has_value()) {
            const mlir::ArrayAttr arrayAttrs = kernelRun.getAttrs().value();
            const auto& attrs = arrayAttrs.getValue();
            for (const auto& attr : attrs) {
                addAttrsToVector(paramsVector, attr);
            }
        }
    }

    return paramsVector;
}

}  // namespace VPUMI40XX
}  // namespace vpux
