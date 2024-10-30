//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once
#include <gtest/gtest.h>

#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include "common/utils.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/ppe_factory.hpp"

using namespace vpux;

template <class MainOpType>
class LayerWithPostOpModel final :
        public vpux::IE::LayerWithPostOpInterface::ExternalModel<LayerWithPostOpModel<MainOpType>, MainOpType> {
public:
    bool isSupportedPostOp(mlir::Operation*, mlir::Operation*, const vpux::LogCb&) const {
        return true;
    }

    bool isSupportedClampOp(mlir::Operation*, mlir::Operation*, const vpux::LogCb&) const {
        return false;
    }

    void setLayerClampOp(mlir::Operation*, mlir::Operation*) const {
    }
};

class MLIR_PpeRegistry : public testing::Test {
public:
    MLIR_PpeRegistry() {
        registry = createDialectRegistry();
        registry.addExtension(+[](mlir::MLIRContext* ctx, vpux::IE::IEDialect*) {
            vpux::IE::ConvolutionOp::attachInterface<LayerWithPostOpModel<vpux::IE::ConvolutionOp>>(*ctx);
            vpux::IE::TransposedConvolutionOp::attachInterface<LayerWithPostOpModel<vpux::IE::TransposedConvolutionOp>>(
                    *ctx);
            vpux::IE::GroupConvolutionOp::attachInterface<LayerWithPostOpModel<vpux::IE::GroupConvolutionOp>>(*ctx);
            vpux::IE::MaxPoolOp::attachInterface<LayerWithPostOpModel<vpux::IE::MaxPoolOp>>(*ctx);
            vpux::IE::AvgPoolOp::attachInterface<LayerWithPostOpModel<vpux::IE::AvgPoolOp>>(*ctx);
            vpux::IE::AddOp::attachInterface<LayerWithPostOpModel<vpux::IE::AddOp>>(*ctx);
            vpux::IE::SubtractOp::attachInterface<LayerWithPostOpModel<vpux::IE::SubtractOp>>(*ctx);
        });
    }

protected:
    mlir::DialectRegistry registry;
};

class IE_OperationFactory {
public:
    mlir::Operation* createAddWithFp16Input(mlir::Location loc, mlir::MLIRContext* ctx,
                                            vpux::IE::PostOpAttr postOpAttr) {
        mlir::OpBuilder builder(ctx);

        mlir::tensor::EmptyOp constantOp = builder.create<mlir::tensor::EmptyOp>(
                mlir::UnknownLoc::get(ctx), ArrayRef<int64_t>{1, 16, 32, 32},
                mlir::RankedTensorType::get(ArrayRef<int64_t>{1, 16, 32, 32}, mlir::Float16Type::get(ctx))
                        .getElementType());

        return builder.create<IE::AddOp>(
                loc, constantOp.getResult(), constantOp.getResult(),
                vpux::IE::AutoBroadcastTypeAttr::get(ctx, vpux::IE::AutoBroadcastType::NONE_OR_EXPLICIT), postOpAttr,
                nullptr, nullptr, nullptr);
    }

    mlir::Operation* createSubtractWithI4Input(mlir::Location loc, mlir::MLIRContext* ctx,
                                               vpux::IE::PostOpAttr postOpAttr) {
        mlir::OpBuilder builder(ctx);

        auto quantType = mlir::quant::UniformQuantizedType::getChecked(loc, mlir::quant::QuantizationFlags::Signed,
                                                                       vpux::getSInt4Type(ctx),
                                                                       mlir::Float16Type::get(ctx), 0.05, 0.0, -7, 7);

        mlir::tensor::EmptyOp constantOp = builder.create<mlir::tensor::EmptyOp>(
                mlir::UnknownLoc::get(ctx), ArrayRef<int64_t>{1, 16, 32, 32},
                mlir::RankedTensorType::get(ArrayRef<int64_t>{1, 16, 32, 32}, quantType).getElementType());

        return builder.create<IE::SubtractOp>(
                loc, mlir::RankedTensorType::get(ArrayRef<int64_t>{1, 16, 32, 32}, quantType), constantOp.getResult(),
                constantOp.getResult(),
                vpux::IE::AutoBroadcastTypeAttr::get(ctx, vpux::IE::AutoBroadcastType::NONE_OR_EXPLICIT), postOpAttr,
                nullptr, nullptr, nullptr);
    }

    mlir::Operation* createSubtractWithFp16Input(mlir::Location loc, mlir::MLIRContext* ctx,
                                                 vpux::IE::PostOpAttr postOpAttr) {
        mlir::OpBuilder builder(ctx);

        mlir::tensor::EmptyOp constantOp = builder.create<mlir::tensor::EmptyOp>(
                mlir::UnknownLoc::get(ctx), ArrayRef<int64_t>{1, 16, 32, 32},
                mlir::RankedTensorType::get(ArrayRef<int64_t>{1, 16, 32, 32}, mlir::Float16Type::get(ctx))
                        .getElementType());

        return builder.create<IE::SubtractOp>(
                loc, constantOp.getResult(), constantOp.getResult(),
                vpux::IE::AutoBroadcastTypeAttr::get(ctx, vpux::IE::AutoBroadcastType::NONE_OR_EXPLICIT), postOpAttr,
                nullptr, nullptr, nullptr);
    }

    mlir::Operation* createAddWithI4Input(mlir::Location loc, mlir::MLIRContext* ctx, vpux::IE::PostOpAttr postOpAttr) {
        mlir::OpBuilder builder(ctx);

        auto quantType = mlir::quant::UniformQuantizedType::getChecked(loc, mlir::quant::QuantizationFlags::Signed,
                                                                       vpux::getSInt4Type(ctx),
                                                                       mlir::Float16Type::get(ctx), 0.05, 0.0, -7, 7);

        mlir::tensor::EmptyOp constantOp = builder.create<mlir::tensor::EmptyOp>(
                mlir::UnknownLoc::get(ctx), ArrayRef<int64_t>{1, 16, 32, 32},
                mlir::RankedTensorType::get(ArrayRef<int64_t>{1, 16, 32, 32}, quantType).getElementType());

        return builder.create<IE::AddOp>(
                loc, mlir::RankedTensorType::get(ArrayRef<int64_t>{1, 16, 32, 32}, quantType), constantOp.getResult(),
                constantOp.getResult(),
                vpux::IE::AutoBroadcastTypeAttr::get(ctx, vpux::IE::AutoBroadcastType::NONE_OR_EXPLICIT), postOpAttr,
                nullptr, nullptr, nullptr);
    }

    mlir::Operation* createAvgPoolWithFp16Input(mlir::Location loc, mlir::MLIRContext* ctx,
                                                vpux::IE::PostOpAttr postOpAttr) {
        const std::vector<int64_t> maxPoolStrides = {1, 1};
        const std::vector<int64_t> maxPoolKernels = {2, 2};
        const std::vector<int64_t> pads = {0, 0};
        const auto padsAttr = getIntArrayAttr(ctx, pads);
        mlir::OpBuilder builder(ctx);

        mlir::tensor::EmptyOp constantOp = builder.create<mlir::tensor::EmptyOp>(
                mlir::UnknownLoc::get(ctx), ArrayRef<int64_t>{1, 16, 32, 32},
                mlir::RankedTensorType::get(ArrayRef<int64_t>{1, 16, 32, 32}, mlir::Float16Type::get(ctx))
                        .getElementType());

        return builder.create<IE::AvgPoolOp>(loc, constantOp.getResult(), getIntArrayAttr(ctx, maxPoolKernels),
                                             getIntArrayAttr(ctx, maxPoolStrides), padsAttr, padsAttr,
                                             vpux::IE::RoundingTypeAttr::get(ctx, vpux::IE::RoundingType::FLOOR),
                                             mlir::UnitAttr::get(ctx), postOpAttr, nullptr, nullptr, nullptr, nullptr);
    }

    mlir::Operation* createAvgPoolWithFp16InputI8Output(mlir::Location loc, mlir::MLIRContext* ctx,
                                                        vpux::IE::PostOpAttr postOpAttr) {
        const std::vector<int64_t> avgPoolStrides = {1, 1};
        const std::vector<int64_t> avgPoolKernels = {2, 2};
        const std::vector<int64_t> pads = {0, 0};
        const auto padsAttr = getIntArrayAttr(ctx, pads);
        mlir::OpBuilder builder(ctx);

        auto quantOutType = mlir::quant::UniformQuantizedType::getChecked(
                loc, mlir::quant::QuantizationFlags::Signed, vpux::getSInt8Type(ctx), mlir::Float16Type::get(ctx), 0.05,
                0.0, -128, 127);
        auto outType = mlir::RankedTensorType::get(ArrayRef<int64_t>{1, 16, 32, 32}, quantOutType);

        mlir::tensor::EmptyOp constantOp = builder.create<mlir::tensor::EmptyOp>(
                mlir::UnknownLoc::get(ctx), ArrayRef<int64_t>{1, 16, 32, 32},
                mlir::RankedTensorType::get(ArrayRef<int64_t>{1, 16, 32, 32}, mlir::Float16Type::get(ctx))
                        .getElementType());

        return builder.create<IE::AvgPoolOp>(loc, outType, constantOp.getResult(), getIntArrayAttr(ctx, avgPoolKernels),
                                             getIntArrayAttr(ctx, avgPoolStrides), padsAttr, padsAttr,
                                             vpux::IE::RoundingTypeAttr::get(ctx, vpux::IE::RoundingType::FLOOR),
                                             mlir::UnitAttr::get(ctx), postOpAttr, nullptr, nullptr, nullptr, nullptr);
    }

    mlir::Operation* createAvgPoolWithI4Input(mlir::Location loc, mlir::MLIRContext* ctx,
                                              vpux::IE::PostOpAttr postOpAttr) {
        const std::vector<int64_t> maxPoolStrides = {1, 1};
        const std::vector<int64_t> maxPoolKernels = {2, 2};
        const std::vector<int64_t> pads = {0, 0};
        const auto padsAttr = getIntArrayAttr(ctx, pads);
        mlir::OpBuilder builder(ctx);

        auto quantType = mlir::quant::UniformQuantizedType::getChecked(
                loc, mlir::quant::QuantizationFlags::Signed, vpux::getSInt4Type(ctx), mlir::Float16Type::get(ctx), 0.05,
                0.0, -7.0, 7.0);

        mlir::tensor::EmptyOp constantOp = builder.create<mlir::tensor::EmptyOp>(
                mlir::UnknownLoc::get(ctx), ArrayRef<int64_t>{1, 16, 32, 32},
                mlir::RankedTensorType::get(ArrayRef<int64_t>{1, 16, 32, 32}, quantType).getElementType());

        return builder.create<IE::AvgPoolOp>(
                loc, mlir::RankedTensorType::get(ArrayRef<int64_t>{1, 16, 32, 32}, quantType), constantOp.getResult(),
                vpux::getIntArrayAttr(ctx, maxPoolKernels), getIntArrayAttr(ctx, maxPoolStrides), padsAttr, padsAttr,
                vpux::IE::RoundingTypeAttr::get(ctx, vpux::IE::RoundingType::FLOOR), mlir::UnitAttr::get(ctx),
                postOpAttr, nullptr, nullptr, nullptr, nullptr);
    }

    mlir::Operation* createMaxPoolWithFp16Input(mlir::Location loc, mlir::MLIRContext* ctx,
                                                vpux::IE::PostOpAttr postOpAttr) {
        const std::vector<int64_t> maxPoolStrides = {1, 1};
        const std::vector<int64_t> maxPoolKernels = {1, 1};
        const std::vector<int64_t> pads = {0, 0};
        const auto padsAttr = getIntArrayAttr(ctx, pads);
        mlir::OpBuilder builder(ctx);

        mlir::tensor::EmptyOp constantOp = builder.create<mlir::tensor::EmptyOp>(
                mlir::UnknownLoc::get(ctx), ArrayRef<int64_t>{1, 16, 32, 32},
                mlir::RankedTensorType::get(ArrayRef<int64_t>{1, 16, 32, 32}, mlir::Float16Type::get(ctx))
                        .getElementType());

        return builder.create<IE::MaxPoolOp>(loc, constantOp.getResult(), getIntArrayAttr(ctx, maxPoolKernels),
                                             getIntArrayAttr(ctx, maxPoolStrides), padsAttr, padsAttr,
                                             vpux::IE::RoundingTypeAttr::get(ctx, vpux::IE::RoundingType::FLOOR),
                                             postOpAttr, nullptr, nullptr, nullptr);
    }

    mlir::Operation* createMaxPoolWithI4Input(mlir::Location loc, mlir::MLIRContext* ctx,
                                              vpux::IE::PostOpAttr postOpAttr) {
        const std::vector<int64_t> maxPoolStrides = {1, 1};
        const std::vector<int64_t> maxPoolKernels = {1, 1};
        const std::vector<int64_t> pads = {0, 0};
        const auto padsAttr = getIntArrayAttr(ctx, pads);
        mlir::OpBuilder builder(ctx);

        auto inQuantType = mlir::quant::UniformQuantizedType::getChecked(loc, mlir::quant::QuantizationFlags::Signed,
                                                                         vpux::getSInt4Type(ctx),
                                                                         mlir::Float16Type::get(ctx), 0.04, 1, -4, 6);
        auto outQuantType = mlir::quant::UniformQuantizedType::getChecked(loc, mlir::quant::QuantizationFlags::Signed,
                                                                          vpux::getSInt4Type(ctx),
                                                                          mlir::Float16Type::get(ctx), 0.02, 1, -2, 3);

        mlir::tensor::EmptyOp constantOp = builder.create<mlir::tensor::EmptyOp>(
                mlir::UnknownLoc::get(ctx), ArrayRef<int64_t>{1, 16, 32, 32},
                mlir::RankedTensorType::get(ArrayRef<int64_t>{1, 16, 32, 32}, inQuantType).getElementType());

        return builder.create<IE::MaxPoolOp>(
                loc, mlir::RankedTensorType::get(ArrayRef<int64_t>{1, 16, 32, 32}, outQuantType),
                constantOp.getResult(), vpux::getIntArrayAttr(ctx, maxPoolKernels),
                getIntArrayAttr(ctx, maxPoolStrides), padsAttr, padsAttr,
                vpux::IE::RoundingTypeAttr::get(ctx, vpux::IE::RoundingType::FLOOR), postOpAttr, nullptr, nullptr,
                nullptr);
    }

    mlir::Operation* createConvolutionFp16Input(mlir::Location loc, mlir::MLIRContext* ctx,
                                                vpux::IE::PostOpAttr postOpAttr) {
        auto strides = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});
        auto padsBegin = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
        auto padsEnd = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
        auto dilations = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});
        mlir::OpBuilder builder(ctx);

        mlir::tensor::EmptyOp constantOp = builder.create<mlir::tensor::EmptyOp>(
                mlir::UnknownLoc::get(ctx), ArrayRef<int64_t>{1, 16, 32, 32},
                mlir::RankedTensorType::get(ArrayRef<int64_t>{1, 16, 32, 32}, mlir::Float16Type::get(ctx))
                        .getElementType());

        return builder.create<IE::ConvolutionOp>(loc, constantOp.getResult(), constantOp.getResult(),
                                                 constantOp.getResult(), strides, padsBegin, padsEnd, dilations,
                                                 postOpAttr, nullptr, nullptr, nullptr, nullptr);
    }

    mlir::Operation* createConvolutionI4Input(mlir::Location loc, mlir::MLIRContext* ctx,
                                              vpux::IE::PostOpAttr postOpAttr) {
        auto strides = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});
        auto padsBegin = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
        auto padsEnd = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
        auto dilations = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});
        mlir::OpBuilder builder(ctx);

        auto quantType = mlir::quant::UniformQuantizedType::getChecked(loc, mlir::quant::QuantizationFlags::Signed,
                                                                       vpux::getSInt4Type(ctx),
                                                                       mlir::Float16Type::get(ctx), 0.05, 0, -7, 7);

        mlir::tensor::EmptyOp constantOp = builder.create<mlir::tensor::EmptyOp>(
                mlir::UnknownLoc::get(ctx), ArrayRef<int64_t>{1, 16, 32, 32},
                mlir::RankedTensorType::get(ArrayRef<int64_t>{1, 16, 32, 32}, quantType).getElementType());

        return builder.create<IE::ConvolutionOp>(
                loc, mlir::RankedTensorType::get(ArrayRef<int64_t>{1, 16, 32, 32}, quantType), constantOp.getResult(),
                constantOp.getResult(), constantOp.getResult(), strides, padsBegin, padsEnd, dilations, postOpAttr,
                nullptr, nullptr, nullptr, nullptr);
    }

    mlir::Operation* createMultiplyWithFp16Input(mlir::Location loc, mlir::MLIRContext* ctx,
                                                 vpux::IE::PostOpAttr postOpAttr) {
        mlir::OpBuilder builder(ctx);

        mlir::tensor::EmptyOp constantOp = builder.create<mlir::tensor::EmptyOp>(
                mlir::UnknownLoc::get(ctx), ArrayRef<int64_t>{1, 16, 32, 32},
                mlir::RankedTensorType::get(ArrayRef<int64_t>{1, 16, 32, 32}, mlir::Float16Type::get(ctx))
                        .getElementType());

        return builder.create<IE::MultiplyOp>(
                loc, constantOp.getResult(), constantOp.getResult(),
                vpux::IE::AutoBroadcastTypeAttr::get(ctx, vpux::IE::AutoBroadcastType::NONE_OR_EXPLICIT), postOpAttr,
                nullptr, nullptr, nullptr);
    }

    mlir::Operation* createMultiplyWithI4Input(mlir::Location loc, mlir::MLIRContext* ctx,
                                               vpux::IE::PostOpAttr postOpAttr) {
        mlir::OpBuilder builder(ctx);

        auto quantType = mlir::quant::UniformQuantizedType::getChecked(loc, mlir::quant::QuantizationFlags::Signed,
                                                                       vpux::getSInt4Type(ctx),
                                                                       mlir::Float16Type::get(ctx), 0.04, 1, -4, 6);

        mlir::tensor::EmptyOp constantOp = builder.create<mlir::tensor::EmptyOp>(
                mlir::UnknownLoc::get(ctx), ArrayRef<int64_t>{1, 16, 32, 32},
                mlir::RankedTensorType::get(ArrayRef<int64_t>{1, 16, 32, 32}, quantType).getElementType());

        return builder.create<IE::MultiplyOp>(
                loc, mlir::RankedTensorType::get(ArrayRef<int64_t>{1, 16, 32, 32}, quantType), constantOp.getResult(),
                constantOp.getResult(),
                vpux::IE::AutoBroadcastTypeAttr::get(ctx, vpux::IE::AutoBroadcastType::NONE_OR_EXPLICIT), postOpAttr,
                nullptr, nullptr, nullptr);
    }
};

class VPU_PpeUnitBase : public MLIR_PpeRegistry {
public:
    VPU_PpeUnitBase(std::unique_ptr<vpux::VPU::IPpeFactory>&& ppeIfc)
            : MLIR_PpeRegistry(),
              _ctx(registry),
              _location(mlir::UnknownLoc::get(&_ctx)),
              _ppeIfc(std::move(ppeIfc)),
              _output(_ppeAttrIR) {
        _ctx.loadDialect<vpux::VPU::VPUDialect>();
        _ctx.loadDialect<vpux::Const::ConstDialect>();
    }

    vpux::IE::PostOpAttr createPostOpAttrRELU() {
        const auto operationName = vpux::IE::ReLUOp::getOperationName();
        return vpux::IE::PostOpAttr::get(&_ctx, mlir::StringAttr::get(&_ctx, operationName),
                                         mlir::DictionaryAttr::get(&_ctx));
    }

    vpux::IE::PostOpAttr createPostOpAttrCLAMP(double min, double max) {
        const auto operationName = vpux::IE::ClampOp::getOperationName();

        SmallVector<mlir::NamedAttribute> dicAttrFields;
        auto minClampName = vpux::IE::ClampOp::getMinAttrName(mlir::OperationName(operationName, &_ctx));
        auto minClampValue = vpux::getFPAttr(&_ctx, min);
        dicAttrFields.emplace_back(minClampName, minClampValue);

        auto maxClampName = vpux::IE::ClampOp::getMaxAttrName(mlir::OperationName(operationName, &_ctx));
        auto maxClampValue = vpux::getFPAttr(&_ctx, max);
        dicAttrFields.emplace_back(maxClampName, maxClampValue);
        auto dicAttr = mlir::DictionaryAttr::get(&_ctx, dicAttrFields);

        return vpux::IE::PostOpAttr::get(&_ctx, mlir::StringAttr::get(&_ctx, operationName), dicAttr);
    }
    vpux::IE::PostOpAttr createPostOpAttrLPRELU(double negativeSlope) {
        const auto operationName = vpux::IE::LeakyReluOp::getOperationName();

        SmallVector<mlir::NamedAttribute> dicAttrFields;
        auto negativeSlopeName =
                vpux::IE::LeakyReluOp::getNegativeSlopeAttrName(mlir::OperationName(operationName, &_ctx));
        auto negativeSlopeValue = vpux::getFPAttr(&_ctx, negativeSlope);
        dicAttrFields.emplace_back(negativeSlopeName, negativeSlopeValue);
        auto dicAttr = mlir::DictionaryAttr::get(&_ctx, dicAttrFields);

        return vpux::IE::PostOpAttr::get(&_ctx, mlir::StringAttr::get(&_ctx, operationName), dicAttr);
    }

protected:
    mlir::MLIRContext _ctx;
    mlir::Location _location;
    VPU::PpeIfcPtr _ppeIfc;
    std::string _ppeAttrIR;
    llvm::raw_string_ostream _output;
    IE_OperationFactory _operationFactory;
};
