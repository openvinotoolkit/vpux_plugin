//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/type_interfaces.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/init.hpp"

#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include "common/ppe_utils.hpp"
#include "common/utils.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include "vpux/compiler/NPU37XX/dialect/VPU/impl/ppe_factory.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/ppe_factory.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_utils.hpp"

#include <gtest/gtest.h>

using namespace vpux;

class NPU37xxPpeIfcUnitTest : public VPU_PpeUnitBase {
public:
    NPU37xxPpeIfcUnitTest(): VPU_PpeUnitBase(std::make_unique<vpux::VPU::arch37xx::PpeFactory>()) {
    }
};

TEST_F(NPU37xxPpeIfcUnitTest, MaxPool_InputFp16_WithPostOp_RELU) {
    auto maxPoolOp =
            _operationFactory.createMaxPoolWithFp16Input(_location, &_ctx, VPU_PpeUnitBase::createPostOpAttrRELU());
    ASSERT_NE(maxPoolOp, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(maxPoolOp);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ;
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), std::numeric_limits<int32_t>::min());
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), std::numeric_limits<int32_t>::max());
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 0);
    EXPECT_EQ(intPpeAttr.getFpPreluAlpha().getValueAsDouble(), 1.0f);
    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LRELU);

    // test update

    if (auto adapter = dynamic_cast<const vpux::VPU::IPpeAdapterClamp*>(_ppeIfc.get())) {
        auto newClampsAttr = vpux::VPU::PPEIntAttr::get(
                &_ctx, intPpeAttr.getMode(), vpux::getIntAttr(&_ctx, std::numeric_limits<int32_t>::min()),
                vpux::getIntAttr(&_ctx, 14.0), intPpeAttr.getLreluMult(), intPpeAttr.getLreluShift(),
                intPpeAttr.getQuantScale(), intPpeAttr.getQuantMult(), intPpeAttr.getQuantShift(),
                intPpeAttr.getQuantPostShift(), intPpeAttr.getIn1QuantMult(), intPpeAttr.getIn2QuantMult(),
                intPpeAttr.getFpPreluAlpha());

        auto newPpe = adapter->updateClamps(intPpeAttr, newClampsAttr);
        auto newIntPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(newPpe);

        EXPECT_EQ(newIntPpeAttr.getClampLow().getValue().getSExtValue(), std::numeric_limits<int32_t>::min());
        EXPECT_EQ(newIntPpeAttr.getClampHigh().getValue().getSExtValue(), 14);
        EXPECT_EQ(newIntPpeAttr.getLreluMult().getValue().getSExtValue(), 1);
        EXPECT_EQ(newIntPpeAttr.getLreluShift().getValue().getSExtValue(), 0);
        EXPECT_EQ(newIntPpeAttr.getFpPreluAlpha().getValueAsDouble(), 1.0f);
        EXPECT_EQ(newIntPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LRELU);
    }

    if (auto adapter = dynamic_cast<const vpux::VPU::IPpeAdapterScale*>(_ppeIfc.get())) {
        auto newPpe = adapter->updateScale(intPpeAttr, {0.1f});
        auto newIntPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(newPpe);

        EXPECT_EQ(newIntPpeAttr.getClampLow().getValue().getSExtValue(), std::numeric_limits<int32_t>::min());
        EXPECT_EQ(newIntPpeAttr.getClampHigh().getValue().getSExtValue(), std::numeric_limits<int32_t>::max());
        EXPECT_EQ(newIntPpeAttr.getLreluMult().getValue().getSExtValue(), 1);
        EXPECT_EQ(newIntPpeAttr.getLreluShift().getValue().getSExtValue(), 0);
        EXPECT_EQ(newIntPpeAttr.getFpPreluAlpha().getValueAsDouble(), 1.0f);
        EXPECT_EQ(newIntPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LRELU);
        EXPECT_EQ(parseFPArrayAttr<double>(newIntPpeAttr.getQuantScale())[0], static_cast<double>(0.1f));
    }

    if (auto adapter = dynamic_cast<const vpux::VPU::IPpeAdapterMode*>(_ppeIfc.get())) {
        auto newPpe = adapter->updateMode(intPpeAttr, vpux::VPU::PPEMode::LRELUX);
        auto newIntPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(newPpe);

        EXPECT_EQ(newIntPpeAttr.getClampLow().getValue().getSExtValue(), std::numeric_limits<int32_t>::min());
        EXPECT_EQ(newIntPpeAttr.getClampHigh().getValue().getSExtValue(), std::numeric_limits<int32_t>::max());
        EXPECT_EQ(newIntPpeAttr.getLreluMult().getValue().getSExtValue(), 1);
        EXPECT_EQ(newIntPpeAttr.getLreluShift().getValue().getSExtValue(), 0);
        EXPECT_EQ(newIntPpeAttr.getFpPreluAlpha().getValueAsDouble(), 1.0f);
        EXPECT_EQ(newIntPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LRELUX);
    }
}

TEST_F(NPU37xxPpeIfcUnitTest, MaxPool_InputI4_WithPostOp_RELU) {
    auto maxPoolOp =
            _operationFactory.createMaxPoolWithI4Input(_location, &_ctx, VPU_PpeUnitBase::createPostOpAttrRELU());
    ASSERT_NE(maxPoolOp, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(maxPoolOp);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), 1);
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), 3);
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 0);
    EXPECT_EQ(intPpeAttr.getFpPreluAlpha().getValueAsDouble(), 1.0f);
    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
}

TEST_F(NPU37xxPpeIfcUnitTest, MaxPool_InputFp16_WithoutPostOp) {
    vpux::IE::PostOpAttr postOpAttr = nullptr;
    auto maxPoolOp = _operationFactory.createMaxPoolWithFp16Input(_location, &_ctx, postOpAttr);
    ASSERT_NE(maxPoolOp, nullptr);

    auto ppeAttr = _ppeIfc->retrievePPEAttribute(maxPoolOp);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), std::numeric_limits<int32_t>::min());
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), std::numeric_limits<int32_t>::max());
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 0);
    EXPECT_EQ(intPpeAttr.getFpPreluAlpha().getValueAsDouble(), 1.0f);
    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
}

TEST_F(NPU37xxPpeIfcUnitTest, MaxPool_InputI4_WithoutPostOp) {
    vpux::IE::PostOpAttr postOpAttr = nullptr;

    auto maxPoolOp = _operationFactory.createMaxPoolWithI4Input(_location, &_ctx, postOpAttr);
    ASSERT_NE(maxPoolOp, nullptr);

    auto ppeAttr = _ppeIfc->retrievePPEAttribute(maxPoolOp);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), -2);
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), 3);
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 0);
    EXPECT_EQ(intPpeAttr.getFpPreluAlpha().getValueAsDouble(), 1.0f);
    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
}

TEST_F(NPU37xxPpeIfcUnitTest, MaxPool_InputFp16_WithPostOp_LPRELU) {
    auto maxPoolOp = _operationFactory.createMaxPoolWithFp16Input(_location, &_ctx,
                                                                  VPU_PpeUnitBase::createPostOpAttrLPRELU(0.25f));
    ASSERT_NE(maxPoolOp, nullptr);

    auto ppeAttr = _ppeIfc->retrievePPEAttribute(maxPoolOp);
    ASSERT_NE(ppeAttr, nullptr);

    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), std::numeric_limits<int32_t>::min());
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), std::numeric_limits<int32_t>::max());
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1024);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 12);
    EXPECT_EQ(intPpeAttr.getFpPreluAlpha().getValueAsDouble(), 0.25f);
    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LPRELU);
}

TEST_F(NPU37xxPpeIfcUnitTest, MaxPool_InputI4_WithPostOp_LPRELU) {
    auto maxPoolOp = _operationFactory.createMaxPoolWithI4Input(_location, &_ctx,
                                                                VPU_PpeUnitBase::createPostOpAttrLPRELU(0.25f));
    ASSERT_NE(maxPoolOp, nullptr);

    auto ppeAttr = _ppeIfc->retrievePPEAttribute(maxPoolOp);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), -2);
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), 3);
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1024);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 12);
    EXPECT_EQ(intPpeAttr.getFpPreluAlpha().getValueAsDouble(), 0.25f);
    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LPRELU);
}

TEST_F(NPU37xxPpeIfcUnitTest, MaxPool_InputFp16_WithPostOp_CLAMP) {
    auto maxPoolOp = _operationFactory.createMaxPoolWithFp16Input(_location, &_ctx,
                                                                  VPU_PpeUnitBase::createPostOpAttrCLAMP(0.0f, 1.25f));
    ASSERT_NE(maxPoolOp, nullptr);

    auto ppeAttr = _ppeIfc->retrievePPEAttribute(maxPoolOp);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    const auto clampLowHigh = VPU::unpackClamp<type::float16>(intPpeAttr.getClampHigh().getValue().getSExtValue());

    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), std::numeric_limits<int32_t>::min());
    EXPECT_NEAR(clampLowHigh.first, 0.0f, 1.0e-8);
    EXPECT_NEAR(clampLowHigh.second, 1.25f, 1.0e-8);
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 0);
    EXPECT_EQ(intPpeAttr.getFpPreluAlpha().getValueAsDouble(), 1);
    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LRELUX);
}

TEST_F(NPU37xxPpeIfcUnitTest, MaxPool_InputFp16_WithPostOp_CLAMP_HIGH_MAX_FP16) {
    auto maxPoolOp = _operationFactory.createMaxPoolWithFp16Input(
            _location, &_ctx, VPU_PpeUnitBase::createPostOpAttrCLAMP(0.0f, 65504.0f));
    ASSERT_NE(maxPoolOp, nullptr);

    auto ppeAttr = _ppeIfc->retrievePPEAttribute(maxPoolOp);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), std::numeric_limits<int32_t>::min());
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), std::numeric_limits<int32_t>::max());
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 0);
    EXPECT_EQ(intPpeAttr.getFpPreluAlpha().getValueAsDouble(), 1);
    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LRELU);
}

TEST_F(NPU37xxPpeIfcUnitTest, MaxPool_InputI4_WithPostOp_CLAMP) {
    auto maxPoolOp = _operationFactory.createMaxPoolWithI4Input(_location, &_ctx,
                                                                VPU_PpeUnitBase::createPostOpAttrCLAMP(-1.0f, 1.0f));
    ASSERT_NE(maxPoolOp, nullptr);

    auto ppeAttr = _ppeIfc->retrievePPEAttribute(maxPoolOp);
    ASSERT_NE(ppeAttr, nullptr);

    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), -2);
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), 3);
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 0);
    EXPECT_EQ(intPpeAttr.getFpPreluAlpha().getValueAsDouble(), 1.0f);
    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
}

TEST_F(NPU37xxPpeIfcUnitTest, Convolution_InputFp16_WithPostOp_RELU) {
    auto convolutionOp =
            _operationFactory.createConvolutionFp16Input(_location, &_ctx, VPU_PpeUnitBase::createPostOpAttrRELU());
    ASSERT_NE(convolutionOp, nullptr);

    auto ppeAttr = _ppeIfc->retrievePPEAttribute(convolutionOp);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), std::numeric_limits<int32_t>::min());
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), std::numeric_limits<int32_t>::max());
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 0);
    EXPECT_EQ(intPpeAttr.getFpPreluAlpha().getValueAsDouble(), 1.0f);
    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LRELU);
}

TEST_F(NPU37xxPpeIfcUnitTest, Convolution_InputI4_WithPostOp_RELU) {
    auto convolutionOp =
            _operationFactory.createConvolutionI4Input(_location, &_ctx, VPU_PpeUnitBase::createPostOpAttrRELU());
    ASSERT_NE(convolutionOp, nullptr);

    auto ppeAttr = _ppeIfc->retrievePPEAttribute(convolutionOp);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), 0);
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), 7);
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 0);
    EXPECT_EQ(intPpeAttr.getFpPreluAlpha().getValueAsDouble(), 1.0f);
    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
}

TEST_F(NPU37xxPpeIfcUnitTest, Convolution_InputFp16_WithoutPostOp) {
    vpux::IE::PostOpAttr postOpAttr = nullptr;
    auto convolutionOp = _operationFactory.createConvolutionFp16Input(_location, &_ctx, postOpAttr);
    ASSERT_NE(convolutionOp, nullptr);

    auto ppeAttr = _ppeIfc->retrievePPEAttribute(convolutionOp);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), std::numeric_limits<int32_t>::min());
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), std::numeric_limits<int32_t>::max());
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 0);
    EXPECT_EQ(intPpeAttr.getFpPreluAlpha().getValueAsDouble(), 1.0f);
    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
}

TEST_F(NPU37xxPpeIfcUnitTest, Convolution_InputI4_WithoutPostOp) {
    vpux::IE::PostOpAttr postOpAttr = nullptr;

    auto convolutionOp = _operationFactory.createConvolutionI4Input(_location, &_ctx, postOpAttr);
    ASSERT_NE(convolutionOp, nullptr);

    auto ppeAttr = _ppeIfc->retrievePPEAttribute(convolutionOp);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), -7);
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), 7);
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 0);
    EXPECT_EQ(intPpeAttr.getFpPreluAlpha().getValueAsDouble(), 1.0f);
    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
}

TEST_F(NPU37xxPpeIfcUnitTest, Convolution_InputFp16_WithPostOp_LPRELU) {
    auto convolutionOp = _operationFactory.createConvolutionFp16Input(_location, &_ctx,
                                                                      VPU_PpeUnitBase::createPostOpAttrLPRELU(0.25f));
    ASSERT_NE(convolutionOp, nullptr);

    auto ppeAttr = _ppeIfc->retrievePPEAttribute(convolutionOp);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), std::numeric_limits<int32_t>::min());
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), std::numeric_limits<int32_t>::max());
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1024);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 12);
    EXPECT_EQ(intPpeAttr.getFpPreluAlpha().getValueAsDouble(), 0.25f);
    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LPRELU);
}

TEST_F(NPU37xxPpeIfcUnitTest, Convolution_InputI4_WithPostOp_LPRELU) {
    auto convolutionOp = _operationFactory.createConvolutionI4Input(_location, &_ctx,
                                                                    VPU_PpeUnitBase::createPostOpAttrLPRELU(0.25f));
    ASSERT_NE(convolutionOp, nullptr);

    auto ppeAttr = _ppeIfc->retrievePPEAttribute(convolutionOp);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), -7);
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), 7);
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1024);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 12);
    EXPECT_EQ(intPpeAttr.getFpPreluAlpha().getValueAsDouble(), 0.25f);
    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LPRELU);
}

TEST_F(NPU37xxPpeIfcUnitTest, AvgPool_InputFp16_WithPostOp_RELU) {
    auto avgPoolOp =
            _operationFactory.createAvgPoolWithFp16Input(_location, &_ctx, VPU_PpeUnitBase::createPostOpAttrRELU());
    ASSERT_NE(avgPoolOp, nullptr);

    auto ppeAttr = _ppeIfc->retrievePPEAttribute(avgPoolOp);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), std::numeric_limits<int32_t>::min());
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), std::numeric_limits<int32_t>::max());
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 0);
    EXPECT_EQ(intPpeAttr.getFpPreluAlpha().getValueAsDouble(), 1.0f);
    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LRELU);
}

TEST_F(NPU37xxPpeIfcUnitTest, AvgPool_InputI4_WithPostOp_RELU) {
    auto avgPoolOp =
            _operationFactory.createAvgPoolWithI4Input(_location, &_ctx, VPU_PpeUnitBase::createPostOpAttrRELU());
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(avgPoolOp);

    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), 0);
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), 7);
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 0);
    EXPECT_EQ(intPpeAttr.getFpPreluAlpha().getValueAsDouble(), 1.0f);
    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
}

TEST_F(NPU37xxPpeIfcUnitTest, AvgPool_InputFp16_WithoutPostOp) {
    vpux::IE::PostOpAttr postOpAttr = nullptr;
    auto avgPool = _operationFactory.createAvgPoolWithFp16Input(_location, &_ctx, postOpAttr);
    ASSERT_NE(avgPool, nullptr);

    auto ppeAttr = _ppeIfc->retrievePPEAttribute(avgPool);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), std::numeric_limits<int32_t>::min());
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), std::numeric_limits<int32_t>::max());
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 0);
    EXPECT_EQ(intPpeAttr.getFpPreluAlpha().getValueAsDouble(), 1.0f);
    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
}

TEST_F(NPU37xxPpeIfcUnitTest, AvgPool_InputI4_WithoutPostOp) {
    vpux::IE::PostOpAttr postOpAttr = nullptr;

    auto avgPoolOp = _operationFactory.createAvgPoolWithI4Input(_location, &_ctx, postOpAttr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(avgPoolOp);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), -7);
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), 7);
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 0);
    EXPECT_EQ(intPpeAttr.getFpPreluAlpha().getValueAsDouble(), 1.0f);
    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
}

TEST_F(NPU37xxPpeIfcUnitTest, AvgPool_InputFp16_WithPostOp_LPRELU) {
    auto avgPoolOp = _operationFactory.createAvgPoolWithFp16Input(_location, &_ctx, createPostOpAttrLPRELU(0.25f));
    ASSERT_NE(avgPoolOp, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(avgPoolOp);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), std::numeric_limits<int32_t>::min());
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), std::numeric_limits<int32_t>::max());
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1024);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 12);
    EXPECT_EQ(intPpeAttr.getFpPreluAlpha().getValueAsDouble(), 0.25f);
    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LPRELU);
    EXPECT_EQ(parseFPArrayAttr<double>(intPpeAttr.getQuantScale())[0], 1 / (2.0f * 2.0f));  // quant scale
}

TEST_F(NPU37xxPpeIfcUnitTest, AvgPool_InputFp16I8Output_WithPostOp_LPRELU) {
    auto avgPoolOp =
            _operationFactory.createAvgPoolWithFp16InputI8Output(_location, &_ctx, createPostOpAttrLPRELU(0.1f));
    ASSERT_NE(avgPoolOp, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(avgPoolOp);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), -128);
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), 127);
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1638);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 14);
    EXPECT_NEAR(intPpeAttr.getFpPreluAlpha().getValueAsDouble(), 2.0f, 1.0e-8);
    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LPRELU);
    EXPECT_NEAR(parseFPArrayAttr<double>(intPpeAttr.getQuantScale())[0], 5.0f, 1.0e-8);
    ppeAttr.print(_output);
}

TEST_F(NPU37xxPpeIfcUnitTest, AvgPool_InputI4_WithPostOp_LPRELU) {
    auto avgPoolOp = _operationFactory.createAvgPoolWithI4Input(_location, &_ctx, createPostOpAttrLPRELU(0.25f));
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(avgPoolOp);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), -7);
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), 7);
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1024);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 12);
    EXPECT_EQ(intPpeAttr.getFpPreluAlpha().getValueAsDouble(), 0.25f);
    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LPRELU);
}

TEST_F(NPU37xxPpeIfcUnitTest, AddOp_InputFp16_WithPostOp_RELU) {
    auto addOp = _operationFactory.createAddWithFp16Input(_location, &_ctx, VPU_PpeUnitBase::createPostOpAttrRELU());
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(addOp);

    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), std::numeric_limits<int32_t>::min());
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), std::numeric_limits<int32_t>::max());

    ppeAttr.print(_output);
    EXPECT_EQ(
            _ppeAttrIR,
            "#VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 "
            ": i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>")
            << "Unexpected PPE attribute IR!";
}

TEST_F(NPU37xxPpeIfcUnitTest, AddOp_InputI4_WithPostOp_RELU) {
    auto addOp = _operationFactory.createAddWithI4Input(_location, &_ctx, VPU_PpeUnitBase::createPostOpAttrRELU());
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(addOp);

    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    ppeAttr.print(_output);
    EXPECT_EQ(_ppeAttrIR,
              "#VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 7 : i64, lrelu_mult = 1 : i64, lrelu_shift "
              "= 0 : i64, quant_mult = [20480], quant_shift = [29], quant_post_shift = 0 : i64, in1_quant_mult = "
              "[26214], in2_quant_mult = [26214], fp_prelu_alpha = 1.000000e+00 : f64>")
            << "Unexpected PPE attribute IR!";
}

TEST_F(NPU37xxPpeIfcUnitTest, AddOp_InputFp16_WithoutPostOp) {
    vpux::IE::PostOpAttr postOpAttr = nullptr;
    auto addOp = _operationFactory.createAddWithFp16Input(_location, &_ctx, postOpAttr);

    auto ppeAttr = _ppeIfc->retrievePPEAttribute(addOp);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), std::numeric_limits<int32_t>::min());
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), std::numeric_limits<int32_t>::max());

    ppeAttr.print(_output);
    EXPECT_EQ(_ppeAttrIR,
              "#VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 "
              ": i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>")
            << "Unexpected PPE attribute IR!";
}

TEST_F(NPU37xxPpeIfcUnitTest, AddOp_InputI4_WithoutPostOp) {
    vpux::IE::PostOpAttr postOpAttr = nullptr;

    auto addOp = _operationFactory.createAddWithI4Input(_location, &_ctx, postOpAttr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(addOp);

    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    ppeAttr.print(_output);
    EXPECT_EQ(
            _ppeAttrIR,
            "#VPU.PPEInt<mode = <NOOP>, clamp_low = -7 : i64, clamp_high = 7 : i64, lrelu_mult = 1 : i64, lrelu_shift "
            "= 0 : i64, quant_mult = [20480], quant_shift = [29], quant_post_shift = 0 : i64, in1_quant_mult = "
            "[26214], in2_quant_mult = [26214], fp_prelu_alpha = 1.000000e+00 : f64>")
            << "Unexpected PPE attribute IR!";
}

TEST_F(NPU37xxPpeIfcUnitTest, AddOp_InputFp16_WithPostOp_LPRELU) {
    auto addOp =
            _operationFactory.createAddWithFp16Input(_location, &_ctx, VPU_PpeUnitBase::createPostOpAttrLPRELU(0.25f));
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(addOp);

    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), std::numeric_limits<int32_t>::min());
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), std::numeric_limits<int32_t>::max());

    ppeAttr.print(_output);
    EXPECT_EQ(_ppeAttrIR,
              "#VPU.PPEInt<mode = <LPRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = "
              "1024 : i64, lrelu_shift = 12 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 2.500000e-01 : f64>")
            << "Unexpected PPE attribute IR!";
}

TEST_F(NPU37xxPpeIfcUnitTest, AddOp_InputI4_WithPostOp_LPRELU) {
    auto addOp =
            _operationFactory.createAddWithI4Input(_location, &_ctx, VPU_PpeUnitBase::createPostOpAttrLPRELU(0.25f));
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(addOp);

    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    ppeAttr.print(_output);
    EXPECT_EQ(_ppeAttrIR,
              "#VPU.PPEInt<mode = <LPRELU>, clamp_low = -7 : i64, clamp_high = 7 : i64, lrelu_mult = 1024 : i64, "
              "lrelu_shift = 12 : i64, quant_mult = [20480], quant_shift = [29], quant_post_shift = 0 : i64, "
              "in1_quant_mult = [26214], in2_quant_mult = [26214], fp_prelu_alpha = 2.500000e-01 : f64>")
            << "Unexpected PPE attribute IR!";
}
