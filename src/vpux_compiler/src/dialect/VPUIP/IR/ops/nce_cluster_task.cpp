//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/attributes/dim.hpp"
#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

//
// NCEClusterTaskOp::build
//

void vpux::VPUIP::NCEClusterTaskOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type output,
                                          mlir::Type output_sparsity_map, mlir::Type profiling_output,
                                          mlir::ValueRange operands, ArrayRef<mlir::NamedAttribute> attributes) {
    assert(operands.size() >= 4u && "mismatched number of parameters");
    state.addOperands(operands);

    auto& props = state.getOrAddProperties<vpux::VPUIP::NCEClusterTaskOp::Properties>();

    auto dict = mlir::DictionaryAttr::get(builder.getContext(), attributes);
    VPUX_THROW_UNLESS(vpux::VPUIP::NCEClusterTaskOp::setPropertiesFromAttr(props, dict, nullptr).succeeded(),
                      "Cannot initialize NCEClusterTaskOp::Properties from attribute '{0}'!", dict);

    // Compute value for resultSegmentSizes attribute and add it to the properties
    int32_t output_sparsity_map_val = (output_sparsity_map != nullptr) ? 1 : 0;
    int32_t profiling_output_val = (profiling_output != nullptr) ? 1 : 0;
    props.setResultSegmentSizes({1, output_sparsity_map_val, profiling_output_val});

    for (unsigned i = 0; i != 2; ++i) {
        (void)state.addRegion();
    }

    state.addTypes(output);

    if (output_sparsity_map != nullptr) {
        state.addTypes(output_sparsity_map);
    }

    if (profiling_output != nullptr) {
        state.addTypes(profiling_output);
    }
}

void vpux::VPUIP::NCEClusterTaskOp::build(
        mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input, mlir::Value weights,
        mlir::Value weight_table, mlir::Value instruction_list_table, mlir::Value spr_lookup_table,
        mlir::Value activation_window, mlir::Value parent_input, mlir::Value parent_output, mlir::Value output_buff,
        vpux::VPUIP::NCETaskType task_type, mlir::ArrayAttr kernel_size, mlir::ArrayAttr kernel_strides,
        vpux::VPU::PaddingAttr kernel_padding, mlir::IntegerAttr activation_window_channel_length,
        mlir::UnitAttr is_continued, mlir::IntegerAttr cm_sp_pattern, mlir::UnitAttr is_segmented,
        mlir::IntegerAttr out_channel_offset, mlir::UnitAttr input_channels_compression, mlir::UnitAttr is_superdense,
        mlir::BoolAttr is_inplace, mlir::IntegerAttr input_se_size, mlir::IntegerAttr output_se_size,
        mlir::UnitAttr isPermuteQuantize, mlir::UnitAttr isSmallKernelOptimized) {
    build(builder, state, output_buff.getType(), nullptr, nullptr, input, nullptr, nullptr, weights, nullptr,
          weight_table, instruction_list_table, spr_lookup_table, activation_window, parent_input, nullptr, nullptr,
          parent_output, nullptr, mlir::ValueRange(), output_buff, nullptr, nullptr, task_type, kernel_size,
          kernel_strides, kernel_padding, activation_window_channel_length, is_continued, cm_sp_pattern, is_segmented,
          out_channel_offset, input_channels_compression, is_superdense, is_inplace, input_se_size, output_se_size,
          isPermuteQuantize, isSmallKernelOptimized);
}

void vpux::VPUIP::NCEClusterTaskOp::build(
        mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type output, mlir::Value input,
        mlir::Value weights, mlir::Value weight_table, mlir::Value instruction_list_table, mlir::Value spr_lookup_table,
        mlir::Value activation_window, mlir::Value parent_input, mlir::Value parent_output, mlir::Value output_buff,
        vpux::VPUIP::NCETaskType task_type, mlir::ArrayAttr kernel_size, mlir::ArrayAttr kernel_strides,
        vpux::VPU::PaddingAttr kernel_padding, mlir::IntegerAttr activation_window_channel_length,
        mlir::UnitAttr is_continued, mlir::IntegerAttr cm_sp_pattern, mlir::UnitAttr is_segmented,
        mlir::IntegerAttr out_channel_offset, mlir::UnitAttr input_channels_compression, mlir::UnitAttr is_superdense,
        mlir::BoolAttr is_inplace, mlir::IntegerAttr input_se_size, mlir::IntegerAttr output_se_size,
        mlir::UnitAttr isPermuteQuantize, mlir::UnitAttr isSmallKernelOptimized) {
    build(builder, state, output, nullptr, nullptr, input, nullptr, nullptr, weights, nullptr, weight_table,
          instruction_list_table, spr_lookup_table, activation_window, parent_input, nullptr, nullptr, parent_output,
          nullptr, mlir::ValueRange(), output_buff, nullptr, nullptr, task_type, kernel_size, kernel_strides,
          kernel_padding, activation_window_channel_length, is_continued, cm_sp_pattern, is_segmented,
          out_channel_offset, input_channels_compression, is_superdense, is_inplace, input_se_size, output_se_size,
          isPermuteQuantize, isSmallKernelOptimized);
}

void vpux::VPUIP::NCEClusterTaskOp::build(
        mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input, mlir::Value weights,
        mlir::Value weight_table, mlir::Value instruction_list_table, mlir::Value spr_lookup_table,
        mlir::Value activation_window, mlir::Value parent_input, mlir::Value parent_output, mlir::Value output_buff,
        mlir::Value profiling_data, vpux::VPUIP::NCETaskType task_type, mlir::ArrayAttr kernel_size,
        mlir::ArrayAttr kernel_strides, vpux::VPU::PaddingAttr kernel_padding,
        mlir::IntegerAttr activation_window_channel_length, mlir::UnitAttr is_continued,
        mlir::IntegerAttr cm_sp_pattern, mlir::UnitAttr is_segmented, mlir::IntegerAttr out_channel_offset,
        mlir::UnitAttr input_channels_compression, mlir::UnitAttr is_superdense, mlir::BoolAttr is_inplace,
        mlir::IntegerAttr input_se_size, mlir::IntegerAttr output_se_size, mlir::UnitAttr isPermuteQuantize,
        mlir::UnitAttr isSmallKernelOptimized) {
    build(builder, state, output_buff.getType(), nullptr, profiling_data ? profiling_data.getType() : nullptr, input,
          nullptr, nullptr, weights, nullptr, weight_table, instruction_list_table, spr_lookup_table, activation_window,
          parent_input, nullptr, nullptr, parent_output, nullptr, mlir::ValueRange(), output_buff, nullptr,
          profiling_data, task_type, kernel_size, kernel_strides, kernel_padding, activation_window_channel_length,
          is_continued, cm_sp_pattern, is_segmented, out_channel_offset, input_channels_compression, is_superdense,
          is_inplace, input_se_size, output_se_size, isPermuteQuantize, isSmallKernelOptimized);
}

void vpux::VPUIP::NCEClusterTaskOp::build(
        mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type output, mlir::Type profiling_output,
        mlir::Value input, mlir::Value weights, mlir::Value weight_table, mlir::Value instruction_list_table,
        mlir::Value spr_lookup_table, mlir::Value activation_window, mlir::Value parent_input,
        mlir::Value parent_output, mlir::Value output_buff, mlir::Value profiling_data,
        vpux::VPUIP::NCETaskType task_type, mlir::ArrayAttr kernel_size, mlir::ArrayAttr kernel_strides,
        vpux::VPU::PaddingAttr kernel_padding, mlir::IntegerAttr activation_window_channel_length,
        mlir::UnitAttr is_continued, mlir::IntegerAttr cm_sp_pattern, mlir::UnitAttr is_segmented,
        mlir::IntegerAttr out_channel_offset, mlir::UnitAttr input_channels_compression, mlir::UnitAttr is_superdense,
        mlir::BoolAttr is_inplace, mlir::IntegerAttr input_se_size, mlir::IntegerAttr output_se_size,
        mlir::UnitAttr isPermuteQuantize, mlir::UnitAttr isSmallKernelOptimized) {
    build(builder, state, output, nullptr, profiling_output, input, nullptr, nullptr, weights, nullptr, weight_table,
          instruction_list_table, spr_lookup_table, activation_window, parent_input, nullptr, nullptr, parent_output,
          nullptr, mlir::ValueRange(), output_buff, nullptr, profiling_data, task_type, kernel_size, kernel_strides,
          kernel_padding, activation_window_channel_length, is_continued, cm_sp_pattern, is_segmented,
          out_channel_offset, input_channels_compression, is_superdense, is_inplace, input_se_size, output_se_size,
          isPermuteQuantize, isSmallKernelOptimized);
}

void vpux::VPUIP::NCEClusterTaskOp::build(
        mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input, mlir::Value input_sparsity_map,
        mlir::Value input_storage_element_table, mlir::Value weights, mlir::Value weights_sparsity_map,
        mlir::Value weight_table, mlir::Value instruction_list_table, mlir::Value spr_lookup_table,
        mlir::Value activation_window, mlir::Value parent_input, mlir::Value parent_input_sparsity_map,
        mlir::Value parent_input_storage_element_table, mlir::Value parent_output,
        mlir::Value parent_output_sparsity_map, mlir::Value output_buff, mlir::Value output_sparsity_map_buff,
        mlir::Value profiling_data, vpux::VPUIP::NCETaskType task_type, mlir::ArrayAttr kernel_size,
        mlir::ArrayAttr kernel_strides, vpux::VPU::PaddingAttr kernel_padding,
        mlir::IntegerAttr activation_window_channel_length, mlir::UnitAttr is_continued,
        mlir::IntegerAttr cm_sp_pattern, mlir::UnitAttr is_segmented, mlir::IntegerAttr out_channel_offset,
        mlir::UnitAttr input_channels_compression, mlir::UnitAttr is_superdense, mlir::BoolAttr is_inplace,
        mlir::IntegerAttr input_se_size, mlir::IntegerAttr output_se_size, mlir::UnitAttr isPermuteQuantize,
        mlir::UnitAttr isSmallKernelOptimized) {
    build(builder, state, output_buff.getType(),
          output_sparsity_map_buff ? output_sparsity_map_buff.getType() : nullptr,
          (profiling_data != nullptr) ? profiling_data.getType() : nullptr, input, input_sparsity_map,
          input_storage_element_table, weights, weights_sparsity_map, weight_table, instruction_list_table,
          spr_lookup_table, activation_window, parent_input, parent_input_sparsity_map,
          parent_input_storage_element_table, parent_output, parent_output_sparsity_map, mlir::ValueRange(),
          output_buff, output_sparsity_map_buff, profiling_data, task_type, kernel_size, kernel_strides, kernel_padding,
          activation_window_channel_length, is_continued, cm_sp_pattern, is_segmented, out_channel_offset,
          input_channels_compression, is_superdense, is_inplace, input_se_size, output_se_size, isPermuteQuantize,
          isSmallKernelOptimized);
}

void vpux::VPUIP::NCEClusterTaskOp::build(
        mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type output, mlir::Type output_sparsity_map,
        mlir::Type profiling_output, mlir::Value input, mlir::Value input_sparsity_map,
        mlir::Value input_storage_element_table, mlir::Value weights, mlir::Value weights_sparsity_map,
        mlir::Value weight_table, mlir::Value instruction_list_table, mlir::Value spr_lookup_table,
        mlir::Value activation_window, mlir::Value parent_input, mlir::Value parent_input_sparsity_map,
        mlir::Value parent_input_storage_element_table, mlir::Value parent_output,
        mlir::Value parent_output_sparsity_map, mlir::Value output_buff, mlir::Value output_sparsity_map_buff,
        mlir::Value profiling_data, vpux::VPUIP::NCETaskType task_type, mlir::ArrayAttr kernel_size,
        mlir::ArrayAttr kernel_strides, vpux::VPU::PaddingAttr kernel_padding,
        mlir::IntegerAttr activation_window_channel_length, mlir::UnitAttr is_continued,
        mlir::IntegerAttr cm_sp_pattern, mlir::UnitAttr is_segmented, mlir::IntegerAttr out_channel_offset,
        mlir::UnitAttr input_channels_compression, mlir::UnitAttr is_superdense, mlir::BoolAttr is_inplace,
        mlir::IntegerAttr input_se_size, mlir::IntegerAttr output_se_size, mlir::UnitAttr isPermuteQuantize,
        mlir::UnitAttr isSmallKernelOptimized) {
    build(builder, state, output, output_sparsity_map, profiling_output, input, input_sparsity_map,
          input_storage_element_table, weights, weights_sparsity_map, weight_table, instruction_list_table,
          spr_lookup_table, activation_window, parent_input, parent_input_sparsity_map,
          parent_input_storage_element_table, parent_output, parent_output_sparsity_map, mlir::ValueRange(),
          output_buff, output_sparsity_map_buff, profiling_data, task_type, kernel_size, kernel_strides, kernel_padding,
          activation_window_channel_length, is_continued, cm_sp_pattern, is_segmented, out_channel_offset,
          input_channels_compression, is_superdense, is_inplace, input_se_size, output_se_size, isPermuteQuantize,
          isSmallKernelOptimized);
}

void vpux::VPUIP::NCEClusterTaskOp::build(
        mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type output, mlir::Type output_sparsity_map,
        mlir::Type profiling_output, mlir::Value input, mlir::Value input_sparsity_map,
        mlir::Value input_storage_element_table, mlir::Value weights, mlir::Value weights_sparsity_map,
        mlir::Value weight_table, mlir::Value instruction_list_table, mlir::Value spr_lookup_table,
        mlir::Value activation_window, mlir::Value parent_input, mlir::Value parent_input_sparsity_map,
        mlir::Value parent_input_storage_element_table, mlir::Value parent_output,
        mlir::Value parent_output_sparsity_map, mlir::ValueRange output_ITI_buff, mlir::Value output_buff,
        mlir::Value output_sparsity_map_buff, mlir::Value profiling_data, vpux::VPUIP::NCETaskType task_type,
        mlir::ArrayAttr kernel_size, mlir::ArrayAttr kernel_strides, vpux::VPU::PaddingAttr kernel_padding,
        mlir::IntegerAttr activation_window_channel_length, mlir::UnitAttr is_continued,
        mlir::IntegerAttr cm_sp_pattern, mlir::UnitAttr is_segmented, mlir::IntegerAttr out_channel_offset,
        mlir::UnitAttr input_channels_compression, mlir::UnitAttr is_superdense, mlir::BoolAttr is_inplace,
        mlir::IntegerAttr input_se_size, mlir::IntegerAttr output_se_size, mlir::UnitAttr isPermuteQuantize,
        mlir::UnitAttr isSmallKernelOptimized) {
    auto taskTypeAttr = vpux::VPUIP::NCETaskTypeAttr::get(builder.getContext(), task_type);

    build(builder, state, output, output_sparsity_map, profiling_output, input, input_sparsity_map,
          input_storage_element_table, weights, weights_sparsity_map, weight_table, instruction_list_table,
          spr_lookup_table, activation_window, parent_input, parent_input_sparsity_map,
          parent_input_storage_element_table, parent_output, parent_output_sparsity_map, output_ITI_buff, output_buff,
          output_sparsity_map_buff, profiling_data, taskTypeAttr, kernel_size, kernel_strides, kernel_padding,
          activation_window_channel_length, is_continued, cm_sp_pattern, is_segmented, out_channel_offset,
          input_channels_compression, is_superdense, is_inplace, input_se_size, output_se_size, isPermuteQuantize,
          isSmallKernelOptimized, /*profilingMetadata=*/nullptr);

    // The auto-generated builders don't populate the regions even if SizedRegion<1> is specified.
    for (auto& region : state.regions) {
        region->emplaceBlock();
    }
}

//
// NCEClusterTaskOp::addDPUTask
//

VPUIP::DPUTaskOp vpux::VPUIP::NCEClusterTaskOp::addDPUTask(mlir::OpBuilder& builder, mlir::ArrayAttr outStart,
                                                           mlir::ArrayAttr outEnd, mlir::ArrayAttr inStart,
                                                           mlir::ArrayAttr inEnd, VPU::PaddingAttr pad,
                                                           VPU::MPEMode mpeMode, mlir::IntegerAttr clusterId) {
    if (getVariants().empty()) {
        getVariants().emplaceBlock();
    }

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&getVariants().front());

    return builder.create<VPUIP::DPUTaskOp>(getLoc(), outStart, outEnd, inStart, inEnd, pad, mpeMode, clusterId);
}

VPUIP::DPUTaskOp vpux::VPUIP::NCEClusterTaskOp::addDPUTask(mlir::OpBuilder& builder, mlir::ArrayAttr outStart,
                                                           mlir::ArrayAttr outEnd, mlir::ArrayAttr inStart,
                                                           mlir::ArrayAttr inEnd, VPU::PaddingAttr pad,
                                                           VPU::MPEMode mpeMode, mlir::IntegerAttr clusterId,
                                                           mlir::ArrayAttr haloRegions) {
    if (getVariants().empty()) {
        getVariants().emplaceBlock();
    }

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&getVariants().front());

    return builder.create<VPUIP::DPUTaskOp>(getLoc(), outStart, outEnd, inStart, inEnd, pad, mpeMode, clusterId,
                                            haloRegions);
}

VPUIP::DPUTaskOp vpux::VPUIP::NCEClusterTaskOp::addDPUTask(mlir::OpBuilder& builder, mlir::ArrayAttr outStart,
                                                           mlir::ArrayAttr outEnd, VPU::PaddingAttr pad,
                                                           VPU::MPEMode mpeMode, mlir::IntegerAttr clusterId) {
    if (getVariants().empty()) {
        getVariants().emplaceBlock();
    }

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&getVariants().front());

    return builder.create<VPUIP::DPUTaskOp>(getLoc(), outStart, outEnd, pad, mpeMode, clusterId);
}

//
// NCEClusterTaskOp::getNumVariants
//

int64_t vpux::VPUIP::NCEClusterTaskOp::getNumVariants() {
    return getVariants().getBlocks().front().getOperations().size();
}

size_t vpux::VPUIP::NCEClusterTaskOp::getOperationCycleCost(std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    auto nceOp = mlir::cast<VPUIP::NCEClusterTaskOp>(this->getOperation());

    auto module = nceOp->getParentOfType<mlir::ModuleOp>();
    // TODO: Expose API to get arch from cost model
    auto arch = VPU::getArch(module);
    auto tileOp = IE::getTileExecutor(module);
    VPUX_THROW_WHEN(tileOp == nullptr, "Couldn't get TileExecutor for module");

    vpux::Logger log = Logger::global();
    auto dpuCount = tileOp.getSubExecutor(VPU::ExecutorKind::DPU).getCount();
    return checked_cast<size_t>(calculateNceCycles(nceOp, costModel, arch, log, dpuCount));
}

//
// NCEClusterTaskOp::inferReturnTypes
//

mlir::LogicalResult vpux::VPUIP::NCEClusterTaskOp::inferReturnTypes(
        mlir::MLIRContext*, std::optional<mlir::Location>, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::OpaqueProperties props, mlir::RegionRange ranges,
        llvm::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    VPUIP::NCEClusterTaskOpAdaptor adaptor(operands, attrs, props, ranges);
    inferredReturnTypes.push_back(adaptor.getOutputBuff().getType());
    if (adaptor.getOutputSparsityMapBuff() != nullptr) {
        inferredReturnTypes.push_back(adaptor.getOutputSparsityMapBuff().getType());
    }
    if (adaptor.getProfilingData() != nullptr) {
        inferredReturnTypes.push_back(adaptor.getProfilingData().getType());
    }
    return mlir::success();
}

//
// verify
//

namespace {

mlir::LogicalResult verifyInOutOrder(mlir::Operation* op, const VPU::ArchKind& arch, const std::string& opName) {
    if (arch != VPU::ArchKind::NPU37XX && arch != VPU::ArchKind::NPU40XX) {
        if (vpux::VPUIP::verifySameInOutSpecificDimsOrder(op, {DimsOrder::NHWC}).failed()) {
            return errorAt(op, "{0} expected the same input/output layout", opName);
        }
    } else {
        const auto inOrder = DimsOrder::fromValue(op->getOperand(0));
        if (inOrder != DimsOrder::NHWC) {
            return errorAt(op, "{0} input must have NHWC layout, got '{1}'", opName, inOrder);
        }
    }

    return mlir::success();
}

mlir::LogicalResult verifyNCEConv(VPUIP::NCEClusterTaskOp op, VPU::ArchKind arch) {
    VPUX_THROW_UNLESS(op.getTaskType() == VPUIP::NCETaskType::CONV || op.getTaskType() == VPUIP::NCETaskType::CMCONV,
                      "Expected task type '{0}' or '{1}', but got '{2}'", VPUIP::NCETaskType::CONV,
                      VPUIP::NCETaskType::CMCONV, op.getTaskType());

    if (op.getWeights() == nullptr) {
        return errorAt(op, "weights is required for NCETaskType : '{0}'", op.getTaskType());
    }
    if (op.getWeightTable() == nullptr) {
        return errorAt(op, "weight_table is required for NCETaskType : '{0}'", op.getTaskType());
    }
    if (op.getTaskType() == VPUIP::NCETaskType::CMCONV) {
        if (op.getActivationWindow() == nullptr) {
            return errorAt(op, "activation_window is required for NCETaskType : '{0}'", op.getTaskType());
        }
    }

    if (op.getKernelSizeAttr() == nullptr) {
        return errorAt(op, "kernel_size is required for NCETaskType : '{0}'", op.getTaskType());
    }
    if (op.getKernelStridesAttr() == nullptr) {
        return errorAt(op, "kernel_strides is required for NCETaskType : '{0}'", op.getTaskType());
    }
    if (op.getKernelPaddingAttr() == nullptr) {
        return errorAt(op, "kernel_padding is required for NCETaskType : '{0}'", op.getTaskType());
    }
    const auto kernelSize = parseIntArrayAttr<int64_t>(op.getKernelSizeAttr());
    const auto KY = kernelSize[0];
    const auto KX = kernelSize[1];

    const auto kernelStrides = parseIntArrayAttr<int64_t>(op.getKernelStridesAttr());
    const auto SY = kernelStrides[0];
    const auto SX = kernelStrides[1];

    const auto kernelPadding = op.getKernelPaddingAttr();
    const auto padLeft = kernelPadding.getLeft().getInt();
    const auto padRight = kernelPadding.getRight().getInt();
    const auto padTop = kernelPadding.getTop().getInt();
    const auto padBottom = kernelPadding.getBottom().getInt();
    if (mlir::failed(VPU::NCEInvariant::verifyKernel(op->getLoc(), KY, KX, SY, SX, padTop, padBottom, padLeft, padRight,
                                                     arch))) {
        return errorAt(op, "Kernel verification failed");
    }

    const auto weightsShape = getShape(op.getWeights());
    const auto OC = weightsShape[Dims4D::Filter::OC];

    const auto weightTableShape = getShape(op.getWeightTable());
    const auto weightTableNumElements = weightTableShape.totalSize();

    if (OC * VPUIP::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC > weightTableNumElements) {
        return errorAt(op, "Weight table must have elements greater than or equal to '{0}', got '{1}'",
                       OC * VPUIP::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC, weightTableNumElements);
    }

    const auto inOrder = DimsOrder::fromValue(op.getInput());
    const auto weightsOrder = DimsOrder::fromValue(op.getWeights());
    const auto outOrder = DimsOrder::fromValue(op.getOutputBuff());

    if (op.getTaskType() == VPUIP::NCETaskType::CONV && (inOrder != DimsOrder::NHWC && inOrder != DimsOrder::GNHWC)) {
        return errorAt(op, "For NCE z-major convolution input must have NHWC or GNHWC layout, got '{0}'", inOrder);
    }
    if (op.getTaskType() == VPUIP::NCETaskType::CMCONV && inOrder != DimsOrder::NCHW) {
        return errorAt(op, "For NCE c-major convolution input must have NCHW layout, got '{0}'", inOrder);
    }
    if (weightsOrder != DimsOrder::OYXI && weightsOrder != DimsOrder::GOYXI) {
        return errorAt(op, "For NCE convolution weights must have OYXI layout, got '{0}'", weightsOrder);
    }
    if (arch != VPU::ArchKind::NPU37XX && arch != VPU::ArchKind::NPU40XX && outOrder != DimsOrder::NHWC &&
        outOrder != DimsOrder::GNHWC) {
        return errorAt(op, "For NCE convolution output must have NHWC or GNHWC layout, got '{0}'", outOrder);
    }

    const auto outputShape = getShape(op.getOutput());

    const auto isOutput5d = outputShape.size() == 5;

    if (isOutput5d) {  // if 5D, then that is grouped matmul and checks below are not applicable
        return mlir::success();
    }

    const auto batch = outputShape[Dims4D::Act::N];
    if (batch != vpux::VPU::NCEInvariant::SUPPORTED_BATCH_SIZE) {
        if (arch < VPU::ArchKind::NPU37XX) {
            return errorAt(op, "Got unsupported input batch '{0}' expected '{1}'", batch,
                           vpux::VPU::NCEInvariant::SUPPORTED_BATCH_SIZE);
        }
        if (batch > vpux::VPU::getMaxArchDPUClusterNum(arch)) {
            return errorAt(op, "Got unsupported input batch '{0}' expected to be less than or equal to '{1}'", batch,
                           vpux::VPU::getMaxArchDPUClusterNum(arch));
        }
    }

    return mlir::success();
}

mlir::LogicalResult verifyNCEPool(VPUIP::NCEClusterTaskOp op, VPU::ArchKind arch) {
    VPUX_THROW_UNLESS(
            op.getTaskType() == VPUIP::NCETaskType::AVEPOOL || op.getTaskType() == VPUIP::NCETaskType::MAXPOOL,
            "Expected task type '{0}' or '{1}', but got '{2}'", VPUIP::NCETaskType::AVEPOOL,
            VPUIP::NCETaskType::MAXPOOL, op.getTaskType());

    if (op.getKernelSizeAttr() == nullptr) {
        return errorAt(op, "kernel_size is required for NCETaskType : '{0}'", op.getTaskType());
    }
    if (op.getKernelStridesAttr() == nullptr) {
        return errorAt(op, "kernel_strides is required for NCETaskType : '{0}'", op.getTaskType());
    }
    if (op.getKernelPaddingAttr() == nullptr) {
        return errorAt(op, "kernel_padding is required for NCETaskType : '{0}'", op.getTaskType());
    }

    const auto kernelSize = parseIntArrayAttr<int64_t>(op.getKernelSizeAttr());
    const auto KY = kernelSize[0];
    const auto KX = kernelSize[1];

    const auto kernelStrides = parseIntArrayAttr<int64_t>(op.getKernelStridesAttr());
    const auto SY = kernelStrides[0];
    const auto SX = kernelStrides[1];

    const auto kernelPadding = op.getKernelPaddingAttr();
    const auto padLeft = kernelPadding.getLeft().getInt();
    const auto padRight = kernelPadding.getRight().getInt();
    const auto padTop = kernelPadding.getTop().getInt();
    const auto padBottom = kernelPadding.getBottom().getInt();

    if (mlir::failed(VPU::NCEInvariant::verifyKernel(op->getLoc(), KY, KX, SY, SX, padTop, padBottom, padLeft, padRight,
                                                     arch))) {
        return errorAt(op, "Kernel verification failed");
    }

    return verifyInOutOrder(op, arch, "Pooling");
}

bool hasZeroPadding(const VPU::PaddingAttr padAttr) {
    if (padAttr == nullptr) {
        return true;
    }
    const auto top = padAttr.getTop().getInt();
    const auto bottom = padAttr.getBottom().getInt();
    const auto left = padAttr.getLeft().getInt();
    const auto right = padAttr.getRight().getInt();
    return top == 0 && bottom == 0 && left == 0 && right == 0;
}

mlir::LogicalResult verifyNCEEltwise(VPUIP::NCEClusterTaskOp op, VPU::ArchKind) {
    VPUX_THROW_UNLESS(op.getTaskType() == VPUIP::NCETaskType::ELTWISE, "Expected task type '{0}', but got '{1}'",
                      VPUIP::NCETaskType::ELTWISE, op.getTaskType());

    if (op.getActivationWindow() != nullptr) {
        return errorAt(op, "activation_window should be empty for NCETaskType : '{0}'", op.getTaskType());
    }

    if (op.getKernelSizeAttr() != nullptr) {
        return errorAt(op, "kernel_size should be empty for NCETaskType : '{0}'", op.getTaskType());
    }
    if (op.getKernelStridesAttr() != nullptr) {
        return errorAt(op, "kernel_strides should be empty for NCETaskType : '{0}'", op.getTaskType());
    }
    if (!hasZeroPadding(op.getKernelPaddingAttr())) {
        return errorAt(op, "kernel_padding should be empty for NCETaskType : '{0}'", op.getTaskType());
    }

    return mlir::success();
}

mlir::LogicalResult verifyNCEDWConv(VPUIP::NCEClusterTaskOp op, VPU::ArchKind arch) {
    VPUX_THROW_UNLESS(op.getTaskType() == VPUIP::NCETaskType::DWCONV, "Expected task type '{0}', but got '{1}'",
                      VPUIP::NCETaskType::CONV, op.getTaskType());

    if (op.getWeights() == nullptr) {
        return errorAt(op, "weights is required for NCETaskType : '{0}'", op.getTaskType());
    }
    if (op.getWeightTable() == nullptr) {
        return errorAt(op, "weight_table is required for NCETaskType : '{0}'", op.getTaskType());
    }
    if (op.getKernelSizeAttr() == nullptr) {
        return errorAt(op, "kernel_size is required for NCETaskType : '{0}'", op.getTaskType());
    }
    if (op.getKernelStridesAttr() == nullptr) {
        return errorAt(op, "kernel_strides is required for NCETaskType : '{0}'", op.getTaskType());
    }
    if (op.getKernelPaddingAttr() == nullptr) {
        return errorAt(op, "kernel_padding is required for NCETaskType : '{0}'", op.getTaskType());
    }

    const auto kernelSize = parseIntArrayAttr<int64_t>(op.getKernelSizeAttr());
    const auto KY = kernelSize[0];
    const auto KX = kernelSize[1];

    const auto kernelStrides = parseIntArrayAttr<int64_t>(op.getKernelStridesAttr());
    const auto SY = kernelStrides[0];
    const auto SX = kernelStrides[1];

    const auto kernelPadding = op.getKernelPaddingAttr();
    const auto padLeft = kernelPadding.getLeft().getInt();
    const auto padRight = kernelPadding.getRight().getInt();
    const auto padTop = kernelPadding.getTop().getInt();
    const auto padBottom = kernelPadding.getBottom().getInt();

    if (mlir::failed(VPU::NCEInvariant::verifyKernel(op->getLoc(), KY, KX, SY, SX, padTop, padBottom, padLeft, padRight,
                                                     arch))) {
        return errorAt(op, "Kernel verification failed");
    }

    const auto weightsShape = getShape(op.getWeights());
    const auto OC = weightsShape[Dims4D::Filter::OC];

    const auto weightTableShape = getShape(op.getWeightTable());
    const auto weightTableNumElements = weightTableShape.totalSize();

    if (OC * VPUIP::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC > weightTableNumElements) {
        return errorAt(op, "Weight table must have elements greater than or equal to '{0}' elements, got '{1}'",
                       OC * VPUIP::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC, weightTableNumElements);
    }

    const auto weightsLayout = DimsOrder::fromValue(op.getWeights());
    if (weightsLayout != DimsOrder::NHWC) {
        return errorAt(op, "weights layout must be NHWC, got {0}", weightsLayout);
    }

    return verifyInOutOrder(op, arch, "DWCONV");
}

}  // namespace

mlir::LogicalResult vpux::VPUIP::DPUTaskOp::verify() {
    const auto op = getOperation();
    static const size_t NUM_WORKLOAD_DIMS = 3;

    if (getOutStart().size() != NUM_WORKLOAD_DIMS) {
        return errorAt(op, "output start coords should {0}-D, but got {1}-D", NUM_WORKLOAD_DIMS, getOutStart().size());
    }
    if (getOutEnd().size() != NUM_WORKLOAD_DIMS) {
        return errorAt(op, "output end coords should {0}-D, but got {1}-D", NUM_WORKLOAD_DIMS, getOutEnd().size());
    }
    if (getInStart().has_value() && getInStart().value().size() != NUM_WORKLOAD_DIMS) {
        return errorAt(op, "input start coords should {0}-D, but got {1}-D", NUM_WORKLOAD_DIMS,
                       getInStart().value().size());
    }
    if (getInEnd().has_value() && getInEnd().value().size() != NUM_WORKLOAD_DIMS) {
        return errorAt(op, "input end coords should {0}-D, but got {1}-D", NUM_WORKLOAD_DIMS,
                       getInEnd().value().size());
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEClusterTaskOp::verify() {
    const auto op = getOperation();
    auto module = op->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);

    for (const auto& operand : getOpOperands()) {
        const auto val = operand.get();
        const auto type = val.getType().cast<vpux::NDTypeInterface>().getElementType();

        if (arch != VPU::ArchKind::NPU37XX && arch != VPU::ArchKind::NPU40XX && type.isBF16()) {
            return errorAt(op, "BF16 is only supported by NPU37XX, NPU40XX");
        }
    }

    if (getTaskType() == VPUIP::NCETaskType::CONV || getTaskType() == VPUIP::NCETaskType::CMCONV) {
        if (mlir::failed(verifyNCEConv(*this, arch))) {
            return mlir::failure();
        }
    } else if (getTaskType() == VPUIP::NCETaskType::MAXPOOL || getTaskType() == VPUIP::NCETaskType::AVEPOOL) {
        if (mlir::failed(verifyNCEPool(*this, arch))) {
            return mlir::failure();
        }
    } else if (getTaskType() == VPUIP::NCETaskType::ELTWISE) {
        if (mlir::failed(verifyNCEEltwise(*this, arch))) {
            return mlir::failure();
        }
    } else if (getTaskType() == VPUIP::NCETaskType::DWCONV) {
        if (mlir::failed(verifyNCEDWConv(*this, arch))) {
            return mlir::failure();
        }
    } else {
        return errorAt(op, "NCE Task Type '{0}' in not supported", getTaskType());
    }

    size_t numDPUTasks = 0;
    for (auto& dpuOp : getVariants().getOps()) {
        if (!mlir::isa<VPUIP::DPUTaskOp>(dpuOp)) {
            return errorAt(op, "Got unsupported Operation '{0}' in 'variants' region", dpuOp.getName());
        }

        ++numDPUTasks;
    }

    static const size_t MIN_NUM_DPUS_PER_CLUSTER = 1;

    if (numDPUTasks < MIN_NUM_DPUS_PER_CLUSTER) {
        return errorAt(op, "There should be at least {0} DPU Tasks per NCEClusterTask, but got {1}",
                       MIN_NUM_DPUS_PER_CLUSTER, numDPUTasks);
    }

    for (auto& ppeOp : getPpe().getOps()) {
        if (!mlir::isa<VPUIP::PPETaskOp>(ppeOp)) {
            return errorAt(op, "Got unsupported Operation '{0}' in 'PPE' region", ppeOp.getName());
        }
    }

    const auto appendToVector = [](SmallVector<mlir::Value>& operands, mlir::Value val) {
        if (val != nullptr)
            operands.push_back(val);
    };
    auto nnCMXOperands = SmallVector<mlir::Value>();

    const auto inputShape = getShape(getInput());

    const auto isInput4d = inputShape.size() == 4;
    if (isInput4d) {
        const auto inputBatch = inputShape[Dims4D::Act::N];
        if (inputBatch != vpux::VPU::NCEInvariant::SUPPORTED_BATCH_SIZE) {
            if (arch < VPU::ArchKind::NPU37XX) {
                return errorAt(op, "Got unsupported input batch '{0}' expected '{1}'", inputBatch,
                               vpux::VPU::NCEInvariant::SUPPORTED_BATCH_SIZE);
            }

            // Verify NPU37XX, NPU40XX and future version with batch tiling feature
            if (inputBatch > vpux::VPUIP::getNumTilesUsed(module)) {
                return errorAt(op, "Got unsupported input batch '{0}' expected to be less than or equal to '{1}'",
                               inputBatch, getNumTilesUsed(module));
            }
            if (auto nceTilingParent = op->getParentOfType<VPUIP::NCEClusterTilingOp>()) {
                auto outputType = nceTilingParent->getResult(0).getType().cast<VPUIP::DistributedBufferType>();
                const auto numClusters = outputType.getDistribution().getNumClusters().getInt();
                if (inputBatch != numClusters) {
                    return errorAt(op, "Got unsupported input batch '{0}' expected '{1}'", inputBatch, numClusters);
                }
            } else if (auto outputType = getOutput().getType().dyn_cast_or_null<VPUIP::DistributedBufferType>()) {
                const auto numClusters = outputType.getDistribution().getNumClusters().getInt();
                if (inputBatch != numClusters) {
                    return errorAt(op, "Got unsupported input batch '{0}' expected '{1}'", inputBatch, numClusters);
                }
            }
        }
    }  // else if (is5D) no limitation

    appendToVector(nnCMXOperands, getInput());
    appendToVector(nnCMXOperands, getWeights());
    appendToVector(nnCMXOperands, getWeightTable());
    appendToVector(nnCMXOperands, getActivationWindow());
    appendToVector(nnCMXOperands, getOutputBuff());
    appendToVector(nnCMXOperands, getProfilingData());

    const auto checkMemoryKind = [&op](mlir::ValueRange operands, EnumSet<VPU::MemoryKind> acceptedMemoryKinds) {
        for (const auto& val : operands) {
            const auto type = val.getType().cast<vpux::NDTypeInterface>();

            const auto mem = type.getMemoryKind();
            if (llvm::find(acceptedMemoryKinds, mem) == acceptedMemoryKinds.end())
                return errorAt(op, "Can't operate with '{0}' MemoryKind.", mem);
        }
        return mlir::success();
    };

    const auto nncmxStatus = checkMemoryKind(
            nnCMXOperands, EnumSet<VPU::MemoryKind>({VPU::MemoryKind::CMX_NN, VPU::MemoryKind::Register}));
    if (nncmxStatus.failed())
        return nncmxStatus;

    // TODO revisit memory checks for parent operands

    for (const auto& val : getOperands()) {
        const auto type = val.getType().cast<vpux::NDTypeInterface>();
        const auto strideReqs = StrideReqs().add(DimStrideReq::compact(MemDim(type.getRank() - 1)));

        if (!strideReqs.checkStrides(val)) {
            return errorAt(op, "Value '{0}' strides do not match requirements '{1}'", val, strideReqs);
        }
    }

    if (arch == VPU::ArchKind::NPU40XX) {
        auto outputType = getOutput().getType().dyn_cast<VPUIP::ITIBufferType>();
        auto outputItiBuffs = getOutput_ITIBuff();

        if (outputType == nullptr && !outputItiBuffs.empty()) {
            return errorAt(op, "Output is not of VPUIP::ITIBufferType, but output_ITI_buffs is not empty.");
        }

        for (const auto itiOutput : outputItiBuffs) {
            if (!itiOutput.getType().isa<VPUIP::ITIBufferType>()) {
                return errorAt(op, "ITI Output is not of VPUIP::ITIBufferType: {0}", itiOutput);
            }
        }

        if (getOutputSparsityMap() != nullptr && outputType != nullptr) {
            if (!getOutputSparsityMap().getType().isa<ITIBufferType>()) {
                return errorAt(op, "Output is of VPUIP::ITIBufferType, but output sparsity map is not.");
            }
        }
    }

    return mlir::success();
}

//
// NCEClusterTaskOp::serialize
//

namespace {

MVCNN::MPE_Mode getMPEMode(VPU::MPEMode mpeMode) {
    switch (mpeMode) {
    case VPU::MPEMode::VECTOR:
        return MVCNN::MPE_Mode_VECTOR;
    case VPU::MPEMode::MATRIX:
        return MVCNN::MPE_Mode_MATRIX;
    case VPU::MPEMode::VECTOR_FP16:
        return MVCNN::MPE_Mode_VECTOR_FP16;
    case VPU::MPEMode::CUBOID_16x16:
        return MVCNN::MPE_Mode_CUBOID_16x16;
    case VPU::MPEMode::CUBOID_8x16:
        return MVCNN::MPE_Mode_CUBOID_8x16;
    case VPU::MPEMode::CUBOID_4x16:
        return MVCNN::MPE_Mode_CUBOID_4x16;
    case VPU::MPEMode::NOP:
        return MVCNN::MPE_Mode_NOP;
    default:
        VPUX_THROW("Unsupported MPE mode type: '{0}'", mpeMode);
    }
}

MVCNN::DPULayerType getDPULayerType(VPUIP::NCETaskType taskType) {
    switch (taskType) {
    case VPUIP::NCETaskType::CONV:
        return MVCNN::DPULayerType_CONV;
    case VPUIP::NCETaskType::DWCONV:
        return MVCNN::DPULayerType_DWCONV;
    case VPUIP::NCETaskType::MAXPOOL:
        return MVCNN::DPULayerType_MAXPOOL;
    case VPUIP::NCETaskType::AVEPOOL:
        return MVCNN::DPULayerType_AVEPOOL;
    case VPUIP::NCETaskType::FCL:
        return MVCNN::DPULayerType_FCL;
    case VPUIP::NCETaskType::ELTWISE:
        return MVCNN::DPULayerType_ELTWISE;
    case VPUIP::NCETaskType::IDENTITY:
        return MVCNN::DPULayerType_IDENTITY;
    case VPUIP::NCETaskType::CMCONV:
        return MVCNN::DPULayerType_CMCONV;
    default:
        VPUX_THROW("Unsupported DPU Layer type: '{0}'", taskType);
    }
}

MVCNN::Permutation getODUPermutationType(DimsOrder outputDimsOrder) {
    if (outputDimsOrder == vpux::DimsOrder::NHWC) {
        return MVCNN::Permutation_ZXY;
    } else if (outputDimsOrder == vpux::DimsOrder::NWHC) {
        return MVCNN::Permutation_ZYX;
    } else if (outputDimsOrder == vpux::DimsOrder::NWCH) {
        return MVCNN::Permutation_YZX;
    } else if (outputDimsOrder == vpux::DimsOrder::NCWH) {
        return MVCNN::Permutation_YXZ;
    } else if (outputDimsOrder == vpux::DimsOrder::NHCW) {
        return MVCNN::Permutation_XZY;
    } else if (outputDimsOrder == vpux::DimsOrder::NCHW) {
        return MVCNN::Permutation_XYZ;
    } else if (outputDimsOrder == vpux::DimsOrder::GNHWC) {
        return MVCNN::Permutation_ZXY;
    } else {
        VPUX_THROW("Can't get ODU permutation by output dimsOrder: '{0}'", outputDimsOrder);
    }
}

MVCNN::PPELayerType getPPELayerType(VPU::PPEMode ppeType) {
    switch (ppeType) {
    case VPU::PPEMode::STORE:
        return MVCNN::PPELayerType_STORE;
    case VPU::PPEMode::LOAD:
        return MVCNN::PPELayerType_LOAD;
    case VPU::PPEMode::CLEAR:
        return MVCNN::PPELayerType_CLEAR;
    case VPU::PPEMode::NOOP:
        return MVCNN::PPELayerType_NOOP;
    case VPU::PPEMode::HALT:
        return MVCNN::PPELayerType_HALT;
    case VPU::PPEMode::ADD:
        return MVCNN::PPELayerType_ADD;
    case VPU::PPEMode::SUB:
        return MVCNN::PPELayerType_SUB;
    case VPU::PPEMode::MULT:
        return MVCNN::PPELayerType_MULT;
    case VPU::PPEMode::MAXIMUM:
        return MVCNN::PPELayerType_MAXIMUM;
    case VPU::PPEMode::MINIMUM:
        return MVCNN::PPELayerType_MINIMUM;
    case VPU::PPEMode::AND:
        return MVCNN::PPELayerType_AND;
    case VPU::PPEMode::OR:
        return MVCNN::PPELayerType_OR;
    case VPU::PPEMode::XOR:
        return MVCNN::PPELayerType_XOR;
    case VPU::PPEMode::LRELU:
        return MVCNN::PPELayerType_LRELU;
    case VPU::PPEMode::LRELUX:
        return MVCNN::PPELayerType_LRELUX;
    case VPU::PPEMode::LPRELU:
        return MVCNN::PPELayerType_LPRELU;
    case VPU::PPEMode::CEIL:
        return MVCNN::PPELayerType_CEIL;
    case VPU::PPEMode::FLOOR:
        return MVCNN::PPELayerType_FLOOR;
    case VPU::PPEMode::EXP:
        return MVCNN::PPELayerType_EXP;
    case VPU::PPEMode::SIGMOID:
        return MVCNN::PPELayerType_SIGMOID;
    case VPU::PPEMode::TANH:
        return MVCNN::PPELayerType_TANH;
    case VPU::PPEMode::SQRT:
        return MVCNN::PPELayerType_SQRT;
    case VPU::PPEMode::RSQRT:
        return MVCNN::PPELayerType_RSQRT;
    case VPU::PPEMode::FLEXARB:
        return MVCNN::PPELayerType_FLEXARB;
    case VPU::PPEMode::NOT:
        return MVCNN::PPELayerType_NOT;
    case VPU::PPEMode::ABS:
        return MVCNN::PPELayerType_ABS;
    case VPU::PPEMode::NEG:
        return MVCNN::PPELayerType_NEG;
    default:
        VPUX_THROW("Unsupported PPE Layer type: '{0}'", ppeType);
    }
}

VPU::MPEMode getMPEFrequentModeFromDPUTasks(mlir::Region& dpuTaskOps) {
    std::unordered_map<VPU::MPEMode, size_t> umap;
    for (auto dpuTaskOp : dpuTaskOps.getOps<VPUIP::DPUTaskOp>()) {
        ++umap[dpuTaskOp.getMpeMode()];
        if (umap.size() > 1) {
            VPUX_THROW("Non-uniform DPU task MPE modes is not supported yet.");
        }
    }
    return umap.begin()->first;
}

// This is a helper routine to build new TensorReference out of NCE task for input, weights and output with provided
// quantization scale parameters
vpux::VPUIP::BlobWriter::TensorReference getTensorReferenceWithUpdatedQuantParams(
        VPUIP::BlobWriter& writer, ArrayRef<int64_t> ppeQuantMult, ArrayRef<int64_t> ppeQuantShift,
        int64_t ppeQuantPostShift, mlir::Value operand, mlir::Value operandSparsityMap = nullptr,
        mlir::Value operandSETable = nullptr, std::optional<int64_t> storageElementSize = std::nullopt) {
    // Get also ZP
    SmallVector<uint8_t> quantZeroPoints;

    auto type = operand.getType().cast<vpux::NDTypeInterface>();

    auto elementType = type.getElementType();
    if (const auto uniformQuantType = elementType.dyn_cast<mlir::quant::UniformQuantizedType>()) {
        quantZeroPoints.push_back(checked_cast<uint8_t>(uniformQuantType.getZeroPoint()));
    } else if (const auto uniformQuantPerAxisType = elementType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        auto zp = uniformQuantPerAxisType.getZeroPoints();
        quantZeroPoints.resize(zp.size());
        std::transform(zp.begin(), zp.end(), quantZeroPoints.begin(), [](int64_t a) {
            return checked_cast<uint8_t>(a);
        });
    } else {
        quantZeroPoints.push_back(0);
    }

    VPUX_THROW_UNLESS(ppeQuantShift.size() == quantZeroPoints.size(),
                      "Mismatch of size between quant shift/mult vector and quant ZP:  {0} != {1}",
                      ppeQuantShift.size(), quantZeroPoints.size());

    // Find corresponding DeclareBufferOp to get all the data needed to build new TensorReference
    auto bufferOp = operand.getDefiningOp<VPURT::DeclareBufferOp>();
    VPUX_THROW_UNLESS(bufferOp != nullptr, "Unable to find parent DeclareBufferOp to build new TensorReference");

    auto sectionIndex = bufferOp.getNonEmptySectionIndex();

    std::optional<int64_t> sparsityMapOffset = std::nullopt;
    std::optional<int64_t> seTableOffset = std::nullopt;
    if (operandSparsityMap != nullptr) {
        auto sparsityMapBufferOp = operandSparsityMap.getDefiningOp<VPURT::DeclareBufferOp>();
        VPUX_THROW_UNLESS(sparsityMapBufferOp != nullptr, "Unable to find DeclareBufferOp for sparsity map");
        sparsityMapOffset = sparsityMapBufferOp.getByteOffset();
    }
    if (operandSETable != nullptr) {
        auto seTableBufferOp = operandSETable.getDefiningOp<VPURT::DeclareBufferOp>();
        VPUX_THROW_UNLESS(seTableBufferOp != nullptr, "Unable to find DeclareBufferOp for storage element table");
        seTableOffset = seTableBufferOp.getByteOffset();
    }

    return writer.createTensorRef("tensor_scale_updated", type, bufferOp.getSection(), sectionIndex,
                                  bufferOp.getByteOffset(), ppeQuantMult, ppeQuantShift, ppeQuantPostShift,
                                  quantZeroPoints, sparsityMapOffset, seTableOffset, storageElementSize,
                                  bufferOp.getSwizzlingKey());
}

// This is a helper routine to build new TensorReference of individual variant with profiling data
vpux::VPUIP::BlobWriter::TensorReference getTensorReferenceForVariantProfiling(VPUIP::NCEClusterTaskOp nceTask,
                                                                               VPUIP::BlobWriter& writer,
                                                                               size_t variantId,
                                                                               uint16_t workloadSize) {
    static size_t tempTensorId = 0;

    auto outputType = nceTask.getProfilingData().getType().cast<vpux::NDTypeInterface>();
    // Find corresponding DeclareBufferOp to get all the data needed to build new TensorReference
    auto bufferOp = nceTask.getProfilingData().getDefiningOp<VPURT::DeclareBufferOp>();
    VPUX_THROW_UNLESS(bufferOp != nullptr, "Unable to find parent DeclareBufferOp to build new TensorReference");

    auto sectionIndex = bufferOp.getNonEmptySectionIndex();
    const auto refMeta = llvm::formatv("_{0}_dpu_{1}", tempTensorId, variantId).str();
    tempTensorId++;

    return writer.createTensorRef(refMeta, outputType, bufferOp.getSection(), sectionIndex,
                                  bufferOp.getByteOffset() + workloadSize * variantId);
}

}  // namespace

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::NCEClusterTaskOp::serialize(VPUIP::BlobWriter& writer) {
    const auto module = getOperation()->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);

    const bool isSwProfiling = (arch != VPU::ArchKind::NPU37XX && arch != VPU::ArchKind::NPU40XX);
    const bool isProfEnabled = getProfilingData() != nullptr;

    const uint16_t wlSize = VPUIP::getProfWorkloadSize(module);
    int32_t profBufferBaseAddress = -1;
    if (isProfEnabled && !isSwProfiling) {
        profBufferBaseAddress =
                checked_cast<int32_t>(getProfilingData().getDefiningOp<VPURT::DeclareBufferOp>().getByteOffset());
    }

    SmallVector<flatbuffers::Offset<MVCNN::NCEVariantFields>> variantList;
    size_t profiledDpuTasksCount = 0;
    for (auto dpuTaskOp : getVariants().getOps<VPUIP::DPUTaskOp>()) {
        auto inStart = SmallVector<int64_t>{0, 0, 0};
        auto inSize = SmallVector<int64_t>{0, 0, 0};

        // Currently, input start/size attributes are computed only for NPU40XX.
        // For NPU37XX, input workload start/size are back-inferred by runtime from
        // output workload start/end.
        // TODO: Add input workload computation for NPU37XX - E#63055
        if (arch == VPU::ArchKind::NPU40XX) {
            VPUX_THROW_UNLESS(dpuTaskOp.getInStart().has_value() && dpuTaskOp.getInEnd().has_value(),
                              "DPUTask does not have input workload start/end for arch 40XX");

            const auto inEnd = parseIntArrayAttr<int64_t>(dpuTaskOp.getInEnd().value());

            inStart = parseIntArrayAttr<int64_t>(dpuTaskOp.getInStart().value());
            inSize = SmallVector<int64_t>{inEnd[0] - inStart[0] + 1, inEnd[1] - inStart[1] + 1,
                                          inEnd[2] - inStart[2] + 1};
        }

        const auto outStart = parseIntArrayAttr<int64_t>(dpuTaskOp.getOutStart());
        const auto outEnd = parseIntArrayAttr<int64_t>(dpuTaskOp.getOutEnd());
        const auto pad = dpuTaskOp.getPad();

        flatbuffers::Offset<MVCNN::TensorReference> profilingData = {0};
        if (isProfEnabled) {
            if (isSwProfiling) {
                // For software DPU profiling we don't care about workload_id, it may be -1.
                // Calculating here correct tensor reference with a shift for particular variant.
                profilingData = getTensorReferenceForVariantProfiling(*this, writer, profiledDpuTasksCount, wlSize);
            } else {
                // Hardware profiling used. Invariant uses CMX base address and variant needs a workload_id
                // (calculated per NCE cluster task + shift per variant) which is obtained from the DPU task attribute
                VPUX_THROW_WHEN(!dpuTaskOp.getWorkloadId().has_value(), "workload_id value has not been assigned");
            }
        }

        SmallVector<VPUIP::BlobWriter::HaloRegion> haloRegionsVec;
        const mlir::ArrayAttr haloRegions = dpuTaskOp.getHaloRegions().value_or(nullptr);
        if (haloRegions != nullptr) {
            for (const auto& attr : haloRegions) {
                auto haloAttr = attr.dyn_cast_or_null<VPUIP::DPUHaloRegionAttr>();
                VPUX_THROW_UNLESS(haloAttr != nullptr, "Got non DPUHaloRegion Attribute '{0}' in Array", haloAttr);

                const auto targetClusters = parseIntArrayAttr<uint8_t>(haloAttr.getTargetClusters());

                const auto xStart = static_cast<uint16_t>(haloAttr.getXStart().getInt());
                const auto xEnd = static_cast<uint16_t>(haloAttr.getXEnd().getInt());
                const auto yStart = static_cast<uint16_t>(haloAttr.getYStart().getInt());
                const auto yEnd = static_cast<uint16_t>(haloAttr.getYEnd().getInt());
                const auto zStart = static_cast<uint16_t>(haloAttr.getZStart().getInt());
                const auto zEnd = static_cast<uint16_t>(haloAttr.getZEnd().getInt());
                const auto targetOffset = static_cast<int32_t>(haloAttr.getTargetOffset().getInt());
                const auto targetClustersVec = writer.createVector(targetClusters);
                const auto sparsityOffset = haloAttr.getSparsityOffset() != nullptr
                                                    ? static_cast<int32_t>(haloAttr.getSparsityOffset().getInt())
                                                    : 0;
                const auto targetWidth = static_cast<uint16_t>(haloAttr.getTargetWidth().getInt());

                haloRegionsVec.push_back(MVCNN::CreateHaloRegion(writer, xStart, xEnd, yStart, yEnd, zStart, zEnd,
                                                                 targetOffset, targetClustersVec, sparsityOffset,
                                                                 targetWidth));
            }
        }

        const VPUIP::BlobWriter::Vector<VPUIP::BlobWriter::HaloRegion> serializedHaloRegions =
                !haloRegionsVec.empty() ? writer.createVector(haloRegionsVec) : 0;

        // workload_start_X/Y/Z and workload_end_X/Y/Z are used to serialize the values for the te_beg_x/y/z and
        // te_end_x/y/z registers, which are defining the start and end of the output workload.
        // For NPU40XX, idu_workload_start_x/y/z and idu_workload_size_x/y/z are used to serialize the values for the
        // workload_start_x/y/z and workload_size_x/y/z registers, which are defining the start and size of the
        // input workload. For all other architectures, idu_workloads* fields are ignored in runtime.
        const auto variant = MVCNN::CreateNCEVariantFields(
                writer,
                0,                                                              // Barriers
                getMPEMode(dpuTaskOp.getMpeMode()),                             // MPE mode
                static_cast<int16_t>(pad.getLeft().getInt()),                   // padLeft
                static_cast<int16_t>(pad.getRight().getInt()),                  // padRight
                static_cast<int16_t>(pad.getTop().getInt()),                    // padTop
                static_cast<int16_t>(pad.getBottom().getInt()),                 // padBottom
                static_cast<int16_t>(outStart[0]),                              // workload_start_X
                static_cast<int16_t>(outStart[1]),                              // workload_start_Y
                static_cast<int16_t>(outStart[2]),                              // workload_start_Z
                static_cast<int16_t>(outEnd[0]),                                // workload_end_X
                static_cast<int16_t>(outEnd[1]),                                // workload_end_Y
                static_cast<int16_t>(outEnd[2]),                                // workload_end_Z
                0,                                                              // flex_map_column
                0,                                                              // flex_map_array
                0,                                                              // flex_inner
                0,                                                              // flex_outer
                0,                                                              // flex_outer_order
                profilingData,                                                  // profiling_data
                serializedHaloRegions,                                          // halo_regions
                checked_cast<int32_t>(dpuTaskOp.getWorkloadId().value_or(-1)),  // workload_id
                static_cast<int16_t>(inStart[0]),                               // idu_workload_start_x
                static_cast<int16_t>(inStart[1]),                               // idu_workload_start_y
                static_cast<int16_t>(inStart[2]),                               // idu_workload_start_z
                static_cast<int16_t>(inSize[0]),                                // idu_workload_size_x
                static_cast<int16_t>(inSize[1]),                                // idu_workload_size_y
                static_cast<int16_t>(inSize[2])                                 // idu_workload_size_z
        );
        ++profiledDpuTasksCount;
        variantList.push_back(variant);
    }
    const auto variant = writer.createVector(variantList);

    SmallVector<uint8_t> ppeList;
    int32_t clampLow = std::numeric_limits<int32_t>::min();
    int32_t clampHigh = std::numeric_limits<int32_t>::max();
    int32_t LreluMult = 1;
    uint32_t LreluShift = 0;
    ::std::optional<SmallVector<int64_t>> ppeQuantMult;
    ::std::optional<SmallVector<int64_t>> ppeQuantShift;
    ::std::optional<int64_t> ppeQuantPostShift;
    ::std::optional<float> ppeQuantScale;
    ::std::optional<SmallVector<int64_t>> in1QuantMult;
    ::std::optional<SmallVector<int64_t>> in2QuantMult;
    float fpPReluAlpha = 1.f;

    for (auto ppeOp : getPpe().getOps<VPUIP::PPETaskOp>()) {
        const auto type = getPPELayerType(ppeOp.getPpeLayerType());
        if (type != MVCNN::PPELayerType_NOOP) {
            ppeList.push_back(type);
        }
        if (ppeOp.getClampLow().has_value()) {
            clampLow = checked_cast<int32_t>(ppeOp.getClampLowAttr().dyn_cast<mlir::IntegerAttr>().getInt());
        }
        if (ppeOp.getClampHigh().has_value()) {
            clampHigh = checked_cast<int32_t>(ppeOp.getClampHighAttr().dyn_cast<mlir::IntegerAttr>().getInt());
        }
        if (ppeOp.getLreluMult().has_value()) {
            LreluMult = checked_cast<int32_t>(ppeOp.getLreluMult().value());
        }
        if (ppeOp.getLreluShift().has_value()) {
            LreluShift = checked_cast<uint32_t>(ppeOp.getLreluShift().value());
        }
        if (ppeOp.getQuantMult().has_value()) {
            ppeQuantMult = parseIntArrayAttr<int64_t>(ppeOp.getQuantMult().value());
        }
        if (ppeOp.getQuantShift().has_value()) {
            ppeQuantShift = parseIntArrayAttr<int64_t>(ppeOp.getQuantShift().value());
        }
        if (ppeOp.getQuantPostShift().has_value()) {
            ppeQuantPostShift = checked_cast<int64_t>(ppeOp.getQuantPostShift().value());
        }
        if (ppeOp.getQuantScale().has_value()) {
            auto floatScaleAttr = ppeOp.getQuantScaleAttr().getValue()[0];
            ppeQuantScale = static_cast<float>(floatScaleAttr.dyn_cast_or_null<mlir::FloatAttr>().getValueAsDouble());
        }
        if (ppeOp.getIn1QuantMult().has_value()) {
            in1QuantMult = parseIntArrayAttr<int64_t>(ppeOp.getIn1QuantMult().value().cast<mlir::ArrayAttr>());
        }
        if (ppeOp.getIn2QuantMult().has_value()) {
            in2QuantMult = parseIntArrayAttr<int64_t>(ppeOp.getIn2QuantMult().value().cast<mlir::ArrayAttr>());
        }
        if (ppeOp.getFpPreluAlpha().has_value()) {
            // For values like prelu_alpha=0.1, checked_cast fails, due to loss in precision when converting
            // from double to float and back, due to the static_cast<double>(static_cast<float>(value)) == value check
            // Use static_cast instead
            fpPReluAlpha = static_cast<float>(ppeOp.getFpPreluAlpha().value().convertToDouble());
        }
    }
    VPUX_THROW_UNLESS(ppeList.size() <= 1, "Cannot set more than one PPE task");

    auto ppeLayerTypes = writer.createVector(ppeList);
    // TODO: Lrelu_Mult, Lrelu_Shift
    auto ppeFixedFunction =
            MVCNN::CreatePPEFixedFunction(writer, ppeLayerTypes, clampLow, clampHigh, LreluMult, LreluShift);
    // TODO: scale_data, rounding
    const auto instructionListTable =
            getInstructionListTable() != nullptr ? writer.getTensorRef(getInstructionListTable()) : 0;

    auto ppeTask = MVCNN::CreatePPETask(writer, 0, ppeFixedFunction, MVCNN::PPERoundingMode_RNE, instructionListTable,
                                        ppeQuantScale.value_or(1.0), fpPReluAlpha);

    int16_t kernelSizeH = 1, kernelSizeW = 1;
    int16_t kernelStridesH = 1, kernelStridesW = 1;
    int16_t kernelPadL = 0, kernelPadR = 0, kernelPadT = 0, kernelPadB = 0;
    flatbuffers::Offset<flatbuffers::Vector<int8_t>> enabled_optimizations = 0;
    int32_t odu_offset = 0;
    int32_t out_channel_offset = 0;
    bool is_segmented = false;
    bool is_continued = false;
    bool isSuperdense = false;
    bool isL1OptOn = false;
    uint16_t cm_sp_pattern = 0;

    if (getKernelSizeAttr() != nullptr) {
        const auto kernelSize = parseIntArrayAttr<int64_t>(getKernelSizeAttr());
        kernelSizeH = checked_cast<int16_t>(kernelSize[0]);
        kernelSizeW = checked_cast<int16_t>(kernelSize[1]);
    }

    if (getKernelStridesAttr() != nullptr) {
        const auto kernelStrides = parseIntArrayAttr<int64_t>(getKernelStridesAttr());
        kernelStridesH = checked_cast<int16_t>(kernelStrides[0]);
        kernelStridesW = checked_cast<int16_t>(kernelStrides[1]);
    }

    if (getKernelPaddingAttr() != nullptr) {
        const auto kernelPadding = getKernelPaddingAttr();
        kernelPadL = checked_cast<int16_t>(kernelPadding.getLeft().getInt());
        kernelPadR = checked_cast<int16_t>(kernelPadding.getRight().getInt());
        kernelPadT = checked_cast<int16_t>(kernelPadding.getTop().getInt());
        kernelPadB = checked_cast<int16_t>(kernelPadding.getBottom().getInt());
    }

    is_continued = (getIsContinuedAttr() != nullptr);
    is_segmented = (getIsSegmentedAttr() != nullptr);
    isSuperdense = (getIsSuperdenseAttr() != nullptr);
    isL1OptOn = (getIsSmallKernelOptimizedAttr() != nullptr);

    // Extract output permutation from output layout
    MVCNN::Permutation oduPermutation = getODUPermutationType(DimsOrder::fromValue(getOutput()));

    if (getCmSpPatternAttr() != nullptr) {
        cm_sp_pattern = checked_cast<uint16_t>(getCmSpPatternAttr().getValue().getSExtValue());
    }

    if (getOutChannelOffsetAttr() != nullptr) {
        out_channel_offset = checked_cast<int32_t>(getOutChannelOffsetAttr().getValue().getSExtValue());
    }

    int8_t input_channels_compression = (getInputChannelsCompressionAttr() != nullptr) ? 1 : 0;

    auto inputData = writer.getTensorRef(getInput());
    auto weightsData = getWeights() != nullptr ? writer.getTensorRef(getWeights()) : 0;
    const auto weightsTable = getWeightTable() != nullptr ? writer.getTensorRef(getWeightTable()) : 0;

    auto outputData = writer.getTensorRef(getOutput());

    // If quant scale (mult, shift) settings were provided as part of PPE block then use it to build new
    // output TensorReference. This is required for Eltwise operation which doesn't have weights table
    // and PPE quantization settings (Mult, Shift) need to be provided for NN runtime in output tensor descriptor
    const auto isQuantizationProvided =
            ppeQuantMult.has_value() && ppeQuantShift.has_value() && ppeQuantPostShift.has_value();
    const auto isQuantizationNotProvided =
            !ppeQuantMult.has_value() && !ppeQuantShift.has_value() && !ppeQuantPostShift.has_value();
    const auto isInputQuantizationProvided = in1QuantMult.has_value() && in2QuantMult.has_value();
    VPUX_THROW_WHEN(!isQuantizationProvided && !isQuantizationNotProvided, "Missing quantization scale settings.");

    if (isQuantizationProvided) {
        outputData = getTensorReferenceWithUpdatedQuantParams(
                writer, ppeQuantMult.value(), ppeQuantShift.value(), ppeQuantPostShift.value(), getOutputBuff(),
                getOutputSparsityMapBuff(), /*operandSETable=*/nullptr, getOutputSeSize());
    }
    if (isInputQuantizationProvided) {
        // Shifts must be set 0 for NPU37XX and NPU40XX runtime to be considered, otherwise runtime will ignore inputs
        // MULT.
        inputData = getTensorReferenceWithUpdatedQuantParams(writer, in1QuantMult.value(), {0}, 0, getInput(),
                                                             getInputSparsityMap(), getInputStorageElementTable(),
                                                             getInputSeSize());
        weightsData = getTensorReferenceWithUpdatedQuantParams(writer, in2QuantMult.value(), {0}, 0, getWeights(),
                                                               getWeightsSparsityMap(), /*operandSETable=*/nullptr);
    }

    // Parent input override is required for PermuteQuantize.
    // Input dimensions are read from parent input in runtime code.
    const auto isPermuteQuantize = getIsPermuteQuantizeAttr() != nullptr;
    const auto overrideParentIn = isPermuteQuantize && (getParentInput().getType() != getInput().getType());
    const auto parentInputTensor = overrideParentIn ? inputData : writer.getTensorRef(getParentInput());
    const auto parentOutputTensor = writer.getTensorRef(getParentOutput());

    const auto invariantMPEMode = getMPEFrequentModeFromDPUTasks(getVariants());

    const auto invariant = MVCNN::CreateNCEInvariantFields(writer,
                                                           getDPULayerType(getTaskType()),  // dpu_task_type
                                                           ppeTask,                         // ppe_task
                                                           getMPEMode(invariantMPEMode),    // mpe_frequent_mode
                                                           kernelSizeH,                     // kernelH
                                                           kernelSizeW,                     // kernelW
                                                           kernelStridesH,                  // kernel_strideH
                                                           kernelStridesW,                  // kernel_strideW
                                                           kernelPadL,                      // kernel_padLeft
                                                           kernelPadR,                      // kernel_padRight
                                                           kernelPadT,                      // kernel_padTop
                                                           kernelPadB,                      // kernel_padBottom
                                                           parentInputTensor,               // parent_input_tensor
                                                           parentOutputTensor,              // parent_output_tensor
                                                           0,                               // parent_weights_tensor
                                                           inputData,                       // input_data
                                                           outputData,                      // output_data
                                                           weightsData,                     // weights_data
                                                           weightsTable,                    // weights_table
                                                           enabled_optimizations,           // enabled_optimizations
                                                           odu_offset,                      // odu_offset
                                                           out_channel_offset,              // out_channel_offset
                                                           is_segmented,                    // is_segmented
                                                           is_continued,                    // is_continued
                                                           isSuperdense,                    // is_superdense
                                                           0,                               // segment_height
                                                           oduPermutation,                  // odu_permutation
                                                           cm_sp_pattern,                   // cm_sp_pattern
                                                           profBufferBaseAddress,           // cmx_local_slice_base
                                                           input_channels_compression,  // input_channels_compression
                                                           false,                       // explicit_input_workloads
                                                           isL1OptOn  // depthwise_conv_3x3s1_optimization
    );

    MVCNN::NCE2TaskBuilder builder(writer);
    builder.add_variant(variant);
    builder.add_invariant(invariant);

    return {builder.Finish().Union(), MVCNN::SpecificTask_NCE2Task};
}
