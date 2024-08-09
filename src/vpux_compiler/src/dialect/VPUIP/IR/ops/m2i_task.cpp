//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/utils.hpp"

using namespace vpux;

//
// M2ITaskOp::serialize
//

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::M2ITaskOp::serialize(VPUIP::BlobWriter& writer) {
    const auto getRawFP16 = [](auto val) {
        const auto valFP16 = vpux::type::float16(val);
        return valFP16.to_bits();
    };

    const auto getVecFP16 = [&](auto range) {
        return writer.createVector(range | transformed(getRawFP16));
    };

    VPUIP::BlobWriter::Vector<uint16_t> serializedCoefs;

    if (getNorm().has_value()) {
        const auto coefs = parseFPArrayAttr<double>(getNorm().value());
        serializedCoefs = getVecFP16(coefs);
    }

    const auto getTensorCb = [&writer](mlir::Value val) {
        return writer.getTensorRef(val);
    };
    const auto inputs = writer.createVector(getInputs() | transformed(getTensorCb));
    const auto outputs = writer.createVector(getOutputs() | transformed(getTensorCb));

    MVCNN::M2ITaskBuilder builder(writer);
    builder.add_src(inputs);
    builder.add_dst(outputs);
    builder.add_do_csc(getDoCsc());
    builder.add_do_norm(getDoNorm());
    builder.add_in_fmt(convertM2iColor2MVCNN(getInFmt()));
    builder.add_out_fmt(convertM2iColor2MVCNN(getOutFmt()));
    builder.add_chroma_in_reverse_channels(getChromaInReverseChannels() ? 1 : 0);
    builder.add_chroma_out_reverse_channels(getChromaOutReverseChannels() ? 1 : 0);
    builder.add_luma_in_reverse_channels(getLumaInReverseChannels() ? 1 : 0);
    builder.add_luma_out_reverse_channels(getLumaOutReverseChannels() ? 1 : 0);
    builder.add_scale_factor_x(getScaleFactorX());
    builder.add_scale_factor_y(getScaleFactorY());
    if (getTileOffsetX().has_value())
        builder.add_tile_offset_x(getTileOffsetX().value());
    if (getTileOffsetY().has_value())
        builder.add_tile_offset_y(getTileOffsetY().value());
    builder.add_interp(convertM2iInterp2MVCNN(getInterp()));

    if (getNorm().has_value()) {
        builder.add_norm_coefs(serializedCoefs);
    }

    return {builder.Finish().Union(), MVCNN::SpecificTask_M2ITask};
}

mlir::LogicalResult vpux::VPUIP::M2ITaskOp::inferReturnTypes(mlir::MLIRContext* /*ctx*/,
                                                             std::optional<mlir::Location> /*optLoc*/,
                                                             mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                             mlir::OpaqueProperties props,
                                                             mlir::RegionRange /*regions*/,
                                                             mlir::SmallVectorImpl<mlir::Type>& inferredTypes) {
    VPUIP::M2ITaskOpAdaptor adaptor(operands, attrs, props);
    inferredTypes.push_back(adaptor.getOutputBuff().getType());
    if (adaptor.getProfilingData() != nullptr) {
        inferredTypes.push_back(adaptor.getProfilingData().getType());
    }
    return mlir::success();
}
