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

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::LRN_IEOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::LRN_IEOpAdaptor lrn_ie(operands, attrs);
    if (mlir::failed(lrn_ie.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = lrn_ie.input().getType().cast<mlir::ShapedType>();
    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType());

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::IE::LRN_IEOp::serialize(EMU::BlobWriter& writer) {
    MVCNN::NormParamsBuilder builder(writer);

    EMU::BlobWriter::String region;
    switch (this->region()) {
    case IE::LRN_IERegion::across:
        region = writer.createString("across");
        break;
    case IE::LRN_IERegion::same:
        region = writer.createString("same");
        break;
    default:
        VPUX_THROW("Unsupported LRN_IERegion {0}", this->region());
    }

    builder.add_alpha(static_cast<float>(alpha().convertToDouble()));
    builder.add_beta(static_cast<float>(beta().convertToDouble()));
    builder.add_local_size(checked_cast<int32_t>(size()));
    builder.add_region(region);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_NormParams});
}
