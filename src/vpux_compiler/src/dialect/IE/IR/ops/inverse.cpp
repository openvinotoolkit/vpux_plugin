#include "vpux/compiler/dialect/IE/IR/ops.hpp"

using namespace vpux;

mlir::LogicalResult IE::InverseOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::InverseOpAdaptor inverse(operands, attrs, prop);
    if (mlir::failed(inverse.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = inverse.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto outDesc = vpux::getTensorAttr(ctx, inType.getDimsOrder(), inType.getMemSpace());
    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType(), outDesc);

    return mlir::success();
}
