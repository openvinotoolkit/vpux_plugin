//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/loop.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/subspaces.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/IR/DialectImplementation.h>

using namespace vpux;

//
// SubViewAttr::verify
//

mlir::LogicalResult vpux::Const::SubViewAttr::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                                     mlir::ArrayAttr offset, mlir::ArrayAttr shape) {
    if (offset == nullptr) {
        return printTo(emitError(), "Got NULL 'offset' in 'SubViewAttr'");
    }
    if (shape == nullptr) {
        return printTo(emitError(), "Got NULL 'shape' in 'SubViewAttr'");
    }

    if (offset.size() != shape.size()) {
        return printTo(emitError(), "Got inconsistent 'offset' and 'shape' values in 'SubViewAttr'");
    }

    for (const auto dimAttr : offset.getValue()) {
        if (!dimAttr.isa<mlir::IntegerAttr>()) {
            return printTo(emitError(), "Got non-integer value '{0}' in 'offset' for 'SubViewAttr'", dimAttr);
        }
        if (dimAttr.cast<mlir::IntegerAttr>().getInt() < 0) {
            return printTo(emitError(), "Got unsupported dimension value '{0}' in 'offset' for 'SubViewAttr'", dimAttr);
        }
    }

    for (const auto dimAttr : shape.getValue()) {
        if (!dimAttr.isa<mlir::IntegerAttr>()) {
            return printTo(emitError(), "Got non-integer value '{0}' in 'shape' for 'SubViewAttr'", dimAttr);
        }
        if (dimAttr.cast<mlir::IntegerAttr>().getInt() <= 0) {
            return printTo(emitError(), "Got unsupported dimension value '{0}' in 'shape' for 'SubViewAttr'", dimAttr);
        }
    }

    return mlir::success();
}

//
// SubViewAttr::print
//

void vpux::Const::SubViewAttr::print(mlir::AsmPrinter& printer) const {
    printer << "<";
    printer.printAttribute(getOffset());
    printer << ", ";
    printer.printAttribute(getShape());
    printer << ">";
}

//
// SubViewAttr::parse
//

mlir::Attribute vpux::Const::SubViewAttr::parse(mlir::AsmParser& parser, mlir::Type) {
    if (mlir::failed(parser.parseLess())) {
        return nullptr;
    }

    mlir::ArrayAttr offset;
    if (mlir::failed(parser.parseAttribute(offset))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseComma())) {
        return nullptr;
    }

    mlir::ArrayAttr shape;
    if (mlir::failed(parser.parseAttribute(shape))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseGreater())) {
        return nullptr;
    }

    return parser.getChecked<Const::SubViewAttr>(offset, shape);
}

//
// SubViewAttr::inferOutputType
//

vpux::NDTypeInterface vpux::Const::SubViewAttr::inferOutputType(vpux::NDTypeInterface input) const {
    const auto shape = parseIntArrayAttr<int64_t>(getShape());
    const auto offset = parseIntArrayAttr<int64_t>(getOffset());

    VPUX_THROW_UNLESS(shape.size() == checked_cast<size_t>(input.getRank()),
                      "View shape and input shape are not consistent in 'SubViewAttr'");

    return input.extractDenseTile(ShapeRef(offset), ShapeRef(shape));
}

bool vpux::Const::SubViewAttr::inferOutputSplat(bool inputIsSplat, vpux::NDTypeInterface) {
    return inputIsSplat;
}

//
// SubViewAttr::transform
//

Const::Content vpux::Const::SubViewAttr::transform(vpux::Const::Content& input) const {
    const auto storageElemTypeSize = vpux::getElemTypeSize(input.getStorageElemType()).count();
    VPUX_THROW_UNLESS(storageElemTypeSize >= CHAR_BIT || storageElemTypeSize == 1,
                      "SubViewAttr does not support sub-byte element types except i1, got {0}",
                      input.getStorageElemType());

    auto output = Const::Content::allocTempBuffer(inferOutputType(input.getType()), input.getStorageElemType(),
                                                  inferOutputSplat(input.isSplat(), input.getType()));

    const auto inBuf = input.getRawStorageBuf();
    auto outBuf = output.getRawTempBuf();

    if (input.isSplat()) {
        std::copy_n(inBuf.data(), inBuf.size(), outBuf.data());
    } else {
        const auto order = input.getType().getDimsOrder();

        const auto inShape = input.getType().getShape();
        const auto inMemShape = order.toMemoryOrder(inShape);

        const auto outShape = output.getType().getShape();
        const auto outMemShape = order.toMemoryOrder(outShape);

        const auto offset = Shape(parseIntArrayAttr<int64_t>(getOffset()));
        const auto memOffset = order.toMemoryOrder(offset);

        // Type specific implementation: i1
        if (getElemTypeSize(input.getStorageElemType()).count() == 1) {
            std::fill(outBuf.begin(), outBuf.end(), 0x00);
            auto outBufData = outBuf.data();
            /*
             * The body of this loop works with bits packed into a byte.
             * It overwrites the entire byte in output each time any bit is written into that byte.
             * It cannot be run as parallel loop as straightforward as other implementations since
             * different threads may write into the same bytes.
             */
            for (int64_t outMemInd1D = 0; outMemInd1D < output.getType().getNumElements(); ++outMemInd1D) {
                const auto outMemIndND = getMemIndexND(outMemInd1D, outMemShape);
                MemShape inMemIndND(outMemIndND.size());
                for (auto ind : irange(inMemIndND.size())) {
                    const auto md = MemDim(ind);
                    inMemIndND[md] = outMemIndND[md] + memOffset[md];
                }
                const auto inMemInd1D = getMemIndex1D(inMemIndND, inMemShape);
                // Extract bit value
                const auto inputCoord = std::div(inMemInd1D, checked_cast<int64_t>(CHAR_BIT));
                const auto bitValue = static_cast<bool>(inBuf.data()[inputCoord.quot] & (1 << inputCoord.rem));
                if (!bitValue) {
                    continue;
                }
                // If non zero then set corresponding bit in output
                const auto outputCoord = std::div(outMemInd1D, checked_cast<int64_t>(CHAR_BIT));
                outBufData[outputCoord.quot] = static_cast<char>(outBufData[outputCoord.quot] | (1 << outputCoord.rem));
            }

            return output;
        }

        const Byte elemSize = getElemTypeSize(input.getStorageElemType());

        if (memOffset.size() == 1) {
            // Opitimized 1D case

            std::copy_n(inBuf.data() + memOffset.front() * elemSize.count(),
                        checked_cast<size_t>(output.getType().getNumElements() * elemSize.count()), outBuf.data());
        } else if (memOffset.size() == 2) {
            // Opitimized 2D case

            const auto md0 = MemDim(0);
            const auto md1 = MemDim(1);

            const auto OUT0 = outMemShape[md0];
            const auto OUT1 = outMemShape[md1];

            const auto IN1 = inMemShape[md1];

            const auto off0 = memOffset[md0];
            const auto off1 = memOffset[md1];

            loop_2d(LoopExecPolicy::Parallel, input.getStorageElemType().getContext(), OUT0, OUT1,
                    [&](int64_t out0, int64_t out1) {
                        const auto in0 = out0 + off0;
                        const auto in1 = out1 + off1;

                        const auto outRawInd = out1 + out0 * OUT1;
                        const auto inRawInd = in1 + in0 * IN1;

                        std::copy_n(inBuf.data() + checked_cast<size_t>(inRawInd * elemSize.count()),
                                    checked_cast<size_t>(elemSize.count()),
                                    outBuf.data() + checked_cast<size_t>(outRawInd * elemSize.count()));
                    });
        } else if (memOffset.size() == 3) {
            // Opitimized 3D case

            const auto md0 = MemDim(0);
            const auto md1 = MemDim(1);
            const auto md2 = MemDim(2);

            const auto OUT0 = outMemShape[md0];
            const auto OUT1 = outMemShape[md1];
            const auto OUT2 = outMemShape[md2];

            const auto IN1 = inMemShape[md1];
            const auto IN2 = inMemShape[md2];

            const auto off0 = memOffset[md0];
            const auto off1 = memOffset[md1];
            const auto off2 = memOffset[md2];

            loop_3d(LoopExecPolicy::Parallel, input.getStorageElemType().getContext(), OUT0, OUT1, OUT2,
                    [&](int64_t out0, int64_t out1, int64_t out2) {
                        const auto in0 = out0 + off0;
                        const auto in1 = out1 + off1;
                        const auto in2 = out2 + off2;

                        const auto outRawInd = out2 + out1 * OUT2 + out0 * OUT2 * OUT1;
                        const auto inRawInd = in2 + in1 * IN2 + in0 * IN2 * IN1;

                        std::copy_n(inBuf.data() + checked_cast<size_t>(inRawInd * elemSize.count()),
                                    checked_cast<size_t>(elemSize.count()),
                                    outBuf.data() + checked_cast<size_t>(outRawInd * elemSize.count()));
                    });
        } else if (memOffset.size() == 4) {
            // Opitimized 4D case

            const auto md0 = MemDim(0);
            const auto md1 = MemDim(1);
            const auto md2 = MemDim(2);
            const auto md3 = MemDim(3);

            const auto OUT0 = outMemShape[md0];
            const auto OUT1 = outMemShape[md1];
            const auto OUT2 = outMemShape[md2];
            const auto OUT3 = outMemShape[md3];

            const auto IN1 = inMemShape[md1];
            const auto IN2 = inMemShape[md2];
            const auto IN3 = inMemShape[md3];

            const auto off0 = memOffset[md0];
            const auto off1 = memOffset[md1];
            const auto off2 = memOffset[md2];
            const auto off3 = memOffset[md3];

            loop_4d(LoopExecPolicy::Parallel, getContext(), OUT0, OUT1, OUT2, OUT3,
                    [&](int64_t out0, int64_t out1, int64_t out2, int64_t out3) {
                        const auto in0 = out0 + off0;
                        const auto in1 = out1 + off1;
                        const auto in2 = out2 + off2;
                        const auto in3 = out3 + off3;

                        const auto outRawInd = out3 + out2 * OUT3 + out1 * OUT3 * OUT2 + out0 * OUT3 * OUT2 * OUT1;
                        const auto inRawInd = in3 + in2 * IN3 + in1 * IN3 * IN2 + in0 * IN3 * IN2 * IN1;

                        std::copy_n(inBuf.data() + checked_cast<size_t>(inRawInd * elemSize.count()),
                                    checked_cast<size_t>(elemSize.count()),
                                    outBuf.data() + checked_cast<size_t>(outRawInd * elemSize.count()));
                    });
        } else {
            // Generic case

            loop_1d(LoopExecPolicy::Parallel, getContext(), output.getType().getNumElements(),
                    [&](int64_t outMemInd1D) {
                        const auto outMemIndND = getMemIndexND(outMemInd1D, outMemShape);

                        MemShape inMemIndND(outMemIndND.size());
                        for (auto ind : irange(inMemIndND.size())) {
                            const auto md = MemDim(ind);
                            inMemIndND[md] = outMemIndND[md] + memOffset[md];
                        }

                        const auto inMemInd1D = getMemIndex1D(inMemIndND, inMemShape);

                        const auto inMemRawInd = checked_cast<size_t>(inMemInd1D * elemSize.count());
                        VPUX_THROW_UNLESS(inMemRawInd < inBuf.size(), "Out-of-bound access in 'SubViewAttr'");

                        const auto outMemRawInd = checked_cast<size_t>(outMemInd1D * elemSize.count());
                        VPUX_THROW_UNLESS(outMemRawInd < outBuf.size(), "Out-of-bound access in 'SubViewAttr'");

                        std::copy_n(inBuf.data() + inMemRawInd, checked_cast<size_t>(elemSize.count()),
                                    outBuf.data() + outMemRawInd);
                    });
        }
    }

    return output;
}

//
// ContentAttr::subview
//

Const::ContentAttr vpux::Const::ContentAttr::subview(ShapeRef offset, ShapeRef shape) const {
    return ContentAttr::addTransformation(
            *this, Const::SubViewAttr::get(getIntArrayAttr(getContext(), offset), getIntArrayAttr(getContext(), shape))
                           .cast<Const::TransformAttrInterface>());
}

//
// SubViewAttr::supportsSubByteStorageType
//

bool Const::SubViewAttr::supportsSubByteStorageType() const {
    return true;
}
