//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "mlir/Support/LogicalResult.h"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/utils/core/logger.hpp"

#include <functional>

namespace vpux {
namespace IE {

mlir::LogicalResult broadcastContentAttrs(vpux::Const::ContentAttr& inLowContentAttr,
                                          vpux::Const::ContentAttr& inHighContentAttr,
                                          vpux::Const::ContentAttr& transformContentAttr, const Logger& log);

mlir::FailureOr<std::tuple<vpux::Const::ContentAttr, vpux::Const::ContentAttr, mlir::RankedTensorType>>
applyTransformation(vpux::Const::ContentAttr inLowContentAttr, vpux::Const::ContentAttr inHighContentAttr,
                    vpux::Const::ContentAttr transformContentAttr,
                    const std::function<float(float, float)>& transformCb, const Logger& log);

mlir::LogicalResult applyScaleShift(const Const::ContentAttr& scale, const Const::ContentAttr& shift,
                                    Const::ContentAttr& low, Const::ContentAttr& high,
                                    vpux::NDTypeInterface& storageType, const Logger& log);

mlir::LogicalResult revertScaleShift(const Const::ContentAttr& scale, const Const::ContentAttr& shift,
                                     Const::ContentAttr& low, Const::ContentAttr& high,
                                     vpux::NDTypeInterface& storageType, const Logger& log);

template <typename NewT>
Const::ContentAttr castStorageType(const Const::Content& content);

class WeightsDequantizeStructureInfo final {
    //                     --- Constant Input Case ---
    //
    //   +----------------------------------------------------------------+
    //   | Weights Const - i8 with transformations                        |
    //   |  [#const.ConvertElemType<i4>] || [#const.ConvertElemType<u4>]  |
    //   | [#const.ConvertElemType<f16>] || [#const.ConvertElemType<f32>] |
    //   | Weights Const - u8 with transformations                        |
    //   |  [#const.ConvertElemType<i4>] || [#const.ConvertElemType<u4>]  |
    //   | [#const.ConvertElemType<f16>] || [#const.ConvertElemType<f32>] |
    //   | Weights Const - f16 with transformations                       |
    //   |  [#const.ConvertElemType<i4>] || [#const.ConvertElemType<u4>]  |
    //   +----------------------------------------------------------------+
    //             |
    //             |      +-------------+
    //             |      | Shift Const |
    //             |      +-------------+
    //             |           |
    //          +-------------------+
    //          | Optional Subtract |
    //          +-------------------+
    //                    |
    //                    |   +-------------+
    //                    |   | Scale Const |
    //                    |   +-------------+
    //                    |          |
    //                +-------------------+
    //                | Optional Multiply |
    //                +-------------------+
    //                          |

    //        --- Block Argument Input Case ---
    //
    //      [Block Argument]    (si8/ui8/si4/ui4)
    //             |
    //   +--------------------+
    //   | Convert to f16/f32 |
    //   +--------------------+
    //             |
    //             |      +-------------+
    //             |      | Shift Const |
    //             |      +-------------+
    //             |           |
    //          +-------------------+
    //          | Optional Subtract |
    //          +-------------------+
    //                    |
    //                    |   +-------------+
    //                    |   | Scale Const |
    //                    |   +-------------+
    //                    |          |
    //                +-------------------+
    //                | Optional Multiply |
    //                +-------------------+
    //                          |

    //               --- Result ---
    //
    //                      |
    //            +--------------------+
    //            | Convert to f16/f32 |
    //            | (kept if present)  |
    //            +--------------------+
    //                      |
    //   +--------------------------------------+
    //   |             FakeQuantize             |
    //   |  inLow   = type_min                  |
    //   |  inHigh  = type_max                  |
    //   |  outLow  = (inLow - shift) * scale   |
    //   |  outHigh = (inHigh - shift) * scale  |
    //   |  levels  = 256/255 (i8), 16/15 (i4)  |
    //   +--------------------------------------+
    //                      |

    //   Subtract and Multiply operation are optional in the dequantization pattern, because they can be folded

    //   Storing i4 values as constants is not possible in MLIR; so even when we import the model we should import
    //   the model in our frontend we should create a Constant that has higher level storage such as SI8 and a
    //   transformation which converts the expressed type to I4/U4
    //   Example: ConvertElemType<si4>, ConvertElemType<ui4>

private:
    mlir::Type inputElemBaseType = nullptr;
    mlir::Type inputElemConvertType = nullptr;
    mlir::Type inputVirtualI4ElemType = nullptr;       // Requried because I4 is represented using ConvertElemType
    mlir::Type initialInputElemStorageType = nullptr;  // Storage type before high precision cast [const only]

    Const::ContentAttr shift = nullptr;  // From subtract op (if present)
    Const::ContentAttr scale = nullptr;  // From multiply op (if present)

    mlir::Operation* firstOp = nullptr;  // The first op of the WD structure
    mlir::Operation* lastOp = nullptr;   // The last op of the WD structure

    mlir::Value inputValue = nullptr;          // With applied ConvertOps effects [block arg. only]
    mlir::BlockArgument inputBlock = nullptr;  // [block arg. only]

    Const::ContentAttr inputAttr = nullptr;                     // [const only]
    std::optional<Const::Content> inputContent = std::nullopt;  // caches constant folding [const only]

    bool isValid = false;

    [[nodiscard]] mlir::LogicalResult initializeStructure(IE::MultiplyOp& multiplyOp);
    [[nodiscard]] mlir::LogicalResult initializeStructure(IE::SubtractOp& subtractOp);
    [[nodiscard]] mlir::LogicalResult initializeStructure(IE::ConvertOp& convertOp);
    [[nodiscard]] mlir::LogicalResult initializeStructure(Const::DeclareOp& declareOp);

public:
    const Logger log;

    WeightsDequantizeStructureInfo(Const::DeclareOp& origOp, const Logger& log) noexcept;
    WeightsDequantizeStructureInfo(IE::ConvertOp& origOp, const Logger& log) noexcept;

    mlir::MLIRContext* getContext() const;
    const mlir::Location getLocation() const;

    mlir::Operation* getFirstOp() const;
    mlir::Operation* getLastOp() const;

    bool isSuccessfulMatch() const;

    bool hasConstInput() const;
    bool has8BitIntegerInput() const;
    bool has4BitIntegerInput() const;
    bool has8BitFloatInput() const;
    bool hasSignedInput() const;

    mlir::ShapedType getInputShapedType() const;
    mlir::Type getInputElemBaseType() const;
    mlir::Type getInputElemConvertType() const;
    mlir::Type getInputFinalElemConvertType() const;
    mlir::Type getInputElemStorageType() const;
    int64_t getInputShapeRank() const;
    int64_t getQuantizationLevels() const;

    // --- Const only ---
    Const::DeclareOp getInputDeclareOp() const;
    const Const::Content& getInputContent() const;
    const Const::ContentAttr& getInputContentAttr() const;

    // --- Block arg only ---
    const mlir::Value getInputValue() const;

    mlir::LogicalResult ensureHighPrecisionStorage();

    [[nodiscard]] std::pair<Const::ContentAttr, Const::ContentAttr> getInputQuantizationInterval(const float low,
                                                                                                 const float high);
    [[nodiscard]] std::pair<Const::ContentAttr, Const::ContentAttr> getOutputQuantizationInterval(
            std::pair<Const::ContentAttr, Const::ContentAttr> inputInterval);
};

std::set<int64_t> findAxes(IE::FakeQuantizeOp origOp);

}  // namespace IE
}  // namespace vpux
