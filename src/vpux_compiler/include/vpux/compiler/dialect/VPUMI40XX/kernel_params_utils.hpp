//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

#include <kernels/inc/common_types.h>

#include <mlir/IR/Value.h>

namespace vpux {
namespace VPUMI40XX {

class KernelParamsSerializer {
public:
    KernelParamsSerializer() = delete;

    static SmallVector<uint8_t> createKernelParams(VPUIP::SwKernelOp swKernelOp);

    static sw_params::Location getSwParamsLocationFromMemKind(VPU::MemoryKind memKind);
    static sw_params::DataType getDataTypeFromMlirType(mlir::Type type);

private:
    static void addTensorArgToVector(SmallVector<uint8_t>& vec, std::optional<uint32_t> tileMaskForBroadcast,
                                     mlir::Value value, bool isDynamic);
    static void addAttrsToVector(SmallVector<uint8_t>& vec, mlir::Attribute attr);
    static void addBasicAttrToVector(SmallVector<uint8_t>& vec, mlir::Attribute attr);

    template <class T>
    static void appendValueToVector(SmallVector<uint8_t>& vec, const T& anyValue) {
        ArrayRef<uint8_t> valueAsArray(reinterpret_cast<const uint8_t*>(&anyValue), sizeof(anyValue));
        vec.insert(vec.end(), valueAsArray.begin(), valueAsArray.end());
    }
};

}  // namespace VPUMI40XX
}  // namespace vpux
