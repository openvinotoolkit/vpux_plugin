#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/types.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

#include <npu_40xx_nnrt.hpp>
using namespace npu40xx;

using namespace vpux;

void vpux::NPUReg40XX::WorkItemOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    nn_public::VpuWorkItem workItem = {};
    auto workItemDesc = getWorkItemDescriptor().getRegMapped();
    auto serializedworkItemDesc = workItemDesc.serialize();

    memcpy(reinterpret_cast<uint8_t*>(&workItem), serializedworkItemDesc.data(), serializedworkItemDesc.size());

    auto ptrCharTmp = reinterpret_cast<uint8_t*>(&workItem);

    binDataSection.appendData(ptrCharTmp, getBinarySize());
}

size_t vpux::NPUReg40XX::WorkItemOp::getBinarySize() {
    return sizeof(nn_public::VpuWorkItem);
}
