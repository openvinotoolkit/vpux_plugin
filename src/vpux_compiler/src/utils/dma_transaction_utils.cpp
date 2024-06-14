//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/dma_transaction_utils.hpp"

namespace vpux {

//
// Below function reduceDimsForDma, tries to collapse contiguous dimensions in memory
// depending if the strides are compact; such that we'll end up with the most reduced
// transfer in terms of dimensionality;
// This is beneficial since the NPU DMA engines are limited in terms of the rank of the transfers
// 3D for NPU37XX; 6D for NPU40XX
//
// The general relation between sizes and strides for memrefs is
// Size[X] is the amount of elements at the index of X;
// Stride[X] is the amount of elements to jump in memory IN BETWEEN different elements of Size[X]
// Size:   [ 1,  2, 3, 4]
// Stride: [24, 12, 4, 1]
//
// When dealing with memory shapes and memory strides; the highest index represent the
// innermost data which is contiguous in memory, to which strides can also be applied;
// Plus memStrides also include the element type size in bits factored in.
//
// When factoring in layout, memory shapes and strides will result from the layout permutation
// applied to the logical shapes and strides.
//
// Example:
// Element type FP16
// Size:   [ 1,  2, 3, 4]
// Stride: [24,  1, 8, 2]
// Layout #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// MemSize   [1, 3, 4, 2]
// MemStride [24 * 16, 8 * 16, 2 * 16, 1 * 16]
//
// Then for the task of reducing the transfer dimensionality; we're doing the following steps:
// For example memref<1x2x3x4xf16, {order = #NCHW, strides = [64, 32, 8, 2]}
// MemSize:   [ 1,       2,      3,      4]
// MemStride: [64 * 16, 32 * 16, 8 * 16, 2 * 16]
// Here we have a two non contiguous levels of stride at indexes [1, 3]
//
// While when dealing with DMA transactions and concepts of sizes and strides;
//   Size[X] is the amount of data to transfer at the index of X, in terms of bytes
// Stride[X] is the amount of bytes to jump in memory AFTER all Size[X] were transferred
// Because we have a discrepancy with regards to what strides mean for memrefs and what
//  they mean for DMA transactions, we need to arrive at an aligned representation.
//
// So if we'd take the previous example, we'd extend them as such for DMA specific logic;
// Size:   [ 1,       2,       3,      4,     16] // by adding a new dim at end representing
//                                                type size in bits
// Stride: [64 * 16, 64 * 16, 32 * 16, 8 * 16, 2 * 16] // by adding a new dim at start
//                                  for the stride over all elements of size[0]
//
// Further taking this into consideration; if we'd like to reduce the array of size/strides
// based on the indexes which are contiguous, we'd be left with the following:
// Size:   [ 2,      12,      16]
// Stride: [64 * 16, 32 * 16, 2 * 16]
//
// At the same dimension reduction stage, we align the dimensions on byte boundary; such that
// it can be transferred by DMA; the engine does not support sub byte access
// Size:   [ 2,  12, 2] // representing dimensions of DMA transfer in bytes
// Stride: [128, 64, 4] // representing strides in bytes
//
// This is generic enough and can be used for all HW platforms to define the DMA transaction
//
// A pseudo code logic for how DMA engine executes the transfer would be:
//
// offset0 = 0;
// for(i0 = 0; i0 < SIZE[0]; i0++) {
//     offset1 = 0
//     for(i1 = 0; i1 < SIZE[1]; i1++) {
//         offset2 = 0
//         for(i2 = 0; i2 < SIZE[2]; i2++) {
//             Do_memory_access(SRC_ADDR + sum(offset<2,1,0>));
//             offset2 ++;
//         }
//         offset1 += STRIDE[2]
//     }
//     offset0 += STRIDE[1]
// }
//
// This fits well with our representation of DMA transaction dims where the innermost dim
// represents the amount of bytes to transfer and the outermost dims are just plane dimensions
// to further iterate over;
//
// Having only the individual reduced dimensions defined as above fits perfectly with the logic
// of how NPU40XX DMA works, since it does nesting for loops over the dimensions to transfer
// all data.
//
// Some extra adaptations need to be done for NPU37XX see function patchDimsForNPU37XX
//
// For NPU37XX style DMA it's a bit more peculiar, for which reason we need to product
// accumulate the sizes from innermost to outermost dimension.
// So we'd have:
// Size:   [2 * 12 * 2, 12 * 2, 2]
//   V
// Size:   [48, 24, 2]
//
// Then specifically for generation of NPU37XX descriptors which are driven by a total length and
// distributing that length in memory with up to 2 levels of strides
// We'd further reduce the above case to
// Total length: 48
// Size:   [24, 2]
// Stride: [64, 4]
//

DMATransaction reduceDimsForDma(vpux::NDTypeInterface ndType) {
    // Get memory view of shape and strides
    // Store them all as bits so to handle sub byte types
    auto memShape = to_small_vector(ndType.getMemShape());
    auto memStrides = to_small_vector(ndType.getMemStrides());

    // Extend shape and strides to accommodate for element type size and batch stride
    const auto elemSizeCount = ndType.getElemTypeSize().count();
    memShape.push_back(elemSizeCount);
    memStrides.insert(memStrides.begin(), memStrides.front() * memShape.front());
    auto innerMostIndex = memShape.size() - 1;

    llvm::SmallVector<uint64_t> reducedDims;
    llvm::SmallVector<uint64_t> reducedStrides;

    const auto alignToByteBoundary = [&](Bit val) {
        return alignMemSize(Bit(val), Byte(1)).to<Byte>().count();
    };

    // Iterate over dim/stride pairs and push cases of non compact strides
    int64_t accumulatedSize = 1;
    auto previousStrideInBits = Bit(1);
    for (int64_t dim = innerMostIndex; dim >= 0; --dim) {
        auto currentSize = memShape[dim];
        auto currentStrideInBits = memStrides[dim];

        accumulatedSize *= currentSize;
        // Found non-compact stride
        if (checked_cast<int64_t>(currentSize) * previousStrideInBits < currentStrideInBits) {
            reducedDims.push_back(alignToByteBoundary(Bit(accumulatedSize)));
            reducedStrides.push_back(alignToByteBoundary(currentStrideInBits));
            accumulatedSize = elemSizeCount;
        }

        previousStrideInBits = currentStrideInBits;
    }

    // Flush out remaining accumulated sizes.
    // Also handle scalar cases of 1 byte transfers.
    if (accumulatedSize > elemSizeCount || reducedDims.size() == 0) {
        reducedDims.push_back(alignToByteBoundary(Bit(accumulatedSize)));
        reducedStrides.push_back(alignToByteBoundary(memStrides.front()));
    }

    // Patch stages for easier later usage

    // Divide by elemSizeCount except innermost dim
    const auto elemSizeByteCount = static_cast<float>(elemSizeCount) / CHAR_BIT;
    for (size_t idx = 1; idx < reducedDims.size(); ++idx) {
        reducedDims[idx] = checked_cast<int64_t>(reducedDims[idx] / elemSizeByteCount);
    }

    // Reverse the arrays such that we keep the same logical order of dimensions as memrefs
    std::reverse(reducedDims.begin(), reducedDims.end());
    std::reverse(reducedStrides.begin(), reducedStrides.end());

    return DMATransaction(reducedDims, reducedStrides);
}

void patchDimsForNPU37XX(DMATransaction& dmaTransactionDetails) {
    // NPU37XX hacks from here on

    // Re - accumulate the reduced shape dims, for NPU37XX purpose;
    size_t accum = dmaTransactionDetails.dims.back();
    for (auto rIt = dmaTransactionDetails.dims.rbegin() + 1; rIt != dmaTransactionDetails.dims.rend(); ++rIt) {
        *rIt *= accum;
        accum = *rIt;
    }

    // Drop first pair of dims strides, those will be deduced based on totalLength
    if (dmaTransactionDetails.dims.size() > 1) {
        dmaTransactionDetails.dims.erase(dmaTransactionDetails.dims.begin());
        dmaTransactionDetails.strides.erase(dmaTransactionDetails.strides.begin());
    }
}

}  // namespace vpux
