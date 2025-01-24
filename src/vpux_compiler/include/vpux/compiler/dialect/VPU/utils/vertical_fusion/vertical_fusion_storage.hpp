//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

namespace vpux {
namespace VPU {

/*
 Container for storage connection between object and information about it
 separated by VF tile

 VFKey - object (ex. block argument, operation)
 VFValue - additional information about object (ex. TileInfo)

 Compare - comparator for VFValue, storage keeps max element of info
*/
template <class VFKey, class VFValue, class Compare = std::less<VFValue>>
class VFContainer {
public:
    // connection between number of tile and info
    using VFTileContainer = DenseMap<size_t, VFValue>;

    // pointer to container
    using UPtr = std::unique_ptr<VFContainer<VFKey, VFValue, Compare>>;

    // merge two containers, in case both containers have info
    // for same tile, max element is chosen based on comparator
    void merge(const VFContainer<VFKey, VFValue, Compare>& src);

    // insert new element in container, in case there is already
    // info for object and tile, max element is chosen based on comparator
    void insert(VFKey key, size_t tile, const VFValue& src);

    // get information about object for exact tile
    std::optional<VFValue> get(VFKey key, size_t tile);

    // function returns information gathered together for all tiles
    std::vector<VFValue> gatherValue(VFKey key);

    // get whole inner container
    const llvm::DenseMap<VFKey, VFTileContainer>& getAll() const {
        return vfContainer;
    };

private:
    // inner container for storage connection
    DenseMap<VFKey, VFTileContainer> vfContainer;

    // comparator for elements of info
    Compare vfComparator;
};

template <class VFKey, class VFValue, class Compare>
void vpux::VPU::VFContainer<VFKey, VFValue, Compare>::merge(const VFContainer<VFKey, VFValue, Compare>& src) {
    for (auto item : src.getAll()) {
        if (vfContainer.count(item.first) == 0) {
            vfContainer[item.first] = item.second;
        } else {
            for (auto tileItem : item.second) {
                insert(item.first, tileItem.first, tileItem.second);
            }
        }
    }
}

template <class VFKey, class VFValue, class Compare>
void vpux::VPU::VFContainer<VFKey, VFValue, Compare>::insert(VFKey key, size_t tile, const VFValue& src) {
    auto foundTileItem = llvm::find_if(vfContainer[key], [&](const auto& i) {
        return tile == i.first;
    });

    if (foundTileItem == vfContainer[key].end()) {
        vfContainer[key].try_emplace(tile, src);
    } else {
        foundTileItem->second = std::max(foundTileItem->second, src, vfComparator);
    }
}

template <class VFKey, class VFValue, class Compare>
std::optional<VFValue> vpux::VPU::VFContainer<VFKey, VFValue, Compare>::get(VFKey key, size_t tile) {
    auto foundItem = vfContainer.find(key);

    if (foundItem == vfContainer.end()) {
        return std::nullopt;
    }

    auto foundTile = foundItem->second.find(tile);

    if (foundTile == foundItem->second.end()) {
        return std::nullopt;
    }

    return foundTile->second;
}

template <class VFKey, class VFValue, class Compare>
std::vector<VFValue> vpux::VPU::VFContainer<VFKey, VFValue, Compare>::gatherValue(VFKey key) {
    return to_std_vector(vfContainer[key] | transformed([](const auto& item) {
                             return item.second;
                         }));
}

}  // namespace VPU
}  // namespace vpux
