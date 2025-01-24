//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/strings.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/profiling/location.hpp"

#include <iostream>

using namespace vpux;

namespace {

std::pair<mlir::SmallVector<std::string>, mlir::DictionaryAttr> getPrimaryLocationComponents(mlir::Location location) {
    mlir::SmallVector<std::string> locParts;
    auto metadata = mlir::DictionaryAttr::get(location.getContext());

    if (auto fusedLoc = location.dyn_cast<mlir::FusedLoc>()) {
        if (auto fusedMeta = fusedLoc.getMetadata()) {
            auto metaDict = fusedMeta.dyn_cast<mlir::DictionaryAttr>();
            VPUX_THROW_UNLESS(metaDict, "Metadata is not a DictionaryAttribute", fusedMeta);
            if (!metaDict.empty()) {
                metadata = metaDict;
            }
        }
        bool seenNestedFusedLoc = false;
        for (mlir::Location subLoc : fusedLoc.getLocations()) {
            // if fusedLoc has multiple nested FusedLocs -- we only handle the first one
            // and ignore other fused locs
            if (subLoc.isa<mlir::FusedLoc>()) {
                if (seenNestedFusedLoc) {
                    continue;
                }
                seenNestedFusedLoc = true;
            }
            const auto partsAndMeta = getPrimaryLocationComponents(subLoc);
            locParts.append(partsAndMeta.first);
            if (metadata.empty() && !partsAndMeta.second.empty()) {
                metadata = partsAndMeta.second;
            }
        }
    } else if (auto nameLoc = location.dyn_cast<mlir::NameLoc>()) {
        locParts.push_back(nameLoc.getName().str());
    } else if (auto callSiteLoc = location.dyn_cast<mlir::CallSiteLoc>()) {
        // CallSiteLoc consist of Caller location and Callee location. At this moment compile doesn't support recursive
        // and nested calls, so Caller locations is @main func loc with some extra suffix. Caller metadata will be
        // always same and doesn't represent actual layer type, so we will forward metadata from the Callee
        const auto calleeParts = getPrimaryLocationComponents(callSiteLoc.getCallee());
        const auto callerParts = getPrimaryLocationComponents(callSiteLoc.getCaller());

        locParts = callerParts.first;
        locParts.append(calleeParts.first.begin(), calleeParts.first.end());
        metadata = calleeParts.second;
    } else if (location.dyn_cast<mlir::UnknownLoc>() || location.dyn_cast<mlir::FileLineColLoc>()) {
        // NOTE: This behavior is here to prevent breaking 100+ lit tests which assume that
        // stringifyPrimaryLocation() for FileLineColLoc and UnknownLoc returns an empty string.
        // TODO: E#93652
        locParts.push_back("");
    } else {
        VPUX_THROW("Unknown location type: {0}", location);
    }

    return std::make_pair(locParts, metadata);
}

std::string encodeLocationAsString(mlir::ArrayRef<std::string> locationParts) {
    VPUX_THROW_WHEN(locationParts.size() == 0, "Can't stringify empty location");
    std::ostringstream ostr;
    ostr << locationParts[0];

    bool isFirst = true;
    auto separator = LOCATION_ORIGIN_SEPARATOR;
    for (const auto& part : locationParts.drop_front()) {
        ostr << separator << part;
        if (isFirst) {
            separator = LOCATION_SEPARATOR;
            isFirst = false;
        }
    }

    return ostr.str();
}

std::string getLayerTypeFromMeta(mlir::DictionaryAttr meta) {
    VPUX_THROW_WHEN(meta.empty(), "Missing location metadata.");
    auto typeAttr = meta.get("type");
    VPUX_THROW_UNLESS(typeAttr, "Location metadata is expected to provide type: {0}", meta);
    auto stringAttr = typeAttr.dyn_cast<mlir::StringAttr>();
    VPUX_THROW_UNLESS(stringAttr, "type is supposed to be StringAttr: {0}", typeAttr);
    return stringAttr.getValue().str();
}

}  // namespace

/**
 * Convert mlir::Location into a string, putting LOCATION_ORIGIN_SEPARATOR ('?') after the
 * part denoting the layer name and LOCATION_SEPARATOR ('/') between other parts.
 *
 * Example:
 *   stringifyPrimaryLocation(loc(fused{type = "Conv"}["Convolution", "tile_1", "cluster_1"])) ==
 *     "Convolution?t_Conv/cluster_1/tile_1"
 *
 *
 * This function also handles special cases where two locations get fused together and the resulting
 * location has mulitple source layer name/type components. The conflict is resolved by keeping the
 * first (primary) layer name/type and ignoring the others.
 *
 *   stringifyPrimaryLocation(loc(fused[fused<{type = "T1"}>["L1"], fused<{type = "T2"}>["L2"], "sfx1", "sfx2"])) ==
 * L1?t_T1/sfx1/sfx2
 *
 * @see MLIR_ProfilingUtils unit tests
 */
std::string vpux::stringifyPrimaryLocation(mlir::Location location) {
    auto partsAndMeta = getPrimaryLocationComponents(location);
    auto locParts = partsAndMeta.first;
    auto meta = partsAndMeta.second;

    VPUX_THROW_WHEN(locParts.empty(), "No primary location components: {0}", location);

    mlir::SmallVector<std::string> parts{locParts[0]};

    if (!meta.empty()) {  // sync_dma, etc.
        parts.push_back("t_" + getLayerTypeFromMeta(meta));
    }
    parts.append(locParts.begin() + 1, locParts.end());

    return encodeLocationAsString(parts);
}

/**
 * Retrieves the original OV layer type from the task's location.
 */
std::string vpux::getLayerTypeFromLocation(mlir::Location location) {
    auto partsAndMeta = getPrimaryLocationComponents(location);
    auto meta = partsAndMeta.second;
    if (meta.empty()) {  // sync_dma, etc.
        return "<Unknown>";
    }
    return getLayerTypeFromMeta(meta);
}
