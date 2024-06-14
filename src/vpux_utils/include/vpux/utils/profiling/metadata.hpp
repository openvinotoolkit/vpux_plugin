//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <flatbuffers/flatbuffers.h>

#include <stdint.h>
#include <cstddef>
#include <vector>

namespace ProfilingFB {
struct ProfilingMeta;
}
namespace MVCNN {
struct GraphFile;
}

namespace vpux::profiling {

constexpr uint32_t PROFILING_SECTION_ENCODING = 1;  // Profiling metadata encoded in FB format

constexpr uint32_t PROFILING_METADATA_VERSION_MAJOR = 2;  // Initial major version of FB schema

constexpr uint32_t PROFILING_METADATA_VERSION_MINOR = 0;  // Initial minor version of FB schema

// The layout is:
// +----------------------------+-----------------------
// | SECTION_ENCODING: uint32_t |  DATA...
// +----------------------------+-----------------------
uint32_t getProfilingSectionEncoding(const uint8_t* data, size_t size);

// Creates profiling metadata section. Profiling metadata is stored in flatbuffer(FB) object, where first
// fields are major/minor versions. FB object is aligned by 8 bytes boundary
// +----------------------------+---------------------+------------
// | SECTION_ENCODING: uint32_t |  DATA_LEN: uint32_t | DATA....
// +----------------------------+---------------------+------------
std::vector<uint8_t> constructProfilingSectionWithHeader(flatbuffers::DetachedBuffer rawMetadataFb);

bool isElfBinary(const uint8_t* data, size_t size);

const MVCNN::GraphFile* getGraphFileVerified(const uint8_t* buffer, size_t size);

const ProfilingFB::ProfilingMeta* getProfilingSectionMeta(const uint8_t* blobData, size_t blobSize);

const uint8_t* getProfilingSectionPtr(const uint8_t* blobData, size_t blobSize);

}  // namespace vpux::profiling
