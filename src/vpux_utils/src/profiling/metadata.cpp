//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/utils/profiling/metadata.hpp"

#include "schema/graphfile_generated.h"
#include "schema/profiling_generated.h"

#include <vpux_elf/accessor.hpp>
#include <vpux_elf/reader.hpp>

#include <sstream>
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/profiling/common.hpp"

using vpux::checked_cast;

namespace {

// To avoid another level of buffers nesting, schema versions is declared as first fields in FB schema, so any
// subsequent changes won't break binary compatibility between parser and data. This assert is needed to prevent reorder
// of fields during development
static_assert(ProfilingFB::ProfilingMeta::FlatBuffersVTableOffset::VT_MAJORVERSION == 4,
              "Schema major version isn't first field in FB metadata. Fix it or change major version");
static_assert(ProfilingFB::ProfilingMeta::FlatBuffersVTableOffset::VT_MINORVERSION == 6,
              "Schema minor version isn't second field in FB metadata. Fix it or change major version");

using SchemaAndPtrType = std::pair<const ProfilingFB::ProfilingMeta*, const uint8_t*>;
using namespace vpux::profiling;

constexpr size_t HEADER_SIZE_V1 = /*version_size*/ sizeof(uint32_t) + /*pad_size*/ sizeof(uint32_t);

const SchemaAndPtrType getProfilingMetaBufferVerified(const uint8_t* buffer, size_t size) {
    auto verifier = flatbuffers::Verifier(buffer, size);
    VPUX_THROW_UNLESS(ProfilingFB::VerifyProfilingMetaBuffer(verifier), "Cannot verify profiling metadata integrity");
    return {ProfilingFB::GetProfilingMeta(buffer), buffer};
}

const SchemaAndPtrType extractProfilingMetadata(const uint8_t* data, size_t size) {
    const uint32_t metadataEncoding = getProfilingSectionEncoding(data, size);
    VPUX_THROW_UNLESS(metadataEncoding == PROFILING_SECTION_ENCODING, "Incompatible profiling metadata encoding");

    const auto schemaAndPtr = getProfilingMetaBufferVerified(data + HEADER_SIZE_V1, size);
    const auto schemaMajorVersion = schemaAndPtr.first->majorVersion();
    // Mismatch of schema major version leads to full incompatibility
    VPUX_THROW_UNLESS(schemaMajorVersion == PROFILING_METADATA_VERSION_MAJOR,
                      "Can't deserialize profiling schema of version {0} by v{1} parser", schemaMajorVersion,
                      PROFILING_METADATA_VERSION_MAJOR);

    const auto schemaMinorVersion = schemaAndPtr.first->minorVersion();
    if (schemaMinorVersion != PROFILING_METADATA_VERSION_MINOR) {
        vpux::Logger::global().warning(
                "Trying to parse blob with profiling metadata version v{0}.{1} with v{2}.{3} parser. Some information "
                "may be unavailable, re-compile blob to get all features",
                schemaMajorVersion, schemaMinorVersion, PROFILING_METADATA_VERSION_MAJOR,
                PROFILING_METADATA_VERSION_MINOR);
    }

    return schemaAndPtr;
}

const SchemaAndPtrType getProfilingSectionDataElf(const uint8_t* blobData, size_t blobSize) {
    VPUX_THROW_WHEN(blobData == nullptr, "Empty input data");

    elf::DDRAccessManager<elf::DDRAlwaysEmplace> elfAccess(blobData, blobSize);
    elf::Reader<elf::ELF_Bitness::Elf64> reader(&elfAccess);
    for (size_t i = 0; i < reader.getSectionsNum(); ++i) {
        const auto section = reader.getSection(i);
        const std::string secName = section.getName();
        if (secName == ".profiling") {
            const uint8_t* profMetaSection = section.getData<uint8_t>();
            const size_t sectionSize = section.getEntriesNum();
            return extractProfilingMetadata(profMetaSection, sectionSize);
        }
    }
    VPUX_THROW("Cannot find profiling section");
}

SchemaAndPtrType getProfilingSectionDataImpl(const uint8_t* blobData, size_t blobSize) {
    return getProfilingSectionDataElf(blobData, blobSize);
}

};  // namespace

namespace vpux {
namespace profiling {

uint32_t getProfilingSectionEncoding(const uint8_t* data, size_t size) {
    VPUX_THROW_WHEN(size < sizeof(uint32_t), "Malformed profiling metadata section");
    return *reinterpret_cast<const uint32_t*>(data);
}

std::vector<uint8_t> constructProfilingSectionWithHeader(flatbuffers::DetachedBuffer rawMetadataFb) {
    constexpr size_t SECTION_ENCODING_INDEX = 0;
    constexpr size_t LEN_INDEX = 1;

    const size_t metadataSize = rawMetadataFb.size();
    const size_t extendedBufferSize = metadataSize + HEADER_SIZE_V1;

    std::vector<uint8_t> buffer(extendedBufferSize);

    uint32_t* header = reinterpret_cast<uint32_t*>(buffer.data());
    header[SECTION_ENCODING_INDEX] = PROFILING_SECTION_ENCODING;
    header[LEN_INDEX] = checked_cast<uint32_t>(rawMetadataFb.size());

    uint8_t* body = buffer.data() + HEADER_SIZE_V1;
    std::memcpy(body, rawMetadataFb.data(), rawMetadataFb.size());
    return buffer;
}

const ProfilingFB::ProfilingMeta* getProfilingSectionMeta(const uint8_t* blobData, size_t blobSize) {
    return getProfilingSectionDataImpl(blobData, blobSize).first;
}

const uint8_t* getProfilingSectionPtr(const uint8_t* blobData, size_t blobSize) {
    return getProfilingSectionDataImpl(blobData, blobSize).second;
}

}  // namespace profiling
}  // namespace vpux
