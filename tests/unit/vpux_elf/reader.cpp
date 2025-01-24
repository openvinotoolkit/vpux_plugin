//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <cstring>

#include <vpux_elf/accessor.hpp>
#include <vpux_elf/reader.hpp>
#include "common_test_utils/test_assertions.hpp"

#include <gtest/gtest.h>

using namespace elf;

namespace {

ELFHeader createTemplateFileHeader() {
    ELFHeader fileHeader{};

    fileHeader.e_ident[EI_MAG0] = ELFMAG0;
    fileHeader.e_ident[EI_MAG1] = ELFMAG1;
    fileHeader.e_ident[EI_MAG2] = ELFMAG2;
    fileHeader.e_ident[EI_MAG3] = ELFMAG3;
    fileHeader.e_ident[EI_CLASS] = ELFCLASS64;
    fileHeader.e_ident[EI_DATA] = ELFDATA2LSB;
    fileHeader.e_ident[EI_VERSION] = EV_NONE;
    fileHeader.e_ident[EI_OSABI] = 0;
    fileHeader.e_ident[EI_ABIVERSION] = 0;

    fileHeader.e_type = ET_REL;
    fileHeader.e_machine = EM_NONE;
    fileHeader.e_version = EV_NONE;

    fileHeader.e_entry = 0;
    fileHeader.e_flags = 0;
    fileHeader.e_shoff = sizeof(ELFHeader);
    fileHeader.e_shstrndx = 0;
    fileHeader.e_shnum = 0;

    fileHeader.e_ehsize = sizeof(ELFHeader);
    fileHeader.e_shentsize = sizeof(SectionHeader);

    return fileHeader;
}

constexpr size_t headerTableSize = 3;
constexpr size_t indexToCheck = 1;

}  // namespace

TEST(ELFReaderTests, ELFReaderThrowsOnIncorrectMagic) {
    auto fileHeader = createTemplateFileHeader();
    fileHeader.e_ident[EI_MAG3] = 'D';
    auto accessor =
            DDRAccessManager<elf::DDRAlwaysEmplace>(reinterpret_cast<uint8_t*>(&fileHeader), sizeof(fileHeader));

    ASSERT_ANY_THROW(auto reader = Reader<ELF_Bitness::Elf64>(&accessor));
}

TEST(ELFReaderTests, ReadingTheCorrectELFHeaderDoesntThrow) {
    std::vector<SectionHeader> sectionHeaders(headerTableSize);

    auto fileHeader = createTemplateFileHeader();
    fileHeader.e_shnum = headerTableSize;
    sectionHeaders[indexToCheck].sh_offset = sizeof(fileHeader);

    std::vector<uint8_t> buffer;
    buffer.insert(buffer.end(), reinterpret_cast<uint8_t*>(&fileHeader),
                  reinterpret_cast<uint8_t*>(&fileHeader) + sizeof(fileHeader));
    buffer.insert(buffer.end(), reinterpret_cast<uint8_t*>(sectionHeaders.data()),
                  reinterpret_cast<uint8_t*>(sectionHeaders.data()) + sizeof(SectionHeader) * headerTableSize);
    auto accessor = DDRAccessManager<elf::DDRAlwaysEmplace>(buffer.data(), buffer.size());

    OV_ASSERT_NO_THROW(auto reader = Reader<ELF_Bitness::Elf64>(&accessor));
}

TEST(ELFReaderTests, ELFHeaderIsReadCorrectly) {
    std::vector<SectionHeader> sectionHeaders(headerTableSize);

    auto fileHeader = createTemplateFileHeader();
    fileHeader.e_shnum = headerTableSize;
    sectionHeaders[indexToCheck].sh_offset = sizeof(fileHeader);

    std::vector<uint8_t> buffer;
    buffer.insert(buffer.end(), reinterpret_cast<uint8_t*>(&fileHeader),
                  reinterpret_cast<uint8_t*>(&fileHeader) + sizeof(fileHeader));
    buffer.insert(buffer.end(), reinterpret_cast<uint8_t*>(sectionHeaders.data()),
                  reinterpret_cast<uint8_t*>(sectionHeaders.data()) + sizeof(SectionHeader) * headerTableSize);
    auto accessor = DDRAccessManager<elf::DDRAlwaysEmplace>(buffer.data(), buffer.size());

    const auto reader = Reader<ELF_Bitness::Elf64>(&accessor);
    auto parsedFileHeader = *reader.getHeader();

    ASSERT_TRUE(sizeof(fileHeader) == sizeof(parsedFileHeader));
    ASSERT_TRUE(!memcmp(&fileHeader, &parsedFileHeader, sizeof(parsedFileHeader)));
}

TEST(ELFReaderTests, ELFReaderThrowsOnInvalidSectionHeaderCount) {
    auto fileHeader = createTemplateFileHeader();
    fileHeader.e_shnum = 1;
    auto accessor =
            DDRAccessManager<elf::DDRAlwaysEmplace>(reinterpret_cast<uint8_t*>(&fileHeader), sizeof(fileHeader));

    ASSERT_ANY_THROW(auto reader = Reader<ELF_Bitness::Elf64>(&accessor));
}

TEST(ELFReaderTests, SectionHeadersAreReadCorrectly) {
    std::vector<SectionHeader> sectionHeaders(headerTableSize);

    for (size_t idx = 0; idx < sectionHeaders.size(); idx++) {
        sectionHeaders[idx].sh_name = idx;
        sectionHeaders[idx].sh_size = headerTableSize;
    }

    auto fileHeader = createTemplateFileHeader();
    fileHeader.e_shnum = headerTableSize;
    fileHeader.e_shstrndx = headerTableSize - 1;

    std::vector<uint8_t> buffer;
    buffer.insert(buffer.end(), reinterpret_cast<uint8_t*>(&fileHeader),
                  reinterpret_cast<uint8_t*>(&fileHeader) + sizeof(fileHeader));
    buffer.insert(buffer.end(), reinterpret_cast<uint8_t*>(sectionHeaders.data()),
                  reinterpret_cast<uint8_t*>(sectionHeaders.data()) + sizeof(SectionHeader) * headerTableSize);

    auto accessor = DDRAccessManager<elf::DDRAlwaysEmplace>(buffer.data(), buffer.size());
    auto reader = Reader<ELF_Bitness::Elf64>(&accessor);

    ASSERT_TRUE(sizeof(sectionHeaders[0]) == sizeof(*reader.getSection(0).getHeader()));

    for (uint64_t idx = 0; idx < sectionHeaders.size(); ++idx) {
        ASSERT_TRUE(!memcmp(&sectionHeaders[idx], reader.getSection(idx).getHeader(), sizeof(sectionHeaders[0])));
    }
}

TEST(ELFReaderTests, PointerToSectionDataIsResolvedCorrectly) {
    std::vector<SectionHeader> sectionHeaders(headerTableSize);

    for (size_t idx = 0; idx < sectionHeaders.size(); idx++) {
        sectionHeaders[idx].sh_name = idx;
        sectionHeaders[idx].sh_size = headerTableSize;
    }

    auto fileHeader = createTemplateFileHeader();
    fileHeader.e_shnum = headerTableSize;
    fileHeader.e_shstrndx = headerTableSize - 1;
    sectionHeaders[indexToCheck].sh_offset = sizeof(fileHeader);

    std::vector<uint8_t> buffer;
    buffer.insert(buffer.end(), reinterpret_cast<uint8_t*>(&fileHeader),
                  reinterpret_cast<uint8_t*>(&fileHeader) + sizeof(fileHeader));
    buffer.insert(buffer.end(), reinterpret_cast<uint8_t*>(sectionHeaders.data()),
                  reinterpret_cast<uint8_t*>(sectionHeaders.data()) + sizeof(SectionHeader) * headerTableSize);

    auto accessor = DDRAccessManager<elf::DDRAlwaysEmplace>(buffer.data(), buffer.size());
    auto reader = Reader<ELF_Bitness::Elf64>(&accessor);
    ASSERT_EQ(reader.getSection(indexToCheck).getData<uint8_t>(), buffer.data() + sizeof(fileHeader));
}

TEST(ELFReaderTests, PtrToSectionDataIsResolvedCorrectlyWithGetSectionNoData) {
    std::vector<SectionHeader> sectionHeaders(headerTableSize);

    for (size_t idx = 0; idx < sectionHeaders.size(); idx++) {
        sectionHeaders[idx].sh_name = idx;
        sectionHeaders[idx].sh_size = headerTableSize;
    }

    auto fileHeader = createTemplateFileHeader();
    fileHeader.e_shnum = headerTableSize;
    fileHeader.e_shstrndx = headerTableSize - 1;
    sectionHeaders[indexToCheck].sh_offset = sizeof(fileHeader);

    std::vector<uint8_t> buffer;
    buffer.insert(buffer.end(), reinterpret_cast<uint8_t*>(&fileHeader),
                  reinterpret_cast<uint8_t*>(&fileHeader) + sizeof(fileHeader));
    buffer.insert(buffer.end(), reinterpret_cast<uint8_t*>(sectionHeaders.data()),
                  reinterpret_cast<uint8_t*>(sectionHeaders.data()) + sizeof(SectionHeader) * headerTableSize);

    auto accessor = DDRAccessManager<elf::DDRAlwaysEmplace>(buffer.data(), buffer.size());
    auto reader = Reader<ELF_Bitness::Elf64>(&accessor);
    ASSERT_EQ(reader.getSection(indexToCheck).getData<uint8_t>(), buffer.data() + sizeof(fileHeader));
}
