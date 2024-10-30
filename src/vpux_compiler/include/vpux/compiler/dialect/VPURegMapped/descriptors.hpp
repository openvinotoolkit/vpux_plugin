//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPURegMapped/types.hpp"
#include "vpux/utils/core/mem_size.hpp"

namespace vpux::VPURegMapped::detail {

template <class... Registers>
struct Union {
    static constexpr auto start = vpux::Byte{std::min({
            vpux::Byte{0},
            Registers::offset...,
    })};
    static constexpr auto end = vpux::Byte{std::max({
            Registers::offset + Registers::size...,
    })};
    static_assert(end >= start, "Invalid registers pack");
    static constexpr auto size = end - start;
};

std::pair<mlir::ParseResult, std::optional<elf::Version>> parseVersion(mlir::AsmParser&);
template <const char* fieldName, class RegisterType, size_t fieldOffsetInBits, size_t fieldSizeInBits,
          ::vpux::VPURegMapped::RegFieldDataType fieldType, uint32_t major, uint32_t minor, uint32_t patch>
struct Field {
    static constexpr auto name = std::string_view{fieldName};
    using Register = RegisterType;
    static constexpr auto offset = vpux::Bit{int64_t{fieldOffsetInBits}};
    static constexpr auto size = vpux::Bit{int64_t{fieldSizeInBits}};
    static constexpr auto type = fieldType;
    static constexpr auto defaultVersion = elf::Version{major, minor, patch};

    static_assert(size.count() != 0, "Field of zero size is unsupported");
    static_assert(size.count() <= sizeof(uint64_t) * CHAR_BIT, "Field of size more than 8 bytes is unsupported");
    static_assert((type != ::vpux::VPURegMapped::RegFieldDataType::SINT) || size.count() > 1,
                  "Signed field must have more than one bit");
    static_assert((type != ::vpux::VPURegMapped::RegFieldDataType::FP) || size.count() == 16 || size.count() == 32 ||
                          size.count() == 64,
                  "Floating-point field must have size of 16, 32 or 64 bits");
    static_assert((type != ::vpux::VPURegMapped::RegFieldDataType::BF) || size.count() == 16,
                  "bfloat16 field must have size of 16 bits");
};

// E#135144: registers seem to be unnecessary overall
template <const char* registerName, class DescriptorType, size_t registerOffsetInBytes, size_t registerSizeInBits,
          class... Fields>
struct Register {
    static_assert(registerSizeInBits % CHAR_BIT == 0, "Register must have size in bytes");

    static constexpr auto name = std::string_view{registerName};
    using Descriptor = DescriptorType;
    static constexpr auto offset = vpux::Byte{registerOffsetInBytes};
    // E#135148: set register size in bytes in the first place or even deduce from Fields
    static constexpr auto size = vpux::Byte{registerSizeInBits / CHAR_BIT};
    // until C++26 there's no indexing of parameter pack
    // use std::tuple and std::tuple_element_t to index pack
    using FieldsTuple = std::tuple<Fields...>;

    static mlir::ParseResult parse(mlir::AsmParser& parser, Descriptor& descriptor) {
        // E#135150: revisit descriptors assembly format
        // some data (e.g. offsets and sizes) seem to be unnecessary
        std::string parsedType;
        if (parser.parseKeywordOrString(&parsedType).failed()) {
            return mlir::failure();
        }

        const auto maybeType = vpux::VPURegMapped::symbolizeEnum<vpux::VPURegMapped::RegFieldDataType>(parsedType);
        if (!maybeType.has_value()) {
            return mlir::failure();
        }

        std::string parsedName;
        if (parser.parseKeywordOrString(&parsedName).failed()) {
            return mlir::failure();
        }

        if (parser.parseKeyword("at").failed()) {
            return mlir::failure();
        }

        auto parsedOffset = uint64_t{0};
        if (parser.parseInteger(parsedOffset).failed()) {
            return mlir::failure();
        }

        if (parser.parseKeyword("size").failed()) {
            return mlir::failure();
        }

        auto parsedSize = uint64_t{0};
        if (parser.parseInteger(parsedSize).failed()) {
            return mlir::failure();
        }

        if (parser.parseEqual().failed()) {
            return mlir::failure();
        }

        auto parsedValue = uint64_t{0};
        if (parser.parseInteger(parsedValue).failed()) {
            return mlir::failure();
        }

        const auto parseResult = parseVersion(parser);
        // C++17 doesn't support capturing structure bindings in lambda
        // use pre-C++17 std::tie approach instead
        auto parsingStatus = mlir::failure();
        std::optional<elf::Version> maybeVersion;
        std::tie(parsingStatus, maybeVersion) = parseResult;
        if (parsingStatus.failed()) {
            return mlir::failure();
        }

        bool isAlreadyParsed = false;
        auto parsing = mlir::success();

        (
                [&] {
                    if (Fields::name != parsedName || static_cast<uint64_t>(Fields::offset.count()) != parsedOffset ||
                        static_cast<uint64_t>(Fields::size.count()) != parsedSize ||
                        Fields::type != maybeType.value()) {
                        return;
                    }

                    if (isAlreadyParsed) {
                        parsing = mlir::failure();
                        return;
                    }
                    isAlreadyParsed = true;

                    descriptor.template write<Fields>(parsedValue, maybeVersion);
                }(),
                ...);

        if (parsing.failed() || !isAlreadyParsed) {
            return mlir::failure();
        }

        return mlir::success();
    }

    static void print(mlir::AsmPrinter& printer, const Descriptor& descriptor) {
        // E#135150: if decided to keep size as part of assembly format
        // conversion to bits seems unnecessary for a register
        printer << name << " offset " << offset.count() << " size " << size.to<vpux::Bit>().count();

        using FirstFieldType = std::tuple_element_t<0, FieldsTuple>;
        if constexpr (name == FirstFieldType::name && size.to<vpux::Bit>() == FirstFieldType::size) {
            printer << " = " << vpux::VPURegMapped::stringifyEnum(FirstFieldType::type) << " "
                    << getFormattedValue(descriptor.template read<FirstFieldType>());
            printFieldVersionIfCustom<FirstFieldType>(printer, descriptor);
        } else {
            printer << " {";
            printer.increaseIndent();

            size_t i = 0;
            constexpr auto fieldsCount = sizeof...(Fields);
            (
                    [&] {
                        printer.printNewline();
                        printer << vpux::VPURegMapped::stringifyEnum(Fields::type) << " " << Fields::name << " at "
                                << Fields::offset.count() << " size " << Fields::size.count() << " = "
                                << getFormattedValue(descriptor.template read<Fields>());
                        printFieldVersionIfCustom<Fields>(printer, descriptor);
                        if (i < fieldsCount - 1) {
                            printer << ',';
                        }
                        ++i;
                    }(),
                    ...);

            printer.decreaseIndent();
            printer.printNewline();
            printer << "}";
        }
    }

private:
    template <class Field>
    static void printFieldVersionIfCustom(mlir::AsmPrinter& printer, const Descriptor& descriptor) {
        const auto& customVersions = descriptor.customFieldsVersions;
        if (!customVersions.contains(Field::name)) {
            return;
        }
        const auto& customVersion = customVersions.at(Field::name);
        printer << " requires " << customVersion.getMajor() << ':' << customVersion.getMinor() << ':'
                << customVersion.getPatch();
    }
};

template <class SpecificDescriptor, const char* descriptorName, class... Registers>
class Descriptor {
public:
    static constexpr auto name = std::string_view{descriptorName};
    using RegistersTuple = std::tuple<Registers...>;

    Descriptor() {
        storage.resize(Union<Registers...>::size.count());
    }

    size_t size() const {
        return storage.size();
    }

    template <class Field, typename = std::enable_if_t<Field::type == ::vpux::VPURegMapped::RegFieldDataType::UINT>>
    void write(uint64_t value, const std::optional<elf::Version>& version = {}) {
        using Register = typename Field::Register;
        static_assert(std::is_same_v<SpecificDescriptor, typename Register::Descriptor>);
        assert(value <= getBitsSet<Field::size.count()>());
        write<Register::offset.count(), Field::offset.count(), Field::size.count()>(value);
        updateVersion<Field>(version);
    }

    template <class Field, typename = std::enable_if_t<Field::type == ::vpux::VPURegMapped::RegFieldDataType::SINT>>
    void write(int64_t value, const std::optional<elf::Version>& version = {}) {
        using Register = typename Field::Register;
        static_assert(std::is_same_v<SpecificDescriptor, typename Register::Descriptor>);
        [[maybe_unused]] constexpr auto max = static_cast<int64_t>(getBitsSet<Field::size.count() - 1>());
        [[maybe_unused]] constexpr auto min = -1 * max - 1;
        assert((min <= value) && (value <= max));
        write<Register::offset.count(), Field::offset.count(), Field::size.count()>(static_cast<uint64_t>(value));
        updateVersion<Field>(version);
    }

    template <class Field>
    uint64_t read() const {
        // see write implementation about part0, part1 and part2 patterns

        using Register = typename Field::Register;
        static_assert(std::is_same_v<SpecificDescriptor, typename Register::Descriptor>);

        // don't convert Field::offset from Bit to Byte via to<vpux::Byte> as it'll throw
        // if Field::offset isn't divisible by CHAR_BIT
        const auto address = storage.data() + Register::offset.count() + Field::offset.count() / CHAR_BIT;
        constexpr auto inByteFieldOffset = Field::offset.count() % CHAR_BIT;
        constexpr auto part0Size = std::min(Field::size.count(), CHAR_BIT - inByteFieldOffset);
        constexpr auto part1n2Size = Field::size.count() - part0Size;

        auto value = uint64_t{0};
        if constexpr (part0Size != 0) {
            constexpr auto part0Mask = getBitsSet<part0Size>() << inByteFieldOffset;
            const auto part0Value = static_cast<uint64_t>(((*address) & part0Mask) >> inByteFieldOffset);
            value |= part0Value;
        }

        constexpr auto part2Size = part1n2Size % 8;
        constexpr auto part1Size = part1n2Size - part2Size;
        static_assert(part1Size % 8 == 0);
        // seems like some compilers (e.g. clang) may complain variable isn't used
        // if both "if constexpr" below evaluate to false
        [[maybe_unused]] constexpr auto part0ByteOffset = size_t{1};
        [[maybe_unused]] constexpr auto part1ByteCount = part1Size / 8;

        if constexpr (part1Size != 0) {
            for (size_t i = 0; i < part1ByteCount; ++i) {
                const auto part1Value = static_cast<uint64_t>(address[part0ByteOffset + i]);
                value |= part1Value << (part0Size + i * 8);
            }
        }

        if constexpr (part2Size != 0) {
            constexpr auto part2Mask = getBitsSet<part2Size>();
            const auto part2Value = static_cast<uint64_t>(address[part0ByteOffset + part1ByteCount] & part2Mask);
            value |= part2Value << (part0Size + part1Size);
        }

        return value;
    }

    bool operator==(const Descriptor& rhs) const {
        // no need to take into account custom versions
        // if values are the same, version will be as well
        return storage == rhs.storage;
    }

    llvm::ArrayRef<uint8_t> getStorage() const {
        return storage;
    }

    void print(mlir::AsmPrinter& printer) const {
        printer << '<';
        printer.increaseIndent();
        printer.printNewline();
        printer << name << " {";
        printer.increaseIndent();

        size_t i = 0;
        constexpr auto registersCount = sizeof...(Registers);

        (
                [&] {
                    printer.printNewline();
                    Registers::print(printer, static_cast<const SpecificDescriptor&>(*this));
                    if (i < registersCount - 1) {
                        printer << ',';
                    }
                    ++i;
                }(),
                ...);

        printer.decreaseIndent();
        printer.printNewline();
        printer << "}";
        printer.decreaseIndent();
        printer.printNewline();
        printer << '>';
    }

    static std::optional<SpecificDescriptor> parse(mlir::AsmParser& parser) {
        if (parser.parseLess().failed()) {
            return {};
        }

        std::string parsedDescriptorName;
        if (parser.parseKeywordOrString(&parsedDescriptorName)) {
            return {};
        }

        if (SpecificDescriptor::name != parsedDescriptorName) {
            return {};
        }

        auto result = SpecificDescriptor{};
        auto parseReg = [&] {
            std::string name;
            if (parser.parseKeywordOrString(&name).failed()) {
                return mlir::failure();
            }

            if (parser.parseKeyword("offset").failed()) {
                return mlir::failure();
            }

            auto offsetInBytes = uint64_t{0};
            if (parser.parseInteger(offsetInBytes).failed()) {
                return mlir::failure();
            }

            if (parser.parseKeyword("size").failed()) {
                return mlir::failure();
            }

            auto sizeInBits = uint64_t{0};
            if (parser.parseInteger(sizeInBits).failed()) {
                return mlir::failure();
            }

            if (parser.parseOptionalKeyword("allowOverlap").succeeded()) {
                // ignoring
                ;
            }

            bool isAlreadyParsed = false;
            auto parsing = mlir::success();

            (
                    [&] {
                        constexpr auto registerName = Registers::name;
                        constexpr auto registerSize = Registers::size;
                        constexpr auto registerOffset = Registers::offset;
                        if (registerName != name ||
                            registerSize != vpux::Bit{static_cast<int64_t>(sizeInBits)}.to<vpux::Byte>() ||
                            static_cast<uint64_t>(registerOffset.count()) != offsetInBytes) {
                            return;
                        }

                        if (isAlreadyParsed) {
                            parsing = mlir::failure();
                            return;
                        }
                        isAlreadyParsed = true;

                        if (parser.parseOptionalEqual().succeeded()) {
                            assert(std::tuple_size_v<typename Registers::FieldsTuple> == 1);
                            using SingleFieldType = std::tuple_element_t<0, typename Registers::FieldsTuple>;

                            std::string parsedType;
                            if (parser.parseKeywordOrString(&parsedType).failed()) {
                                parsing = mlir::failure();
                                return;
                            }

                            const auto maybeType =
                                    vpux::VPURegMapped::symbolizeEnum<vpux::VPURegMapped::RegFieldDataType>(parsedType);
                            if (!maybeType.has_value()) {
                                parsing = mlir::failure();
                                return;
                            }

                            if (maybeType.value() != SingleFieldType::type) {
                                parsing = mlir::failure();
                                return;
                            }

                            auto parsedValue = uint64_t{0};
                            if (parser.parseInteger(parsedValue).failed()) {
                                parsing = mlir::failure();
                                return;
                            }

                            const auto [parsingStatus, maybeVersion] = parseVersion(parser);
                            if (parsingStatus.failed()) {
                                parsing = mlir::failure();
                                return;
                            }

                            result.template write<SingleFieldType>(parsedValue);
                            if (maybeVersion.has_value()) {
                                result.customFieldsVersions[SingleFieldType::name] = maybeVersion.value();
                            }
                        } else {
                            parsing = parser.parseCommaSeparatedList(mlir::AsmParser::Delimiter::Braces, [&] {
                                return Registers::parse(parser, result);
                            });
                        }
                    }(),
                    ...);

            return mlir::success();
        };

        if (parser.parseCommaSeparatedList(mlir::AsmParser::Delimiter::Braces, parseReg).failed()) {
            return {};
        }

        if (parser.parseGreater().failed()) {
            return {};
        }

        return result;
    }

    llvm::hash_code hash_value() const {
        return llvm::hash_value(getStorage());
    }

private:
    template <size_t registerOffsetInBytes, size_t fieldOffsetInBits, size_t fieldSizeInBits>
    void write(uint64_t value) {
        // note: schemas below follow convention of having Least Significant Bits (LSB)
        //       on the right side and Most Significant Bits (MSB) on the left
        //       Intel uses little-endian format where LSB read first
        //
        //   Byte 0   Byte 1   Byte 2   Byte 3   Byte 4   Byte 5   Byte 6   Byte 7
        // |xxxxxxxx|xxxxxxxx|xxxxxxxx|xxxxxxxx|xxxxxxxx|xxxxxxxx|xxxxxxxx|xxxxxxxx|
        //  76543210       98 <- bits ordering
        //
        // value represents sequence of bits to be written to the descriptor
        // specifically, to the descriptor in position occupied by a given field
        // since field location isn't necessary byte-aligned, we split value into 3 parts
        //
        // example:
        //
        // descriptor slice containing bits occupied by the field
        // S - start of the descriptor
        // RO - Register offset (always byte-aligned) = 2 bytes
        // FO - Field offset (not byte-aligned) = 11 bits = 1 byte + 3 bit
        // P0 - Part 0 of the field of size 5 bit (it can be up to 8 bits)
        // (FO % 8) + sizeof(P0) = 8 bit = 1 byte
        // P1 - Part 1 of the field of size 2 bytes (always byte-aligned)
        // P2 - Part 2 of the field of size 4 bits (it can be up to 7 bits)
        // sizeof(field) = sizeof(P0) + sizeof(P1) + sizeof(P2)
        //
        // S                 RO           P0   FO          P1               P2
        // |xxxxxxxx|xxxxxxxx|xxxxxxxx|[xxxxx]xxx|[xxxxxxxx|xxxxxxxx]|xxxx[xxxx]|xxxxxxxx|
        //
        // depending on field configuration (size, offset), some of the parts maybe omitted
        // e.g. if sizeof(field) = 5, FO = 0, then sizeof(P0) = 5, sizeof(P1) = sizeof(P2) = 0
        //
        // value generic case (64 bit): [unused][P2][P1][P0]
        //                  unused                         P2            P1           P0
        // |[xxxxxxxx|xxxxxxxx|xxxxxxxx|xxxxxxxx|xxxxxxx][x|xxx][xxxxx|xxxxxxxx|xxx][xxxxx]|
        //
        // unused bits maybe present if sizeof(field) < 64
        //
        // function basically matches P0, P1 and P2 in the value with corresponding positions
        // in the descriptor

        auto address = storage.data() + registerOffsetInBytes + fieldOffsetInBits / CHAR_BIT;
        constexpr auto inByteFieldOffset = fieldOffsetInBits % CHAR_BIT;
        constexpr auto part0Size = std::min(fieldSizeInBits, CHAR_BIT - inByteFieldOffset);
        constexpr auto part1n2Size = fieldSizeInBits - part0Size;

        if constexpr (part0Size != 0) {
            constexpr auto part0Mask = getBitsSet<part0Size>();
            const auto part0Value = value & part0Mask;

            // clear out whatever were in P0 position originally
            // so bitwise OR would result in bits from value only
            address[0] &= ~(part0Mask << inByteFieldOffset);
            address[0] |= static_cast<uint8_t>(part0Value << inByteFieldOffset);
        }

        constexpr auto part2Size = part1n2Size % 8;
        constexpr auto part1Size = part1n2Size - part2Size;
        static_assert(part1Size % 8 == 0);
        [[maybe_unused]] constexpr auto part0ByteOffset = size_t{1};
        [[maybe_unused]] constexpr auto part1ByteCount = part1Size / 8;

        if constexpr (part2Size != 0) {
            constexpr auto part2Mask = getBitsSet<part2Size>();
            const auto part2Value = value >> (part0Size + part1Size);
            // clear out whatever were in P1 position originally
            // so bitwise OR would result in bits from value only
            address[part0ByteOffset + part1ByteCount] &= ~part2Mask;
            address[part0ByteOffset + part1ByteCount] |= part2Value;
        }

        if constexpr (part1Size != 0) {
            value >>= part0Size;

            for (size_t i = 0; i < part1ByteCount; ++i) {
                address[part0ByteOffset + i] = static_cast<uint8_t>(value);
                value >>= 8;
            }
        }
    }

    template <size_t size>
    static constexpr uint64_t getBitsSet() {
        static_assert(size <= 64, "No support for more than 64 bits");
        if constexpr (size == 64) {
            return static_cast<uint64_t>(int64_t{-1});
        } else {
            return (uint64_t{1} << size) - 1;
        }
    }

    template <class Field>
    void updateVersion(const std::optional<elf::Version>& version) {
        if (!version.has_value() || version.value() == Field::defaultVersion) {
            return;
        }

        customFieldsVersions[Field::name] = version.value();
    }

    mlir::SmallVector<std::uint8_t> storage;
    llvm::SmallDenseMap<llvm::StringRef, elf::Version> customFieldsVersions;

    template <const char*, class, size_t, size_t, class...>
    friend struct Register;
};

template <class Descriptor>
llvm::hash_code hash_value(const Descriptor& descriptor) {
    return descriptor.hash_value();
}

}  // namespace vpux::VPURegMapped::detail
