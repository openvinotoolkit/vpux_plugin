//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "fileIO.hpp"
#include <cassert>
#include <fstream>
#include <iostream>

bool FileIO::write(const std::string& filename, const void* p_data, const uint32_t& size) {
    try {
        if (!size || p_data == nullptr) {
            throw(-1);
        }

        std::ofstream binary_data_stream{};
        binary_data_stream.open(filename, std::ios_base::binary);

        if (!binary_data_stream.is_open()) {
            throw(-2);
        }

        binary_data_stream.write(reinterpret_cast<const char*>(p_data), size);

        binary_data_stream.close();
        return true;
    } catch (int status) {
        std::cerr << "FileIO::save error code " << status << std::endl;
        return false;
    }
}

bool FileIO::read(const std::string& filename, std::vector<uint8_t>& input_data) {
    std::ifstream binary_data_stream{};
    binary_data_stream.open(filename, std::ios_base::binary);

    if (binary_data_stream.is_open()) {
        binary_data_stream.seekg(0, std::ios_base::end);
        const uint32_t input_size{static_cast<uint32_t>(binary_data_stream.tellg())};
        binary_data_stream.seekg(0, std::ios_base::beg);

        input_data.resize(input_size);

        binary_data_stream.read(reinterpret_cast<char*>(input_data.data()), input_size);

        binary_data_stream.close();
        return true;
    }

    return false;
}

void FileIO::write_hex_buffer(std::ofstream& tensor_file, std::vector<uint8_t>& buffer) {
    const auto buffer_size{buffer.size()};
    tensor_file << "  ";
    const auto last_indx{(buffer_size - 1)};

    for (size_t i{}; i < buffer_size; ++i) {
        tensor_file << std::hex << "0x" << static_cast<uint32_t>(buffer[i]);
        if (i != last_indx) {
            tensor_file << ",";
        }

        if (i && (i % 16) == 0) {
            tensor_file << "\n  ";
        }
    }
}
