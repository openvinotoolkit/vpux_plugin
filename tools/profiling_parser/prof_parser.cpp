//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>

#include <gflags/gflags.h>

#include <flatbuffers/minireflect.h>
#include "schema/profiling_generated.h"

#include "openvino/core/version.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/profiling/metadata.hpp"
#include "vpux/utils/profiling/parser/api.hpp"
#include "vpux/utils/profiling/reports/api.hpp"

using namespace vpux::profiling;

namespace {

enum class OutputFormat { TEXT, JSON, DEBUG };

DEFINE_string(b, "", "Precompiled blob that was profiled");
DEFINE_string(p, "", "Profiling result binary");
DEFINE_string(f, "json", "Format to use (text, json or debug)");
DEFINE_string(o, "", "Output file, stdout by default");
DEFINE_bool(g, false, "Profiling data is from FPGA");
DEFINE_bool(v, false, "Increased verbosity of DPU tasks parsing (include variant level tasks)");
DEFINE_bool(vv, false, "Highest verbosity of tasks parsing (Currently same as -v)");
DEFINE_bool(m, false, "Dump profiling metadata");
DEFINE_bool(fast_clk, false, "Assume perf_clk of 400MHz");

bool validateFile(const char* flagName, const std::string& pathToFile) {
    if (pathToFile.empty()) {
        return false;
    }
    std::ifstream ifile;
    ifile.open(pathToFile);

    const bool isValid = ifile.good();
    if (isValid) {
        ifile.close();
    } else {
        std::cerr << "Got error when parsing argument \"" << flagName << "\" with value " << pathToFile << std::endl;
    }
    return isValid;
}

VerbosityLevel getVerbosity() {
    if (FLAGS_vv) {
        return VerbosityLevel::HIGH;
    } else if (FLAGS_v) {
        return VerbosityLevel::MEDIUM;
    } else {
        return VerbosityLevel::LOW;
    }
}

std::string verbosityToStr(VerbosityLevel verbosity) {
    std::map<VerbosityLevel, std::string> labels = {
            {VerbosityLevel::LOW, "Low"},
            {VerbosityLevel::MEDIUM, "Medium"},
            {VerbosityLevel::HIGH, "High"},
    };
    return labels[verbosity];
}

OutputFormat getOutputFormat() {
    if (FLAGS_f == "text") {
        return OutputFormat::TEXT;
    } else if (FLAGS_f == "json") {
        return OutputFormat::JSON;
    } else if (FLAGS_f == "debug") {
        return OutputFormat::DEBUG;
    }
    VPUX_THROW("Unknown output format: {0}.", FLAGS_f);
}

void parseCommandLine(int argc, char* argv[], const std::string& usage) {
    gflags::SetUsageMessage(usage);
    gflags::RegisterFlagValidator(&FLAGS_b, &validateFile);
    std::ostringstream version;
    version << ov::get_openvino_version();
    gflags::SetVersionString(version.str());

    gflags::ParseCommandLineFlags(&argc, &argv, true);

    if (!FLAGS_m && !validateFile("-p", FLAGS_p)) {
        throw std::runtime_error("Invalid -p parameter value");
    }
}

void printCommandLineParameters() {
    std::cout << "Parameters:" << std::endl;
    std::cout << "    Network blob file:         " << FLAGS_b << std::endl;
    std::cout << "    Profiling result file:     " << FLAGS_p << std::endl;
    std::cout << "    Format (text/json):        " << FLAGS_f << std::endl;
    std::cout << "    Output file:               " << FLAGS_o << std::endl;
    std::cout << "    Verbosity:                 " << verbosityToStr(getVerbosity()) << std::endl;
    std::cout << "    FPGA:                      " << FLAGS_g << std::endl;
    std::cout << "    Dump metadata:             " << FLAGS_m << std::endl;
    std::cout << "    Assume perf_clk of 400MHz: " << FLAGS_fast_clk << std::endl;
    std::cout << std::endl;
}

void dumpProfilingMetadata(const uint8_t* blobData, size_t blobSize, std::ostream& output) {
    const uint8_t* sectionData = vpux::profiling::getProfilingSectionPtr(blobData, blobSize);

    const auto prettyProfilingMeta =
            flatbuffers::FlatBufferToString(sectionData, ProfilingFB::ProfilingMetaTypeTable(), /*multi_line*/ true,
                                            /*vector_delimited*/ false);
    output << prettyProfilingMeta << std::endl;
}

std::vector<uint8_t> readBinaryFile(std::string path) {
    std::ifstream file;
    file.open(path, std::ios::in | std::ios::binary);
    file.seekg(0, file.end);
    size_t blob_length = file.tellg();
    file.seekg(0, file.beg);
    std::vector<uint8_t> data(blob_length);
    file.read(reinterpret_cast<char*>(data.data()), data.size());
    return data;
}

void writeProfilingOutput(const OutputFormat format, const uint8_t* blobData, size_t blobSize, const uint8_t* profData,
                          size_t profSize, std::ostream& output, VerbosityLevel verbosity, bool fpga,
                          bool highFreqPerfClk) {
    if (format == OutputFormat::DEBUG) {
        writeDebugProfilingInfo(output, blobData, blobSize, profData, profSize);
        return;
    }
    ProfInfo profInfo = getProfInfo(blobData, blobSize, profData, profSize, verbosity, fpga, highFreqPerfClk);

    switch (format) {
    case OutputFormat::TEXT:
        printProfilingAsText(profInfo.tasks, profInfo.layers, output);
        break;
    case OutputFormat::JSON:
        printProfilingAsTraceEvent(profInfo.tasks, profInfo.layers, profInfo.dpuFreq, output);
        break;
    default:
        VPUX_THROW("Unsupported profiling output type.");
    }
};

}  // namespace

int main(int argc, char** argv) {
    static const char* usage = "Usage: prof_parser -b <blob> -p <profiling.bin> [-f json|text] "
                               "[-o <output_file>] [-v|vv] [-g] [-m] [-fast_clk]";
    try {
        parseCommandLine(argc, argv, usage);
        printCommandLineParameters();

        auto blob = readBinaryFile(FLAGS_b);

        std::ofstream outfile;
        const auto filename = FLAGS_o;
        if (!filename.empty()) {
            outfile.open(filename, std::ios::out | std::ios::trunc);
            VPUX_THROW_WHEN(!outfile, "Cannot write to '{0}'", filename);
        }

        std::ostream& output = outfile.is_open() ? outfile : std::cout;
        output.exceptions(std::ios::badbit | std::ios::failbit);

        if (FLAGS_m) {
            if (!FLAGS_p.empty()) {
                throw std::runtime_error("Cannot use -p when -m is specified");
            }
            dumpProfilingMetadata(blob.data(), blob.size(), output);
            return 0;
        }

        auto profdata = readBinaryFile(FLAGS_p);
        auto format = getOutputFormat();

        writeProfilingOutput(format, blob.data(), blob.size(), profdata.data(), profdata.size(), output, getVerbosity(),
                             FLAGS_g, FLAGS_fast_clk);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cerr << usage << std::endl;
        return 1;
    }

    return 0;
}
