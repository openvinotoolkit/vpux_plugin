//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cassert>
#ifdef GNU_LESS_9_1
#include <experimental/filesystem>
#else
#include <filesystem>
#endif
#include <string>
#include <unordered_set>

#include <gflags/gflags.h>
#include <transformations/rt_info/fused_names_attribute.hpp>

#include "openvino/openvino.hpp"

static const char help_message[] = "Print a usage message.";
static const char model_message[] = "Required. Path to an IR .xml file.";
static const char target_device_message[] =
        "Optional. Specify the target device to infer on (the list of available devices is shown below). "
        "Default value is VPUX.3720. Sample will look for a suitable plugin for device specified.";

DEFINE_bool(h, false, help_message);
DEFINE_string(m, "", model_message);
DEFINE_string(d, "NPU.3720", target_device_message);

static void showUsage() {
    std::cout << std::endl;
    std::cout << "query_model [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                      " << help_message << std::endl;
    std::cout << "    -m \"<path>\"           " << model_message << std::endl;
    std::cout << "    -d \"<device>\"         " << target_device_message << std::endl;
}

void ParseAndCheckCommandLine(int argc, char* argv[]) {
    const bool empty_args = (argc == 1);

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);

    if (FLAGS_h || empty_args) {
        showUsage();
        exit(1);
    }

    if (FLAGS_d.empty()) {
        throw std::logic_error("Parameter -d is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    if (std::getenv("NPU_SERIALIZE_CANONICAL_MODEL") == NULL) {
        throw std::logic_error("NPU_SERIALIZE_CANONICAL_MODEL environment variable is not set.");
    }
}

std::string find_dumped_model() {
#ifdef GNU_LESS_9_1
    namespace fs = std::experimental::filesystem;
#else
    namespace fs = std::filesystem;
#endif

    const auto ir_ends_with = std::string("_canonical.xml");

    const auto cur = fs::current_path();
    for (const auto& entry : fs::directory_iterator(cur)) {
        const auto path = entry.path().string();

        // check for unguarded std::equal
        if (ir_ends_with.size() > path.size()) {
            continue;
        }

        if (std::equal(ir_ends_with.rbegin(), ir_ends_with.rend(), path.rbegin())) {
            return path;
        }
    }

    throw std::logic_error("Failed to find dumped IR after performing the graph passes.");
}

std::unordered_set<std::string> extract_supported_ops(const ov::SupportedOpsMap& ops_map) {
    auto supported_ops = std::unordered_set<std::string>();
    for (const auto& [op, plugin] : ops_map) {
        supported_ops.insert(op);
    }
    return supported_ops;
}

std::unordered_set<std::string> find_unsupported_layers(const std::shared_ptr<ov::Model>& dumped_model,
                                                        const std::unordered_set<std::string>& supported_ops) {
    auto unsupported_ops = std::unordered_set<std::string>{};

    const auto not_supported = [&supported_ops](const std::string& layer_name) {
        return supported_ops.find(layer_name) == supported_ops.end();
    };

    for (const auto& op : dumped_model->get_ordered_ops()) {
        const auto fused_layers = ov::getFusedNamesVector(op);
        if (std::any_of(fused_layers.begin(), fused_layers.end(), not_supported)) {
            unsupported_ops.insert(op->get_type_name());
        }
    }

    // Turns out it is usually a part of unsupported operation, removed manually
    unsupported_ops.erase("Constant");

    return unsupported_ops;
}

int main(int argc, char* argv[]) {
    try {
        ParseAndCheckCommandLine(argc, argv);
        const std::string model_path = FLAGS_m;
        const std::string device = FLAGS_d;

        ov::Core core;

        const auto model = core.read_model(model_path);
        const auto supported_ops_map = core.query_model(model, device);
        const auto supported_ops = extract_supported_ops(supported_ops_map);

        const auto dumped_model_path = find_dumped_model();
        const auto dumped_model = core.read_model(dumped_model_path);

        const auto ops = find_unsupported_layers(dumped_model, supported_ops);

        if (!ops.empty()) {
            std::cout << "Unsupported Layers: ";
            std::cout << std::accumulate(std::next(std::begin(ops)), std::end(ops), std::string{*std::begin(ops)},
                                         [](std::string& ss, const std::string& s) {
                                             return ss + "," + s;
                                         })
                      << '\n';
        } else {
            std::cout << "All layers are supported\n";
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown/internal exception happened." << std::endl;
        return 1;
    }

    return 0;
}
