//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// Task name handling utilities

#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace vpux::profiling {

// Suffix used to create cluster name from task name
extern const std::string CLUSTER_LEVEL_PROFILING_SUFFIX;
// Suffix used to create variant name from cluster name
extern const std::string VARIANT_LEVEL_PROFILING_SUFFIX;

struct TokenizedTaskName {
    std::string layerName;
    std::vector<std::string> tokens;
};

struct ParsedTaskName {
    std::string layerName;
    std::string layerType;
};

TokenizedTaskName tokenizeTaskName(const std::string& taskName);

// Parses the full task nameinto ParsedTaskName, extracting task name, layer type and cluster id
ParsedTaskName deserializeTaskName(const std::string& taskName);

/**
 * @brief Extract suffix from original task name
 *
 * @param name - task name
 * @return task name suffixes after original name separator (if present)
 *
 * Eg. for name = Subtract_1751?t_Add/cluster_0/variant_0
 * the function returns "t_Add/cluster_0/variant_0"
 * The original task name before ? is ignored.
 */
std::string getTaskNameSuffixes(const std::string& name);

std::string getLayerName(const std::string& taskName);

/**
 * @brief Extract cluster id from task name suffixes
 *
 * @param name - task name
 * @return cluster id suffix
 *
 * Eg. for name = Subtract_1751?t_Add/cluster_0/variant_1
 * the function returns "0"
 * Original task name (i.e. string before ?) is ignored.
 */
std::string getClusterFromName(const std::string& name);

/**
 * @brief Extract variant id from task name suffixes
 *
 * @param name - task name
 * @return variant id suffix
 *
 * Eg. for name = Subtract_1751?t_Add/cluster_0/variant_0
 * the function returns "0"
 * Original task name (i.e. string before ?) is ignored.
 */
std::string getVariantFromName(const std::string& name);

/**
 * @brief Extract a value from a structured task name string
 *
 * @param name - structured task name string in format prefix1/prefix2/key1_val1/key2_val2
 * @param key - keyword to have value extracted eg: "key1"
 * @return std::string - extracted value starting a character after '_' and ending on either the end of the string
 * or a keyword delimiter '/'
 *
 * Eg.
 *
 * For "origTaskName?key1_val1/key2_val2" and key "key1", the function yields "val1",
 * for "origTaskName?key1_val1/key2_val2" and key "key2", the function yields "val2",
 * for "origTaskName?key1_val1/key2_val2" and key "key3", the function yields ""
 *
 * Original task name (i.e. string before ?) is ignored.
 */
std::string getValueFromStructuredTaskName(const std::string& name, const std::string& key);

}  // namespace vpux::profiling
