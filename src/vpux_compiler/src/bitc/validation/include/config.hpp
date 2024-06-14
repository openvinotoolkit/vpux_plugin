//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once
#include <cctype>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <variant>
#include <vector>

#define ANSI_RESET "\033[0m"
#define ANSI_RED "\033[31m"
#define ANSI_GREEN "\033[32m"
#define ANSI_YELLOW "\033[33m"
#define ANSI_BLUE "\033[34m"

using std::string_literals::operator""s;
using string_vector = std::vector<std::string>;
using config_map = std::map<std::string, std::variant<std::string, string_vector>>;

config_map parse_config(std::ifstream& config_file);
void print_config(const config_map& config);
