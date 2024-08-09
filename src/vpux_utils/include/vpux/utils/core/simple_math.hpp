//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

//
// Simple integer arithmetic to be used for the work sizes calculation.
// Supported operations : +,-,*,/,%,(,)
// no unary -,+
// Variables defined as single chars and should not include one of the ops, whitespaces or 0-9
//

#pragma once

#include "vpux/utils/core/hash.hpp"
#include "vpux/utils/core/optional.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <istream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace vpux {

template <typename T>
std::enable_if_t<std::is_integral<T>::value, std::optional<T>> parseNumber(StringRef str) {
    T res = 0;
    if (str.getAsInteger(10, res)) {
        return res;
    }
    return std::nullopt;
}

template <typename T>
std::enable_if_t<std::is_floating_point<T>::value, std::optional<T>> parseNumber(StringRef str) {
    double res = 0;
    if (str.getAsDouble(res)) {
        return static_cast<T>(res);
    }
    return std::nullopt;
}

//
// IntOrFloat
//

class IntOrFloat final {
public:
    explicit IntOrFloat(int x);
    explicit IntOrFloat(float x);
    explicit IntOrFloat(StringRef str);

public:
    bool isInt() const {
        return _isInt;
    }

    int asInt() const {
        return _isInt ? _value.i : static_cast<int>(_value.f);
    }

    float asFloat() const {
        return _isInt ? static_cast<float>(_value.i) : _value.f;
    }

private:
    union {
        int i;
        float f;
    } _value = {};

    bool _isInt = true;
};

IntOrFloat operator+(const IntOrFloat& a, const IntOrFloat& b);
IntOrFloat operator-(const IntOrFloat& a, const IntOrFloat& b);
IntOrFloat operator*(const IntOrFloat& a, const IntOrFloat& b);
IntOrFloat operator/(const IntOrFloat& a, const IntOrFloat& b);
IntOrFloat operator%(const IntOrFloat& a, const IntOrFloat& b);

//
// MathExpression
//

class MathExpression final {
public:
    using VarMap = std::map<std::string, std::string>;

public:
    void setVariables(const VarMap& variables);
    void parse(StringRef expression);
    IntOrFloat evaluate() const;

private:
    enum class TokenType { Value, Operator, Function };

    struct Token final {
        TokenType _type;
        IntOrFloat _value;
        std::string _opName;

        Token(TokenType type, IntOrFloat value, StringRef opName): _type(type), _value(value), _opName(opName.str()) {
        }
    };

private:
    std::unordered_map<std::string, IntOrFloat> _vars;
    std::vector<Token> _tokens;
};

}  // namespace vpux
