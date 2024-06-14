//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scenario/accuracy_metrics.hpp"

#include "utils/error.hpp"

Norm::Norm(const double tolerance): m_tolerance(tolerance){};

Result Norm::compare(const cv::Mat& lhs, const cv::Mat& rhs) {
    cv::Mat lhsf32, rhsf32;
    lhs.convertTo(lhsf32, CV_32F);
    rhs.convertTo(rhsf32, CV_32F);

    ASSERT(lhsf32.total() == rhsf32.total());
    auto value = cv::norm(lhsf32, rhsf32);

    if (value > m_tolerance) {
        std::stringstream ss;
        ss << value << " > " << m_tolerance;
        return Error{ss.str()};
    }
    return Success{};
}

std::string Norm::str() {
    std::stringstream ss;
    ss << "Norm{tolerance: " << m_tolerance << "}";
    return ss.str();
}

Cosine::Cosine(const double threshold): m_threshold(threshold){};

Result Cosine::compare(const cv::Mat& lhs, const cv::Mat& rhs) {
    cv::Mat lhsf32, rhsf32;
    lhs.convertTo(lhsf32, CV_32F);
    rhs.convertTo(rhsf32, CV_32F);

    ASSERT(lhsf32.total() == rhsf32.total());
    const auto* lhsptr = lhsf32.ptr<float>();
    const auto* rhsptr = rhsf32.ptr<float>();

    double lhsdot = 0.0, rhsdot = 0.0, numr = 0.0;
    for (size_t i = 0; i < lhsf32.total(); ++i) {
        numr += lhsptr[i] * rhsptr[i];
        lhsdot += lhsptr[i] * lhsptr[i];
        rhsdot += rhsptr[i] * rhsptr[i];
    }

    const double eps = 1e-9;
    if (lhsdot < eps || rhsdot < eps) {
        return Error{"Division by zero!"};
    }

    const double similarity = numr / (std::sqrt(lhsdot) * std::sqrt(rhsdot));
    if (similarity > (1.0 + eps) || similarity < -(1.0 + eps)) {
        std::stringstream ss;
        ss << "Invalid result " << similarity << " (valid range [-1 : +1])";
        return Error{ss.str()};
    }

    if (m_threshold - eps > similarity) {
        std::stringstream ss;
        ss << similarity << " < " << m_threshold;
        return Error{ss.str()};
    }
    return Success{};
}

std::string Cosine::str() {
    std::stringstream ss;
    ss << "Cosine{threshold: " << m_threshold << "}";
    return ss.str();
}
