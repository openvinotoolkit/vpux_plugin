//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/utils/models/tensor.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/custom_float.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/range.hpp"

#include <fstream>

namespace {

template <typename InT, typename OutT>
void convertTensorPrecisionImpl(const ov::Tensor& in, const ov::Tensor& out) {
    const ov::element::Type& inPrecision = in.get_element_type();
    const ov::element::Type& outPrecision = out.get_element_type();

    VPUX_THROW_UNLESS(inPrecision.size() == sizeof(InT), "Wrong tensor precision : {0}", inPrecision.get_type_name());
    VPUX_THROW_UNLESS(outPrecision.size() == sizeof(OutT), "Wrong tensor precision : {0}",
                      outPrecision.get_type_name());

    const auto inputBuffer = static_cast<const InT*>(in.data());
    VPUX_THROW_UNLESS(inputBuffer != nullptr, "Tensor was not allocated");

    const auto outputBuffer = static_cast<OutT*>(out.data());
    VPUX_THROW_UNLESS(outputBuffer != nullptr, "Tensor was not allocated");

    for (size_t index = 0; index < in.get_size(); ++index) {
        outputBuffer[index] = vpux::checked_cast<OutT>(inputBuffer[index]);
    }
}

}  // namespace

namespace vpux {

void copyTensor(const ov::Tensor& in, const ov::Tensor& out) {
    VPUX_THROW_UNLESS(in.get_element_type() == out.get_element_type(), "Precision mismatch");
    VPUX_THROW_UNLESS(in.get_shape() == out.get_shape(), "Shape mismatch");

    const auto inputBuffer = in.data<const uint8_t>();
    VPUX_THROW_UNLESS(inputBuffer != nullptr, "Tensor was not allocated");

    const auto outputBuffer = out.data<uint8_t>();
    VPUX_THROW_UNLESS(outputBuffer != nullptr, "Tensor was not allocated");

    std::copy_n(inputBuffer, in.get_byte_size(), outputBuffer);
}

void convertTensorPrecision(const ov::Tensor& in, const ov::Tensor& out) {
    VPUX_THROW_UNLESS(in.get_shape() == out.get_shape(), "Mismatch in Dims");

    const ov::element::Type& inPrecision = in.get_element_type();
    const ov::element::Type& outPrecision = out.get_element_type();

    if (inPrecision == outPrecision) {
        copyTensor(in, out);
        return;
    }

#define CASE(InT, OutT)                             \
    convertTensorPrecisionImpl<InT, OutT>(in, out); \
    break

    switch (inPrecision) {
    case ov::element::Type_t::f64: {
        switch (outPrecision) {
        case ov::element::Type_t::f32:
            CASE(double, float);
        case ov::element::Type_t::u64:
            CASE(double, uint64_t);
        case ov::element::Type_t::i64:
            CASE(double, int64_t);
        case ov::element::Type_t::u32:
            CASE(double, uint32_t);
        case ov::element::Type_t::i32:
            CASE(double, int32_t);
        case ov::element::Type_t::u16:
            CASE(double, uint16_t);
        case ov::element::Type_t::i16:
            CASE(double, int16_t);
        case ov::element::Type_t::u8:
            CASE(double, uint8_t);
        case ov::element::Type_t::i8:
            CASE(double, int8_t);
        case ov::element::Type_t::f16:
            CASE(double, vpux::type::float16);
        case ov::element::Type_t::bf16:
            CASE(double, vpux::type::bfloat16);
        default:
            VPUX_THROW("Unsupported combination of precisions {0} -> {1}", inPrecision.get_type_name(),
                       outPrecision.get_type_name());
        }
        break;
    }
    case ov::element::Type_t::f32: {
        switch (outPrecision) {
        case ov::element::Type_t::f64:
            CASE(float, double);
        case ov::element::Type_t::u64:
            CASE(float, uint64_t);
        case ov::element::Type_t::i64:
            CASE(float, int64_t);
        case ov::element::Type_t::u32:
            CASE(float, uint32_t);
        case ov::element::Type_t::i32:
            CASE(float, int32_t);
        case ov::element::Type_t::u16:
            CASE(float, uint16_t);
        case ov::element::Type_t::i16:
            CASE(float, int16_t);
        case ov::element::Type_t::u8:
            CASE(float, uint8_t);
        case ov::element::Type_t::i8:
            CASE(float, int8_t);
        case ov::element::Type_t::f16:
            CASE(float, vpux::type::float16);
        case ov::element::Type_t::bf16:
            CASE(float, vpux::type::bfloat16);
        default:
            VPUX_THROW("Unsupported combination of precisions {0} -> {1}", inPrecision.get_type_name(),
                       outPrecision.get_type_name());
        }
        break;
    }
    case ov::element::Type_t::f16: {
        switch (outPrecision) {
        case ov::element::Type_t::f64:
            CASE(vpux::type::float16, double);
        case ov::element::Type_t::f32:
            CASE(vpux::type::float16, float);
        case ov::element::Type_t::bf16:
            CASE(vpux::type::float16, vpux::type::bfloat16);
        case ov::element::Type_t::u64:
            CASE(vpux::type::float16, uint64_t);
        case ov::element::Type_t::i64:
            CASE(vpux::type::float16, int64_t);
        case ov::element::Type_t::u32:
            CASE(vpux::type::float16, uint32_t);
        case ov::element::Type_t::i32:
            CASE(vpux::type::float16, int32_t);
        case ov::element::Type_t::u16:
            CASE(vpux::type::float16, uint16_t);
        case ov::element::Type_t::i16:
            CASE(vpux::type::float16, int16_t);
        case ov::element::Type_t::u8:
            CASE(vpux::type::float16, uint8_t);
        case ov::element::Type_t::i8:
            CASE(vpux::type::float16, int8_t);
        default:
            VPUX_THROW("Unsupported combination of precisions {0} -> {1}", inPrecision.get_type_name(),
                       outPrecision.get_type_name());
        }
        break;
    }
    case ov::element::Type_t::bf16: {
        switch (outPrecision) {
        case ov::element::Type_t::f64:
            CASE(vpux::type::bfloat16, double);
        case ov::element::Type_t::f32:
            CASE(vpux::type::bfloat16, float);
        case ov::element::Type_t::f16:
            CASE(vpux::type::bfloat16, vpux::type::float16);
        case ov::element::Type_t::u64:
            CASE(vpux::type::bfloat16, uint64_t);
        case ov::element::Type_t::i64:
            CASE(vpux::type::bfloat16, int64_t);
        case ov::element::Type_t::u32:
            CASE(vpux::type::bfloat16, uint32_t);
        case ov::element::Type_t::i32:
            CASE(vpux::type::bfloat16, int32_t);
        case ov::element::Type_t::u16:
            CASE(vpux::type::bfloat16, uint16_t);
        case ov::element::Type_t::i16:
            CASE(vpux::type::bfloat16, int16_t);
        case ov::element::Type_t::u8:
            CASE(vpux::type::bfloat16, uint8_t);
        case ov::element::Type_t::i8:
            CASE(vpux::type::bfloat16, int8_t);
        default:
            VPUX_THROW("Unsupported combination of precisions {0} -> {1}", inPrecision.get_type_name(),
                       outPrecision.get_type_name());
        }
        break;
    }
    case ov::element::Type_t::u64: {
        switch (outPrecision) {
        case ov::element::Type_t::f64:
            CASE(uint64_t, double);
        case ov::element::Type_t::f32:
            CASE(uint64_t, float);
        case ov::element::Type_t::i64:
            CASE(uint64_t, int64_t);
        case ov::element::Type_t::u32:
            CASE(uint64_t, uint32_t);
        case ov::element::Type_t::i32:
            CASE(uint64_t, int32_t);
        case ov::element::Type_t::u16:
            CASE(uint64_t, uint16_t);
        case ov::element::Type_t::i16:
            CASE(uint64_t, int16_t);
        case ov::element::Type_t::u8:
            CASE(uint64_t, uint8_t);
        case ov::element::Type_t::i8:
            CASE(uint64_t, int8_t);
        case ov::element::Type_t::f16:
            CASE(uint64_t, vpux::type::float16);
        case ov::element::Type_t::bf16:
            CASE(uint64_t, vpux::type::bfloat16);
        default:
            VPUX_THROW("Unsupported combination of precisions {0} -> {1}", inPrecision.get_type_name(),
                       outPrecision.get_type_name());
        }
        break;
    }
    case ov::element::Type_t::i64: {
        switch (outPrecision) {
        case ov::element::Type_t::f64:
            CASE(int64_t, double);
        case ov::element::Type_t::f32:
            CASE(int64_t, float);
        case ov::element::Type_t::u64:
            CASE(int64_t, uint64_t);
        case ov::element::Type_t::u32:
            CASE(int64_t, uint32_t);
        case ov::element::Type_t::i32:
            CASE(int64_t, int32_t);
        case ov::element::Type_t::u16:
            CASE(int64_t, uint16_t);
        case ov::element::Type_t::i16:
            CASE(int64_t, int16_t);
        case ov::element::Type_t::u8:
            CASE(int64_t, uint8_t);
        case ov::element::Type_t::i8:
            CASE(int64_t, int8_t);
        case ov::element::Type_t::f16:
            CASE(int64_t, vpux::type::float16);
        case ov::element::Type_t::bf16:
            CASE(int64_t, vpux::type::bfloat16);
        default:
            VPUX_THROW("Unsupported combination of precisions {0} -> {1}", inPrecision.get_type_name(),
                       outPrecision.get_type_name());
        }
        break;
    }
    case ov::element::Type_t::u32: {
        switch (outPrecision) {
        case ov::element::Type_t::f64:
            CASE(uint32_t, double);
        case ov::element::Type_t::f32:
            CASE(uint32_t, float);
        case ov::element::Type_t::u64:
            CASE(uint32_t, uint64_t);
        case ov::element::Type_t::i64:
            CASE(uint32_t, int64_t);
        case ov::element::Type_t::i32:
            CASE(uint32_t, int32_t);
        case ov::element::Type_t::u16:
            CASE(uint32_t, uint16_t);
        case ov::element::Type_t::i16:
            CASE(uint32_t, int16_t);
        case ov::element::Type_t::u8:
            CASE(uint32_t, uint8_t);
        case ov::element::Type_t::i8:
            CASE(uint32_t, int8_t);
        case ov::element::Type_t::f16:
            CASE(uint32_t, vpux::type::float16);
        case ov::element::Type_t::bf16:
            CASE(uint32_t, vpux::type::bfloat16);
        default:
            VPUX_THROW("Unsupported combination of precisions {0} -> {1}", inPrecision.get_type_name(),
                       outPrecision.get_type_name());
        }
        break;
    }
    case ov::element::Type_t::i32: {
        switch (outPrecision) {
        case ov::element::Type_t::f64:
            CASE(int32_t, double);
        case ov::element::Type_t::f32:
            CASE(int32_t, float);
        case ov::element::Type_t::u64:
            CASE(int32_t, uint64_t);
        case ov::element::Type_t::i64:
            CASE(int32_t, int64_t);
        case ov::element::Type_t::u32:
            CASE(int32_t, uint32_t);
        case ov::element::Type_t::u16:
            CASE(int32_t, uint16_t);
        case ov::element::Type_t::i16:
            CASE(int32_t, int16_t);
        case ov::element::Type_t::u8:
            CASE(int32_t, uint8_t);
        case ov::element::Type_t::i8:
            CASE(int32_t, int8_t);
        case ov::element::Type_t::f16:
            CASE(int32_t, vpux::type::float16);
        case ov::element::Type_t::bf16:
            CASE(int32_t, vpux::type::bfloat16);
        default:
            VPUX_THROW("Unsupported combination of precisions {0} -> {1}", inPrecision.get_type_name(),
                       outPrecision.get_type_name());
        }
        break;
    }
    case ov::element::Type_t::u16: {
        switch (outPrecision) {
        case ov::element::Type_t::f64:
            CASE(uint16_t, double);
        case ov::element::Type_t::f32:
            CASE(uint16_t, float);
        case ov::element::Type_t::u64:
            CASE(uint16_t, uint64_t);
        case ov::element::Type_t::i64:
            CASE(uint16_t, int64_t);
        case ov::element::Type_t::u32:
            CASE(uint16_t, uint32_t);
        case ov::element::Type_t::i32:
            CASE(uint16_t, int32_t);
        case ov::element::Type_t::i16:
            CASE(uint16_t, int16_t);
        case ov::element::Type_t::u8:
            CASE(uint16_t, uint8_t);
        case ov::element::Type_t::i8:
            CASE(uint16_t, int8_t);
        case ov::element::Type_t::f16:
            CASE(uint16_t, vpux::type::float16);
        case ov::element::Type_t::bf16:
            CASE(uint16_t, vpux::type::bfloat16);
        default:
            VPUX_THROW("Unsupported combination of precisions {0} -> {1}", inPrecision.get_type_name(),
                       outPrecision.get_type_name());
        }
        break;
    }
    case ov::element::Type_t::i16: {
        switch (outPrecision) {
        case ov::element::Type_t::f64:
            CASE(int16_t, double);
        case ov::element::Type_t::f32:
            CASE(int16_t, float);
        case ov::element::Type_t::u64:
            CASE(int16_t, uint64_t);
        case ov::element::Type_t::i64:
            CASE(int16_t, int64_t);
        case ov::element::Type_t::u32:
            CASE(int16_t, uint32_t);
        case ov::element::Type_t::i32:
            CASE(int16_t, int32_t);
        case ov::element::Type_t::u16:
            CASE(int16_t, uint16_t);
        case ov::element::Type_t::u8:
            CASE(int16_t, uint8_t);
        case ov::element::Type_t::i8:
            CASE(int16_t, int8_t);
        case ov::element::Type_t::f16:
            CASE(int16_t, vpux::type::float16);
        case ov::element::Type_t::bf16:
            CASE(int16_t, vpux::type::bfloat16);
        default:
            VPUX_THROW("Unsupported combination of precisions {0} -> {1}", inPrecision.get_type_name(),
                       outPrecision.get_type_name());
        }
        break;
    }
    case ov::element::Type_t::u8: {
        switch (outPrecision) {
        case ov::element::Type_t::f64:
            CASE(uint8_t, double);
        case ov::element::Type_t::f32:
            CASE(uint8_t, float);
        case ov::element::Type_t::u64:
            CASE(uint8_t, uint64_t);
        case ov::element::Type_t::i64:
            CASE(uint8_t, int64_t);
        case ov::element::Type_t::u32:
            CASE(uint8_t, uint32_t);
        case ov::element::Type_t::i32:
            CASE(uint8_t, int32_t);
        case ov::element::Type_t::u16:
            CASE(uint8_t, uint16_t);
        case ov::element::Type_t::i16:
            CASE(uint8_t, int16_t);
        case ov::element::Type_t::i8:
            CASE(uint8_t, int8_t);
        case ov::element::Type_t::f16:
            CASE(uint8_t, vpux::type::float16);
        case ov::element::Type_t::bf16:
            CASE(uint8_t, vpux::type::bfloat16);
        default:
            VPUX_THROW("Unsupported combination of precisions {0} -> {1}", inPrecision.get_type_name(),
                       outPrecision.get_type_name());
        }
        break;
    }
    case ov::element::Type_t::i8: {
        switch (outPrecision) {
        case ov::element::Type_t::f64:
            CASE(int8_t, double);
        case ov::element::Type_t::f32:
            CASE(int8_t, float);
        case ov::element::Type_t::u64:
            CASE(int8_t, uint64_t);
        case ov::element::Type_t::i64:
            CASE(int8_t, int64_t);
        case ov::element::Type_t::u32:
            CASE(int8_t, uint32_t);
        case ov::element::Type_t::i32:
            CASE(int8_t, int32_t);
        case ov::element::Type_t::u16:
            CASE(int8_t, uint16_t);
        case ov::element::Type_t::i16:
            CASE(int8_t, int16_t);
        case ov::element::Type_t::u8:
            CASE(int8_t, uint8_t);
        case ov::element::Type_t::f16:
            CASE(int8_t, vpux::type::float16);
        case ov::element::Type_t::bf16:
            CASE(int8_t, vpux::type::bfloat16);
        default:
            VPUX_THROW("Unsupported combination of precisions {0} -> {1}", inPrecision.get_type_name(),
                       outPrecision.get_type_name());
        }
        break;
    }
    default:
        VPUX_THROW("Unsupported combination of precisions {0} -> {1}", inPrecision.get_type_name(),
                   outPrecision.get_type_name());
    }

#undef CASE
}

ov::Tensor toPrecision(const ov::Tensor& in, const ov::element::Type& precision, void* ptr) {
    if (in.get_element_type() == precision && ptr == nullptr) {
        return in;
    }

    ov::Tensor out;

    if (ptr == nullptr) {
        out = ov::Tensor(precision, in.get_shape());
    } else {
        out = ov::Tensor(precision, in.get_shape(), ptr);
    }

    convertTensorPrecision(in, out);

    return out;
}

std::vector<std::vector<float>> parseTensorsAsFP32(const std::map<std::string, ov::Tensor>& tensors) {
    std::vector<std::vector<float>> results;

    for (const auto& tensor : tensors) {
        const ov::Tensor tensorFP32 = toFP32(tensor.second);
        const auto dataBuffer = tensorFP32.data<float>();
        OPENVINO_ASSERT(dataBuffer != nullptr);

        const size_t size = tensorFP32.get_size();
        std::vector<float> result(size);
        std::copy_n(dataBuffer, size, result.begin());

        results.push_back(result);
    }

    return results;
}

}  // namespace vpux
