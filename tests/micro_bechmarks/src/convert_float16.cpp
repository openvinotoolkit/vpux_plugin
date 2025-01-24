//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <benchmark/benchmark.h>

#include "vpux/utils/core/type/float16.hpp"

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace {
constexpr size_t dataLen = 1'048'576;

using ToF16Ptr = void (*)(const float*, uint16_t*, const size_t);
using ToF32Ptr = void (*)(const uint16_t*, float*, const size_t);

void naiveToF16(const float* in, vpux::type::float16* out, const size_t size) {
    for (size_t i = 0; i < size; ++i) {
        out[i] = in[i];
    }
}

void naiveToF32(const vpux::type::float16* in, float* out, const size_t size) {
    for (size_t i = 0; i < size; ++i) {
        out[i] = in[i];
    }
}

#ifdef __AVX2__
void scalarToF16(const float* f32, uint16_t* f16, const size_t size) {
    constexpr auto flags = _MM_FROUND_TO_NEAREST_INT;
    for (size_t i = 0; i < size; ++i) {
        f16[i] = static_cast<uint16_t>(_mm_cvtsi128_si32(_mm256_cvtps_ph(_mm256_set1_ps(f32[i]), flags)));
    }
}

void vectorToF16(const float* f32, uint16_t* f16, const size_t size) {
    constexpr size_t vecSize = 8;
    const size_t numIter = size / vecSize;
    for (size_t i = 0; i < numIter; i++) {
        const auto inPtr = f32 + i * vecSize;
        auto outPtr = reinterpret_cast<__m128i*>(f16 + i * vecSize);
        __m256 float32x8 = _mm256_loadu_ps(inPtr);
        __m128i float16x8 = _mm256_cvtps_ph(float32x8, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128(outPtr, float16x8);
    }

    const size_t tailStart = numIter * vecSize;
    for (size_t i = tailStart; i < size; i++) {
        __m128 float32x1 = _mm_load_ss(f32 + i);
        __m128i float16x1 = _mm_cvtps_ph(float32x1, _MM_FROUND_TO_NEAREST_INT);
        f16[i] = static_cast<uint16_t>(_mm_extract_epi16(float16x1, 0));
    }
}

void vectorToF16StepBack(const float* f32, uint16_t* f16, const size_t size) {
    constexpr size_t vecSize = 8;
    const size_t numIter = size / vecSize;
    for (size_t i = 0; i < numIter; i++) {
        const auto inPtr = f32 + i * vecSize;
        auto outPtr = reinterpret_cast<__m128i*>(f16 + i * vecSize);
        __m256 float32x8 = _mm256_loadu_ps(inPtr);
        __m128i float16x8 = _mm256_cvtps_ph(float32x8, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128(outPtr, float16x8);
    }

    const size_t tailStart = numIter * vecSize;
    if (tailStart < size) {
        const auto idx = size - vecSize;
        const auto inPtr = f32 + idx;
        auto outPtr = reinterpret_cast<__m128i*>(f16 + idx);
        __m256 float32x8 = _mm256_loadu_ps(inPtr);
        __m128i float16x8 = _mm256_cvtps_ph(float32x8, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128(outPtr, float16x8);
    }
}

void scalarToF32(const uint16_t* f16, float* f32, const size_t size) {
    for (size_t i = 0; i < size; ++i) {
        f32[i] = _mm256_cvtss_f32(_mm256_cvtph_ps(_mm_cvtsi32_si128(f16[i])));
    }
}

void vectorToF32(const uint16_t* f16, float* f32, const size_t size) {
    constexpr size_t vecSize = 128 / 8;
    const size_t numIter = size / vecSize;
    for (size_t i = 0; i < numIter; i++) {
        const auto inPtr = reinterpret_cast<const __m128i*>(f16 + i * vecSize);
        auto outPtr = f32 + i * vecSize;
        __m128i float16x8 = _mm_loadu_si128(inPtr);
        __m256 float32x8 = _mm256_cvtph_ps(float16x8);
        _mm256_storeu_ps(outPtr, float32x8);
    }

    const size_t tailStart = numIter * vecSize;
    __m128i dummyVal{};
    dummyVal = _mm_xor_si128(dummyVal, dummyVal);
    for (size_t i = tailStart; i < size; i++) {
        const auto* inPtr = f16 + i;
        __m128i float16x1 = _mm_insert_epi16(dummyVal, *inPtr, 0);
        __m128 float32x1 = _mm_cvtph_ps(float16x1);
        f32[i] = _mm_cvtss_f32(float32x1);
    }
}

void vectorToF32StepBack(const uint16_t* f16, float* f32, const size_t size) {
    constexpr size_t vecSize = 128 / 8;
    const size_t numIter = size / vecSize;
    for (size_t i = 0; i < numIter; i++) {
        const auto inPtr = reinterpret_cast<const __m128i*>(f16 + i * vecSize);
        auto outPtr = f32 + i * vecSize;
        __m256 float32x8 = _mm256_cvtph_ps(*inPtr);
        _mm256_storeu_ps(outPtr, float32x8);
    }

    const size_t tailStart = numIter * vecSize;
    if (tailStart < size) {
        const auto idx = size - vecSize;
        const auto inPtr = reinterpret_cast<const __m128i*>(f16 + idx);
        auto outPtr = f32 + idx;
        __m256 float32x8 = _mm256_cvtph_ps(*inPtr);
        _mm256_storeu_ps(outPtr, float32x8);
    }
}
#endif
}  // namespace

static void BM_F32toF16(benchmark::State& state) {
    const std::vector<float> f32Array(dataLen);
    std::vector<vpux::type::float16> f16Array(dataLen);
    benchmark::DoNotOptimize(f16Array);
    for (auto _ : state) {
        naiveToF16(f32Array.data(), f16Array.data(), dataLen);
        benchmark::ClobberMemory();
    }
}

static void BM_F16toF32(benchmark::State& state) {
    const std::vector<vpux::type::float16> f16Array(dataLen);
    std::vector<float> f32Array(dataLen);
    benchmark::DoNotOptimize(f32Array);
    for (auto _ : state) {
        naiveToF32(f16Array.data(), f32Array.data(), dataLen);
        benchmark::ClobberMemory();
    }
}

#ifdef __AVX2__
static void BM_F32toF16_avx2(benchmark::State& state, const ToF16Ptr& callback) {
    const std::vector<float> f32Array(dataLen);
    std::vector<uint16_t> f16Array(dataLen);
    benchmark::DoNotOptimize(f16Array);
    for (auto _ : state) {
        callback(f32Array.data(), f16Array.data(), dataLen);
        benchmark::ClobberMemory();
    }
}

static void BM_F16toF32_avx2(benchmark::State& state, const ToF32Ptr& callback) {
    const std::vector<uint16_t> f16Array(dataLen);
    std::vector<float> f32Array(dataLen);
    benchmark::DoNotOptimize(f32Array);
    for (auto _ : state) {
        callback(f16Array.data(), f32Array.data(), dataLen);
        benchmark::ClobberMemory();
    }
}
#endif

BENCHMARK(BM_F32toF16);
BENCHMARK(BM_F16toF32);
#ifdef __AVX2__
BENCHMARK_CAPTURE(BM_F32toF16_avx2, SingleValue, scalarToF16);
BENCHMARK_CAPTURE(BM_F32toF16_avx2, PackedValue, vectorToF16);
BENCHMARK_CAPTURE(BM_F32toF16_avx2, PackedValueStepBack, vectorToF16StepBack);
BENCHMARK_CAPTURE(BM_F16toF32_avx2, SingleValue, scalarToF32);
BENCHMARK_CAPTURE(BM_F16toF32_avx2, PackedValue, vectorToF32);
BENCHMARK_CAPTURE(BM_F16toF32_avx2, PackedValueStepBack, vectorToF32StepBack);
#endif
