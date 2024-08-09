//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//
#include "vpux/compiler/utils/prelu_pwl_table.hpp"

namespace vpux {

std::optional<vpux::PWLTableEntry> getPWLEntryForAlpha0(const int64_t zeroPoint) {
    static std::map<int64_t, PWLTableEntry> reluPwlTableAlpha0 = {
            {0, PWLTableEntry{/*range=*/{0, 255},
                              /*shift=*/{-5},
                              /*bias=*/{0},
                              /*floatRange=*/std::make_pair(-65504.0, 65504.0),
                              /*postShift=*/5}},
    };

    if (reluPwlTableAlpha0.count(zeroPoint) > 0) {
        return reluPwlTableAlpha0[zeroPoint];
    }

    return std::nullopt;
}

std::optional<vpux::PWLTableEntry> getPWLEntryForAlpha1(const int64_t zeroPoint) {
    static std::map<int64_t, PWLTableEntry> reluPwlTableAlpha1 = {
            {128, PWLTableEntry{/*range=*/{-128, -109, -90, -72, -54, -36, -18, 0, 127},
                                /*shift=*/{1, -1, 0, 0, 0, -1, -1, -4},
                                /*bias=*/{-119, 44, -43, -31, -19, 18, 10, 0},
                                /*floatRange=*/std::make_pair(-65504.0, 65504.0),
                                /*postShift=*/4}},
    };

    if (reluPwlTableAlpha1.count(zeroPoint) > 0) {
        return reluPwlTableAlpha1[zeroPoint];
    }

    return std::nullopt;
}

std::optional<vpux::PWLTableEntry> getPWLEntryForAlpha2(const int64_t zeroPoint) {
    static std::map<int64_t, PWLTableEntry> reluPwlTableAlpha2 = {
            {0, PWLTableEntry{/*range=*/{0, 255},
                              /*shift=*/{-5},
                              /*bias=*/{0},
                              /*floatRange=*/std::make_pair(-65504.0, 65504.0),
                              /*postShift=*/5}},
            {1, PWLTableEntry{/*range=*/{-5, -3, -1, 1, 254},
                              /*shift=*/{0, -1, 0, -1},
                              /*bias=*/{3, 4, 1, 0},
                              /*floatRange=*/std::make_pair(-65504.0, 65504.0),
                              /*postShift=*/1}},
            {2, PWLTableEntry{/*range=*/{-10, -8, -6, -4, 0, 253},
                              /*shift=*/{1, 0, 1, 1, 0},
                              /*bias=*/{3, 6, 2, 1, 0},
                              /*floatRange=*/std::make_pair(-65504.0, 65504.0),
                              /*postShift=*/0}},
            {3, PWLTableEntry{/*range=*/{-15, -13, -11, -9, -7, -5, -3, 0, 252},
                              /*shift=*/{2, 0, 2, 4, 2, 3, 1, 0},
                              /*bias=*/{1, 10, 1, -1, 1, 0, 1, 0},
                              /*floatRange=*/std::make_pair(-65504.0, 65504.0),
                              /*postShift=*/0}},
            {4, PWLTableEntry{/*range=*/{-20, -12, -5, -4, -3, -2, -1, 0, 251},
                              /*shift=*/{0, 0, -5, -5, -5, -5, -5, -3},
                              /*bias=*/{-7, -1, 152, 120, 88, 64, 32, 0},
                              /*floatRange=*/std::make_pair(-65504.0, 65504.0),
                              /*postShift=*/3}},
            {5, PWLTableEntry{/*range=*/{-25, -17, -9, -4, -3, -2, -1, 0, 250},
                              /*shift=*/{1, 1, 0, -5, -5, -5, -5, -2},
                              /*bias=*/{-5, -2, 3, 124, 92, 64, 32, 0},
                              /*floatRange=*/std::make_pair(-65504.0, 65504.0),
                              /*postShift=*/2}},
            {6, PWLTableEntry{/*range=*/{-30, -24, -18, -12, -6, -2, -1, 0, 249},
                              /*shift=*/{-1, -1, 0, 0, -1, -5, -5, -3},
                              /*bias=*/{14, 12, -7, -1, 4, 64, 32, 0},
                              /*floatRange=*/std::make_pair(-65504.0, 65504.0),
                              /*postShift=*/3}},
            {7, PWLTableEntry{/*range=*/{-35, -29, -23, -17, -11, -5, -1, 0, 248},
                              /*shift=*/{0, 0, 1, 1, 0, 0, -5, -2},
                              /*bias=*/{8, 7, -5, -2, 3, 2, 32, 0},
                              /*floatRange=*/std::make_pair(-65504.0, 65504.0),
                              /*postShift=*/2}},
            {8, PWLTableEntry{/*range=*/{-40, -34, -28, -22, -16, -10, -4, 0, 247},
                              /*shift=*/{-1, -1, 0, 0, -1, -1, -2, -3},
                              /*bias=*/{18, 16, -13, -7, 8, 6, 8, 0},
                              /*floatRange=*/std::make_pair(-65504.0, 65504.0),
                              /*postShift=*/3}},
            {9, PWLTableEntry{/*range=*/{-45, -37, -29, -22, -14, -7, -1, 0, 246},
                              /*shift=*/{0, 0, 0, 0, 0, 0, -5, -3},
                              /*bias=*/{-22, -16, -13, -7, -4, 2, 32, 0},
                              /*floatRange=*/std::make_pair(-65504.0, 65504.0),
                              /*postShift=*/3}},
            {10, PWLTableEntry{/*range=*/{-50, -42, -34, -27, -19, -12, -4, 0, 245},
                               /*shift=*/{0, 0, 0, 0, 0, 0, -2, -3},
                               /*bias=*/{-25, -19, -16, -10, -7, -1, 8, 0},
                               /*floatRange=*/std::make_pair(-65504.0, 65504.0),
                               /*postShift=*/3}},
            {11, PWLTableEntry{/*range=*/{-55, -47, -38, -32, -23, -17, -7, 0, 244},
                               /*shift=*/{0, 0, 0, 0, 0, 0, 0, -3},
                               /*bias=*/{-28, -22, -19, -13, -10, -4, 2, 0},
                               /*floatRange=*/std::make_pair(-65504.0, 65504.0),
                               /*postShift=*/3}},
            {12, PWLTableEntry{/*range=*/{-60, -52, -42, -32, -22, -12, -2, 0, 243},
                               /*shift=*/{0, 0, 0, 0, 0, 0, -2, -3},
                               /*bias=*/{-31, -25, -19, -13, -7, -1, 8, 0},
                               /*floatRange=*/std::make_pair(-65504.0, 65504.0),
                               /*postShift=*/3}},
            {13, PWLTableEntry{/*range=*/{-65, -57, -47, -37, -27, -17, -7, 0, 242},
                               /*shift=*/{1, 1, 1, 1, 1, 1, 1, -2},
                               /*bias=*/{-17, -14, -11, -8, -5, -2, 1, 0},
                               /*floatRange=*/std::make_pair(-65504.0, 65504.0),
                               /*postShift=*/2}},
            {14, PWLTableEntry{/*range=*/{-70, -58, -47, -36, -24, -13, -2, 0, 241},
                               /*shift=*/{-1, 0, 0, -1, -1, 0, -2, -3},
                               /*bias=*/{28, -28, -22, 16, 10, -1, 8, 0},
                               /*floatRange=*/std::make_pair(-65504.0, 65504.0),
                               /*postShift=*/3}},
            {15, PWLTableEntry{/*range=*/{-75, -63, -52, -41, -29, -18, -7, 0, 240},
                               /*shift=*/{-1, 0, 0, -1, -1, 0, 0, -3},
                               /*bias=*/{30, -31, -25, 18, 12, -4, 2, 0},
                               /*floatRange=*/std::make_pair(-65504.0, 65504.0),
                               /*postShift=*/3}},
            {16, PWLTableEntry{/*range=*/{-80, -68, -56, -45, -33, -21, -10, 0, 239},
                               /*shift=*/{0, 0, 0, 0, 0, 0, 0, -2},
                               /*bias=*/{16, 14, 12, 9, 7, 5, 2, 0},
                               /*floatRange=*/std::make_pair(-65504.0, 65504.0),
                               /*postShift=*/2}},
            {17, PWLTableEntry{/*range=*/{-85, -72, -59, -46, -33, -20, -7, 0, 238},
                               /*shift=*/{0, 0, 0, 0, 0, 0, 1, -2},
                               /*bias=*/{17, 15, 12, 9, 7, 4, 1, 0},
                               /*floatRange=*/std::make_pair(-65504.0, 65504.0),
                               /*postShift=*/2}},
            {18, PWLTableEntry{/*range=*/{-90, -77, -64, -51, -38, -26, -14, 0, 237},
                               /*shift=*/{0, 0, 0, 0, 0, 0, 0, -2},
                               /*bias=*/{18, 16, 13, 10, 8, 6, 3, 0},
                               /*floatRange=*/std::make_pair(-65504.0, 65504.0),
                               /*postShift=*/2}},
            {19, PWLTableEntry{/*range=*/{-95, -81, -67, -53, -39, -26, -13, 0, 236},
                               /*shift=*/{0, 0, 0, 0, 0, 0, 0, -2},
                               /*bias=*/{19, 16, 13, 11, 8, 5, 3, 0},
                               /*floatRange=*/std::make_pair(-65504.0, 65504.0),
                               /*postShift=*/2}},
            {20, PWLTableEntry{/*range=*/{-100, -85, -70, -55, -41, -28, -15, 0, 235},
                               /*shift=*/{0, 0, 0, 0, 0, 0, 0, -2},
                               /*bias=*/{20, 17, 14, 11, 8, 6, 3, 0},
                               /*floatRange=*/std::make_pair(-65504.0, 65504.0),
                               /*postShift=*/2}},
            {21, PWLTableEntry{/*range=*/{-105, -72, -44, -32, -22, -12, -6, 0, 234},
                               /*shift=*/{-1, -1, -1, 0, 0, 0, -1, -3},
                               /*bias=*/{38, 26, 18, -13, -7, -1, 4, 0},
                               /*floatRange=*/std::make_pair(-65504.0, 65504.0),
                               /*postShift=*/3}},
            {22, PWLTableEntry{/*range=*/{-110, -72, -44, -32, -22, -12, -6, 0, 233},
                               /*shift=*/{-1, -1, -1, 0, 0, 0, -1, -3},
                               /*bias=*/{40, 26, 18, -13, -7, -1, 4, 0},
                               /*floatRange=*/std::make_pair(-65504.0, 65504.0),
                               /*postShift=*/3}},
            {23, PWLTableEntry{/*range=*/{-115, -72, -44, -32, -22, -12, -6, 0, 232},
                               /*shift=*/{-1, -1, -1, 0, 0, 0, -1, -3},
                               /*bias=*/{40, 26, 18, -13, -7, -1, 4, 0},
                               /*floatRange=*/std::make_pair(-65504.0, 65504.0),
                               /*postShift=*/3}},
            {24, PWLTableEntry{/*range=*/{-120, -94, -66, -42, -24, -12, -6, 0, 231},
                               /*shift=*/{-1, -1, -1, -1, -1, 0, -1, -3},
                               /*bias=*/{46, 36, 24, 16, 10, -1, 4, 0},
                               /*floatRange=*/std::make_pair(-65504.0, 65504.0),
                               /*postShift=*/3}},
            {25, PWLTableEntry{/*range=*/{-125, -98, -70, -44, -24, -12, -6, 0, 230},
                               /*shift=*/{-1, -1, -1, -1, -1, 0, -1, -3},
                               /*bias=*/{48, 36, 26, 16, 10, -1, 4, 0},
                               /*floatRange=*/std::make_pair(-65504.0, 65504.0),
                               /*postShift=*/3}},
            {26, PWLTableEntry{/*range=*/{-130, -96, -60, -32, -22, -12, -6, 0, 229},
                               /*shift=*/{-1, -1, -1, 0, 0, 0, -1, -3},
                               /*bias=*/{48, 34, 22, -13, -7, -1, 4, 0},
                               /*floatRange=*/std::make_pair(-65504.0, 65504.0),
                               /*postShift=*/3}},
            {27, PWLTableEntry{/*range=*/{-135, -104, -76, -52, -32, -12, -6, 0, 228},
                               /*shift=*/{-1, -1, -1, -1, -1, 0, -1, -3},
                               /*bias=*/{50, 40, 28, 20, 12, -1, 4, 0},
                               /*floatRange=*/std::make_pair(-65504.0, 65504.0),
                               /*postShift=*/3}},
            {28, PWLTableEntry{/*range=*/{-140, -110, -80, -56, -32, -12, -6, 0, 227},
                               /*shift=*/{-1, -1, -1, -1, -1, 0, -1, -3},
                               /*bias=*/{52, 40, 30, 20, 12, -1, 4, 0},
                               /*floatRange=*/std::make_pair(-65504.0, 65504.0),
                               /*postShift=*/3}},
            {29, PWLTableEntry{/*range=*/{-145, -114, -84, -56, -32, -12, -6, 0, 226},
                               /*shift=*/{-1, -1, -1, -1, -1, 0, -1, -3},
                               /*bias=*/{54, 42, 32, 20, 12, -1, 4, 0},
                               /*floatRange=*/std::make_pair(-65504.0, 65504.0),
                               /*postShift=*/3}},
            {30, PWLTableEntry{/*range=*/{-150, -128, -106, -84, -62, -40, -18, 0, 225},
                               /*shift=*/{0, 0, 0, 0, 0, 0, 0, -2},
                               /*bias=*/{29, 25, 20, 16, 12, 7, 3, 0},
                               /*floatRange=*/std::make_pair(-65504.0, 65504.0),
                               /*postShift=*/2}},
            {31, PWLTableEntry{/*range=*/{-155, -130, -105, -80, -55, -30, -5, 0, 224},
                               /*shift=*/{0, 0, 0, 0, 0, 0, 0, -2},
                               /*bias=*/{30, 25, 20, 15, 10, 5, 2, 0},
                               /*floatRange=*/std::make_pair(-65504.0, 65504.0),
                               /*postShift=*/2}},
            {32, PWLTableEntry{/*range=*/{-160, -135, -110, -85, -60, -35, -10, 0, 223},
                               /*shift=*/{0, 0, 0, 0, 0, 0, 0, -2},
                               /*bias=*/{31, 26, 21, 16, 11, 6, 2, 0},
                               /*floatRange=*/std::make_pair(-65504.0, 65504.0),
                               /*postShift=*/2}},
            {33, PWLTableEntry{/*range=*/{-165, -140, -115, -90, -65, -40, -15, 0, 222},
                               /*shift=*/{0, 0, 0, 0, 0, 0, 0, -2},
                               /*bias=*/{32, 27, 22, 17, 12, 7, 3, 0},
                               /*floatRange=*/std::make_pair(-65504.0, 65504.0),
                               /*postShift=*/2}},
            {34, PWLTableEntry{/*range=*/{-170, -145, -120, -95, -70, -45, -20, 0, 221},
                               /*shift=*/{0, 0, 0, 0, 0, 0, 0, -2},
                               /*bias=*/{33, 28, 23, 18, 13, 8, 4, 0},
                               /*floatRange=*/std::make_pair(-65504.0, 65504.0),
                               /*postShift=*/2}},
            {35, PWLTableEntry{/*range=*/{-175, -150, -125, -100, -75, -50, -25, 0, 220},
                               /*shift=*/{0, 0, 0, 0, 0, 0, 0, -2},
                               /*bias=*/{34, 29, 24, 19, 14, 9, 4, 0},
                               /*floatRange=*/std::make_pair(-65504.0, 65504.0),
                               /*postShift=*/2}},
            {36, PWLTableEntry{/*range=*/{-180, -155, -130, -105, -80, -55, -30, 0, 219},
                               /*shift=*/{0, 0, 0, 0, 0, 0, 0, -2},
                               /*bias=*/{35, 30, 25, 20, 15, 10, 4, 0},
                               /*floatRange=*/std::make_pair(-65504.0, 65504.0),
                               /*postShift=*/2}},
            {37, PWLTableEntry{/*range=*/{-185, -159, -133, -107, -81, -55, -29, 0, 218},
                               /*shift=*/{0, 0, 0, 0, 0, 0, 0, -2},
                               /*bias=*/{36, 31, 25, 20, 15, 10, 4, 0},
                               /*floatRange=*/std::make_pair(-65504.0, 65504.0),
                               /*postShift=*/2}},
            {38, PWLTableEntry{/*range=*/{-190, -163, -136, -109, -82, -55, -28, 0, 217},
                               /*shift=*/{0, 0, 0, 0, 0, 0, 0, -2},
                               /*bias=*/{37, 31, 26, 21, 15, 10, 4, 0},
                               /*floatRange=*/std::make_pair(-65504.0, 65504.0),
                               /*postShift=*/2}},
            {39, PWLTableEntry{/*range=*/{-195, -167, -139, -111, -83, -55, -27, 0, 216},
                               /*shift=*/{0, 0, 0, 0, 0, 0, 0, -2},
                               /*bias=*/{38, 32, 27, 21, 15, 10, 4, 0},
                               /*floatRange=*/std::make_pair(-65504.0, 65504.0),
                               /*postShift=*/2}},
            {40, PWLTableEntry{/*range=*/{-200, -171, -142, -113, -84, -55, -26, 0, 215},
                               /*shift=*/{0, 0, 0, 0, 0, 0, 0, -2},
                               /*bias=*/{39, 33, 27, 21, 15, 10, 4, 0},
                               /*floatRange=*/std::make_pair(-65504.0, 65504.0),
                               /*postShift=*/2}},
            {41, PWLTableEntry{/*range=*/{-205, -176, -147, -118, -89, -60, -31, 0, 214},
                               /*shift=*/{0, 0, 0, 0, 0, 0, 0, -2},
                               /*bias=*/{40, 34, 28, 22, 16, 11, 4, 0},
                               /*floatRange=*/std::make_pair(-65504.0, 65504.0),
                               /*postShift=*/2}},
            {42, PWLTableEntry{/*range=*/{-210, -180, -150, -120, -90, -60, -30, 0, 213},
                               /*shift=*/{1, 1, 1, 1, 1, 1, 1, -1},
                               /*bias=*/{20, 17, 14, 11, 8, 5, 2, 0},
                               /*floatRange=*/std::make_pair(-65504.0, 65504.0),
                               /*postShift=*/1}},
            {43, PWLTableEntry{/*range=*/{-215, -183, -151, -119, -87, -55, -23, 0, 212},
                               /*shift=*/{0, 0, 0, 0, 0, 0, 0, -2},
                               /*bias=*/{41, 35, 28, 22, 15, 9, 4, 0},
                               /*floatRange=*/std::make_pair(-65504.0, 65504.0),
                               /*postShift=*/2}},
            {44, PWLTableEntry{/*range=*/{-220, -188, -156, -124, -92, -60, -28, 0, 211},
                               /*shift=*/{1, 1, 1, 1, 1, 1, 1, -1},
                               /*bias=*/{21, 18, 15, 12, 8, 5, 2, 0},
                               /*floatRange=*/std::make_pair(-65504.0, 65504.0),
                               /*postShift=*/1}},
            {45, PWLTableEntry{/*range=*/{-225, -192, -159, -126, -93, -60, -27, 0, 210},
                               /*shift=*/{0, 0, 0, 0, 0, 0, 0, -2},
                               /*bias=*/{43, 36, 30, 23, 17, 10, 4, 0},
                               /*floatRange=*/std::make_pair(-65504.0, 65504.0),
                               /*postShift=*/2}},
    };

    if (reluPwlTableAlpha2.count(zeroPoint) > 0) {
        return reluPwlTableAlpha2[zeroPoint];
    }

    return std::nullopt;
}

std::optional<vpux::PWLTableEntry> getPWLEntryForAlpha25(const int64_t zeroPoint) {
    // FIX ME: The PWL table requires fixed input and output quantization ranges for i13 data.
    // So the low boundry of PWL input should be -4096. It means the value will be clamped if lower than -4096.
    // Here the clampLow and clampHigh is corresponding to the FQ range of leakyRelu's output.
    // If zeroPoint is higher than 128, clampLow will exceed the lower bound.
    // But clampLow only shows the theoretical lowest value, actual values could still stay in a reasonable range.
    //
    // For example, if FQ range is [-192.0, 63.0] and zeropoint is 192.
    // Then values within [-128.0, 63.0] are accurate but others are clamped to -128.
    if (zeroPoint < 0 || zeroPoint > 128)
        return std::nullopt;

    const auto clampLow = checked_cast<int32_t>(-zeroPoint * 4) * 8 - 0;
    const auto clampHigh = checked_cast<int32_t>(255 - zeroPoint) * 8 + 0;
    const int32_t ps = 0;
    const int32_t bn = 4;
    const int32_t bp = 4;
    VPUX_THROW_WHEN(clampHigh < 6, "Wrong clamp high: '{0}', expected to be greater than 6", clampHigh);
    return PWLTableEntry{/*range=*/{clampLow, 0, clampHigh * 1 / 7, clampHigh * 2 / 7, clampHigh * 3 / 7,
                                    clampHigh * 4 / 7, clampHigh * 5 / 7, clampHigh * 6 / 7, clampHigh * 7 / 7},
                         /*shift=*/{-ps + 2, -ps, -ps, -ps, -ps, -ps, -ps, -ps},
                         /*bias=*/{bn, bp, bp, bp, bp, bp, bp, bp},
                         /*floatRange=*/std::make_pair(-65504.0, 65504.0),
                         /*postShift=*/ps + 3};
}

}  // namespace vpux
