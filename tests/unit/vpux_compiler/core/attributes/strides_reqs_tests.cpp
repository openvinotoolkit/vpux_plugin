//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/core/attributes/stride_reqs.hpp"

#include <gtest/gtest.h>

using namespace vpux;

namespace {

std::vector<std::pair<StrideReqKind, Bit>> getValidAttrs() {
    return {std::make_pair(StrideReqKind::Aligned, 2_Byte), std::make_pair(StrideReqKind::Aligned, 4_Byte),
            std::make_pair(StrideReqKind::Aligned, 8_Byte), std::make_pair(StrideReqKind::Compact, 0_Byte),
            std::make_pair(StrideReqKind::Fixed, 1_Byte),   std::make_pair(StrideReqKind::Fixed, 2_Byte),
            std::make_pair(StrideReqKind::Fixed, 3_Byte)};
}

std::vector<std::pair<StrideReqKind, Bit>> getInvalidAttrs() {
    return {
            std::make_pair(StrideReqKind::Aligned, Byte(-2)), std::make_pair(StrideReqKind::Aligned, 0_Byte),
            std::make_pair(StrideReqKind::Aligned, 3_Byte),   std::make_pair(StrideReqKind::Compact, Byte(-1)),
            std::make_pair(StrideReqKind::Compact, 1_Byte),   std::make_pair(StrideReqKind::Compact, 2_Byte),
            std::make_pair(StrideReqKind::Fixed, Byte(-1)),   std::make_pair(StrideReqKind::Fixed, 0_Byte),
    };
}

std::vector<std::tuple<MemShape, Bit, MemStrides>> getShapes2Strides() {
    return {
            std::make_tuple(MemShape({2, 3}), 1_Byte, MemStrides({Bit(3_Byte), Bit(1_Byte)})),
            std::make_tuple(MemShape({2, 3}), 4_Byte, MemStrides({Bit(12_Byte), Bit(4_Byte)})),
            std::make_tuple(MemShape({2, 3, 4, 5}), 1_Byte,
                            MemStrides({Bit(60_Byte), Bit(20_Byte), Bit(5_Byte), Bit(1_Byte)})),
            std::make_tuple(MemShape({2, 3, 4, 5}), 4_Byte,
                            MemStrides({Bit(240_Byte), Bit(80_Byte), Bit(20_Byte), Bit(4_Byte)})),
    };
}

std::vector<std::tuple<MemShape, Bit, MemStrides>> getIncorrectShapes2Strides() {
    return {
            std::make_tuple(MemShape({2, 3}), 4_Byte, MemStrides({Bit(1_Byte), Bit(2_Byte)})),
            std::make_tuple(MemShape({2, 3}), 1_Byte, MemStrides({Bit(4_Byte), Bit(8_Byte)})),
            std::make_tuple(MemShape({2, 3, 4, 5}), 4_Byte,
                            MemStrides({Bit(1_Byte), Bit(2_Byte), Bit(6_Byte), Bit(24_Byte)})),
            std::make_tuple(MemShape({2, 3, 4, 5}), 1_Byte,
                            MemStrides({Bit(4_Byte), Bit(8_Byte), Bit(24_Byte), Bit(96_Byte)})),
            std::make_tuple(MemShape({2, 3, 4, 5}), 4_Byte,
                            MemStrides({Bit(3_Byte), Bit(8_Byte), Bit(24_Byte), Bit(96_Byte)})),
            std::make_tuple(MemShape({2, 3, 4, 5}), 4_Byte,
                            MemStrides({Bit(4_Byte), Bit(9_Byte), Bit(24_Byte), Bit(96_Byte)})),
            std::make_tuple(MemShape({2, 3, 4, 5}), 4_Byte,
                            MemStrides({Bit(4_Byte), Bit(8_Byte), Bit(25_Byte), Bit(96_Byte)})),
            std::make_tuple(MemShape({2, 3, 4, 5}), 4_Byte,
                            MemStrides({Bit(4_Byte), Bit(8_Byte), Bit(24_Byte), Bit(97_Byte)})),
    };
}

}  //  namespace

TEST(MLIR_EnumTraitsTest, StringifyEnumTest) {
    std::vector<std::pair<StrideReqKind, std::string>> enum2StrView{std::make_pair(StrideReqKind::Compact, "Compact"),
                                                                    std::make_pair(StrideReqKind::Aligned, "Aligned"),
                                                                    std::make_pair(StrideReqKind::Fixed, "Fixed")};

    std::for_each(enum2StrView.begin(), enum2StrView.end(), [](const auto& enum2str) {
        EXPECT_EQ(stringifyEnum(enum2str.first), enum2str.second);
    });
}

TEST(MLIR_DimStrideReqTest, verifyAttrsTest) {
    // check valid cases
    auto validAttrs = getValidAttrs();
    std::for_each(validAttrs.begin(), validAttrs.end(), [](const auto& attr) {
        EXPECT_NO_THROW(DimStrideReq::verifyAttrs(attr.first, attr.second));
    });

    // check invalid cases
    auto invalidAttrs = getInvalidAttrs();
    std::for_each(invalidAttrs.begin(), invalidAttrs.end(), [](const auto& attr) {
        EXPECT_ANY_THROW(DimStrideReq::verifyAttrs(attr.first, attr.second));
    });
}

TEST(MLIR_DimStrideReqTest, memDimTest) {
    auto validAttrs = getValidAttrs();
    std::for_each(validAttrs.begin(), validAttrs.end(), [](const auto& attr) {
        for (size_t dim = 0; dim < MAX_NUM_DIMS; ++dim) {
            EXPECT_EQ(DimStrideReq(MemDim(dim), attr.first, attr.second).memDim(), MemDim(dim));
        }
    });
}

TEST(MLIR_DimStrideReqTest, kindTest) {
    auto validAttrs = getValidAttrs();
    std::for_each(validAttrs.begin(), validAttrs.end(), [](const auto& attr) {
        for (size_t dim = 0; dim < MAX_NUM_DIMS; ++dim) {
            EXPECT_EQ(DimStrideReq(MemDim(dim), attr.first, attr.second).kind(), attr.first);
        }
    });
}

TEST(MLIR_DimStrideReqTest, extraValueTest) {
    auto validAttrs = getValidAttrs();
    std::for_each(validAttrs.begin(), validAttrs.end(), [](const auto& attr) {
        for (size_t dim = 0; dim < MAX_NUM_DIMS; ++dim) {
            EXPECT_EQ(DimStrideReq(MemDim(dim), attr.first, attr.second).extraValue(), attr.second);
        }
    });
}

TEST(MLIR_StrideReqRefTest, addRemoveTest) {
    auto validAttrs = getValidAttrs();
    std::for_each(validAttrs.begin(), validAttrs.end(), [](const auto& attr) {
        StrideReqs strideReq;
        for (size_t dim = 0; dim < MAX_NUM_DIMS; ++dim) {
            EXPECT_NO_THROW(strideReq.add(DimStrideReq(MemDim(dim), attr.first, attr.second)));
        }
        for (size_t dim = 0; dim < MAX_NUM_DIMS; ++dim) {
            EXPECT_NO_THROW(strideReq.remove(MemDim(dim)));
        }
    });
}

TEST(MLIR_StrideReqRefTest, hasReqForTest) {
    auto validAttrs = getValidAttrs();
    std::for_each(validAttrs.begin(), validAttrs.end(), [](const auto& attr) {
        StrideReqs strideReq;
        for (size_t dim = 0; dim < MAX_NUM_DIMS; ++dim) {
            EXPECT_NO_THROW(strideReq.add(DimStrideReq(MemDim(dim), attr.first, attr.second)));
            EXPECT_TRUE(strideReq.hasReqFor(MemDim(dim)));
        }
        for (size_t dim = 0; dim < MAX_NUM_DIMS; ++dim) {
            EXPECT_NO_THROW(strideReq.remove(MemDim(dim)));
            EXPECT_FALSE(strideReq.hasReqFor(MemDim(dim)));
        }
    });
}

TEST(MLIR_StrideReqRefTest, sizeTest) {
    auto validAttrs = getValidAttrs();
    std::for_each(validAttrs.begin(), validAttrs.end(), [](const auto& attr) {
        StrideReqs strideReq;
        for (size_t dim = 0; dim < MAX_NUM_DIMS; ++dim) {
            EXPECT_NO_THROW(strideReq.add(DimStrideReq(MemDim(dim), attr.first, attr.second)));
            EXPECT_EQ(strideReq.size(), dim + 1);
        }
        for (size_t dim = 0; dim < MAX_NUM_DIMS; ++dim) {
            EXPECT_NO_THROW(strideReq.remove(MemDim(dim)));
            EXPECT_EQ(strideReq.size(), MAX_NUM_DIMS - (dim + 1));
        }
    });
}

TEST(MLIR_StrideReqRefTest, checkStridesTest) {
    // check strides for correct cases
    auto shapes2strides = getShapes2Strides();
    std::for_each(shapes2strides.begin(), shapes2strides.end(), [](const auto& shape2stride) {
        auto stridesReq = StrideReqs::compact(std::get<0>(shape2stride).size());

        EXPECT_TRUE(stridesReq.checkStrides(std::get<2>(shape2stride), std::get<1>(shape2stride),
                                            std::get<0>(shape2stride)));
    });
    // check strides for incorrect cases
    auto shapes2stridesIncorrect = getIncorrectShapes2Strides();
    std::for_each(shapes2stridesIncorrect.begin(), shapes2stridesIncorrect.end(), [](const auto& shape2stride) {
        auto stridesReq = StrideReqs::compact(std::get<0>(shape2stride).size());

        EXPECT_FALSE(stridesReq.checkStrides(std::get<2>(shape2stride), std::get<1>(shape2stride),
                                             std::get<0>(shape2stride)));
    });
}

TEST(MLIR_StrideReqRefTest, calcStridesTest) {
    auto shapes2strides = getShapes2Strides();
    std::for_each(shapes2strides.begin(), shapes2strides.end(), [](const auto& shape2stride) {
        auto stridesReq = StrideReqs::compact(std::get<0>(shape2stride).size());

        MemStrides memStrides;
        stridesReq.calcStrides(memStrides, std::get<1>(shape2stride), std::get<0>(shape2stride));

        EXPECT_EQ(stridesReq.calcStrides(std::get<1>(shape2stride), std::get<0>(shape2stride)),
                  std::get<2>(shape2stride));
        EXPECT_EQ(memStrides, std::get<2>(shape2stride));
    });
}

TEST(MLIR_StrideReqRefTest, emptyTest) {
    auto validAttrs = getValidAttrs();
    std::for_each(validAttrs.begin(), validAttrs.end(), [](const auto& attr) {
        StrideReqs strideReq;
        EXPECT_TRUE(strideReq.empty());
        for (size_t dim = 0; dim < MAX_NUM_DIMS; ++dim) {
            EXPECT_NO_THROW(strideReq.add(DimStrideReq(MemDim(dim), attr.first, attr.second)));
            EXPECT_FALSE(strideReq.empty());
        }
        for (size_t dim = 0; dim < MAX_NUM_DIMS; ++dim) {
            EXPECT_FALSE(strideReq.empty());
            EXPECT_NO_THROW(strideReq.remove(MemDim(dim)));
        }
        EXPECT_TRUE(strideReq.empty());
    });
}
