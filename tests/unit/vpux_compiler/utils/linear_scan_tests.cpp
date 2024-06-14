//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/utils/linear_scan.hpp"

#include <gtest/gtest.h>

using namespace vpux;

class MLIR_LinearScanTests : public ::testing::Test {
protected:
    struct Handler;

    struct LiveRange {
        bool alive = true;
        bool fixed = false;
        AddressType size = 0;
        AddressType alignment = 1;
        AddressType addr = InvalidAddress;
        int spillWeight = 0;
        bool spilled = false;
    };

    struct Handler {
        bool isAlive(LiveRange* r) const {
            return r->alive;
        }

        bool isFixedAlloc(LiveRange* r) const {
            return r->fixed;
        }

        AddressType getSize(LiveRange* r) const {
            return r->size;
        }

        AddressType getAlignment(LiveRange* r) const {
            return r->alignment;
        }

        AddressType getAddress(LiveRange* r) const {
            return r->addr;
        }

        void allocated(LiveRange* r, AddressType addr) const {
            r->addr = addr;
        }

        void freed(LiveRange*) const {
        }

        int getSpillWeight(LiveRange* r) const {
            return r->spillWeight;
        }

        bool spilled(LiveRange* r) const {
            r->spilled = true;
            return true;
        }
    };
};

TEST_F(MLIR_LinearScanTests, Alloc) {
    {
        LinearScan<LiveRange*, Handler> s(4);

        LiveRange r1;
        r1.size = 2;

        LiveRange r2;
        r2.size = 2;
        r2.spillWeight = -1;

        LiveRange r3;
        r3.size = 2;

        ASSERT_TRUE(s.alloc({&r1, &r2}));
        ASSERT_EQ(s.liveRanges(), (SmallVector<LiveRange*>{&r1, &r2}));

        ASSERT_TRUE(s.alloc({&r3}));
        ASSERT_EQ(s.liveRanges(), (SmallVector<LiveRange*>{&r1, &r3}));

        ASSERT_FALSE(r1.spilled);
        ASSERT_TRUE(r2.spilled);
        ASSERT_FALSE(r3.spilled);

        r1.alive = false;
        r2.alive = false;
        r3.alive = false;
        s.freeNonAlive();

        ASSERT_EQ(r1.addr, 0);
        ASSERT_EQ(r2.addr, 2);
        ASSERT_EQ(r3.addr, 2);

        ASSERT_TRUE(s.liveRanges().empty());
        ASSERT_EQ(s.gaps().size(), 1);
    }

    {
        LinearScan<LiveRange*, Handler> s(4);

        LiveRange r1;
        r1.size = 2;

        LiveRange r2;
        r2.size = 2;
        r2.spillWeight = -1;

        LiveRange r3;
        r3.size = 2;

        ASSERT_TRUE(s.alloc({&r1, &r2}, false));
        ASSERT_EQ(s.liveRanges(), (SmallVector<LiveRange*>{&r1, &r2}));

        ASSERT_FALSE(s.alloc({&r3}, false));
        ASSERT_EQ(s.liveRanges(), (SmallVector<LiveRange*>{&r1, &r2}));

        ASSERT_FALSE(r1.spilled);
        ASSERT_FALSE(r2.spilled);

        r1.alive = false;
        r2.alive = false;
        r3.alive = false;
        s.freeNonAlive();

        ASSERT_EQ(r1.addr, 0);
        ASSERT_EQ(r2.addr, 2);

        ASSERT_TRUE(s.liveRanges().empty());
    }

    {
        LinearScan<LiveRange*, Handler> s(4);

        LiveRange r1;
        r1.size = 1;
        r1.spillWeight = 1;

        LiveRange r2;
        r2.size = 1;
        r2.spillWeight = 3;

        LiveRange r3;
        r3.size = 1;
        r3.spillWeight = 4;

        LiveRange r4;
        r4.size = 1;
        r4.spillWeight = 2;

        LiveRange r5;
        r5.size = 2;
        r5.spillWeight = 5;

        ASSERT_TRUE(s.alloc({&r1, &r2, &r3, &r4}));
        ASSERT_EQ(r1.addr, 0);
        ASSERT_EQ(r2.addr, 1);
        ASSERT_EQ(r3.addr, 2);
        ASSERT_EQ(r4.addr, 3);

        ASSERT_FALSE(r1.spilled);
        ASSERT_FALSE(r2.spilled);
        ASSERT_FALSE(r3.spilled);
        ASSERT_FALSE(r4.spilled);

        ASSERT_TRUE(s.alloc({&r5}));
        ASSERT_EQ(s.liveRanges().size(), 3);
        ASSERT_EQ(r1.addr, 0);
        ASSERT_EQ(r2.addr, 1);
        ASSERT_EQ(r3.addr, 2);
        ASSERT_EQ(r4.addr, 3);
        ASSERT_EQ(r5.addr, 0);

        ASSERT_TRUE(r1.spilled);
        ASSERT_TRUE(r2.spilled);
        ASSERT_FALSE(r3.spilled);
        ASSERT_FALSE(r4.spilled);
        ASSERT_FALSE(r5.spilled);

        r1.alive = false;
        r2.alive = false;
        r3.alive = false;
        r4.alive = false;
        r5.alive = false;
        s.freeNonAlive();

        ASSERT_TRUE(s.liveRanges().empty());
    }

    {
        LinearScan<LiveRange*, Handler> s(1);

        LiveRange r1;
        r1.size = 1;

        LiveRange r2;
        r2.size = 1;

        ASSERT_FALSE(s.alloc({&r1, &r2}));

        ASSERT_FALSE(r1.spilled);
        ASSERT_FALSE(r2.spilled);

        ASSERT_EQ(r1.addr, InvalidAddress);
        ASSERT_EQ(r2.addr, InvalidAddress);

        ASSERT_TRUE(s.alloc({&r1}));

        ASSERT_FALSE(r1.spilled);
        ASSERT_FALSE(r2.spilled);

        ASSERT_EQ(r1.addr, 0);
        ASSERT_EQ(r2.addr, InvalidAddress);

        ASSERT_TRUE(s.alloc({&r2}));

        ASSERT_TRUE(r1.spilled);
        ASSERT_FALSE(r2.spilled);

        ASSERT_EQ(r1.addr, 0);
        ASSERT_EQ(r2.addr, 0);
    }
}

TEST_F(MLIR_LinearScanTests, AllocFixed) {
    LinearScan<LiveRange*, Handler> s(4);

    LiveRange r1;
    r1.size = 1;

    LiveRange r2;
    r2.size = 1;

    LiveRange r3;
    r3.size = 1;

    LiveRange r4;
    r4.size = 1;

    ASSERT_TRUE(s.alloc({&r1, &r2, &r3, &r4}));

    ASSERT_FALSE(r1.spilled);
    ASSERT_FALSE(r2.spilled);
    ASSERT_FALSE(r3.spilled);
    ASSERT_FALSE(r4.spilled);

    LiveRange r5;
    r5.size = 2;
    r5.addr = 1;
    r5.fixed = true;

    ASSERT_FALSE(s.alloc({&r5}, false));
    ASSERT_FALSE(r1.spilled);
    ASSERT_FALSE(r2.spilled);
    ASSERT_FALSE(r3.spilled);
    ASSERT_FALSE(r4.spilled);

    ASSERT_TRUE(s.alloc({&r5}, true));
    ASSERT_FALSE(r1.spilled);
    ASSERT_TRUE(r2.spilled);
    ASSERT_TRUE(r3.spilled);
    ASSERT_FALSE(r4.spilled);
    ASSERT_FALSE(r5.spilled);

    ASSERT_EQ(r5.addr, 1);
}
