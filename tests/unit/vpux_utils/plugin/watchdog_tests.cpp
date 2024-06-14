//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vpux/utils/plugin/watchdog.hpp>

using ::testing::StrictMock;
using namespace std::chrono;
using ::testing::InvokeWithoutArgs;

class MockAbortCallback {
public:
    MOCK_METHOD0(onAbort, void());
    operator std::function<void()>() {
        return [this]() {
            onAbort();
        };
    }
};

class WatchDogSlowDestruct : public vpux::WatchDog {
public:
    using vpux::WatchDog::WatchDog;
    // we do not call cv notify one, which increases thread stopping time
    ~WatchDogSlowDestruct() override {
        _stopWatchdog = true;
        if (_abortThread.joinable()) {
            _abortThread.join();
        }
    }
};

constexpr milliseconds MS10 = milliseconds(10);

class WatchDogTests : public ::testing::Test {
protected:
    vpux::Logger test_logger;
    StrictMock<MockAbortCallback> mock_callback;
    std::condition_variable cv;
    std::mutex mt;

    WatchDogTests(): test_logger("watchdog_tests", vpux::LogLevel::Error) {
    }
};

TEST_F(WatchDogTests, can_create_watchdog) {
    vpux::WatchDog wd1(
            1000, test_logger, []() {}, MS10);
}

TEST_F(WatchDogTests, cannot_pause_invalid_watcher) {
    vpux::WatchDog wd1(
            1000, test_logger, []() {}, MS10);
    ASSERT_ANY_THROW(wd1.Pause());

    wd1.Start();
    ASSERT_ANY_THROW(wd1.Pause(this));
}

TEST_F(WatchDogTests, cannot_double_start) {
    vpux::WatchDog wd1(
            1000, test_logger, []() {}, MS10);
    wd1.Start();
    ASSERT_ANY_THROW(wd1.Start());
}

TEST_F(WatchDogTests, receiving_abort_callback_if_timeout_passes) {
    vpux::WatchDog wd1(10, test_logger, mock_callback, MS10);

    EXPECT_CALL(mock_callback, onAbort()).Times(1).WillOnce(InvokeWithoutArgs([&] {
        cv.notify_one();
    }));

    // checking that thread spawned if timeout happened
    wd1.Start();
    std::unique_lock<std::mutex> lk(mt);
    cv.wait_for(lk, milliseconds(5000));
    wd1.Pause();
}

TEST_F(WatchDogTests, skip_spawning_thread_if_no_interval) {
    vpux::WatchDog wd1(0, test_logger, mock_callback);

    // checking that thread wont call our callback
    wd1.Start();
    std::this_thread::sleep_for(milliseconds(500));
    wd1.Pause();
}

// [Track number: E#49576]
// Unstable, failing from time to time
TEST_F(WatchDogTests, DISABLED_can_stop_thread_faster_than_30_ms) {
    for (size_t i = 0; i != 1000; i++) {
        std::shared_ptr<vpux::WatchDog> wd1 = std::make_shared<vpux::WatchDog>(1000, test_logger, mock_callback);

        wd1->Start();
        if ((i % 2) == 1) {
            std::this_thread::sleep_for(milliseconds(5));
        }
        wd1->Pause();

        auto destructionStart = steady_clock::now();
        wd1.reset();
        auto destructionTime = duration_cast<milliseconds>(steady_clock::now() - destructionStart).count();

        ASSERT_LE(destructionTime, 30);
    }
}

TEST_F(WatchDogTests, can_not_stop_thread_fast_enough_without_cv_notify) {
    milliseconds destructionTime;
    for (size_t i = 0; i != 100; i++) {
        std::shared_ptr<WatchDogSlowDestruct> wd1 =
                std::make_shared<WatchDogSlowDestruct>(1000, test_logger, mock_callback);
        // wait until cv entered into wait phase
        std::this_thread::sleep_for(milliseconds(30));
        auto destructionStart = steady_clock::now();
        wd1.reset();
        destructionTime = duration_cast<milliseconds>(steady_clock::now() - destructionStart);
        if (destructionTime.count() > 30) {
            break;
        }
    }
    ASSERT_GE(destructionTime.count(), 30);
}

TEST_F(WatchDogTests, can_watch_for_multiple_infer_reqs_independently) {
    vpux::WatchDog wd1(100, test_logger, mock_callback, MS10);
    char threads[2];
    void* threadPtrs[2] = {&threads[0], &threads[1]};

    // no notify
    wd1.Start(threadPtrs[0]);
    std::this_thread::sleep_for(milliseconds(1));
    EXPECT_NO_THROW(wd1.Pause(threadPtrs[0]));

    wd1.Start(threadPtrs[1]);
    std::this_thread::sleep_for(milliseconds(1));
    EXPECT_NO_THROW(wd1.Pause(threadPtrs[1]));

    // now we are expecting call
    EXPECT_CALL(mock_callback, onAbort()).Times(1);

    wd1.Start(threadPtrs[0]);
    wd1.Start(threadPtrs[1]);
    std::this_thread::sleep_for(milliseconds(1));
    EXPECT_NO_THROW(wd1.Pause(threadPtrs[0]));
    std::this_thread::sleep_for(milliseconds(500));
}
