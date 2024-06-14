//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "intel_npu/utils/logger/logger.hpp"

#include <gtest/gtest.h>
#include <iostream>
#include <sstream>
#include <string>

using namespace intel_npu;

class NPULoggerTests : public testing::Test {
public:
    std::stringstream testStream;
    std::streambuf* coutBuffer;
    std::streambuf* cerrBuffer;

    void SetUp() override {
        testing::Test::SetUp();

        coutBuffer = std::cout.rdbuf(testStream.rdbuf());
        cerrBuffer = std::cerr.rdbuf(testStream.rdbuf());
    }

    void TearDown() override {
        std::cout.rdbuf(coutBuffer);
        std::cerr.rdbuf(coutBuffer);
        testing::Test::TearDown();
    }
};

TEST_F(NPULoggerTests, LocalLoggerTest) {
    Logger logger = Logger("LocalLoggerTest", ov::log::Level::NO);
    logger.error("Test error msg under None level, shall see no info.");
    std::string content = testStream.str();
    testStream.str("");
    EXPECT_TRUE(content.find("error") == std::string::npos);

    logger.setLevel(ov::log::Level::ERR);
    logger.error("Test error msg under Error level, shall see error.");
    logger.warning("Test warning msg under Error level, shall not see warning.");
    content = testStream.str();
    testStream.str("");
    EXPECT_TRUE(content.find("error") != std::string::npos && content.find("warning") == std::string::npos);

    logger.setLevel(ov::log::Level::WARNING);
    logger.error("Test error msg under Warning level , shall see error.");
    logger.warning("Test warning msg under Warning level , shall see warning.");
    logger.info("Test info msg under Warning level , shall not see info.");
    content = testStream.str();
    testStream.str("");
    EXPECT_TRUE(content.find("error") != std::string::npos && content.find("warning") != std::string::npos &&
                content.find("info") == std::string::npos);

    logger.setLevel(ov::log::Level::INFO);
    logger.warning("Test warning msg under Info level , shall see warning.");
    logger.info("Test info msg under Info level , shall see info.");
    logger.debug("Test debug msg under Info level , shall not see debug.");
    content = testStream.str();
    testStream.str("");
    EXPECT_TRUE(content.find("warning") != std::string::npos && content.find("info") != std::string::npos &&
                content.find("debug") == std::string::npos);

    logger.setLevel(ov::log::Level::DEBUG);
    logger.info("Test info msg under Debug level , shall see info.");
    logger.debug("Test debug msg under Debug level , shall see debug.");
    logger.trace("Test trace msg under Debug level , shall not see trace.");
    content = testStream.str();
    testStream.str("");
    EXPECT_TRUE(content.find("info") != std::string::npos && content.find("debug") != std::string::npos &&
                content.find("trace") == std::string::npos);

    logger.setLevel(ov::log::Level::TRACE);
    logger.debug("Test debug msg under Trace level , shall see debug.");
    logger.trace("Test trace msg under Trace level , shall see trace.");
    content = testStream.str();
    testStream.str("");
    EXPECT_TRUE(content.find("debug") != std::string::npos && content.find("trace") != std::string::npos);
}

TEST_F(NPULoggerTests, CloneLoggerTest) {
    Logger logger = Logger::global().clone("GloablLoggerTest");
    logger.setLevel(ov::log::Level::INFO);
    logger.warning("Test warning msg under Info level , shall see warning.");
    logger.info("Test info msg under Info level , shall see info.");
    logger.debug("Test debug msg under Info level , shall not see debug.");
    std::string content = testStream.str();
    testStream.str("");
    EXPECT_TRUE(content.find("warning") != std::string::npos && content.find("info") != std::string::npos &&
                content.find("debug") == std::string::npos);
}

TEST_F(NPULoggerTests, LongMsgTest) {
    std::string msg(266, 't');
    Logger logger = Logger::global().clone("LongStringTest");
    logger.setLevel(ov::log::Level::INFO);
    logger.info(msg.c_str());
    std::string content = testStream.str();
    testStream.str("");
    EXPECT_TRUE(content.find(msg) != std::string::npos);
}
