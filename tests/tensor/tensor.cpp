#include "gtest/gtest.h"
#include "include/mcm/tensor/tensor.hpp"
#include "include/mcm/tensor/math.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/order/order.hpp"

TEST(tensor, populating)
{

    mv::Shape tShape({5, 5});
    std::vector<double> data = mv::utils::generateSequence<double>(tShape.totalSize());
    mv::Tensor t("t", tShape, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)));
    t.populate(data);

    for (unsigned j = 0; j < tShape[0]; ++j)
        for (unsigned i = 0; i < tShape[1]; ++i)
            ASSERT_EQ(t({i, j}), i + tShape[0] * j);

}

/*TEST(tensor, sub_to_ind_column_major)
{

    mv::Shape tShape({32, 16, 8, 4});
    std::vector<std::size_t> subs[] = {
        {31, 15, 4, 2},
        {15, 7, 2, 0},
        {5, 1, 2, 3},
        {25, 5, 6, 1},
        {0, 12, 0, 3}
    };

    auto idxFcn = [tShape](const std::vector<std::size_t> &s) {
        return s[0] + tShape[0] * (s[1] + tShape[1] * (s[2] + tShape[2] * s[3]));
    };

    mv::Order mv::Order(mv::Order(mv::Order::getColMajorID(3)));

    for (unsigned i = 0; i < 5; ++i)
        ASSERT_EQ(order.subToInd(tShape, subs[i]), idxFcn(subs[i]));



}

TEST(tensor, int_to_sub_column_major)
{

    mv::Shape tShape({32, 16, 8, 4});
    std::vector<double> data = mv::utils::generateSequence<double>(tShape.totalSize());
    mv::Tensor t("t", tShape, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)));
    t.populate(data);

    std::vector<unsigned> idx = {0, 100, 101, 545, 10663};

    for (unsigned i = 0; i < 5; ++i)
    {
        std::vector<std::size_t> sub = t.indToSub(idx[i]);
        ASSERT_EQ(t(sub), idx[i]);
    }

}

TEST(tensor, sub_to_ind_row_major)
{

    mv::Shape tShape({32, 16, 8, 4});
    std::vector<std::size_t> subs[] = {
        {31, 15, 4, 2},
        {15, 7, 2, 0},
        {5, 1, 2, 3},
        {25, 5, 6, 1},
        {0, 12, 0, 3}
    };

    auto idxFcn = [tShape](const std::vector<std::size_t> &s) {
        return s[3] + tShape[3] * (s[2] + tShape[2] * (s[1] + tShape[1] * s[0]));
    };

    mv::Order mv::Order(mv::Order(mv::Order::getRowMajorID(3)));

    for (unsigned i = 0; i < 5; ++i)
        ASSERT_EQ(order.subToInd(tShape, subs[i]), idxFcn(subs[i]));

}

TEST(tensor, ind_to_sub_row_major)
{

    mv::Shape tShape({32, 16, 8, 4});
    std::vector<double> data = mv::utils::generateSequence<double>(tShape.totalSize());
    mv::Tensor t("t", tShape, mv::DTypeType::Float16, mv::Order(mv::Order::getRowMajorID(3)));
    t.populate(data);

    std::vector<unsigned> idx = {0, 100, 101, 545, 10663};

    for (unsigned i = 0; i < 5; ++i)
    {
        std::vector<std::size_t> sub = t.indToSub(idx[i]);
        ASSERT_EQ(t(sub), t(idx[i]));
    }

}

TEST(tensor, sub_to_ind_planar)
{

    mv::Shape tShape({32, 16, 8, 4});
    std::vector<std::size_t> subs[] = {
        {31, 15, 4, 2},
        {15, 7, 2, 0},
        {5, 1, 2, 3},
        {25, 5, 6, 1},
        {0, 12, 0, 3}
    };

    auto idxFcn = [tShape](const std::vector<std::size_t> &s) {
        return s[3] + tShape[3] * (s[2] + tShape[2] * (s[0] + tShape[0] * s[1]));
    };

    mv::Order mv::Order(mv::Order("HWC"));

    for (unsigned i = 0; i < 5; ++i)
        ASSERT_EQ(order.subToInd(tShape, subs[i]), idxFcn(subs[i]));

}

TEST(tensor, ind_to_sub_planar)
{

    mv::Shape tShape({32, 16, 8, 4});
    std::vector<double> data = mv::utils::generateSequence<double>(tShape.totalSize());
    mv::Tensor t("t", tShape, mv::DTypeType::Float16, mv::Order("HWC"));
    t.populate(data);

    std::vector<unsigned> idx = {0, 100, 101, 545, 10663};
    auto idxFcn = [tShape](const std::vector<std::size_t>& s) {
        return s[3] + tShape[3] * (s[2] + tShape[2] * (s[0] + tShape[0] * s[1]));
    };

    for (unsigned i = 0; i < 5; ++i)
    {
        std::vector<std::size_t> sub = t.indToSub(idx[i]);
        ASSERT_EQ(t(sub), t(idx[i]));
    }

}*/

TEST(tensor, column_major_to_row_major)
{

    mv::Shape tShape({3, 3, 3, 3});
    std::vector<double> data = {
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
        17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f,
        25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f,
        33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f, 40.0f,
        41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f,
        49.0f, 50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f,
        57.0f, 58.0f, 59.0f, 60.0f, 61.0f, 62.0f, 63.0f, 64.0f,
        65.0f, 66.0f, 67.0f, 68.0f, 69.0f, 70.0f, 71.0f, 72.0f,
        73.0f, 74.0f, 75.0f, 76.0f, 77.0f, 78.0f, 79.0f, 80.0f
    };

    std::vector<double> reorderedData = {
        0.0f, 27.0f, 54.0f, 9.0f, 36.0f, 63.0f, 18.0f, 45.0f, 72.0f,
        3.0f, 30.0f, 57.0f, 12.0f, 39.0f, 66.0f, 21.0f, 48.0f, 75.0f,
        6.0f, 33.0f, 60.0f, 15.0f, 42.0f, 69.0f, 24.0f, 51.0f, 78.0f,
        1.0f, 28.0f, 55.0f, 10.0f, 37.0f, 64.0f, 19.0f, 46.0f, 73.0f,
        4.0f, 31.0f, 58.0f, 13.0f, 40.0f, 67.0f, 22.0f, 49.0f, 76.0f,
        7.0f, 34.0f, 61.0f, 16.0f, 43.0f, 70.0f, 25.0f, 52.0f, 79.0f,
        2.0f, 29.0f, 56.0f, 11.0f, 38.0f, 65.0f, 20.0f, 47.0f, 74.0f,
        5.0f, 32.0f, 59.0f, 14.0f, 41.0f, 68.0f, 23.0f, 50.0f, 77.0f,
        8.0f, 35.0f, 62.0f, 17.0f, 44.0f, 71.0f, 26.0f, 53.0f, 80.0f
    };

    mv::Tensor t("t", tShape, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)));
    t.populate(data);
    t.setOrder(mv::Order(mv::Order::getRowMajorID(3)));

    for (unsigned i = 0; i < data.size(); ++i)
        ASSERT_EQ(t(i), reorderedData[i]);

}

TEST(tensor, row_major_to_column_major)
{

    mv::Shape tShape({3, 3, 3, 3});

    std::vector<double> data = {
        0.0f, 27.0f, 54.0f, 9.0f, 36.0f, 63.0f, 18.0f, 45.0f, 72.0f,
        3.0f, 30.0f, 57.0f, 12.0f, 39.0f, 66.0f, 21.0f, 48.0f, 75.0f,
        6.0f, 33.0f, 60.0f, 15.0f, 42.0f, 69.0f, 24.0f, 51.0f, 78.0f,
        1.0f, 28.0f, 55.0f, 10.0f, 37.0f, 64.0f, 19.0f, 46.0f, 73.0f,
        4.0f, 31.0f, 58.0f, 13.0f, 40.0f, 67.0f, 22.0f, 49.0f, 76.0f,
        7.0f, 34.0f, 61.0f, 16.0f, 43.0f, 70.0f, 25.0f, 52.0f, 79.0f,
        2.0f, 29.0f, 56.0f, 11.0f, 38.0f, 65.0f, 20.0f, 47.0f, 74.0f,
        5.0f, 32.0f, 59.0f, 14.0f, 41.0f, 68.0f, 23.0f, 50.0f, 77.0f,
        8.0f, 35.0f, 62.0f, 17.0f, 44.0f, 71.0f, 26.0f, 53.0f, 80.0f
    };

    std::vector<double> reorderedData = {
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
        17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f,
        25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f,
        33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f, 40.0f,
        41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f,
        49.0f, 50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f,
        57.0f, 58.0f, 59.0f, 60.0f, 61.0f, 62.0f, 63.0f, 64.0f,
        65.0f, 66.0f, 67.0f, 68.0f, 69.0f, 70.0f, 71.0f, 72.0f,
        73.0f, 74.0f, 75.0f, 76.0f, 77.0f, 78.0f, 79.0f, 80.0f
    };

    mv::Tensor t("t", tShape, mv::DTypeType::Float16, mv::Order(mv::Order::getRowMajorID(3)));
    t.populate(data);
    t.setOrder(mv::Order(mv::Order::getColMajorID(3)));

    for (unsigned i = 0; i < data.size(); ++i)
        ASSERT_EQ(t(i), reorderedData[i]);

}

TEST(tensor, column_major_to_planar)
{

    mv::Shape tShape({3, 3, 3, 3});
    std::vector<double> data = {
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
        17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f,
        25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f,
        33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f, 40.0f,
        41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f,
        49.0f, 50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f,
        57.0f, 58.0f, 59.0f, 60.0f, 61.0f, 62.0f, 63.0f, 64.0f,
        65.0f, 66.0f, 67.0f, 68.0f, 69.0f, 70.0f, 71.0f, 72.0f,
        73.0f, 74.0f, 75.0f, 76.0f, 77.0f, 78.0f, 79.0f, 80.0f
    };

    std::vector<double> reorderedData = {
        0.0f, 27.0f, 54.0f, 9.0f, 36.0f, 63.0f, 18.0f, 45.0f, 72.0f,
        1.0f, 28.0f, 55.0f, 10.0f, 37.0f, 64.0f, 19.0f, 46.0f, 73.0f,
        2.0f, 29.0f, 56.0f, 11.0f, 38.0f, 65.0f, 20.0f, 47.0f, 74.0f,
        3.0f, 30.0f, 57.0f, 12.0f, 39.0f, 66.0f, 21.0f, 48.0f, 75.0f,
        4.0f, 31.0f, 58.0f, 13.0f, 40.0f, 67.0f, 22.0f, 49.0f, 76.0f,
        5.0f, 32.0f, 59.0f, 14.0f, 41.0f, 68.0f, 23.0f, 50.0f, 77.0f,
        6.0f, 33.0f, 60.0f, 15.0f, 42.0f, 69.0f, 24.0f, 51.0f, 78.0f,
        7.0f, 34.0f, 61.0f, 16.0f, 43.0f, 70.0f, 25.0f, 52.0f, 79.0f,
        8.0f, 35.0f, 62.0f, 17.0f, 44.0f, 71.0f, 26.0f, 53.0f, 80.0f
    };

    mv::Tensor t("t", tShape, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)));
    t.populate(data);
    t.setOrder(mv::Order("HWC"));

    for (unsigned i = 0; i < data.size(); ++i)
        ASSERT_EQ(t(i), reorderedData[i]);

}

TEST(tensor, planar_to_column_major)
{

    mv::Shape tShape({3, 3, 3, 3});

    std::vector<double> data = {
        0.0f, 27.0f, 54.0f, 9.0f, 36.0f, 63.0f, 18.0f, 45.0f, 72.0f,
        1.0f, 28.0f, 55.0f, 10.0f, 37.0f, 64.0f, 19.0f, 46.0f, 73.0f,
        2.0f, 29.0f, 56.0f, 11.0f, 38.0f, 65.0f, 20.0f, 47.0f, 74.0f,
        3.0f, 30.0f, 57.0f, 12.0f, 39.0f, 66.0f, 21.0f, 48.0f, 75.0f,
        4.0f, 31.0f, 58.0f, 13.0f, 40.0f, 67.0f, 22.0f, 49.0f, 76.0f,
        5.0f, 32.0f, 59.0f, 14.0f, 41.0f, 68.0f, 23.0f, 50.0f, 77.0f,
        6.0f, 33.0f, 60.0f, 15.0f, 42.0f, 69.0f, 24.0f, 51.0f, 78.0f,
        7.0f, 34.0f, 61.0f, 16.0f, 43.0f, 70.0f, 25.0f, 52.0f, 79.0f,
        8.0f, 35.0f, 62.0f, 17.0f, 44.0f, 71.0f, 26.0f, 53.0f, 80.0f
    };

    std::vector<double> reorderedData = {
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
        17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f,
        25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f,
        33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f, 40.0f,
        41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f,
        49.0f, 50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f,
        57.0f, 58.0f, 59.0f, 60.0f, 61.0f, 62.0f, 63.0f, 64.0f,
        65.0f, 66.0f, 67.0f, 68.0f, 69.0f, 70.0f, 71.0f, 72.0f,
        73.0f, 74.0f, 75.0f, 76.0f, 77.0f, 78.0f, 79.0f, 80.0f
    };

    mv::Tensor t("t", tShape, mv::DTypeType::Float16, mv::Order("HWC"));
    t.populate(data);
    t.setOrder(mv::Order(mv::Order::getColMajorID(3)));

    for (unsigned i = 0; i < data.size(); ++i)
        ASSERT_EQ(t(i), reorderedData[i]);

}

TEST(tensor, row_major_to_planar)
{

    mv::Shape tShape({3, 3, 3, 3});
    std::vector<double> data = {
        0.0f, 27.0f, 54.0f, 9.0f, 36.0f, 63.0f, 18.0f, 45.0f, 72.0f,
        3.0f, 30.0f, 57.0f, 12.0f, 39.0f, 66.0f, 21.0f, 48.0f, 75.0f,
        6.0f, 33.0f, 60.0f, 15.0f, 42.0f, 69.0f, 24.0f, 51.0f, 78.0f,
        1.0f, 28.0f, 55.0f, 10.0f, 37.0f, 64.0f, 19.0f, 46.0f, 73.0f,
        4.0f, 31.0f, 58.0f, 13.0f, 40.0f, 67.0f, 22.0f, 49.0f, 76.0f,
        7.0f, 34.0f, 61.0f, 16.0f, 43.0f, 70.0f, 25.0f, 52.0f, 79.0f,
        2.0f, 29.0f, 56.0f, 11.0f, 38.0f, 65.0f, 20.0f, 47.0f, 74.0f,
        5.0f, 32.0f, 59.0f, 14.0f, 41.0f, 68.0f, 23.0f, 50.0f, 77.0f,
        8.0f, 35.0f, 62.0f, 17.0f, 44.0f, 71.0f, 26.0f, 53.0f, 80.0f
    };

    std::vector<double> reorderedData = {
        0.0f, 27.0f, 54.0f, 9.0f, 36.0f, 63.0f, 18.0f, 45.0f, 72.0f,
        1.0f, 28.0f, 55.0f, 10.0f, 37.0f, 64.0f, 19.0f, 46.0f, 73.0f,
        2.0f, 29.0f, 56.0f, 11.0f, 38.0f, 65.0f, 20.0f, 47.0f, 74.0f,
        3.0f, 30.0f, 57.0f, 12.0f, 39.0f, 66.0f, 21.0f, 48.0f, 75.0f,
        4.0f, 31.0f, 58.0f, 13.0f, 40.0f, 67.0f, 22.0f, 49.0f, 76.0f,
        5.0f, 32.0f, 59.0f, 14.0f, 41.0f, 68.0f, 23.0f, 50.0f, 77.0f,
        6.0f, 33.0f, 60.0f, 15.0f, 42.0f, 69.0f, 24.0f, 51.0f, 78.0f,
        7.0f, 34.0f, 61.0f, 16.0f, 43.0f, 70.0f, 25.0f, 52.0f, 79.0f,
        8.0f, 35.0f, 62.0f, 17.0f, 44.0f, 71.0f, 26.0f, 53.0f, 80.0f
    };

    mv::Tensor t("t", tShape, mv::DTypeType::Float16, mv::Order(mv::Order::getRowMajorID(3)));
    t.populate(data);
    t.setOrder(mv::Order("HWC"));

    for (unsigned i = 0; i < data.size(); ++i)
        ASSERT_EQ(t(i), reorderedData[i]);

}

TEST(tensor, planar_to_row_major)
{

    mv::Shape tShape({3, 3, 3, 3});

    std::vector<double> data = {
        0.0f, 27.0f, 54.0f, 9.0f, 36.0f, 63.0f, 18.0f, 45.0f, 72.0f,
        1.0f, 28.0f, 55.0f, 10.0f, 37.0f, 64.0f, 19.0f, 46.0f, 73.0f,
        2.0f, 29.0f, 56.0f, 11.0f, 38.0f, 65.0f, 20.0f, 47.0f, 74.0f,
        3.0f, 30.0f, 57.0f, 12.0f, 39.0f, 66.0f, 21.0f, 48.0f, 75.0f,
        4.0f, 31.0f, 58.0f, 13.0f, 40.0f, 67.0f, 22.0f, 49.0f, 76.0f,
        5.0f, 32.0f, 59.0f, 14.0f, 41.0f, 68.0f, 23.0f, 50.0f, 77.0f,
        6.0f, 33.0f, 60.0f, 15.0f, 42.0f, 69.0f, 24.0f, 51.0f, 78.0f,
        7.0f, 34.0f, 61.0f, 16.0f, 43.0f, 70.0f, 25.0f, 52.0f, 79.0f,
        8.0f, 35.0f, 62.0f, 17.0f, 44.0f, 71.0f, 26.0f, 53.0f, 80.0f
    };

    std::vector<double> reorderedData = {
        0.0f, 27.0f, 54.0f, 9.0f, 36.0f, 63.0f, 18.0f, 45.0f, 72.0f,
        3.0f, 30.0f, 57.0f, 12.0f, 39.0f, 66.0f, 21.0f, 48.0f, 75.0f,
        6.0f, 33.0f, 60.0f, 15.0f, 42.0f, 69.0f, 24.0f, 51.0f, 78.0f,
        1.0f, 28.0f, 55.0f, 10.0f, 37.0f, 64.0f, 19.0f, 46.0f, 73.0f,
        4.0f, 31.0f, 58.0f, 13.0f, 40.0f, 67.0f, 22.0f, 49.0f, 76.0f,
        7.0f, 34.0f, 61.0f, 16.0f, 43.0f, 70.0f, 25.0f, 52.0f, 79.0f,
        2.0f, 29.0f, 56.0f, 11.0f, 38.0f, 65.0f, 20.0f, 47.0f, 74.0f,
        5.0f, 32.0f, 59.0f, 14.0f, 41.0f, 68.0f, 23.0f, 50.0f, 77.0f,
        8.0f, 35.0f, 62.0f, 17.0f, 44.0f, 71.0f, 26.0f, 53.0f, 80.0f
    };

    mv::Tensor t("t", tShape, mv::DTypeType::Float16, mv::Order("HWC"));
    t.populate(data);
    t.setOrder(mv::Order(mv::Order::getRowMajorID(3)));

    for (unsigned i = 0; i < data.size(); ++i)
        ASSERT_EQ(t(i), reorderedData[i]);

}

/*TEST(tensor, ind_to_sub_1d)
{

    mv::Shape tShape({32});
    std::vector<double> data = mv::utils::generateSequence<double>(tShape.totalSize());
    mv::Tensor tColumnMajor("t", tShape, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)));
    mv::Tensor tRowMajor("t", tShape, mv::DTypeType::Float16, mv::Order(mv::Order::getRowMajorID(3)));
    mv::Tensor tPlanar("t", tShape, mv::DTypeType::Float16, mv::Order("HWC"));
    tColumnMajor.populate(data);
    tRowMajor.populate(data);
    tPlanar.populate(data);

    for (unsigned i = 0; i < data.size(); ++i)
    {
        auto subColumnMajor = tColumnMajor.indToSub(i);
        auto subRowMajor = tRowMajor.indToSub(i);
        auto subPlanar = tPlanar.indToSub(i);
        ASSERT_EQ(tColumnMajor(subColumnMajor), data[i]);
        ASSERT_EQ(tRowMajor(subRowMajor), data[i]);
        ASSERT_EQ(tPlanar(subPlanar), data[i]);
    }

}

TEST(tensor, ind_to_sub_2d)
{

    mv::Shape tShape({8, 4});
    std::vector<double> data = mv::utils::generateSequence<double>(tShape.totalSize());
    mv::Tensor tColumnMajor("t", tShape, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)));
    mv::Tensor tRowMajor("t", tShape, mv::DTypeType::Float16, mv::Order(Order::getRowMajorID(3)));
    mv::Tensor tPlanar("t", tShape, mv::DTypeType::Float16, mv::Order("HWC"));
    tColumnMajor.populate(data);
    tRowMajor.populate(data);
    tPlanar.populate(data);

    for (unsigned i = 0; i < data.size(); ++i)
    {
        auto subColumnMajor = tColumnMajor.indToSub(i);
        auto subRowMajor = tRowMajor.indToSub(i);
        auto subPlanar = tPlanar.indToSub(i);

        ASSERT_EQ(tColumnMajor(subColumnMajor), data[i]);
        ASSERT_EQ(tRowMajor(subRowMajor), data[i]);
        ASSERT_EQ(tPlanar(subPlanar), data[i]);
    }

}*/

TEST(tensor, augment)
{

    mv::Shape tShape({8, 1, 4});
    mv::Shape tShapeAugmented({8, 4, 4});

    std::vector<double> data = mv::utils::generateSequence<double>(tShape.totalSize());
    mv::Tensor t("t", tShape, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)));
    t.populate(data);
    t.broadcast(tShapeAugmented);

    for (unsigned k = 0; k < 4; ++k)
        for (unsigned j = 0; j < 4; ++j)
            for (unsigned i = 0; i < 8; ++i)
                ASSERT_EQ(t({i, j, k}), t({i, 0, k}));

}

TEST(tensor, add)
{

    double start = -100.0f;
    double diff = 0.5f;

    mv::Shape tShape({32, 32, 128});
    std::vector<double> data1 = mv::utils::generateSequence<double>(tShape.totalSize(), start, diff);
    std::vector<double> data2 = mv::utils::generateSequence<double>(tShape.totalSize(), -start, -diff);

    mv::Tensor t1("t1", tShape, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)), data1);
    mv::Tensor t2("t2", tShape, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)), data2);

    auto t3 = mv::math::add(t1, t2);

    std::cout << t3.getShape().totalSize() << std::endl;

    /*for (unsigned i = 0; i < tShape[0]; ++i)
        for (unsigned j = 0; j < tShape[1]; ++j)
            for (unsigned k = 0; k < tShape[2]; ++k)
                ASSERT_FLOAT_EQ(t3({i, j, k}), 0.0f);*/

}

TEST(tensor, add_broadcast_vec)
{

    double start = -100.0f;
    double diff = 0.5f;

    mv::Shape t1Shape({32, 32, 128});
    mv::Shape t2Shape({128});
    std::vector<double> data1 = mv::utils::generateSequence<double>(t1Shape.totalSize(), start, diff);
    std::vector<double> data2 = mv::utils::generateSequence<double>(t2Shape.totalSize());

    mv::Tensor t1("t1", t1Shape, mv::DTypeType::Float16, mv::Order(mv::Order::getRowMajorID(3)), data1);
    mv::Tensor t2("t2", t2Shape, mv::DTypeType::Float16, mv::Order(mv::Order::getRowMajorID(3)), data2);

    auto t3 = mv::math::add(t1, t2);

    std::cout << t3.getShape().totalSize() << std::endl;

    /*for (unsigned i = 0; i < t1Shape[0]; ++i)
        for (unsigned j = 0; j < t1Shape[1]; ++j)
            for (unsigned k = 0; k < t1Shape[2]; ++k)
                ASSERT_FLOAT_EQ(t3({i, j, k}), t1({i, j, k}) + t2(k));*/


}

TEST(tensor, add_broadcast_mat)
{

    double start = -100.0f;
    double diff = 0.5f;

    mv::Shape t1Shape({8, 4, 3});
    mv::Shape t2Shape({4, 3});
    std::vector<double> data1 = mv::utils::generateSequence<double>(t1Shape.totalSize(), start, diff);
    std::vector<double> data2 = mv::utils::generateSequence<double>(t2Shape.totalSize());

    mv::Tensor t1("t1", t1Shape, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)), data1);
    mv::Tensor t2("t2", t2Shape, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)), data2);

    auto t3 = mv::math::add(t1, t2);

    for (unsigned i = 0; i < t1Shape[0]; ++i)
        for (unsigned j = 0; j < t2Shape[0]; ++j)
            for (unsigned k = 0; k < t1Shape[2]; ++k)
                ASSERT_FLOAT_EQ(t3({i, j, k}), t1({i, j, k}) + t2({j, k}));

}

/*TEST(tensor, add_broadcast_eq)
{

    double start = -100.0f;
    double diff = 0.5f;

    mv::Shape t1Shape({32, 3, 3, 16});
    mv::Shape t2Shape({32, 3, 3, 1});
    std::vector<double> data1 = mv::utils::generateSequence<double>(t1Shape.totalSize(), start, diff);
    std::vector<double> data2 = mv::utils::generateSequence<double>(t2Shape.totalSize());

    mv::Tensor t1("t1", t1Shape, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)), data1);
    mv::Tensor t2("t2", t2Shape, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)), data2);

    auto t3 = mv::math::add(t1, t2);

    for (unsigned i = 0; i < t1Shape[0]; ++i)
        for (unsigned j = 0; j < t2Shape[1]; ++j)
            for (unsigned k = 0; k < t1Shape[2]; ++k)
                for (unsigned l = 0; l < t2Shape[3]; ++l)
                    ASSERT_FLOAT_EQ(t3({i, j, k, l}), t1({i, j, k, l}) + t2({i, j, k, 0}));

}*/

TEST(tensor, subtract)
{

    double start = -100.0f;
    double diff = 0.5f;

    mv::Shape tShape({32, 32, 3});
    std::vector<double> data = mv::utils::generateSequence<double>(tShape.totalSize(), start, diff);

    mv::Tensor t1("t1", tShape, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)), data);
    mv::Tensor t2("t2", tShape, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)), data);

    t1.subtract(t2);

    for (unsigned i = 0; i < tShape[0]; ++i)
        for (unsigned j = 0; j < tShape[1]; ++j)
            for (unsigned k = 0; k < tShape[2]; ++k)
                ASSERT_FLOAT_EQ(t1({i, j, k}), 0.0f);

}

TEST(tensor, multiply)
{

    double start = 1.0f;
    double diff = 0.5f;

    mv::Shape tShape({32, 32, 3});
    std::vector<double> data1 = mv::utils::generateSequence<double>(tShape.totalSize(), start, diff);
    std::vector<double> data2(data1.size());

    for (unsigned i = 0; i < data2.size(); ++i)
        data2[i] = 1.0f / data1[i];

    mv::Tensor t1("t1", tShape, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)), data1);
    mv::Tensor t2("t2", tShape, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)), data2);

    t1.multiply(t2);

    for (unsigned i = 0; i < tShape[0]; ++i)
        for (unsigned j = 0; j < tShape[1]; ++j)
            for (unsigned k = 0; k < tShape[2]; ++k)
                ASSERT_FLOAT_EQ(t1({i, j, k}), 1.0f);

}

TEST(tensor, divide)
{

    double start = 2.0f;
    double diff = 0.5f;

    mv::Shape tShape({32, 32, 3});
    std::vector<double> data = mv::utils::generateSequence<double>(tShape.totalSize(), start, diff);

    mv::Tensor t1("t1", tShape, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)), data);
    mv::Tensor t2("t2", tShape, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)), data);

    t1.divide(t2);

    for (unsigned i = 0; i < tShape[0]; ++i)
        for (unsigned j = 0; j < tShape[1]; ++j)
            for (unsigned k = 0; k < tShape[2]; ++k)
                ASSERT_FLOAT_EQ(t1({i, j, k}), 1.0f);

}

TEST(tensor, get_data)
{

    double start = 2.0f;
    double diff = 0.5f;

    mv::Shape tShape({32, 32, 128});
    std::vector<double> data = mv::utils::generateSequence<double>(tShape.totalSize(), start, diff);

    mv::Tensor t1("t1", tShape, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)), data);

    std::cout << t1.getData().size() << std::endl;

}
