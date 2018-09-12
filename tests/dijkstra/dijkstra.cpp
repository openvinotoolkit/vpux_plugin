#include "gtest/gtest.h"
#include "mcm/graph/dijkstra.hpp"
#include "mcm/pass/nce1/mode_selection.hpp"

//This is a simple test case, no splits are involved, 1 step to solve it.
TEST (dijkstra, resnet_first_conv)
{
    ModeSelectionNode source;

    source.parameters.input_height = 224;
    source.parameters.input_width = 224;
    source.parameters.output_height = 112;
    source.parameters.output_width = 112;
    source.parameters.input_channels = 3;
    source.parameters.output_channels = 64;
    source.parameters.kernel_x = 7;
    source.parameters.kernel_y = 7;
    source.parameters.stride_x = 2;
    source.parameters.stride_y = 2;
    source.remaining_output_channels = source.parameters.output_channels;

    ModeSelectionNode target;
    target.remaining_output_channels = 0;

    mv::DijkstraReturnValue<ModeSelectionNode, ModeSelectionDistance> shortestPath = mv::dijkstraRT<ModeSelectionNode, ModeSelectionDistance>(source, target, generateNeighboursComingFromValidModes, computeModeCost);

    //ASSERTION Values given by python compiler
    //First: Check that paths have the same size
    ASSERT_EQ(shortestPath.distances.size(),1);

    //Second: If they have the same size, check that all modes corrispond (no permutation allowed)
    std::vector<unsigned> expected_modes = {2};
    for(unsigned i = 0; i < shortestPath.distances.size(); ++i)
        ASSERT_EQ(shortestPath.distances[i].mode, expected_modes[i]);

    //Third: If all the modes correspond, splits should be equal as well
    std::vector<int> expected_splits = {1};
    for(unsigned i = 0; i < shortestPath.distances.size(); ++i)
        ASSERT_EQ(shortestPath.distances[i].num_splits, expected_splits[i]);

    //Fourth: If all the modes correspond, check that total solution cost correspnds
    unsigned total_cost = shortestPath.distances[shortestPath.distances.size()-1].cost;
    ASSERT_EQ(total_cost, 2458624);

    //Finished
    std::cout << "Finished!" << std::endl;
}

//This is a simple test case, no splits are involved, more steps to solve it.
TEST (dijkstra, resnet_another_conv)
{
    ModeSelectionNode source;

    source.parameters.input_height = 56;
    source.parameters.input_width = 56;
    source.parameters.output_height = 56;
    source.parameters.output_width = 56;
    source.parameters.input_channels = 64;
    source.parameters.output_channels = 224;
    source.parameters.kernel_x = 1;
    source.parameters.kernel_y = 1;
    source.parameters.stride_x = 1;
    source.parameters.stride_y = 1;
    source.remaining_output_channels = source.parameters.output_channels;

    ModeSelectionNode target;
    target.remaining_output_channels = 0;

    mv::DijkstraReturnValue<ModeSelectionNode, ModeSelectionDistance> shortestPath = mv::dijkstraRT<ModeSelectionNode, ModeSelectionDistance>(source, target, generateNeighboursComingFromValidModes, computeModeCost);

    //ASSERTION Values given by python compiler
    //First: Check that paths have the same size
    ASSERT_EQ(shortestPath.distances.size(),3);

    //Second: If they have the same size, check that all modes corrispond (no permutation allowed)
    std::vector<unsigned> expected_modes = {3, 2, 1};
    for(unsigned i = 0; i < shortestPath.distances.size(); ++i)
        ASSERT_EQ(shortestPath.distances[i].mode, expected_modes[i]);

    //Third: If all the modes correspond, splits should be equal as well
    std::vector<int> expected_splits = {1, 1, 1};
    for(unsigned i = 0; i < shortestPath.distances.size(); ++i)
        ASSERT_EQ(shortestPath.distances[i].num_splits, expected_splits[i]);

    //Fourth: If all the modes correspond, check that total solution cost correspnds
    unsigned total_cost = shortestPath.distances[shortestPath.distances.size()-1].cost;
    ASSERT_EQ(total_cost, 189616);

    //Finished
    std::cout << "Finished!" << std::endl;
}

//This is a simple test case, no splits are involved, more steps to solve it.
TEST (dijkstra, resnet_another_conv)
{
    ModeSelectionNode source;

    source.parameters.input_height = 56;
    source.parameters.input_width = 56;
    source.parameters.output_height = 56;
    source.parameters.output_width = 56;
    source.parameters.input_channels = 64;
    source.parameters.output_channels = 224;
    source.parameters.kernel_x = 1;
    source.parameters.kernel_y = 1;
    source.parameters.stride_x = 1;
    source.parameters.stride_y = 1;
    source.remaining_output_channels = source.parameters.output_channels;

    ModeSelectionNode target;
    target.remaining_output_channels = 0;

    mv::DijkstraReturnValue<ModeSelectionNode, ModeSelectionDistance> shortestPath = mv::dijkstraRT<ModeSelectionNode, ModeSelectionDistance>(source, target, generateNeighboursComingFromValidModes, computeModeCost);

    //ASSERTION Values given by python compiler
    //First: Check that paths have the same size
    ASSERT_EQ(shortestPath.distances.size(),3);

    //Second: If they have the same size, check that all modes corrispond (no permutation allowed)
    std::vector<unsigned> expected_modes = {3, 2, 1};
    for(unsigned i = 0; i < shortestPath.distances.size(); ++i)
        ASSERT_EQ(shortestPath.distances[i].mode, expected_modes[i]);

    //Third: If all the modes correspond, splits should be equal as well
    std::vector<int> expected_splits = {1, 1, 1};
    for(unsigned i = 0; i < shortestPath.distances.size(); ++i)
        ASSERT_EQ(shortestPath.distances[i].num_splits, expected_splits[i]);

    //Fourth: If all the modes correspond, check that total solution cost correspnds
    unsigned total_cost = shortestPath.distances[shortestPath.distances.size()-1].cost;
    ASSERT_EQ(total_cost, 189616);

    //Finished
    std::cout << "Finished!" << std::endl;
}


