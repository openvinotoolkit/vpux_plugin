#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{

    mv::CompilationUnit unit("SliceModel");
    mv::OpModel& om = unit.model();

    auto inputShape = mv::Shape({24,24,15,1});
    auto inputOrder = mv::Order("NHWC");
    auto axis = 1;
    auto num_splits = 5;
    // Calculate output begin & size Shapes 
    std::vector<mv::Shape> beginShapes;
    std::vector<mv::Shape> outputShapes;
    for (auto i=0; i<num_splits; ++i)
    {
        auto split_dim = (inputShape.ndims()-1 - axis);
        auto size = inputShape[split_dim] / num_splits;
        beginShapes.push_back(mv::Shape({0,0,size*i,0}));
        outputShapes.push_back(mv::Shape({inputShape[0],inputShape[1],size,1}));
    }

    // Input
    auto input0 = om.input(inputShape, mv::DType("UInt8"), inputOrder, {{0},{1.0},{},{}}, "input0");

    // Slices
    std::vector<mv::Data::TensorIterator> slices;
    std::vector<mv::Data::TensorIterator> maxpools;
    for (auto i=0; i<num_splits; ++i)
    {
        slices.push_back(om.slice(input0, beginShapes.at(i), outputShapes.at(i), {{0},{1.0},{},{}}, "slice" + std::to_string(i)));
        maxpools.push_back(om.maxPool(slices.back(), {1,1}, {1,1}, {0,0,0,0}, true, mv::DType("UInt8"), {{0},{1.0},{},{}}, "identity_maxpool" + std::to_string(i)));
    }

    // Concat
    std::string concat_axis = "C";
    auto concat0 = om.concat(maxpools, concat_axis, mv::DType("UInt8"), {{0},{1.0},{},{}}, "concat0");

    // Output
    auto output0 = om.output(concat0);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}

