
// The file was generated by RecordedOpModel

#include <limits>
#include <include/mcm/op_model.hpp>
#include "include/mcm/compiler/compilation_unit.hpp"

void build_pySwigCU(mv::OpModel& model)
{
    using namespace mv;

    static const auto inf = std::numeric_limits<double>::infinity();

    const std::array<unsigned short, 2UL> ksize = {15, 11};
    const std::array<unsigned short, 2UL> stride = {15, 11};
    const auto input_1_0 = model.input({30, 23, 16, 16}, mv::DType("UInt8"), mv::Order("NHWC"), {{128},{0.007843137718737},{-1.000000000000000},{1.000000000000000},{0},{1}}, "input#1");
    const auto icnet_features_conv5_3_pool2_1_AvgPool_AvgPool_2_0 = model.averagePool(input_1_0, ksize, stride, {0, 0, 0, 0}, false, mv::DType("UInt8"), {{128},{0.007843137718737},{-1.003921627998352},{0.996078431606293},{0},{1}}, "icnet_features/conv5_3_pool2_1/AvgPool/AvgPool#2");
    const auto output = model.output(icnet_features_conv5_3_pool2_1_AvgPool_AvgPool_2_0, mv::DType("Default"), {{},{},{},{}}, true, "");
}

int main()
{
    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();
    build_pySwigCU(om);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}

