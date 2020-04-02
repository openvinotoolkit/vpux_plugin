// The file was generated by RecordedOpModel

#include <limits>
#include <include/mcm/op_model.hpp>
#include "include/mcm/compiler/compilation_unit.hpp"
#include "eltwise_sparsity.data.inc"

void build_pySwigCU(mv::OpModel& model)
{
    using namespace mv;

    static const auto inf = std::numeric_limits<double>::infinity();

    const auto input_8_0 = model.input({32, 32, 16, 1}, mv::DType("UInt8"), mv::Order("NHWC"), {{128},{0.007843137718737},{-1.000000000000000},{1.000000000000000},{0},{1}}, "input#8");
    const auto Conv1_0_weights_1_0 = model.constantInt(Conv1_0_weights_1_0_data, {1, 1, 16, 32}, mv::DType("UInt8"), mv::Order("NCHW"), {{122},{0.002283898880705},{-0.279308497905731},{0.303085714578629},{0},{1}}, "Conv1#0_weights#1");
    const auto Conv1_9_0 = model.conv(input_8_0, Conv1_0_weights_1_0, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.003921568859369},{0.000000000000000},{1.000000000000000},{0},{1}}, "Conv1#9");
    const auto Conv1_0_bias_2weights_0 = model.constantInt(Conv1_0_bias_2weights_0_data, {32}, mv::DType("UInt8"), mv::Order("W"), {{0},{0.000017912932890},{-inf},{inf},{0},{1}}, "Conv1#0_bias#2weights");
    const auto Conv1_0_bias_2_0 = model.bias(Conv1_9_0, Conv1_0_bias_2weights_0, mv::DType("UInt8"), {{0},{0.000017912932890},{-inf},{inf},{0},{1}}, "Conv1#0_bias#2");
    const auto Conv2_3_weights_4_0 = model.constantInt(Conv2_3_weights_4_0_data, {1, 1, 16, 32}, mv::DType("UInt8"), mv::Order("NCHW"), {{132},{0.002380653517321},{-0.315335750579834},{0.291730880737305},{0},{1}}, "Conv2#3_weights#4");
    const auto Conv2_10_0 = model.conv(input_8_0, Conv2_3_weights_4_0, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.003921568859369},{0.000000000000000},{1.000000000000000},{0},{1}}, "Conv2#10");
    const auto Conv2_3_bias_5weights_0 = model.constantInt(Conv2_3_bias_5weights_0_data, {32}, mv::DType("UInt8"), mv::Order("W"), {{0},{0.000018671791622},{-inf},{inf},{0},{1}}, "Conv2#3_bias#5weights");
    const auto Conv2_3_bias_5_0 = model.bias(Conv2_10_0, Conv2_3_bias_5weights_0, mv::DType("UInt8"), {{0},{0.000018671791622},{-inf},{inf},{0},{1}}, "Conv2#3_bias#5");
    const auto add_Add_11_0 = model.eltwise({Conv1_0_bias_2_0, Conv2_3_bias_5_0}, "Add", mv::DType("UInt8"), {{128},{0.007843137718737},{-1.003921627998352},{0.996078431606293},{0},{1}}, "add/Add#11");
    const auto output_max_pool_12_0 = model.maxPool(add_Add_11_0, {1, 1}, {1, 1}, {0, 0, 0, 0}, false, mv::DType("UInt8"), {{128},{0.007843137718737},{-1.003921627998352},{0.996078431606293},{0},{1}}, "output/max_pool#12");
    const auto output = model.output(output_max_pool_12_0, mv::DType("Default"), {{},{},{},{}}, "");
}

int main()
{
    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();
    build_pySwigCU(om);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb_SC-PrefetchAdaptive.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}
