#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include <math.h>

static void propagateParametersFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& compilationDescriptor, mv::Element&);

namespace mv
{
    namespace pass
    {

        MV_REGISTER_PASS(FakeQuantize)
        .setFunc(propagateParametersFcn)
        .setDescription(
            "Propagate quantization parametets from FakeQuantize layer"
        );
    }
}

enum class Precision {
    Default,
    U8,
    I8,
    FP16,
    I32,
    FP32,
};

mv::DType getDType(Precision p) {
    static const std::map<Precision, mv::DType> types {
            {Precision::Default, mv::DType("Default")},
            {Precision::U8, mv::DType("UInt8")},
            {Precision::I8, mv::DType("Int8")},
            {Precision::FP16, mv::DType("Float16")},
            {Precision::I32, mv::DType("Int32")},
            {Precision::FP32, mv::DType("Float32")},
    };

    return types.at(p);
}

int64_t calculateZeroPoint(float low, float high, int levels, mv::DType dtype) {
    int64_t zeroPoint = 0;

    // Typical condition for symmetric case is low < 0, high > 0
    if (dtype == getDType(Precision::U8)) {
        //  MCM team provide this formula, need check
        if ((low <= 0.f) && (high >= 0.f)) {
            auto x = (high / (fabs(low) + high)) * (levels - 1);
            zeroPoint = ceil(levels - 1 - x);  // TODO Why not round?
        } else if (low >= 0.f) {
            zeroPoint = 0;  // TODO Why not assert?
        } else if (high <= 0.f) {
            zeroPoint = (levels - 1);  // TODO Why not assert?
        }
    }
    if (dtype == getDType(Precision::I8)) {
        if ((low <= 0.f) && (high >= 0.f)) {
            float x = -(levels - 1) * ((high + low) * 0.5f) / (high - low);
            zeroPoint = ceil(x);  // TODO Why not round?
        } else if (low > 0.f) {
            zeroPoint = 127 - (levels - 1);  // TODO Why not assert?
        } else if (high < 0.f) {
            zeroPoint = 127;  // TODO Why not assert?
        }
    }

    return zeroPoint;
}

double calculateScales(float low, float high, int levels) {
    return static_cast<double>((high - low) / (levels - 1));
}

mv::QuantizationParams extractQuantParams(mv::Data::OpListIterator fqOp, bool merge_in_one) {
    assert(fqOp->getOpType() == "FakeQuantize");

    auto inputs = fqOp->getInputTensor();
    auto attrs = fqOp->getAttrs();

    auto levels = fqOp->get<unsigned>("levels");

    auto input_min = fqOp->getInputTensor(1)->getDoubleData();
    auto input_max = fqOp->getInputTensor(2)->getDoubleData();
    auto output_min = fqOp->getInputTensor(3)->getDoubleData();
    auto output_max = fqOp->getInputTensor(4)->getDoubleData();

    assert(input_min.size() == input_max.size() == output_min.size() == output_max.size() && input_min.size() != 0);

    std::vector<int64_t> zero_points;
    std::vector<double> scales;
    std::vector<double> mins;
    std::vector<double> maxs;
    if (merge_in_one) {
        float output_min_value = *std::min_element(output_min.begin(), output_min.end());
        float output_max_value = *std::max_element(output_max.begin(), output_max.end());

        zero_points.push_back(calculateZeroPoint(output_min_value, output_max_value, levels, getDType(Precision::U8)));
        scales.push_back(calculateScales(output_min_value, output_max_value, levels));
        mins.push_back(output_min_value);
        maxs.push_back(output_max_value);
    } else {
        for (size_t i = 0; i < output_min.size(); ++i) {
            float min_value = output_min[i];
            float max_value = output_max[i];

            zero_points.push_back(calculateZeroPoint(min_value, max_value, levels, getDType(Precision::U8)));
            scales.push_back(calculateScales(min_value, max_value, levels));
            mins.push_back(min_value);
            maxs.push_back(min_value);
        }
    }

    //TODO: In original code we have -inf, inf insted of mins and max. What variant should we use
    return mv::QuantizationParams{zero_points, scales, mins, maxs};
}

template<typename T>
T clamp(const T& value, const T& min, const T& max) {
    assert(min < max);
    return std::max(min, std::min(max, value));
}

//NOTE: Bias is not a const op.
mv::Data::OpListIterator quantizeConstOp(mv::OpModel& model, mv::Data::OpListIterator operation, mv::QuantizationParams quant_params, mv::DType precision) {
    // TODO: fp16 tensors are constInt. need to handle
    assert(operation->getOpType() == "Constant");
    auto originalTensor = operation->getOutputTensor(0);

    auto shape = operation->getOutputTensor(0)->getShape();
    // TODO: check correctness
    auto OC = shape[mv::KERNEL_OUTPUT_CHANNELS];
    auto IC = shape[mv::KERNEL_INPUT_CHANNELS];
    auto KH = shape[mv::KERNEL_HEIGHT];
    auto KW = shape[mv::KERNEL_WIDTH];

    std::vector<int64_t> quantized_weights(OC * IC * KH *KW);

    auto scales = quant_params.getScale();
    auto zero_points = quant_params.getZeroPoint();
    auto broadcast = scales.size() == 1;
    assert(scales.size() == zero_points.size());

    auto src_data = operation->getOutputTensor(0)->getDoubleData();
    std::vector<int64_t> dst_data(src_data.size());
    for (size_t oc = 0; oc < OC; ++oc) {
        for (size_t ic = 0; ic < IC; ++ic) {
            for (size_t kh = 0; kh < KH; ++kh) {
                for (size_t kw = 0; kw < KW; ++kw) {
                    const size_t idx = oc * IC * KW * KH + ic * KH * KW + kh * KW + kw;
                    auto scale = broadcast ? scales[0] : scales[oc];
                    auto zero_point = broadcast ? zero_points[0] : zero_points[oc];

                    auto new_value = std::round((src_data[idx] + scale * zero_point) / scale);
                    if (precision == getDType(Precision::U8)) {
                        uint8_t value = clamp<double>(new_value, 0, 255);
                        dst_data[idx] = value;
                    } else if (precision == getDType(Precision::U8)) {
                        int8_t value = clamp<double>(new_value, -128, 127);
                        dst_data[idx] = value;
                    } else {
                        throw std::runtime_error("Unexpected dtype");
                    }
                }
            }
        }
    }

    for (size_t i = 0; i < 15; ++i) {
        std::cout << src_data[i] << " " << dst_data[i] << std::endl;
    }

    // TODO: do we need to pass name?
    auto quantized_const_tensor = model.constantInt(dst_data, shape, precision, originalTensor->getOrder(), quant_params);
    if(operation->hasAttr("opId")) {
        unsigned currentOpId = operation->get<unsigned>("opId");
        quantized_const_tensor->set<unsigned>("opId", currentOpId);
        model.getSourceOp(quantized_const_tensor)->set<unsigned>("opId", currentOpId);
    }

    // TODO: what if we don't have parent op?
    mv::linkNewOperationsReplacement(mv::Data::OpListIterator(), quantized_const_tensor, model, operation);
}

// TODO: pass quantParams instead of fqOP;
void quantize_weights(mv::OpModel& model, mv::Data::OpListIterator weights, mv::Data::OpListIterator fqOp) {
    std::cout << "Quantize weights\n";
    auto data_tensor = weights->getOutputTensor(0);

    if (data_tensor->getDType() == getDType(Precision::I8) ||
        data_tensor->getDType() == getDType(Precision::U8)) {
        // Weights would already be quantized
        // TODO: Recalculate quant params
        assert(false && "Not implemented");
        return;
    }

    auto quant_params = extractQuantParams(fqOp, false);


    // TODO: quantize const layer
    quantizeConstOp(model, weights, quant_params, getDType(Precision::U8));
}

mv::Data::OpListIterator quantizeBias(mv::OpModel model, mv::Data::OpListIterator biasOp) {
    std::cout << model.getSourceOp(biasOp->getInputTensor(1))->get<mv::DType>("dType").toString() << std::endl;
    std::cout << model.getSourceOp(biasOp->getInputTensor(1))->getOutputTensor(0)->get<mv::DType>("dType").toString() << std::endl;
    std::cout << biasOp->getInputTensor(1)->get<mv::DType>("dType").toString() << std::endl;

    auto tensor_data = biasOp->getInputTensor(1)->getData();
    for (size_t i = 0; i < 15; ++i) {
        std::cout << static_cast<double>(tensor_data[i]) << " " << static_cast<int64_t>(tensor_data[i]) << std::endl;
    }

    auto activation_params = model.getSourceOp(biasOp->getInputTensor(0))->get<mv::QuantizationParams>("quantParams");
    auto weights_params = model.getSourceOp(biasOp->getInputTensor(1))->get<mv::QuantizationParams>("quantParams");

    // assert(activation_params.isPerTensor() == weights_params.isPerTensor());
    bool is_broadcasted = activation_params.isPerTensor() && weights_params.isPerTensor();

    if (!activation_params.isPerTensor() && activation_params.getScale().size() != tensor_data.size()) {
        throw std::runtime_error("Bias and Activation quant params size mismatch");
    }

    if (!weights_params.isPerTensor() && weights_params.getScale().size() != tensor_data.size()) {
        throw std::runtime_error("Bias and Weights quant params size mismatch");
    }

    std::vector<int64_t> newBiasData(tensor_data.size(), 0);
    std::vector<double> biasScales;
    std::vector<int64_t> zeroPoints;
    auto bias_dtype = biasOp->getInputTensor(1)->getDType();
    if (bias_dtype == getDType(Precision::FP32)) {
        auto bias_data = biasOp->getInputTensor(1)->getDoubleData();

        if (is_broadcasted) {
            auto activation_scale = activation_params.getScale(0);
            auto weights_scale = weights_params.getScale(0);

            auto bias_scale = activation_scale * weights_scale;
            biasScales.push_back(bias_scale);
            zeroPoints.push_back(0);
        } else {
            for (size_t i = 0; i < bias_data.size(); ++i) {
                auto activation_scale = activation_params.getScale(i);
                auto weights_scale = weights_params.getScale(i);

                auto bias_scale = activation_scale * weights_scale;
                biasScales.push_back(bias_scale);
                zeroPoints.push_back(0);
                newBiasData[i] = std::round(bias_data[i] / bias_scale);
            }
        }

        auto original_tensor = biasOp->getInputTensor(1);
        mv::QuantizationParams quant_params{{zeroPoints}, {biasScales}, {}, {}};
        // TODO: pass name
        auto quantized_data = model.constantInt(newBiasData, original_tensor->getShape(), getDType(Precision::I32), original_tensor->getOrder(), quant_params);
        auto quantize_bias_tensor = model.bias(biasOp->getInputTensor(0), quantized_data, quantized_data->getDType(), quant_params);

        return mv::linkNewOperationsReplacement(model.getSourceOp(biasOp->getInputTensor(0)), quantize_bias_tensor, model, biasOp);
    } else if (bias_dtype == getDType(Precision::I32)) {
        // Do nothing
    } else {
        std::runtime_error("Unsupported bias data type");
    }

    return biasOp;
}

void propagate(mv::OpModel& model, mv::Data::OpListIterator fqOp) {
    auto params = extractQuantParams(fqOp, false);
    auto currentOp = model.getSourceOp(fqOp->getInputTensor(0));
    std::cout << currentOp->getOpType() << std::endl;
    std::cout << currentOp->get<mv::DType>("dType").toString() << std::endl;
    // TODO: check if bias alse hae const type. check logic for int buffers
    if (currentOp->getOpType() == "ConstantInt" || currentOp->getOpType() == "Constant") {
        quantize_weights(model, currentOp, fqOp);
        return;
    }

    auto stop_propagation = [](const std::string& op_type) -> bool {
        std::vector<std::string> stop_layers{"Input", "Constant", "ConstantInt", "FakeQuantize"};
        return std::find(stop_layers.begin(), stop_layers.end(), op_type) != stop_layers.end();
    };

    while (!stop_propagation(currentOp->getOpType())) {
        if (currentOp->getOpType() == "Bias") {
            std::cout << "Parse bias\n";
            currentOp = quantizeBias(model, currentOp);
        } else {
//            std::cout << "Layer " << currentOp->getName() << " " << currentOp->getOpType() << " new quant params\n";
//            currentOp->set<mv::QuantizationParams>("quantParams", params);
        }

        currentOp = model.getSourceOp(currentOp->getInputTensor(0));
    }
}

void propagateParameters(const mv::pass::PassEntry& pass, mv::ComputationModel& model) {
    std::cout << "Propagate FQ pass\n";
    mv::OpModel om(model);

    auto fqOps = om.getOps("FakeQuantize");
    std::cout << "Found " << fqOps.size() << std::endl;
    for (auto& fq : fqOps) {
        propagate(om, fq);
    }
}


void removeFQ(const mv::pass::PassEntry& pass, mv::ComputationModel& model) {
    std::cout << "Remove FakeQuantize pass\n";
    mv::OpModel om(model);

    auto fqOps = om.getOps("FakeQuantize");
    std::cout << "Found " << fqOps.size() << std::endl;
    for (auto& fq : fqOps) {
        auto parent = om.getSourceOp(fq->getInputTensor(0));
        // parent->set<mv::QuantizationParams>("quantParams", extractQuantParams(fq));

        //TODO: this function doesn't work correct for weights because in mcm Weight is const tensor and
        // this functions removes all const inputs of the fq op.
        // TODO: Remove fix that was made to prevent this behaviourd
        linkNewOperationsRemove(parent, parent->getOutputTensor(0), om, fq);
    }
}


void propagateParametersFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& compilationDescriptor, mv::Element&) {
    propagateParameters(pass, model);
    removeFQ(pass, model);
}
