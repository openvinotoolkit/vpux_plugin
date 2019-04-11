/**
 * @brief Example presenting composition and compilation of the ResNet50 CNN
 * 
 * In this example ResNet50 model is composed using MCMCompiler's Composition API. Then
 * the compilation for the target device MA2480 is initialized and compilation passes scheduled by 
 * the target descriptor are executed. Included GenerateDot pass will generate *.dot files
 * that visualize the computation model at the end of each accomplished compilation phase.
 * Included GenerateBlob pass will serialize the model to a binary deployable to the target device.
 * 
 * Notes:
 * - This implementation of ResNet50 uses fused batch norm representation - batch norm is expressed
 * as a sequence of scale and bias
 * - This implementation of ResNet50 is aligned with Caffe - batch norm is followed by scale and bias
 * - Weights and other model parameters are initialized as sequences of numbers starting with 0
 * 
 * @file cm_resnet50.cpp
 * @author Stanislaw Maciag
 * @date 2018-07-19
 */

#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"

/**
 * @brief Helper function creates a chain of conv2D and fused batchnorm attached to the selected input tensor
 * 
 * @param model Master compositional model
 * @param input Tensor that is an input data for the conv2D
 * @param kernelShape Shape of conv2D kernel
 * @param stride Stride of conv2D
 * @param padding Padding of conv2D
 * @return mv::Data::TensorIterator Iterator referencing the created batchnorm output 
 */
mv::Data::TensorIterator convBatchNormBlock(mv::CompositionalModel& model, mv::Data::TensorIterator input,  mv::Shape kernelShape, std::array<unsigned short, 2> stride, std::array<unsigned short, 4> padding)
{
    
    std::vector<double> weightsData = mv::utils::generateSequence<double>(kernelShape.totalSize());

    auto weights = model.constant(weightsData, kernelShape, mv::DType("Float16"), mv::Order("HWCN"));
    auto conv = model.conv(input, weights, stride, padding, 1);

    return conv;
    //std::vector<double> biasData = mv::utils::generateSequence<double>(conv->getShape()[-1]);
    //auto biasParam = model.constant(biasData, {conv->getShape()[-1]}, mv::DType("Float16"), mv::Order("W"));
    //return model.bias(conv, biasParam);

}

/**
 * @brief Helper function that attaches a residual block to the selected input tensor
 * @param model Master compositional model
 * @param input Tensor that is an input data for first stages of residual block
 * @param intermediateDepth Number of output channels for the first convolution of the branch 2
 * @return mv::Data::TensorIterator Iterator referencing the created residual block output
 */
mv::Data::TensorIterator residualBlock(mv::CompositionalModel& model, mv::Data::TensorIterator input, unsigned intermediateDepth)
{

    auto inputShape = input->getShape();
    auto branch2a = convBatchNormBlock(model, input, {1, 1, inputShape[2], intermediateDepth}, {1, 1}, {0, 0, 0, 0});
    //branch2a = model.relu(branch2a);
    auto branch2b = convBatchNormBlock(model, branch2a, {3, 3, intermediateDepth, intermediateDepth}, {1, 1}, {1, 1, 1, 1});
    //branch2b = model.relu(branch2b);
    auto branch2c = convBatchNormBlock(model, branch2b, {1, 1, intermediateDepth, inputShape[2]}, {1, 1}, {0, 0, 0, 0});

    auto res = model.add(input, branch2c);
    return res;
    //return model.relu(res);

}

/**
 * @brief Helper function that attaches a residual block (with conv2d on branch b) to the selected input tensor
 * @param model Master compositional model
 * @param input Tensor that is an input data for first stages of residual block 
 * @param intermediateDepth Number of output channels for the first convolution of the branch 2
 * @param outputDepth Number of output channels of the block
 * @param stride Stride applied for the convolution in branch 1 and the first convolution in branch 2
 * @return mv::Data::TensorIterator Iterator referencing the created residual block output
 */
mv::Data::TensorIterator residualConvBlock(mv::CompositionalModel& model, mv::Data::TensorIterator input, unsigned intermediateDepth,
    unsigned outputDepth, std::array<unsigned short, 2> stride)
{

    auto inputShape = input->getShape();
    auto branch1 = convBatchNormBlock(model, input, {1, 1, inputShape[2], outputDepth}, stride, {0, 0, 0, 0});
    auto branch2a = convBatchNormBlock(model, input, {1, 1, inputShape[2], intermediateDepth}, stride, {0, 0, 0, 0});
    //branch2a = model.relu(branch2a);
    auto branch2b = convBatchNormBlock(model, branch2a, {3, 3, intermediateDepth, intermediateDepth}, {1, 1}, {1, 1, 1, 1});
    //branch2b = model.relu(branch2b);
    auto branch2c = convBatchNormBlock(model, branch2b, {1, 1, intermediateDepth, outputDepth}, {1, 1}, {0, 0, 0, 0});

    auto res = model.add(branch1, branch2c);
    return res;
    //return model.relu(res);

}

int main()
{

    mv::Logger::setVerboseLevel(mv::VerboseLevel::Debug);

    // Define the primary compilation unit
    mv::CompilationUnit unit("ResNet50");

    std::string descPath = mv::utils::projectRootPath() + "/config/compilation/resnet50_HW.json";
    std::ifstream compDescFile(descPath);
    // if (compDescFile.good())
    // {
    //     std::cout << "DECLARING COMPILATION UNIT with descriptor json filename: " << descPath << std::endl;
    //     unit.loadCompilationDescriptor(descPath);
    // }

    // Obtain a compositional model from the compilation unit
    mv::CompositionalModel& cm = unit.model();

    // Compose the model for ResNet50
    auto input = cm.input({50, 50, 3}, mv::DType("Float16"), mv::Order("HWC"));
    auto conv1 = convBatchNormBlock(cm, input, {7, 7, 3, 64}, {2, 2}, {3, 3, 3, 3});
    //conv1 = cm.relu(conv1);
    auto pool1 = cm.maxPool(conv1, {3, 3}, {2, 2}, {1, 1, 1, 1});
    auto res2a = residualConvBlock(cm, pool1, 64, 206, {1, 1});
    auto res2b = residualBlock(cm, res2a, 64);
    auto res2c = residualBlock(cm, res2b, 64);
    auto res3a = residualConvBlock(cm, res2c, 128, 206, {2, 2});
    auto res3b = residualBlock(cm, res3a, 128);
    auto res3c = residualBlock(cm, res3b, 128);
    auto res3d = residualBlock(cm, res3c, 128);
    auto res4a = residualConvBlock(cm, res3d, 206, 200, {2, 2});
    auto res4b = residualBlock(cm, res4a, 206);
    auto res4c = residualBlock(cm, res4b, 206);
    auto res4d = residualBlock(cm, res4c, 206);
    auto res4e = residualBlock(cm, res4d, 206);
    auto res4f = residualBlock(cm, res4e, 206);
    auto res5a = residualConvBlock(cm, res4f, 206, 512, {2, 2});
    auto res5b = residualBlock(cm, res5a, 206);
    auto res5c = residualBlock(cm, res5b, 206);
    //auto pool5 = cm.averagePool(res5c, {7, 7}, {1, 1}, {0, 0, 0, 0});
    cm.output(res5c);

    // Load target descriptor for the selected target to the compilation unit
    if (!unit.loadTargetDescriptor(mv::Target::ma2490))
        exit(1);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);

    // Initialize compilation 
    unit.initialize();
    // Run all passes
    auto result = unit.run();

    return 0;

}