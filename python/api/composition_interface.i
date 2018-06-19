
/// SWIG Interface File.

// Include the external numpy swig bindings.
%{
#define SWIG_FILE_WITH_INIT
%}
%include "numpy.i"
%init %{
import_array();
%}

%module composition_api
%{
    #include <include/mcm/computation/model/op_model.hpp>
    #include <include/mcm/computation/model/control_model.hpp>
    #include <include/mcm/computation/model/iterator/data_context.hpp>
    #include <include/mcm/deployer/serializer.hpp>
    #include <string>

    int serialize(mv::OpModel * test_cm){
        mv::ControlModel *cm = new mv::ControlModel(*test_cm);
        mv::Serializer *gs = new mv::Serializer(mv::mvblob_mode);
        uint64_t filesize = gs->serialize(*cm, "cpp.blob");

        return filesize;   // Success, return filesize
    }

    int testSWIG(){
        /// A simple test to ensure the connection between Python and C++ is working
        int test = 1;
        return test;
    }

    mv::OpModel * getOM(){
        /// Get a blank OpModel
        mv::OpModel *om = new mv::OpModel();
        return om;
    }

    mv::UnsignedVector2D * get2DVector(int x, int y){
        mv::UnsignedVector2D *a = new mv::UnsignedVector2D();
        a->e0 = x;
        a->e1 = y;
        return a;
    }


    mv::UnsignedVector4D * get4DVector(int w, int x, int y, int z){
        mv::UnsignedVector4D *a = new mv::UnsignedVector4D();
        a->e0 = w;
        a->e1 = x;
        a->e2 = y;
        a->e3 = z;
        return a;
    }


    mv::Shape * getShape(int x, int y){
        /// Create a c++ shape object from a passed in set of dimension sizes
        mv::Shape* a = new mv::Shape(x, y);
        return a;
    }

    mv::Shape * getShape(int x, int y, int z){
        /// Create a c++ shape object from a passed in set of dimension sizes
        mv::Shape* a = new mv::Shape(x, y, z);
        return a;
    }

    mv::Shape * getShape(int b, int x, int y, int z){
        /// Create a c++ shape object from a passed in set of dimension sizes
        mv::Shape* a = new mv::Shape(b, x, y, z);
        return a;
    }

    mv::dynamic_vector<mv::float_type> * getData(float * d, size_t len){
        /// Populate a Vector with a numpy array.
        mv::dynamic_vector<mv::float_type> * weightsData = new mv::dynamic_vector<mv::float_type>(d, len);
        return weightsData;
    }

    int testConv(
        mv::Data::OpListIterator &target,
        unsigned exp_strideX,
        unsigned exp_strideY,
        unsigned exp_padX,
        unsigned exp_padY
        ){
        /// A couple of simple checks to ensure we have loaded the items correctly.

        int ret_val = 0;    // Success
        mv::UnsignedVector2D stride = target->getAttr("stride").getContent<mv::UnsignedVector2D>();
        mv::UnsignedVector4D pad = target->getAttr("padding").getContent<mv::UnsignedVector4D>();
        if(stride.e0 != exp_strideX)
            ret_val = 1;
        if(stride.e1 != exp_strideY)
            ret_val = 2;
        if(pad.e1 != exp_padX)
            ret_val = 3;
        if(pad.e3 != exp_padY)
            ret_val = 4;
        // TODO Consider assymetric padding

        return ret_val;
    }

    mv::Data::TensorIterator input(mv::OpModel *o, const mv::Shape &shape){
        /// Add an Input Layer to the OpModel and return the relevant iterator
        return o->input(shape, mv::DType::Float, mv::Order::NWHC);
    }

    mv::Data::TensorIterator output(mv::OpModel *o, mv::Data::TensorIterator input){
        /// Add an Output Layer to the OpModel and return the relevant iterator
        return o->output(input);
    }

    mv::Data::TensorIterator maxpool2D(mv::OpModel *o, mv::Data::TensorIterator input, unsigned kernelSizeX,
        unsigned kernelSizeY, unsigned strideX, unsigned strideY, unsigned padX, unsigned padY){
        /// Add a Max Pooling Layer to the OpModel and return the relevant iterator
        return o->maxpool2D(input, {kernelSizeX, kernelSizeY}, {strideX, strideY},
            {padX, padX, padY, padY});
    }

    mv::Data::TensorIterator concat(mv::OpModel *o, mv::Data::TensorIterator input0, mv::Data::TensorIterator input1){
        /// Add a Concat Layer to the OpModel and return the relevant iterator.
        /// Allows only two inputs at a time. More must cascade
        return o->concat(input0, input1);
    }

    mv::Data::TensorIterator conv2D(mv::OpModel *o, mv::Data::TensorIterator input, mv::Data::TensorIterator filters,
        unsigned strideX, unsigned strideY, unsigned padX, unsigned padY){
        /// Add a Convolutional Layer to the OpModel and return the relevant iterator
        return o->conv2D(input, filters, {strideX, strideY}, {padX, padX, padY, padY});
    }

    mv::Data::TensorIterator constant(mv::OpModel *o, const mv::dynamic_vector<mv::float_type>& data, const mv::Shape &shape){
        /// Add a Constant Layer to the OpModel and return the relevant iterator
        return o->constant(data, shape, mv::DType::Float, mv::Order::NWHC);
    }

    mv::Data::OpListIterator getSourceOp(mv::OpModel *o, mv::Data::TensorIterator tensor){
        // Get source operation of a tensor
        return o->getSourceOp(tensor);
    }

    mv::Data::TensorIterator fullyConnected(mv::OpModel *o, mv::Data::TensorIterator input, mv::Data::TensorIterator weights){
        return o->fullyConnected(input, weights);
    }

    mv::Data::TensorIterator avgpool2D(mv::OpModel *o, mv::Data::TensorIterator input, mv::UnsignedVector2D kernelSize, mv::UnsignedVector2D stride, mv::UnsignedVector4D padding){
        return o->avgpool2D(input, kernelSize, stride, padding);
    }

    mv::Data::TensorIterator batchNorm(mv::OpModel *o,mv::Data::TensorIterator input, mv::Data::TensorIterator mean, mv::Data::TensorIterator variance, mv::Data::TensorIterator offset, mv::Data::TensorIterator scale, mv::float_type varianceEps){
        return o->batchNorm(input, mean, variance, offset, scale, varianceEps);
    }
    mv::Data::TensorIterator scale(mv::OpModel *o,mv::Data::TensorIterator input, mv::Data::TensorIterator scale){
        return o->scale(input, scale);
    }
    mv::Data::TensorIterator relu(mv::OpModel *o,mv::Data::TensorIterator input){
        return o->relu(input);
    }
    mv::Data::TensorIterator softmax(mv::OpModel *o,mv::Data::TensorIterator input){
        return o->softmax(input);
    }
    mv::Data::TensorIterator add(mv::OpModel *o,mv::Data::TensorIterator input0, mv::Data::TensorIterator input1){
        return o->add(input0, input1);
    }
    mv::Data::TensorIterator subtract(mv::OpModel *o,mv::Data::TensorIterator input0, mv::Data::TensorIterator input1){
        return o->subtract(input0, input1);
    }
    mv::Data::TensorIterator multiply(mv::OpModel *o,mv::Data::TensorIterator input0, mv::Data::TensorIterator input1){
        return o->multiply(input0, input1);
    }
    mv::Data::TensorIterator divide(mv::OpModel *o,mv::Data::TensorIterator input0, mv::Data::TensorIterator input1){
        return o->divide(input0, input1);
    }
    mv::Data::TensorIterator reshape(mv::OpModel *o,mv::Data::TensorIterator input, const mv::Shape& shape){
        return o->reshape(input, shape);
    }
 %}

#include <include/mcm/computation/model/op_model.hpp>
#include <include/mcm/computation/model/model.hpp>
#include <include/mcm/computation/model/iterator/model_iterator.hpp>
#include <include/mcm/computation/model/iterator/data_context.hpp>
#include <include/mcm/computation/model/control_model.hpp>
#include <include/mcm/computation/tensor/shape.hpp>
#include <include/mcm/computation/tensor/tensor.hpp>
#include <include/mcm/computation/model/attribute.hpp>
#include <include/mcm/api/compositional_model.hpp>
#include <include/mcm/deployer/serializer.hpp>
#include <include/mcm/computation/model/model.hpp>
#include <include/mcm/computation/op/def/input.hpp>
#include <include/mcm/computation/op/def/output.hpp>
#include <include/mcm/computation/op/def/conv.hpp>
#include <include/mcm/computation/op/def/maxpool.hpp>
#include <include/mcm/computation/op/def/concat.hpp>
#include <string>

// The section below is exposing the functions within the included files,
// or the ones defined above in the module.

namespace mv
{

    class OpModel
    {
    public:
        bool isValid() const;
    };

    namespace Data
    {
        class TensorIterator
        {
        public:
            ~TensorIterator();
        };

        class OpListIterator
        {
        public:
            ~OpListIterator();
        };

    }
}

int testSWIG();
mv::OpModel * getOM();
mv::Shape * getShape(int x, int y);
mv::Shape * getShape(int x, int y, int z);
mv::Shape * getShape(int b, int x, int y, int z);
mv::UnsignedVector2D * get2DVector(int x, int y);
mv::UnsignedVector4D * get4DVector(int w, int x, int y, int z);

// Expand a numpy array to a data pointer and a length
%include "stdint.i"
%apply (float* INPLACE_ARRAY1, int DIM1) {(float* d, int len)}
mv::dynamic_vector<mv::float_type> * getData(float * d, int len);


mv::Data::TensorIterator input(mv::OpModel * o, const mv::Shape &shape);
mv::Data::TensorIterator output(mv::OpModel * o, mv::Data::TensorIterator input);
mv::Data::TensorIterator conv2D(mv::OpModel * o, mv::Data::TensorIterator input, mv::Data::TensorIterator filters,
    unsigned strideX, unsigned strideY, unsigned padX, unsigned padY);
mv::Data::TensorIterator maxpool2D(mv::OpModel * o, mv::Data::TensorIterator input, unsigned kernelSizeX,
    unsigned kernelSizeY, unsigned strideX, unsigned strideY, unsigned padX, unsigned padY);
mv::Data::TensorIterator concat(mv::OpModel * o, mv::Data::TensorIterator input0, mv::Data::TensorIterator input1);
mv::Data::OpListIterator getSourceOp(mv::OpModel *o, mv::Data::TensorIterator tensor);

mv::Data::TensorIterator fullyConnected(mv::OpModel *o, mv::Data::TensorIterator input, mv::Data::TensorIterator weights);
mv::Data::TensorIterator avgpool2D(mv::OpModel *o, mv::Data::TensorIterator input, mv::UnsignedVector2D kernelSize, mv::UnsignedVector2D stride, mv::UnsignedVector4D padding);
mv::Data::TensorIterator batchNorm(mv::OpModel *o,mv::Data::TensorIterator input, mv::Data::TensorIterator mean, mv::Data::TensorIterator variance, mv::Data::TensorIterator offset, mv::Data::TensorIterator scale, float varianceEps);
mv::Data::TensorIterator scale(mv::OpModel *o,mv::Data::TensorIterator input, mv::Data::TensorIterator scale);
mv::Data::TensorIterator relu(mv::OpModel *o,mv::Data::TensorIterator input);
mv::Data::TensorIterator softmax(mv::OpModel *o,mv::Data::TensorIterator input);
mv::Data::TensorIterator add(mv::OpModel *o,mv::Data::TensorIterator input0, mv::Data::TensorIterator input1);
mv::Data::TensorIterator subtract(mv::OpModel *o,mv::Data::TensorIterator input0, mv::Data::TensorIterator input1);
mv::Data::TensorIterator multiply(mv::OpModel *o,mv::Data::TensorIterator input0, mv::Data::TensorIterator input1);
mv::Data::TensorIterator divide(mv::OpModel *o,mv::Data::TensorIterator input0, mv::Data::TensorIterator input1);
mv::Data::TensorIterator reshape(mv::OpModel *o,mv::Data::TensorIterator input, const mv::Shape& shape);

mv::Data::TensorIterator constant(mv::OpModel * o, const mv::dynamic_vector<mv::float_type>& data, const mv::Shape &shape);


int testConv(
    mv::Data::OpListIterator &target,
    int exp_strideX,
    int exp_strideY,
    int exp_padX,
    int exp_padY
);

int serialize(mv::OpModel * test_cm);