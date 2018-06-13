#ifndef BATCH_NORM_HPP_
#define BATCH_NORM_HPP_

#include "include/fathom/computation/op/source_op.hpp"
#include "include/fathom/computation/op/sink_op.hpp"


namespace mv
{
    namespace Op
    {

        class BatchNorm : public SourceOp, public SinkOp
        {

        public:

            BatchNorm(float_type varianceEps, const string &name) :
            ComputationOp(OpType::BatchNorm, name),
            SourceOp(OpType::BatchNorm, 1, name),
            SinkOp(OpType::BatchNorm, 5, name)
            {
                addAttr("varianceEps", AttrType::FloatType, varianceEps);
                addAttr("executable", AttrType::BoolType, true);
            }

            Tensor getOutputDef(byte_type idx)
            {
                
                if (idx > 0)
                    return Tensor();

                if (!validOutputDef_())
                    return Tensor();

                auto input = getInput(0);
                auto inputShape = input->getShape(); 

                auto mean = getInput(1);
                auto meanShape = mean->getShape();

                auto variance = getInput(2);
                auto varianceShape = variance->getShape();

                auto offset = getInput(3);
                auto offsetShape = offset->getShape();

                auto scale = getInput(4);
                auto scaleShape = scale->getShape();

                if (!(inputShape == meanShape && inputShape == varianceShape && inputShape == offsetShape && inputShape == scaleShape))
                {

                    if ((meanShape.ndims() != 1 || varianceShape.ndims() != 1 || offsetShape.ndims() != 1 || scaleShape.ndims() != 1) ||
                        (meanShape[0] != inputShape[-1] || varianceShape[0] != inputShape[-1] || offsetShape[0] != inputShape[-1] || scaleShape[0] != inputShape[-1]))
                    {
                        logger_.log(Logger::MessageType::MessageError, "Unable to define output tensor for '" + name_ + 
                            "' because of incorrect shape of mean (" + meanShape.toString() + ") or variance (" + varianceShape.toString() +
                            ") or offset (" + offsetShape.toString() + ") or scale (" + scaleShape.toString() + ") - they need to be either"
                            " equal to shape of the input (" + inputShape.toString() + ") or to be one dimensional tensors of dimension " +
                            Printable::toString(inputShape[-1]));
                        return Tensor();
                    }

                }

                return Tensor(name_ + ":0", inputShape, input->getDType(), input->getOrder());
                
            }

        };

    }

}

#endif // BATCH_NORM_HPP_