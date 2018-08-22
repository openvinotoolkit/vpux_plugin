#include <stdio.h>
#include "include/mcm/deployer/blob_serialization/bTensor.hpp"
#include <string.h>

#define BLOB_INPUT_LOCATION 1
#define BLOB_OUTPUT_LOCATION 2
#define BLOB_INTERNAL_LOCATION 3
#define BLOB_EXTERNAL_LOCATION 4

namespace mv
{
    void Blob_Tensor::write(WBuffer* b)
    {
        b->AddBytes(4, this->dimX);
        b->AddBytes(4, this->dimY);
        b->AddBytes(4, this->dimZ);
        b->AddBytes(4, this->strideX);
        b->AddBytes(4, this->strideY);
        b->AddBytes(4, this->strideZ);
        b->AddBytes(4, this->offset);
        b->AddBytes(4, this->location);
        b->AddBytes(4, this->dataType);
        b->AddBytes(4, this->order);
    }

    Blob_Tensor::Blob_Tensor(int x, int y, int z,
        int sx, int sy, int sz,
        int offsetParam, int locationParam,
        int dtype, int orderParam)
        : dimX(x),
          dimY(y),
          dimZ(z),
          strideX(sx),
          strideY(sy),
          strideZ(sz),
          offset(offsetParam),
          location(locationParam),
          dataType(dtype),
          order(orderParam)
    {
        // DEPRECIATED.
    }

    Blob_Tensor::Blob_Tensor(mv::OpModel* om, RelocationTable* rt , mv::dynamic_vector<mv::float_type>* biasVec)
        : dimX(biasVec->size()),
          dataType(0)
    {
        // TODO: This should be inside the allocator, and this function should not exist.

        if (this->dimX > 0)
        {
            this->strideX = 2;
            this->dimY = 1; this->strideY = 2*biasVec->size();
            this->dimZ = 1; this->strideZ = 2*biasVec->size();
            mv::Data::TensorIterator biasTensor = om->constant(*biasVec, mv::Shape(1, this->dimX, 1, 1), mv::DType::Float, mv::Order::ColumnMajor);

            mv::ControlModel cm(*om);
            mv::DataModel dm(*om);

            auto stageIt = cm.getStage(0);
            mv::Data::BufferIterator mem = dm.allocateTensor("ConstantMemory", stageIt, biasTensor);

            std::cout << "## Allocated Bias Buffer. Offset:  " << mem->offset << std::endl;
            int rt_entry = rt->push_entry(std::pair<int, bLocation>(mem->offset, bLocation::Constant ));
            std::cout << "## Pushed Bias Relocation entry: " << rt_entry << std::endl;
            this->offset = rt_entry;
            this->location = BLOB_INTERNAL_LOCATION;
            this->order = 1;
        }
        else
        {
            // biasTensor = om->constant({0,0,0,0}, mv::Shape(1, 1, 1, 1), mv::DType::Float, mv::Order::ColumnMajor);
            this->dimY = 0;
            this->dimZ = 0;
            this->strideX = 0;
            this->strideY = 0;
            this->strideZ = 0;

            this->offset = 0 ;
            this->location = 0;
            this->order = 0;
        }

    }


    Blob_Tensor::Blob_Tensor(mv::DataModel* dm, mv::ControlModel* cm, RelocationTable * rt , mv::Data::TensorIterator* t){

        int fp16_size = 2;
        this->dataType = 0;


        // std::cout << "Tensor:" << (*t)->getName() << "Layout: " << Printable::toString((*t)->getOrder()) << "Shape: " << Printable::toString((*t)->getShape()) <<  std::endl;

        switch((int)(*t)->getShape().ndims()){
            case 5:
            {
                // Hardware Weights
                this->dimX = (*t)->getShape()[0] * (*t)->getShape()[4];
                this->dimY = (*t)->getShape()[1];
                this->dimZ = (*t)->getShape()[2] * (*t)->getShape()[3];
            }
            break;
            case 4:
            {
                // Most Software Weights
                this->dimZ = (*t)->getShape()[3];
                this->dimY = (*t)->getShape()[2];
                this->dimX = (*t)->getShape()[0] * (*t)->getShape()[1];
            }
            break;
            case 3:
            {
                // I/O
                this->dimX = (*t)->getShape()[0];
                this->dimY = (*t)->getShape()[1];
                this->dimZ = (*t)->getShape()[2];
            }
            break;
            case 2:
            {
                this->dimX = 1;
                this->dimY = 1;
                this->dimZ = (*t)->getShape()[1];
            }
            break;
            default:
            {
                std::cout << "Serialization Error: Shape of Tensor not supported in graphFile serializer" << std::endl;
                assert(0);
            }

        }


        try{
            if (!dm->hasAllocator("ConstantMemory") || !dm->hasAllocator("IntermediateMemory"))
                assert(0);
        }catch(mv::ArgumentError){
            printf("Warning: No Intermediary Buffers\n");
        }
        // if (!dm->hasAllocator("BSS"))
        //     assert(0);

        Data::BufferIterator mem;
        mv::Control::StageIterator stg = cm->getStage(0);

        int blk_stride = 0;
        int block = 0;

        if ((*t)->isPopulated()){
            // std::cout << "Populated Tensor: " << (*t)->getName() << std::endl;

            mem = dm->getBuffer("ConstantMemory", stg, *t);
            this->location = BLOB_INTERNAL_LOCATION;

            blk_stride = (int)mem->stride;
            block = (int)mem->block;

            int rt_entry = rt->push_entry(std::pair<int, bLocation>(mem->offset, bLocation::Constant ));
            std::cout << "## Pushed Taps Relocation entry: " << rt_entry << std::endl;

            this->offset = rt_entry;
        }
        else
        {

            mv::OpModel om(*cm);

            // std::cout << "UnPopulated Tensor: " << (*t)->getName() << std::endl;

            int no_buffers = 0;
            try{
                mem = dm->getBuffer("IntermediateMemory", stg, *t);
            }catch(mv::ArgumentError){
                printf("Warning: No Intermediary Buffers\n");
                no_buffers = 1;
            }

            if (no_buffers || mem == dm->bufferEnd("IntermediateMemory", stg) ){//&& !hack_activated){

                // Not Found - In or Output
                std::vector<mv::string> input_names, output_names;

                for(auto opIterator = om.opBegin(); opIterator != om.opEnd(); ++opIterator)
                {
                    if (opIterator->getOpType() == OpType::Input){
                        auto b = opIterator->getOutputTensor(0)->getName();
                        input_names.push_back(b);
                    }else if(opIterator->getOpType() == OpType::Output){
                        auto b = opIterator->getInputTensor(0)->getName();
                        output_names.push_back(b);
                    }
                }

                if(std::find(input_names.begin(), input_names.end(), (*t)->getName()) != input_names.end()) {
                    // std::cout  << "Network Input. Note: IO Offset not supported by serializer" << std::endl;
                    this->location = BLOB_INPUT_LOCATION;
                    this->offset = 0;
                }else{
                    if(std::find(output_names.begin(), output_names.end(), (*t)->getName()) != output_names.end()) {
                        // std::cout  << "Network Output. Note: IO Offset not supported by serializer" << std::endl;
                        this->location = BLOB_OUTPUT_LOCATION;
                        this->offset = 0;
                    }else{
                        // std::cout << "Serialization Error: Tensor Position not resolved" << std::endl;
                        assert(0);
                    }
                }
            }else{
                // Found
                this->location = BLOB_EXTERNAL_LOCATION;
                blk_stride = (int)mem->stride;
                block = (int)mem->block;
                int rt_entry = rt->push_entry(std::pair<int, bLocation>(mem->offset, bLocation::Variable ));
                this->offset = rt_entry;
            }
        }

        int striding_axis = 0;
        if (block == 0){
            std::cout << "Warning: Zero-Storage Tensor." << std::endl;
            striding_axis = 0;
        }else if (block == fp16_size){
            // X
            striding_axis = 0;
        }else if(block == this->dimX){
            // Y
            striding_axis = 1;
        }else if(block == this->dimX*this->dimY){
            // Z
            striding_axis = 2;
        }else if(block == this->dimX*this->dimY*this->dimZ){
            // N
            striding_axis = 3;
        }else{
            std::cout << block << ", " << this->dimX*this->dimY*this->dimZ << std::endl;
            std::cout << "Serialization Error: Unknown mapping of memory block to mvTensor notations" << std::endl;
            assert(0);
        }

        switch((*t)->getOrder()){
            case Order::RowMajor:
                // UPA Shave
                this->order = 0;
                // printf("ROW MAJOR\n");
                this->strideZ = (striding_axis == 0 && blk_stride != 0)? blk_stride:fp16_size;
                this->strideX = (striding_axis == 0 && blk_stride != 0)? blk_stride:this->dimZ*this->strideZ;
                this->strideY = (striding_axis == 0 && blk_stride != 0)? blk_stride:this->dimX*this->strideX;
                break;
            case Order::Planar:
                // NCE1 - Option 1
                // printf("PLANAR\n");
                this->order = 1;
                this->strideX = (striding_axis == 0 && blk_stride != 0)? blk_stride:fp16_size;
                this->strideY = (striding_axis == 0 && blk_stride != 0)? blk_stride:this->dimX*this->strideX;
                this->strideZ = (striding_axis == 0 && blk_stride != 0)? blk_stride:this->dimY*this->strideY;
                break;
            case Order::ColumnMajor:
                // NCE1 - Option 2
                // printf("Column MAJOR\n");
                this->order = 2;
                this->strideX = (striding_axis == 0 && blk_stride != 0)? blk_stride:fp16_size;
                this->strideZ = (striding_axis == 0 && blk_stride != 0)? blk_stride:this->dimX*this->strideX;
                this->strideY = (striding_axis == 0 && blk_stride != 0)? blk_stride:this->dimZ*this->strideZ;
                break;
            case Order::RowMajorAlt:
                this->order = 3;
                this->strideZ = (striding_axis == 0 && blk_stride != 0)? blk_stride:fp16_size;
                this->strideY = (striding_axis == 0 && blk_stride != 0)? blk_stride:this->dimZ*this->strideZ;
                this->strideX = (striding_axis == 0 && blk_stride != 0)? blk_stride:this->dimY*this->strideY;
                break;

            default:
                std::cout << "Serialization Error: Order of Tensor not supported" << std::endl;
                assert(0);
        }
    }
}
