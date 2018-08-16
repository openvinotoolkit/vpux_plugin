#include <stdio.h>
#include "include/mcm/deployer/blob_serialization/bConv_MX.hpp"

namespace mv
{

    void bConv2D::writeStageInfo(mv::OpModel * om, mv::Blob_buffer* b)
    {
        if (1)
        {

            int fp16_size = 2;

            mv::DataModel dm(*om);
            mv::ControlModel cm(*om);

            // Hardware
            b->AddBytes(4, this->streamingMask);
            b->AddBytes(4, this->input->getShape().totalSize()*fp16_size);
            b->AddBytes(4, this->output->getShape().totalSize()*fp16_size);
            b->AddBytes(4, this->concatOffset);
            b->AddBytes(4, this->unloadCMX);
            b->AddBytes(4, this->overwriteInput);
            b->AddBytes(4, this->CMXSize);
            b->AddBytes(4, this->reluSHVAcc);
            b->AddBytes(4, this->shvNegSlope);
            b->AddBytes(4, this->shvPosSlope);
            b->AddBytes(4, this->desc_count);

            std::cout << "Streaming Mask: " << this->streamingMask << std::endl;
            std::cout << "Total Input Size: " << this->input->getShape().totalSize()*fp16_size << std::endl;
            std::cout << "Total Output Size: " << this->output->getShape().totalSize()*fp16_size << std::endl;
            std::cout << "concatOffset: " << this->concatOffset << std::endl;
            std::cout << "unloadCMX: " << this->unloadCMX << std::endl;
            std::cout << "overwriteInput: " << this->overwriteInput << std::endl;
            std::cout << "CMXSize: " << this->CMXSize << std::endl;
            std::cout << "reluSHVAcc: " << this->reluSHVAcc << std::endl;
            std::cout << "shvNegSlope: " << this->shvNegSlope << std::endl;
            std::cout << "shvPosSlope: " << this->shvPosSlope << std::endl;
            std::cout << "Desc Count: " << this->desc_count << std::endl;


            Blob_Tensor inputBlobTensor = Blob_Tensor(&dm, &cm, &b->reloc_table, &this->input);
            Blob_Tensor outputBlobTensor = Blob_Tensor(&dm, &cm, &b->reloc_table, &this->output);
            Blob_Tensor tapsBlobTensor = Blob_Tensor(&dm, &cm, &b->reloc_table, &this->taps);

            for (unsigned i = 0; i != this->desc_count; i++)
            {


                unsigned int original_height = this->input->getShape()[1];
                unsigned int current_height;
                if (i+1 == this->desc_count){   // Last Descriptor may be an unequal height to the rest.
                    int surplus = ceil(original_height/(float)this->desc_count)*this->desc_count - original_height;
                    current_height = ceil(original_height/(float)this->desc_count) - surplus;
                }else{
                    current_height = ceil(original_height/(float)this->desc_count);
                }


                auto input_shape = this->input->getShape();
                auto output_shape = this->output->getShape();
                // this->descriptors[i].dataBaseAddr = i*0x3f0;    // TODO: Calculate 3f0 (1008)
                this->descriptors[i].dataBaseAddr = i*2*input_shape[1]*current_height;    // TODO: Calculate 3f0 (1008)
                this->descriptors[i].coeffBaseAddr = 0;
                this->descriptors[i].biasBaseAddr = 0;
                this->descriptors[i].scaleBaseAddr = 0;
                this->descriptors[i].outBaseAddr = i*2*output_shape[1]*current_height;  // TODO: Calculate 3f0 (1008)

                this->descriptors[i].dataChStr = inputBlobTensor.strideZ;
                this->descriptors[i].dataLnStr = inputBlobTensor.strideY;

                auto weight_4dshape = this->taps->getShape();
                printf("Coeff Strides: %i %i %i\n", tapsBlobTensor.strideX, tapsBlobTensor.strideY, tapsBlobTensor.strideZ);
                printf("Coeff Dims: %i %i %i\n", tapsBlobTensor.dimX, tapsBlobTensor.dimY, tapsBlobTensor.dimZ);
                std::cout << Printable::toString(weight_4dshape) << std::endl;

                this->descriptors[i].coeffChStrOut = tapsBlobTensor.strideZ /  weight_4dshape[0];
                this->descriptors[i].coeffChStrIn = weight_4dshape[4]*2;

                this->descriptors[i].outLnStr = outputBlobTensor.strideY;
                this->descriptors[i].outChStr = outputBlobTensor.strideZ;

                for(unsigned j = 0; j != 32; j++){
                    b->AddBytes(4, ((int *) &this->descriptors[i])[j]);
                }
            }

            // Blob_Tensor scaleBlobTensor = Blob_Tensor(&dm, &cm, &b->reloc_table, &this->scale);

            printf("Warning: Currently no Scale absorb support in HW Serialization\n");
            Blob_Tensor biasBlobTensor = Blob_Tensor(
                // this->output->getShape().totalSize(),   // X
                // 0x01,   // Y
                // 0x01,   // Z
                0,
                0,
                0,
                0,     // X Stride
                0,
                0,
                // fp16_size*this->output->getShape().totalSize(),    // Y Stride
                // fp16_size*this->output->getShape().totalSize(),    // z Stride
                0, // Offset - Memory Manager
                0, // Location - Memory Manager
                0,
                0
            );

            printf("Warning: Currently no Scale absorb support in HW Serialization\n");
            Blob_Tensor scaleBlobTensor = Blob_Tensor(
                // this->taps->getShape()[0]*this->taps->getShape()[1],  // X
                // this->taps->getShape()[2],   // y
                // this->taps->getShape()[3],   // z
                0,
                0,
                0,
                // fp16_size*this->taps->getShape()[2]*this->taps->getShape()[3],
                // fp16_size*this->taps->getShape()[3], // Taps Sy
                0,
                0,
                0, // SZ
                0, // Offset - Memory Manager
                0, // Location - Memory Manager
                0,
                0
            );
            b->reloc_table.push_entry(std::pair<int, bLocation>(0, bLocation::Constant ));
            b->reloc_table.push_entry(std::pair<int, bLocation>(0, bLocation::Constant ));


            inputBlobTensor.write(b);
            outputBlobTensor.write(b);
            tapsBlobTensor.write(b);
            biasBlobTensor.write(b);
            scaleBlobTensor.write(b);

        }else{
            // Software
        }
    }

    bConv2D::bConv2D(mv::ComputationOp* it)
        :
          Blob_Op_Definition(),
          input((it->getInputTensor(0))),
          output((it->getOutputTensor(0))),
          taps((it->getInputTensor(1)))
    {


        printf("Serializing a HW Conv\n");

        int cmxSize = 256*1024;
        int descriptors_count = 1;

        if (! it->hasAttr("NCE1_AssignedCMX"))
        {
            printf("WARNING: Needs Attribute 'NCE1_AssignedCMX'. Defaulting to 256*1024\n");
        }
        else
        {
            cmxSize = it->getAttr("NCE1_AssignedCMX").getContent<int>();
            printf("WARNING: Overriding attribute 'NCE1_AssignedCMX' to 256*1024\n");
            cmxSize = 256*1024;
        }

        if (! it->hasAttr("NCE1_DescriptorSplits"))
        {
            printf("WARNING: Needs Attribute 'NCE1_DescriptorSplits'. Defaulting to 1\n");
        }
        else
        {
            descriptors_count = it->getAttr("NCE1_DescriptorSplits").getContent<int>();
        }

        if (! it->hasAttr("NCE1_StreamingMask"))
        {
            printf("WARNING: Needs Attribute 'NCE1_StreamingMask'. Defaulting to 1\n");
            this->streamingMask = 1;
        }
        else
        {
            this->streamingMask = it->getAttr("NCE1_StreamingMask").getContent<int>();
        }
        if (! it->hasAttr("NCE1_Mode"))
        {
            printf("WARNING: Needs Attribute 'NCE1_Mode'. Defaulting to 0\n");
            this->opMode = 0;
        }
        else
        {
            this->opMode = it->getAttr("NCE1_Mode").getContent<int>();
        }


        if (it->hasAttr("bias"))
        {
            this->bias = it->getAttr("bias").getContent<mv::dynamic_vector<float>>();
        }
        else
        {
            this->bias = mv::dynamic_vector<float>();
        }

        this->concatOffset = 0; // Concat not supported currently
        this->unloadCMX = 0;
        this->overwriteInput = 0;

        this->CMXSize = cmxSize;
        this->reluSHVAcc = 0;
        float val = 0;
        float val2 = 1;
        this->shvNegSlope = *(int * )(&val);
        this->shvPosSlope = *(int * )(&val2);

        this->desc_count = descriptors_count;

        // this->descriptors = (cnnConvolutionPoolStructure *)malloc(128 * this->desc_count);
        this->descriptors = new cnnConvolutionPoolStructure[this->desc_count];

        int chPerRamBlock = 1;
        int topJunk = 1, bottomJunk = 1;
        int localLS = 1, localCS = 1;
        int LPC = 1;
        int minLines = 1;
        int stride = 1;
        int padEn = 1;


        if (! it->hasAttr("NCE1_InputChannelsPerRamBlock"))
        {
            printf("WARNING: Needs Attribute 'NCE1_InputChannelsPerRamBlock'. Defaulting to 1\n");
        }
        else
        {
            chPerRamBlock = it->getAttr("NCE1_InputChannelsPerRamBlock").getContent<int>();
        }

        if (! it->hasAttr("NCE1_TopOutputJunk"))
        {
            printf("WARNING: Needs Attribute 'NCE1_TopOutputJunk'. Defaulting to 1\n");
        }
        else
        {
            topJunk = it->getAttr("NCE1_TopOutputJunk").getContent<int>();
        }

        if (! it->hasAttr("NCE1_BottomOutputJunk"))
        {
            printf("WARNING: Needs Attribute 'NCE1_BottomOutputJunk'. Defaulting to 1\n");
        }
        else
        {
            bottomJunk = it->getAttr("NCE1_BottomOutputJunk").getContent<int>();
        }

        if (! it->hasAttr("NCE1_LocalLineStride"))
        {
            printf("WARNING: Needs Attribute 'NCE1_LocalLineStride'. Defaulting to 1\n");
        }
        else
        {
            localLS = it->getAttr("NCE1_LocalLineStride").getContent<int>();
        }

        auto weight_4dshape = this->taps->getShape();
        localCS = weight_4dshape[0]*2;

        // if (! it->hasAttr("NCE1_LinesPerChannel"))
        // {
        //     printf("WARNING: Needs Attribute 'NCE1_LinesPerChannel'. Defaulting to 1\n");
        // }
        // else
        // {
        //     LPC = it->getAttr("NCE1_LinesPerChannel").getContent<int>() -1;
        // }

        LPC = weight_4dshape[4];

        if (! it->hasAttr("NCE1_MinLines"))
        {
            printf("WARNING: Needs Attribute 'NCE1_MinLines'. Defaulting to 1\n");
        }
        else
        {
            minLines = it->getAttr("NCE1_MinLines").getContent<int>() -1;
        }

        if (! it->hasAttr("stride"))
        {
            printf("WARNING: Needs Attribute 'stride'. Defaulting to 1\n");
        }
        else
        {
            stride = it->getAttr("stride").getContent<mv::UnsignedVector2D>().e0;
        }

        if (! it->hasAttr("padding"))
        {
            printf("WARNING: Needs Attribute 'padding'. Defaulting to 1\n");
        }
        else
        {
            padEn = it->getAttr("padding").getContent<mv::UnsignedVector4D>().e0;
        }


        for (unsigned i = 0; i != this->desc_count; i++)
        {
            this->descriptors[i] =  cnnConvolutionPoolStructure();

            // Relations to other Descriptors
            if (i+1 == this->desc_count){
                this->descriptors[i].Line0.linkAddress = 0; // Last.
            }else{
                this->descriptors[i].Line0.linkAddress = 32*4;
            }
            printf("linkAddress: %d\n", 32*4);

            this->descriptors[i].Line0.id = 0;

            // Layer Meta Information - Layout & DataTypes
            this->descriptors[i].Line0.type = NCE1_CONV;
            this->descriptors[i].Line0.interleavedInput = 0;
            this->descriptors[i].Line0.interleavedOutput = 0;
            this->descriptors[i].Line0.cm = NCE1_DTYPE_FP16;
            this->descriptors[i].Line0.dm = NCE1_DTYPE_FP16;

            std::cout << "Taps" <<  Printable::toString(this->taps->getShape()) << std::endl;

            // Standard Fields for Convolution
            this->descriptors[i].kernelWidth = this->taps->getShape()[2] -1;
            this->descriptors[i].kernelHeight = this->taps->getShape()[3] -1;

            this->descriptors[i].chStride = stride -1;  // Stride of Kernel (Square only)

            if (padEn > 0)
            {
                this->descriptors[i].padEn = 1;
            }
            else
            {
                this->descriptors[i].padEn = 0;
            }

            this->descriptors[i].padType = 0;   // Zero Padding

            this->descriptors[i].inputWidth = this->input->getShape()[0] -1;
            unsigned int original_height = this->input->getShape()[1];
            unsigned int current_height;
            if (i+1 == this->desc_count){   // Last Descriptor may be an unequal height to the rest.
                int surplus = ceil(original_height/(float)this->desc_count)*this->desc_count - original_height;
                current_height = ceil(original_height/(float)this->desc_count) - surplus;
            }else{
                current_height = ceil(original_height/(float)this->desc_count);
            }

            this->descriptors[i].inputHeight =  current_height - 1;
            this->descriptors[i].inputChannels = this->input->getShape()[2] -1;

            this->descriptors[i].outputChannels = this->output->getShape()[2] -1;

            // std::cout << "Input Size: " << descriptors[i].inputWidth + 1 << std::endl;
            // std::cout << "Input Size: " << descriptors[i].inputHeight + 1 << std::endl;
            // std::cout << "Input Size: " << descriptors[i].inputChannels + 1 << std::endl;
            // std::cout << "Output Size: " << descriptors[i].outputChannels + 1 << std::endl;

            // Descriptor Buffers

            // Resolved at write time.

            // this->descriptors[i].dataBaseAddr = 0;
            // this->descriptors[i].dataChStr = -1;
            // this->descriptors[i].dataLnStr = -1;

            // this->descriptors[i].coeffBaseAddr = 0;
            // this->descriptors[i].coeffChStrOut = -1;
            // this->descriptors[i].coeffChStrIn = -1;

            // this->descriptors[i].outLnStr = 0;
            // this->descriptors[i].outBaseAddr = -1;
            // this->descriptors[i].outChStr = -1;

            // this->descriptors[i].biasBaseAddr = 0;
            // this->descriptors[i].scaleBaseAddr = 0;

            // Myriad X DPU Assignment & Execution Configuration
            this->descriptors[i].Line0.mode = 0 ;//this->opMode;
            this->descriptors[i].Line0.it = 0;  // Interrupt Trigger
            this->descriptors[i].Line0.disInt = 0;  // 0 - Interrupts Enabled, 1 - Interrupts disabled.


            std::cout << "chPerRamBlock:::::::::::::::::::::::::"<< chPerRamBlock << std::endl;
            this->descriptors[i].chPerRamBlock = chPerRamBlock -1;        // Input Channels per Ram Block


            // Myriad X Compensation Fields
            this->descriptors[i].topOutputJunk = topJunk;
            this->descriptors[i].bottomOutputJunk = bottomJunk;

            this->descriptors[i].localLs =  localLS;
            this->descriptors[i].localCs =  localCS - 1;
            this->descriptors[i].linesPerCh = LPC;

            this->descriptors[i].rud = 0;   // Re-Use bit

            this->descriptors[i].minLines = minLines;     // Minimum lines of data required to carry out function

            this->descriptors[i].coeffLpb = (this->descriptors[i].chPerRamBlock+1) * (this->descriptors[i].kernelWidth+1) * (this->descriptors[i].kernelHeight+1) - 1;
            this->descriptors[i].css = (this->descriptors[i].kernelWidth + 1) * (this->descriptors[i].kernelHeight + 1) -1 ;
            this->descriptors[i].outputX = this->output->getShape()[0];

            // Myriad X - Splitting groups
            this->descriptors[i].sohGroup = i;  // TODO: Looped decisions
            this->descriptors[i].sodGroup = 0;

            // Fused ReLU
            this->descriptors[i].t0 = 0;
            this->descriptors[i].a0 = 0;
            this->descriptors[i].a1 = 0;
            this->descriptors[i].reluxEn = 0;
            this->descriptors[i].reluEn = 0;

            // Fused Pooling
            if (0)
            {
                this->descriptors[i].Line0.type = NCE1_CONV_POOL;
            }
            this->descriptors[i].avgPoolX = 0;
            this->descriptors[i].poolType = 0;
            this->descriptors[i].poolEn = 0;
            this->descriptors[i].poolKernelHeight = 0;
            this->descriptors[i].poolKernelWidth = 0;

            // Reserved fields of the hw descriptor. Leave as zero or live in eternal fear.
            this->descriptors[i].Line0.rsvd1 = 0;
            this->descriptors[i].rsvd2 = 0;
            this->descriptors[i].rsvd3 = 0;
            this->descriptors[i].rsvd4 = 0;
            this->descriptors[i].rsvd5 = 0;
            this->descriptors[i].rsvd6 = 0;
            this->descriptors[i].rsvd7 = 0;
            this->descriptors[i].rsvd9 = 0;
            this->descriptors[i].rsvd10 = 0;
            this->descriptors[i].rsvd13 = 0;
            this->descriptors[i].rsvd8 = 0;

            // Palette for Weights Lookup (Currently Unsupported).
            this->descriptors[i].p0 = 0;
            this->descriptors[i].p1 = 0;
            this->descriptors[i].p2 = 0;
            this->descriptors[i].p3 = 0;
            this->descriptors[i].p4 = 0;
            this->descriptors[i].p5 = 0;
            this->descriptors[i].p6 = 0;
            this->descriptors[i].p7 = 0;
            this->descriptors[i].p8 = 0;
            this->descriptors[i].p9 = 0;
            this->descriptors[i].p10 = 0;
            this->descriptors[i].p11 = 0;
            this->descriptors[i].p12 = 0;
            this->descriptors[i].p13 = 0;
            this->descriptors[i].p14 = 0;
            this->descriptors[i].p15 = 0;
        }
    }

}
