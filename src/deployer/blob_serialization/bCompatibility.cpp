#include <stdio.h>
#include "include/mcm/deployer/blob_serialization/bCompatibility.hpp"

namespace mv
{
    void bCompatibility::writeStageInfo(mv::OpModel * om, Blob_buffer* b)
    {
        int fp16_size = 2;

        mv::DataModel dm(*om);
        mv::ControlModel cm(*om);

        Blob_Tensor inputBlobTensor = Blob_Tensor(&dm, &cm, &b->reloc_table, &this->input);
        Blob_Tensor outputBlobTensor = Blob_Tensor(&dm, &cm, &b->reloc_table, &this->output);

        inputBlobTensor.write(b);
        outputBlobTensor.write(b);

    }

    bCompatibility::bCompatibility(mv::ComputationOp* it)
        :
          Blob_Op_Definition(),
          input((it->getInputTensor(0))),
          output((it->getOutputTensor(0)))
    {
    }

}
