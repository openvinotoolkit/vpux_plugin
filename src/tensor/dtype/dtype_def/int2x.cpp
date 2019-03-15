#include "include/mcm/tensor/dtype/dtype_registry.hpp"
#include "include/mcm/tensor/dtype/dtype.hpp"

namespace mv
{

    static std::function<BinaryData(const std::vector<DataElement>&)> toBinaryFunc =
    [](const std::vector<DataElement> & vals)->mv::BinaryData
    {
        (void) vals;
        throw DTypeError("DType", "conversion for Int2X is not supported yet");
    };

    MV_REGISTER_DTYPE(Int2X)
    .setToBinaryFunc(toBinaryFunc)
    .setIsDoubleType(false);
}
